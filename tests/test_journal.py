"""Journal durability, locking, and recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from memory.journal import (
    DEFAULT_LOCK_TIMEOUT,
    Journal,
    LockTimeout,
    OpKind,
    Step,
    atomic_write,
    hash_file,
    new_op_id,
)
from memory.paths import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    p = Paths(tmp_path)
    p.ensure_tree()
    return p


@pytest.fixture
def journal(paths: Paths) -> Journal:
    return Journal(paths, lock_timeout=2.0)


def write_memory_file(paths: Paths, name: str, content: str) -> Path:
    target = paths.type_dir("fact") / name
    target.write_text(content, encoding="utf-8")
    return target


class TestAtomicWrite:
    def test_writes_content(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b.txt"
        atomic_write(target, b"hello")
        assert target.read_bytes() == b"hello"

    def test_leaves_no_temp_files(self, tmp_path: Path) -> None:
        atomic_write(tmp_path / "x.txt", b"data")
        assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "x.txt"
        target.write_bytes(b"old")
        atomic_write(target, b"new")
        assert target.read_bytes() == b"new"

    def test_original_survives_a_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-write must leave the previous content byte-identical."""
        target = tmp_path / "x.txt"
        target.write_bytes(b"original")

        real_replace = os.replace

        def boom(src: object, dst: object) -> None:
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="simulated crash"):
            atomic_write(target, b"replacement")
        monkeypatch.setattr(os, "replace", real_replace)

        assert target.read_bytes() == b"original"
        assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]

    def test_temp_file_is_in_same_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-directory temp keeps os.replace a rename, not a copy."""
        seen: list[Path] = []
        real_replace = os.replace

        def spy(src: object, dst: object) -> None:
            seen.append(Path(str(src)))
            real_replace(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "replace", spy)
        target = tmp_path / "sub" / "x.txt"
        atomic_write(target, b"d")
        assert seen and seen[0].parent == target.parent

    def test_empty_payload(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.txt"
        atomic_write(target, b"")
        assert target.exists() and target.read_bytes() == b""


class TestOpId:
    def test_is_sortable_across_time(self) -> None:
        """Ids carry a millisecond prefix, so they order by creation time."""
        import time as _time

        ids = []
        for _ in range(5):
            ids.append(new_op_id())
            _time.sleep(0.002)
        assert ids == sorted(ids)

    def test_shares_prefix_within_a_millisecond(self) -> None:
        batch = [new_op_id() for _ in range(5)]
        assert len({i[:12] for i in batch}) <= 2

    def test_is_unique(self) -> None:
        assert len({new_op_id() for _ in range(1000)}) == 1000


class TestHashFile:
    def test_missing_file_is_none(self, tmp_path: Path) -> None:
        assert hash_file(tmp_path / "nope") is None

    def test_stable_for_same_content(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"same")
        b.write_bytes(b"same")
        assert hash_file(a) == hash_file(b)

    def test_differs_for_different_content(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        assert hash_file(a) != hash_file(b)


class TestLocking:
    def test_requires_lock_to_write(self, journal: Journal) -> None:
        from memory.journal import JournalError

        with pytest.raises(JournalError, match="require the write lock"):
            journal.begin(OpKind.CREATE_MEMORY, ["x"], [])

    def test_acquire_release(self, journal: Journal) -> None:
        assert not journal.locked
        with journal:
            assert journal.locked
        assert not journal.locked

    def test_is_reentrant(self, journal: Journal) -> None:
        """Nested operations (merge -> per-memory write) must not deadlock."""
        with journal:
            with journal:
                assert journal.locked
            assert journal.locked
        assert not journal.locked

    def test_second_process_times_out(self, paths: Paths) -> None:
        holder = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
            from memory.journal import Journal
            from memory.paths import Paths
            j = Journal(Paths({str(paths.root)!r}))
            j.acquire()
            print("locked", flush=True)
            time.sleep(10)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "locked"
            contender = Journal(paths, lock_timeout=0.3)
            with pytest.raises(LockTimeout, match="another process holds"):
                contender.acquire()
        finally:
            proc.kill()
            proc.wait(timeout=5)
            if proc.stdout is not None:
                proc.stdout.close()

    def test_default_timeout_is_finite(self) -> None:
        assert 0 < DEFAULT_LOCK_TIMEOUT < 120


class TestSequence:
    def test_strictly_increases(self, journal: Journal) -> None:
        with journal:
            seqs = []
            for _ in range(5):
                op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
                journal.commit(op)
                seqs.extend(r.seq for r in journal.show(op))
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    def test_survives_a_lost_counter_file(self, journal: Journal, paths: Paths) -> None:
        """A missing seq file must not restart numbering and collide."""
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
            before = max(r.seq for r in journal.show(op))
            paths.seq_file.unlink()
            op2 = journal.begin(OpKind.CREATE_MEMORY, ["y"], [])
        assert max(r.seq for r in journal.show(op2)) > before

    def test_increases_across_processes(self, paths: Paths) -> None:
        with Journal(paths) as j1:
            op = j1.begin(OpKind.CREATE_MEMORY, ["x"], [])
            j1.commit(op)
            first = max(r.seq for r in j1.show(op))
        with Journal(paths) as j2:
            op2 = j2.begin(OpKind.CREATE_MEMORY, ["y"], [])
            j2.commit(op2)
            second = max(r.seq for r in j2.show(op2))
        assert second > first


class TestOpenOps:
    def test_begin_without_commit_is_open(self, journal: Journal) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            assert [r.op_id for r in journal.open_ops()] == [op]

    def test_committed_op_is_not_open(self, journal: Journal) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
            assert journal.open_ops() == []

    def test_aborted_op_is_not_open(self, journal: Journal) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.abort(op, "changed my mind")
            assert journal.open_ops() == []

    def test_ordered_by_seq(self, journal: Journal) -> None:
        with journal:
            ops = [journal.begin(OpKind.CREATE_MEMORY, [str(i)], []) for i in range(3)]
            assert [r.op_id for r in journal.open_ops()] == ops


class TestCorruption:
    def test_torn_trailing_line_is_dropped(self, journal: Journal, paths: Paths) -> None:
        """A process killed mid-append leaves a partial line; prior records survive."""
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
        log = next(paths.journal_dir.glob("*.jsonl"))
        with open(log, "a", encoding="utf-8") as fh:
            fh.write('{"op_id":"partial","seq":99,"kind":"create_mem')

        with journal:
            report = journal.recover()
        assert report.torn_lines == 1
        assert journal.show(op)  # earlier records still readable

    def test_bad_checksum_is_dropped(self, journal: Journal, paths: Paths) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
        log = next(paths.journal_dir.glob("*.jsonl"))
        lines = log.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[0])
        payload["targets"] = ["tampered"]
        lines[0] = json.dumps(payload, separators=(",", ":"))
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with journal:
            report = journal.recover()
        assert report.dropped_records == 1
        assert any("corrupt" in w for w in report.warnings)

    def test_unparseable_line_is_dropped(self, journal: Journal, paths: Paths) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
        log = next(paths.journal_dir.glob("*.jsonl"))
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("this is not json at all\n")
        with journal:
            assert journal.recover().dropped_records == 1


class TestRecovery:
    def test_clean_tree_reports_nothing(self, journal: Journal) -> None:
        with journal:
            assert journal.recover().clean

    def test_rolls_back_an_untouched_create(self, journal: Journal, paths: Paths) -> None:
        """No step done -> roll back and remove temp residue."""
        target = paths.type_dir("fact") / "new.md"
        with journal:
            op = journal.begin(
                OpKind.CREATE_MEMORY,
                ["mid"],
                [Step(i=0, act="write", path=paths.rel(target), new_hash="sha256:" + "0" * 64)],
            )
            report = journal.recover()
        assert report.rolled_back == [op]
        assert not target.exists()
        assert journal.open_ops() == []

    def test_rolls_forward_a_completed_create(self, journal: Journal, paths: Paths) -> None:
        target = write_memory_file(paths, "done.md", "content")
        with journal:
            op = journal.begin(
                OpKind.CREATE_MEMORY,
                ["mid"],
                [Step(i=0, act="write", path=paths.rel(target), new_hash=hash_file(target))],
                db={"reindex": ["mid"]},
            )
            report = journal.recover()
        assert report.rolled_forward == [op]
        assert target.read_text(encoding="utf-8") == "content"
        assert report.reindex == ["mid"]

    def test_removes_created_file_with_no_pre_image(self, journal: Journal, paths: Paths) -> None:
        """A half-created file that never existed before is removed on rollback."""
        target = write_memory_file(paths, "half.md", "partial")
        with journal:
            op = journal.begin(
                OpKind.CREATE_MEMORY,
                ["mid"],
                [
                    Step(i=0, act="write", path=paths.rel(target), new_hash=hash_file(target)),
                    Step(
                        i=1,
                        act="write",
                        path=paths.rel(paths.type_dir("fact") / "second.md"),
                        new_hash="sha256:" + "1" * 64,
                    ),
                ],
            )
            report = journal.recover()
        assert report.rolled_back == [op]
        assert not target.exists()

    def test_restores_pre_image_on_rollback(self, journal: Journal, paths: Paths) -> None:
        """An update crashed mid-replace must restore byte-identical content."""
        target = write_memory_file(paths, "update.md", "original content")
        original_hash = hash_file(target)

        with journal:
            op_id = new_op_id()
            backup = journal.backup(op_id, 0, target)
            target.write_text("partially written new content", encoding="utf-8")
            op = journal.begin(
                OpKind.UPDATE_MEMORY,
                ["mid"],
                [
                    Step(
                        i=0,
                        act="write",
                        path=paths.rel(target),
                        new_hash="sha256:" + "2" * 64,
                        prev_hash=original_hash,
                        prev_backup=backup,
                    ),
                    Step(
                        i=1,
                        act="write",
                        path=paths.rel(paths.type_dir("fact") / "other.md"),
                        new_hash="sha256:" + "3" * 64,
                    ),
                ],
            )
            report = journal.recover()

        assert report.rolled_back == [op]
        assert target.read_text(encoding="utf-8") == "original content"
        assert hash_file(target) == original_hash

    def test_purge_rolls_forward_when_partially_applied(
        self, journal: Journal, paths: Paths
    ) -> None:
        """Destruction was the intent: the tombstone is durable, so finish the unlink."""
        tombstone = paths.tombstone_file("mid")
        tombstone.write_text('{"memory_id": "mid", "permanent": true}', encoding="utf-8")
        doomed = write_memory_file(paths, "doomed.md", "still here")

        with journal:
            op = journal.begin(
                OpKind.PURGE_MEMORY,
                ["mid"],
                [
                    Step(
                        i=0, act="write", path=paths.rel(tombstone), new_hash=hash_file(tombstone)
                    ),
                    Step(i=1, act="unlink", path=paths.rel(doomed), prev_hash=hash_file(doomed)),
                ],
                db={"unindex": ["mid"]},
            )
            report = journal.recover()

        assert report.rolled_forward == [op]
        assert tombstone.exists()
        assert not doomed.exists(), "purge must complete the unlink"
        assert report.unindex == ["mid"]

    def test_is_idempotent(self, journal: Journal, paths: Paths) -> None:
        target = paths.type_dir("fact") / "x.md"
        with journal:
            journal.begin(
                OpKind.CREATE_MEMORY,
                ["mid"],
                [Step(i=0, act="write", path=paths.rel(target), new_hash="sha256:" + "4" * 64)],
            )
            first = journal.recover()
            second = journal.recover()
            third = journal.recover()
        assert first.settled == 1
        assert second.settled == 0 and second.clean
        assert third.settled == 0 and third.clean

    def test_dry_run_changes_nothing(self, journal: Journal, paths: Paths) -> None:
        target = write_memory_file(paths, "dry.md", "content")
        with journal:
            op = journal.begin(
                OpKind.CREATE_MEMORY,
                ["mid"],
                [Step(i=0, act="write", path=paths.rel(target), new_hash=hash_file(target))],
            )
            report = journal.recover(dry_run=True)
            assert report.rolled_forward == [op]
            assert [r.op_id for r in journal.open_ops()] == [op], "still open after dry run"
            assert target.exists()

    def test_cleans_temp_residue(self, journal: Journal, paths: Paths) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["mid"], [])
            stray = paths.type_dir("fact") / f"leftover.md.tmp-{op}"
            stray.write_text("junk", encoding="utf-8")
            journal.recover()
        assert not stray.exists()

    def test_discards_undo_dir_after_settling(self, journal: Journal, paths: Paths) -> None:
        target = write_memory_file(paths, "u.md", "content")
        with journal:
            op = journal.begin(OpKind.UPDATE_MEMORY, ["mid"], [])
            journal.backup(op, 0, target)
            assert (paths.undo_dir / op).exists()
            journal.commit(op)
        assert not (paths.undo_dir / op).exists()

    def test_multiple_open_ops_settled_in_order(self, journal: Journal, paths: Paths) -> None:
        with journal:
            ops = []
            for i in range(3):
                t = paths.type_dir("fact") / f"m{i}.md"
                ops.append(
                    journal.begin(
                        OpKind.CREATE_MEMORY,
                        [f"mid{i}"],
                        [Step(i=0, act="write", path=paths.rel(t), new_hash="sha256:" + "5" * 64)],
                    )
                )
            report = journal.recover()
        assert sorted(report.rolled_back) == sorted(ops)
        assert journal.open_ops() == []


class TestTruncate:
    def test_keeps_recent_files(self, journal: Journal, paths: Paths) -> None:
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
            journal.commit(op)
            assert journal.truncate(keep_days=7) == 0
        assert list(paths.journal_dir.glob("*.jsonl"))

    def test_removes_old_settled_files(self, journal: Journal, paths: Paths) -> None:
        stale = paths.journal_dir / "20200101.jsonl"
        stale.write_text("", encoding="utf-8")
        with journal:
            assert journal.truncate(keep_days=7) == 1
        assert not stale.exists()

    def test_never_removes_a_file_with_an_open_op(self, journal: Journal, paths: Paths) -> None:
        """An old file still holding an unsettled op must survive truncation."""
        with journal:
            op = journal.begin(OpKind.CREATE_MEMORY, ["x"], [])
        current = next(paths.journal_dir.glob("*.jsonl"))
        stale = paths.journal_dir / "20200101.jsonl"
        stale.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
        current.unlink()

        with journal:
            assert journal.truncate(keep_days=7) == 0
        assert stale.exists()
        assert [r.op_id for r in journal.open_ops()] == [op]
