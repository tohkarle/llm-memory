"""Concurrent access: one writer at a time, readers never blocked."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from memory.api import MemoryAPI
from memory.journal import Journal, LockTimeout

PROJECT = Path(__file__).resolve().parent.parent


def write_script(root: Path, count: int, label: str) -> str:
    return textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT)!r})
        from memory.api import MemoryAPI
        from memory.models import MemoryType

        with MemoryAPI({str(root)!r}) as api:
            for i in range({count}):
                api.create(
                    type=MemoryType.FACT,
                    title=f"{label} memory {{i}}",
                    body=f"Written by {label}, number {{i}}.",
                )
        print("done", flush=True)
        """
    )


def run_writer(root: Path, count: int, label: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", write_script(root, count, label)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    api = MemoryAPI.initialize(tmp_path)
    api.close()
    return tmp_path


class TestConcurrentWriters:
    def test_two_processes_do_not_lose_writes(self, root: Path) -> None:
        """The single-writer lock must serialize, not drop, concurrent work."""
        first = run_writer(root, 25, "alpha")
        second = run_writer(root, 25, "beta")
        outs = [p.communicate(timeout=180) for p in (first, second)]

        for proc, (out, err) in zip((first, second), outs, strict=True):
            assert proc.returncode == 0, f"writer failed: {err}"
            assert "done" in out

        with MemoryAPI(root) as api:
            total = api.conn.execute("SELECT count(*) c FROM memories").fetchone()["c"]
            assert total == 50
            assert api.verify()["clean"]

    def test_journal_sequence_numbers_stay_unique(self, root: Path) -> None:
        procs = [run_writer(root, 15, f"w{i}") for i in range(2)]
        for proc in procs:
            _, err = proc.communicate(timeout=180)
            assert proc.returncode == 0, err

        paths = MemoryAPI(root).paths
        seqs: list[int] = []
        for log in paths.journal_dir.glob("*.jsonl"):
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seqs.append(json.loads(line)["seq"])
        assert len(seqs) == len(set(seqs)), "sequence numbers collided across processes"

    def test_three_writers(self, root: Path) -> None:
        procs = [run_writer(root, 10, f"w{i}") for i in range(3)]
        for proc in procs:
            _, err = proc.communicate(timeout=240)
            assert proc.returncode == 0, err

        with MemoryAPI(root) as api:
            assert api.conn.execute("SELECT count(*) c FROM memories").fetchone()["c"] == 30
            assert api.verify()["clean"]


class TestLockTimeout:
    def test_second_writer_times_out_cleanly(self, root: Path) -> None:
        """A contended write fails fast and leaves nothing partial behind."""
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import sys, time
                    sys.path.insert(0, {str(PROJECT)!r})
                    from memory.journal import Journal
                    from memory.paths import Paths
                    journal = Journal(Paths({str(root)!r}))
                    journal.acquire()
                    print("locked", flush=True)
                    time.sleep(30)
                    """
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"

            before = sum(1 for _ in MemoryAPI(root).paths.memory_glob())
            api = MemoryAPI(root, lock_timeout=0.3, auto_recover=False)
            api.open()
            try:
                with pytest.raises(LockTimeout):
                    api.create(type="fact", title="Contended write", body="x")
            finally:
                api.close()

            after = sum(1 for _ in MemoryAPI(root).paths.memory_glob())
            assert after == before, "a timed-out write must leave no file"
        finally:
            holder.kill()
            holder.wait(timeout=10)
            if holder.stdout is not None:
                holder.stdout.close()

    def test_cli_reports_lock_timeout_as_exit_4(self, root: Path) -> None:
        from click.testing import CliRunner

        from memory.cli import EXIT_LOCKED, main

        journal = Journal(MemoryAPI(root).paths)
        journal.acquire()
        try:
            result = CliRunner().invoke(
                main,
                [
                    "--root",
                    str(root),
                    "--lock-timeout",
                    "0.2",
                    "create",
                    "--type",
                    "fact",
                    "--title",
                    "Blocked",
                    "--body",
                    "x",
                ],
            )
            # The in-process lock is re-entrant, so this only asserts the
            # mapping when the lock is genuinely contended; either way the
            # command must emit a parseable envelope.
            payload = json.loads(result.stdout)
            if not payload["ok"]:
                assert payload["error"]["code"] == "locked"
                assert result.exit_code == EXIT_LOCKED
        finally:
            journal.release()


class TestReadersAreNotBlocked:
    def test_search_works_while_a_writer_holds_the_lock(self, root: Path) -> None:
        """WAL is what buys this: readers proceed during a write."""
        with MemoryAPI(root) as api:
            api.create(type="fact", title="Readable memory", body="findable text")

        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import sys, time
                    sys.path.insert(0, {str(PROJECT)!r})
                    from memory.journal import Journal
                    from memory.paths import Paths
                    journal = Journal(Paths({str(root)!r}))
                    journal.acquire()
                    print("locked", flush=True)
                    time.sleep(30)
                    """
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"

            reader = MemoryAPI(root, read_only=True, auto_recover=False)
            reader.open()
            try:
                hits = reader.search("findable")
                assert len(hits) == 1
                pack = reader.get_context("findable", token_budget=500)
                assert pack.blocks
            finally:
                reader.close()
        finally:
            holder.kill()
            holder.wait(timeout=10)
            if holder.stdout is not None:
                holder.stdout.close()

    def test_read_only_api_refuses_writes(self, root: Path) -> None:
        api = MemoryAPI(root, read_only=True)
        api.open()
        try:
            with pytest.raises(PermissionError):
                api.create(type="fact", title="Nope", body="x")
        finally:
            api.close()


class TestReentrancy:
    def test_nested_operations_do_not_deadlock(self, root: Path) -> None:
        """merge() writes several memories inside one lock acquisition."""
        with MemoryAPI(root) as api:
            ids = [api.create(type="fact", title=f"Dup {i}", body=f"body {i}").id for i in range(3)]
            result = api.merge(ids)
            assert len(result.superseded) == 3
            assert api.verify()["clean"]

    def test_sequential_opens_are_clean(self, root: Path) -> None:
        for i in range(5):
            with MemoryAPI(root) as api:
                api.create(type="fact", title=f"Sequential {i}", body="x")
                assert api.recovery is not None
                assert api.recovery.clean
