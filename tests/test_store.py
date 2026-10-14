"""Filesystem CRUD, blobs, and resurrection guards."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory.journal import Journal, hash_file
from memory.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    SourceKind,
    SourceRef,
    ValidationError,
    new_id,
)
from memory.paths import Paths
from memory.store import Conflict, NotFound, Store, Tombstoned


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    p = Paths(tmp_path)
    p.ensure_tree()
    return p


@pytest.fixture
def store(paths: Paths) -> Iterator[Store]:
    journal = Journal(paths, lock_timeout=2.0)
    journal.acquire()
    try:
        yield Store(paths, journal)
    finally:
        journal.release()


def make_memory(**kw: object) -> Memory:
    defaults: dict[str, object] = {
        "type": MemoryType.FACT,
        "title": "Prefers uv over pip",
        "body": "Uses uv for all Python projects.",
    }
    defaults.update(kw)
    return Memory(**defaults)  # type: ignore[arg-type]


def tree_manifest(paths: Paths) -> dict[str, str]:
    """Hash every file under data/, excluding volatile journal state."""
    out: dict[str, str] = {}
    for path in sorted(paths.data.rglob("*")):
        if not path.is_file():
            continue
        rel = paths.rel(path)
        if rel.startswith(".journal/"):
            continue
        out[rel] = hash_file(path) or ""
    return out


class TestReadWrite:
    def test_create_writes_file_in_type_directory(self, store: Store, paths: Paths) -> None:
        m = make_memory(type=MemoryType.PREFERENCE)
        result = store.write_new(m)
        assert result.path == paths.memories / "preferences" / f"{m.id}.md"
        assert result.path.exists()

    def test_round_trip_through_disk(self, store: Store) -> None:
        m = make_memory(tags=["python"], entities=["uv"], confidence=0.9, profile=True)
        store.write_new(m)
        back = store.read(m.id)
        assert back is not None
        assert back.title == m.title
        assert back.tags == ["python"]
        assert back.profile is True
        assert back.confidence == pytest.approx(0.9)

    def test_read_missing_is_none(self, store: Store) -> None:
        assert store.read(new_id()) is None

    def test_require_missing_raises(self, store: Store) -> None:
        with pytest.raises(NotFound):
            store.require(new_id())

    def test_duplicate_id_is_a_conflict(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        with pytest.raises(Conflict, match="already exists"):
            store.write_new(m)

    def test_locate_finds_across_type_dirs(self, store: Store) -> None:
        ids = {}
        for mem_type in MemoryType:
            m = make_memory(type=mem_type, title=f"A {mem_type.value}")
            store.write_new(m)
            ids[mem_type] = m.id
        for mem_type, mid in ids.items():
            found = store.locate(mid)
            assert found is not None
            assert found.parent.name.startswith(mem_type.value)

    def test_invalid_memory_is_rejected(self, store: Store) -> None:
        with pytest.raises(ValidationError):
            store.write_new(make_memory(title="  "))

    def test_filename_must_match_id(self, store: Store, paths: Paths) -> None:
        """A renamed file would otherwise be indexed under the wrong id."""
        m = make_memory()
        store.write_new(m)
        original = paths.memory_file("fact", m.id)
        renamed = original.with_name(f"{new_id()}.md")
        original.rename(renamed)

        with pytest.raises(ValidationError, match="does not match id"):
            store.read_path(renamed)

    def test_iter_all_skips_corrupt_files(self, store: Store, paths: Paths) -> None:
        """One bad file must not stop a rebuild."""
        store.write_new(make_memory(title="Good one"))
        (paths.type_dir("fact") / f"{new_id()}.md").write_text("not a memory", encoding="utf-8")
        assert [m.title for m in store.iter_all()] == ["Good one"]

    def test_iter_problems_reports_corrupt_files(self, store: Store, paths: Paths) -> None:
        bad = paths.type_dir("fact") / f"{new_id()}.md"
        bad.write_text("no frontmatter here", encoding="utf-8")
        problems = list(store.iter_problems())
        assert len(problems) == 1
        assert problems[0][0] == bad

    def test_iter_all_excludes_deleted_by_default(self, store: Store) -> None:
        live = make_memory(title="Live")
        gone = make_memory(title="Gone", status=MemoryStatus.DELETED)
        store.write_new(live)
        store.write_new(gone)
        assert [m.title for m in store.iter_all()] == ["Live"]
        assert len(list(store.iter_all(include_deleted=True))) == 2


class TestOverwrite:
    def test_updates_content(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        store.overwrite(m.with_updates(title="Revised title"))
        back = store.read(m.id)
        assert back is not None and back.title == "Revised title"

    def test_identical_content_is_a_noop(self, store: Store, paths: Paths) -> None:
        """An idempotent retry must cost nothing and write no journal record."""
        m = make_memory()
        store.write_new(m)
        before = len(list(paths.journal_dir.glob("*.jsonl"))), tree_manifest(paths)

        assert store.overwrite(m) is None

        after = len(list(paths.journal_dir.glob("*.jsonl"))), tree_manifest(paths)
        assert before[1] == after[1]

    def test_missing_memory_raises(self, store: Store) -> None:
        with pytest.raises(NotFound):
            store.overwrite(make_memory())

    def test_expect_hash_guards_against_concurrent_change(self, store: Store) -> None:
        m = make_memory()
        result = store.write_new(m)
        store.overwrite(m.with_updates(body="changed by someone else"))

        with pytest.raises(Conflict, match="changed on disk"):
            store.overwrite(m.with_updates(title="mine"), expect_hash=result.content_hash)

    def test_expect_hash_passes_when_unchanged(self, store: Store) -> None:
        m = make_memory()
        result = store.write_new(m)
        assert (
            store.overwrite(m.with_updates(title="New"), expect_hash=result.content_hash)
            is not None
        )

    def test_type_change_moves_the_file(self, store: Store, paths: Paths) -> None:
        m = make_memory(type=MemoryType.FACT)
        store.write_new(m)
        store.overwrite(m.with_updates(type=MemoryType.DECISION))

        assert not (paths.memories / "facts" / f"{m.id}.md").exists()
        assert (paths.memories / "decisions" / f"{m.id}.md").exists()
        back = store.read(m.id)
        assert back is not None and back.type is MemoryType.DECISION

    def test_no_temp_residue(self, store: Store, paths: Paths) -> None:
        m = make_memory()
        store.write_new(m)
        store.overwrite(m.with_updates(body="new body"))
        assert not list(paths.data.rglob("*.tmp-*"))


class TestSoftDeleteAndPurge:
    def test_soft_delete_keeps_file_and_body(self, store: Store, paths: Paths) -> None:
        m = make_memory()
        store.write_new(m)
        tomb, _ = store.soft_delete(m.id, reason="superseded by life")

        path = paths.memory_file("fact", m.id)
        assert path.exists()
        back = store.read_path(path)
        assert back.status is MemoryStatus.DELETED
        assert back.body == m.body, "body is retained on soft delete"
        assert tomb.permanent is False
        assert tomb.reason == "superseded by life"

    def test_soft_delete_writes_tombstone(self, store: Store, paths: Paths) -> None:
        m = make_memory()
        store.write_new(m)
        store.soft_delete(m.id)
        assert paths.tombstone_file(m.id).exists()
        tomb = store.read_tombstone(m.id)
        assert tomb is not None and tomb.permanent is False

    def test_purge_removes_file_keeps_tombstone(self, store: Store, paths: Paths) -> None:
        m = make_memory()
        store.write_new(m)
        tomb, _ = store.purge(m.id, reason="user asked")

        assert not paths.memory_file("fact", m.id).exists()
        assert paths.tombstone_file(m.id).exists()
        assert tomb.permanent is True
        assert store.read(m.id) is None

    def test_purge_tombstone_records_provenance(self, store: Store) -> None:
        src = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md", locator="L1")
        m = make_memory(sources=[src])
        store.write_new(m)
        tomb, _ = store.purge(m.id)
        assert tomb.content_hash == m.content_hash
        assert tomb.source_keys() == m.source_keys()

    def test_purge_missing_raises(self, store: Store) -> None:
        with pytest.raises(NotFound):
            store.purge(new_id())

    def test_does_not_touch_source_files(self, store: Store, paths: Paths) -> None:
        """Deleting a memory never deletes the conversation it came from."""
        convo = paths.conversations / "2026-10-15.md"
        convo.write_text("the original chat log", encoding="utf-8")
        m = make_memory(
            sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/2026-10-15.md")]
        )
        store.write_new(m)
        store.purge(m.id)
        assert convo.exists()
        assert convo.read_text(encoding="utf-8") == "the original chat log"


class TestResurrectionGuards:
    def test_identical_content_is_refused(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        store.purge(m.id)

        with pytest.raises(Tombstoned, match="content_hash"):
            store.write_new(make_memory())

    def test_reworded_title_with_same_source_is_refused(self, store: Store) -> None:
        """The case that actually matters: re-extraction rewords slightly."""
        src = SourceRef(
            kind=SourceKind.CONVERSATION, ref="conversations/setup.md", locator="L120-L142"
        )
        original = make_memory(title="Prefers uv over pip", sources=[src])
        store.write_new(original)
        store.purge(original.id)

        reworded = make_memory(
            title="Likes uv more than pip for packaging",
            body="Totally different wording here.",
            sources=[src],
        )
        with pytest.raises(Tombstoned, match="source"):
            store.write_new(reworded)

    def test_same_title_different_body_is_refused(self, store: Store) -> None:
        m = make_memory(title="A specific claim", body="one body")
        store.write_new(m)
        store.purge(m.id)
        with pytest.raises(Tombstoned, match="title_hash"):
            store.write_new(make_memory(title="A specific claim", body="another body"))

    def test_force_recreate_overrides(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        store.purge(m.id)
        result = store.write_new(make_memory(), force_recreate=True)
        assert result.path.exists()

    def test_soft_delete_does_not_block_recreation(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        store.soft_delete(m.id)
        # A soft delete is reversible by design, so recreation is allowed.
        store.write_new(make_memory(title="Prefers uv over pip", body="different body"))

    def test_unrelated_content_is_allowed(self, store: Store) -> None:
        m = make_memory()
        store.write_new(m)
        store.purge(m.id)
        store.write_new(make_memory(title="Something else entirely", body="unrelated"))

    def test_tombstone_survives_on_disk(self, store: Store, paths: Paths) -> None:
        """Tombstones are files, so they outlive any index rebuild."""
        m = make_memory()
        store.write_new(m)
        store.purge(m.id)
        raw = json.loads(paths.tombstone_file(m.id).read_text(encoding="utf-8"))
        assert raw["permanent"] is True
        assert raw["memory_id"] == m.id


class TestBlobs:
    def test_stores_bytes_content_addressed(self, store: Store, paths: Paths) -> None:
        result = store.put_blob(b"hello world")
        assert result.path == paths.asset_blob(result.sha256)
        assert result.path.read_bytes() == b"hello world"
        assert result.deduped is False

    def test_shard_layout(self, store: Store) -> None:
        result = store.put_blob(b"shard me")
        parts = result.path.parts
        assert parts[-2] == result.sha256[2:4]
        assert parts[-3] == result.sha256[:2]
        assert parts[-1] == result.sha256

    def test_identical_bytes_dedupe(self, store: Store, paths: Paths) -> None:
        first = store.put_blob(b"same content")
        second = store.put_blob(b"same content")
        assert first.sha256 == second.sha256
        assert second.deduped is True
        assert len(list(paths.blob_glob())) == 1

    def test_from_file(self, store: Store, tmp_path: Path) -> None:
        src = tmp_path / "input.txt"
        src.write_bytes(b"file content")
        result = store.put_blob(src)
        assert result.bytes == len(b"file content")
        assert result.path.read_bytes() == b"file content"

    def test_missing_file_raises(self, store: Store, tmp_path: Path) -> None:
        with pytest.raises(NotFound):
            store.put_blob(tmp_path / "absent.bin")

    def test_empty_blob(self, store: Store) -> None:
        result = store.put_blob(b"")
        assert result.bytes == 0
        assert result.path.exists()

    def test_large_blob(self, store: Store, tmp_path: Path) -> None:
        src = tmp_path / "big.bin"
        src.write_bytes(os.urandom(5 * 1024 * 1024))
        result = store.put_blob(src)
        assert result.bytes == 5 * 1024 * 1024
        assert result.path.stat().st_size == result.bytes

    def test_no_temp_residue(self, store: Store, paths: Paths) -> None:
        store.put_blob(b"data")
        assert not list(paths.data.rglob("*.tmp-*"))

    def test_read_and_unlink(self, store: Store) -> None:
        result = store.put_blob(b"transient")
        assert store.read_blob(result.sha256) == b"transient"
        assert store.unlink_blob(result.sha256) is True
        assert store.unlink_blob(result.sha256) is False
        with pytest.raises(NotFound):
            store.read_blob(result.sha256)


class TestAssetMeta:
    def test_round_trip(self, store: Store) -> None:
        from memory.models import Asset

        blob = store.put_blob(b"pdf bytes")
        asset = Asset(
            sha256=blob.sha256,
            media_type="application/pdf",
            bytes=blob.bytes,
            original_filename="paper.pdf",
            tags=["research"],
        )
        store.write_asset_meta(asset)
        back = store.read_asset_meta(asset.id)
        assert back is not None
        assert back.sha256 == blob.sha256
        assert back.original_filename == "paper.pdf"
        assert back.tags == ["research"]

    def test_missing_is_none(self, store: Store) -> None:
        assert store.read_asset_meta(new_id()) is None

    def test_refcount_counts_versions(self, store: Store) -> None:
        """GC must be conservative: a prior version still pins its blob."""
        from memory.models import Asset

        old_blob = store.put_blob(b"v1")
        new_blob = store.put_blob(b"v2")
        store.write_asset_meta(
            Asset(
                sha256=new_blob.sha256,
                media_type="text/plain",
                bytes=2,
                prev_sha256=old_blob.sha256,
                version=2,
            )
        )
        assert store.blob_refcount(new_blob.sha256) == 1
        assert store.blob_refcount(old_blob.sha256) == 1, "prior version pins the old blob"

    def test_refcount_ignores_deleted_assets(self, store: Store) -> None:
        from memory.models import Asset

        blob = store.put_blob(b"orphan")
        store.write_asset_meta(
            Asset(sha256=blob.sha256, media_type="text/plain", bytes=1, status="deleted")
        )
        assert store.blob_refcount(blob.sha256) == 0

    def test_two_assets_may_share_one_blob(self, store: Store) -> None:
        """Same bytes, two real-world things -> two asset ids, one blob."""
        from memory.models import Asset

        blob = store.put_blob(b"shared bytes")
        a = Asset(sha256=blob.sha256, media_type="text/plain", bytes=1)
        b = Asset(sha256=blob.sha256, media_type="text/plain", bytes=1)
        store.write_asset_meta(a)
        store.write_asset_meta(b)
        assert a.id != b.id
        assert store.blob_refcount(blob.sha256) == 2

    def test_iter_assets_skips_corrupt(self, store: Store, paths: Paths) -> None:
        from memory.models import Asset

        store.write_asset_meta(Asset(sha256="a" * 64, media_type="text/plain", bytes=1))
        bad = paths.asset_meta_dir / "zz" / "broken.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")
        assert len(list(store.iter_assets())) == 1

    def test_delete_meta(self, store: Store) -> None:
        from memory.models import Asset

        asset = Asset(sha256="b" * 64, media_type="text/plain", bytes=1)
        store.write_asset_meta(asset)
        assert store.delete_asset_meta(asset.id) is True
        assert store.delete_asset_meta(asset.id) is False

    def test_media_type_guessing(self, store: Store, tmp_path: Path) -> None:
        assert store.guess_media_type(tmp_path / "a.pdf") == "application/pdf"
        assert store.guess_media_type(tmp_path / "a.txt") == "text/plain"
        assert store.guess_media_type(tmp_path / "a.unknown") == "application/octet-stream"


class TestJournalIntegration:
    def test_create_records_an_op(self, store: Store) -> None:
        result = store.write_new(make_memory())
        records = store.journal.show(result.op_id)
        assert [r.phase for r in records] == ["begin"]
        assert records[0].db["reindex"] == [records[0].targets[0]]

    def test_op_is_open_until_committed(self, store: Store) -> None:
        result = store.write_new(make_memory())
        assert [r.op_id for r in store.journal.open_ops()] == [result.op_id]
        store.journal.commit(result.op_id)
        assert store.journal.open_ops() == []

    def test_purge_plan_writes_tombstone_before_unlink(self, store: Store) -> None:
        """Order matters: a crash between them must roll forward, not back."""
        m = make_memory()
        store.write_new(m)
        _, op_id = store.purge(m.id)
        plan = store.journal.show(op_id)[0].plan
        assert plan[0].act == "write" and ".tombstones" in plan[0].path
        assert plan[1].act == "unlink"

    def test_update_plan_carries_a_pre_image(self, store: Store, paths: Paths) -> None:
        m = make_memory()
        store.write_new(m)
        result = store.overwrite(m.with_updates(title="Changed"))
        assert result is not None
        plan = store.journal.show(result.op_id)[0].plan
        assert plan[0].prev_backup is not None
        assert paths.abs(plan[0].prev_backup).exists()
