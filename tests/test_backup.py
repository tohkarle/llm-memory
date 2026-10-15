"""Backup, export, import, retention, recovery.

The properties worth the most here:

* a round trip loses nothing -- export, import into an empty root, and the
  memories compare equal field by field;
* an archive never carries the index, and a restored root rebuilds to the
  same state;
* a permanently deleted memory cannot come back through an import, which is
  the retention rule that would otherwise be trivially bypassable.
"""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from memory import backup as backup_mod
from memory.api import MemoryAPI
from memory.backup import (
    ARCHIVE_VERSION,
    DEFAULT_KEEP_COUNT,
    ArchiveFormatError,
    BackupError,
    backup,
    backup_dir,
    export_jsonl,
    export_markdown_dir,
    import_jsonl,
    import_markdown_dir,
    list_backups,
    prune,
    read_manifest,
    restore,
)
from memory.models import MemoryStatus, MemoryType, Relation, RelationType, SourceKind, SourceRef


@pytest.fixture
def api(tmp_path: Path) -> Any:
    instance = MemoryAPI.initialize(tmp_path / "root")
    try:
        yield instance
    finally:
        instance.close()


def seed_corpus(api: MemoryAPI) -> list[Any]:
    """A corpus exercising every field an export has to carry."""
    first = api.create(
        type=MemoryType.PREFERENCE,
        title="Prefers uv over pip",
        body="Uses uv for every Python project.\n\nSecond paragraph.",
        tags=["tooling", "python"],
        entities=["uv", "pip"],
        confidence=0.95,
        salience=0.8,
        profile=True,
        sensitivity="private",
        sources=[
            SourceRef(
                kind=SourceKind.CONVERSATION,
                ref="conversations/a.md",
                locator="L1-L20",
                captured_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
        occurred_at=datetime(2026, 2, 3, tzinfo=UTC),
        review_after=datetime(2027, 1, 1, tzinfo=UTC),
        extra={"custom_key": "preserved"},
    )
    second = api.create(
        type=MemoryType.FACT,
        title="Works in Pacific time",
        body="US west coast.",
        relations=[Relation(type=RelationType.RELATES_TO, target=first.id, note="context")],
    )
    return [first, second]


def field_snapshot(api: MemoryAPI) -> dict[str, dict[str, Any]]:
    """Every durable field, keyed by id, for equality comparison."""
    out: dict[str, dict[str, Any]] = {}
    for memory in api.store.iter_all(include_deleted=True):
        payload = memory.to_public()
        out[memory.id] = payload
    return out


# ---------------------------------------------------------------- backup


def test_backup_writes_an_archive(api: MemoryAPI) -> None:
    seed_corpus(api)
    result = backup(api)
    path = Path(result.path)
    assert path.is_file()
    assert result.memories == 2
    assert result.bytes > 0
    assert path.parent == backup_dir(api.paths)


def test_archive_never_carries_the_index(api: MemoryAPI) -> None:
    """The index is derived; shipping it would let a restore disagree with the files."""
    seed_corpus(api)
    result = backup(api)
    with tarfile.open(result.path) as tar:
        names = tar.getnames()
    assert not [n for n in names if "/index/" in n], names
    assert not [n for n in names if ".journal" in n]
    assert not [n for n in names if n.endswith(".lock")]


def test_archive_always_carries_tombstones(api: MemoryAPI) -> None:
    """Without them, a restore silently re-enables resurrection."""
    memories = seed_corpus(api)
    api.delete(memories[0].id, permanent=True, reason="user asked")
    result = backup(api)
    with tarfile.open(result.path) as tar:
        names = tar.getnames()
    assert [n for n in names if ".tombstones" in n]
    assert result.tombstones == 1


def test_archive_carries_a_readable_manifest(api: MemoryAPI) -> None:
    seed_corpus(api)
    manifest = read_manifest(Path(backup(api).path))
    assert manifest["archive_version"] == ARCHIVE_VERSION
    assert manifest["memories"] == 2
    assert "index" in manifest["excludes"]


def test_backup_excludes_atomic_write_leftovers(api: MemoryAPI) -> None:
    """A crash can leave a temp file; it must not restore as a memory."""
    seed_corpus(api)
    stray = api.paths.type_dir("fact") / "abc.md.tmp-deadbeef"
    stray.write_text("not a memory", encoding="utf-8")
    with tarfile.open(backup(api).path) as tar:
        assert not [n for n in tar.getnames() if ".tmp-" in n]


def test_backup_refuses_to_overwrite(api: MemoryAPI) -> None:
    seed_corpus(api)
    first = Path(backup(api).path)
    with pytest.raises(BackupError, match="already exists"):
        backup(api, dest=first)


def test_backup_leaves_no_partial_on_failure(
    api: MemoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_corpus(api)

    def boom(*args: Any, **kw: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(backup_mod, "_add_bytes", boom)
    with pytest.raises(OSError):
        backup(api)
    leftovers = list(backup_dir(api.paths).glob("*.partial"))
    assert leftovers == []


def test_uncompressed_backup_is_supported(api: MemoryAPI) -> None:
    seed_corpus(api)
    result = backup(api, compress=False)
    assert result.path.endswith(".tar")
    assert read_manifest(Path(result.path))["archive_version"] == ARCHIVE_VERSION


def test_labelled_backups_are_distinguishable(api: MemoryAPI) -> None:
    seed_corpus(api)
    result = backup(api, label="Before Big Merge")
    assert "before-big-merge" in Path(result.path).name


def test_backup_result_is_json_serializable(api: MemoryAPI) -> None:
    seed_corpus(api)
    json.dumps(backup(api).to_json())


# ---------------------------------------------------------------- listing


def test_list_backups_is_newest_first(api: MemoryAPI) -> None:
    seed_corpus(api)
    base = backup_dir(api.paths)
    base.mkdir(parents=True, exist_ok=True)
    for stamp in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
        (base / f"memory-backup-{stamp}.tar.gz").write_bytes(b"x")
    names = [entry["name"] for entry in list_backups(api.paths)]
    assert names[0].startswith("memory-backup-20260301")


def test_list_backups_ignores_unrelated_files(api: MemoryAPI) -> None:
    base = backup_dir(api.paths)
    base.mkdir(parents=True, exist_ok=True)
    (base / "notes.txt").write_bytes(b"x")
    (base / "memory-backup-20260101T000000Z.tar.gz").write_bytes(b"x")
    assert len(list_backups(api.paths)) == 1


def test_list_backups_on_a_missing_directory_is_empty(api: MemoryAPI) -> None:
    assert list_backups(api.paths) == []


def test_read_manifest_rejects_a_foreign_archive(tmp_path: Path) -> None:
    archive = tmp_path / "foreign.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("hello.txt")
        info.size = 0
        tar.addfile(info)
    with pytest.raises(ArchiveFormatError):
        read_manifest(archive)


def test_read_manifest_rejects_a_future_archive_version(tmp_path: Path) -> None:
    """Refuse rather than guess at a layout change."""
    archive = tmp_path / "future.tar"
    payload = json.dumps({"archive_version": ARCHIVE_VERSION + 99}).encode()
    with tarfile.open(archive, "w") as tar:
        backup_mod._add_bytes(tar, "manifest.json", payload)
    with pytest.raises(ArchiveFormatError, match="not supported"):
        read_manifest(archive)


# ---------------------------------------------------------------- retention


def _fake_backups(api: MemoryAPI, ages_in_days: list[int]) -> None:
    import os

    base = backup_dir(api.paths)
    base.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    for age in ages_in_days:
        when = now - timedelta(days=age)
        path = base / f"memory-backup-{when.strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
        path.write_bytes(b"x" * 100)
        stamp = when.timestamp()
        os.utime(path, (stamp, stamp))


def test_prune_is_a_dry_run_by_default(api: MemoryAPI) -> None:
    _fake_backups(api, [0, 200, 400])
    report = prune(api.paths, keep_days=30, keep_count=2)
    assert report["dry_run"] is True
    assert report["removed"]
    assert len(list_backups(api.paths)) == 3, "a dry run must not delete"


def test_prune_removes_by_age(api: MemoryAPI) -> None:
    _fake_backups(api, [0, 200, 400])
    prune(api.paths, keep_days=30, keep_count=10, dry_run=False)
    assert len(list_backups(api.paths)) == 1


def test_prune_removes_by_count(api: MemoryAPI) -> None:
    _fake_backups(api, [0, 1, 2, 3, 4])
    prune(api.paths, keep_days=3650, keep_count=2, dry_run=False)
    assert len(list_backups(api.paths)) == 2


def test_prune_always_keeps_the_newest(api: MemoryAPI) -> None:
    """A retention rule that can empty the directory is a data-loss bug."""
    _fake_backups(api, [500, 900])
    report = prune(api.paths, keep_days=0, keep_count=1, dry_run=False)
    remaining = list_backups(api.paths)
    assert len(remaining) == 1
    assert report["kept"] == [remaining[0]["path"]]


def test_prune_refuses_a_policy_that_keeps_nothing(api: MemoryAPI) -> None:
    with pytest.raises(ValueError):
        prune(api.paths, keep_count=0)


def test_prune_reports_bytes_freed(api: MemoryAPI) -> None:
    _fake_backups(api, [0, 400, 500])
    report = prune(api.paths, keep_days=30, dry_run=False)
    assert report["bytes_freed"] == 200


def test_default_retention_is_documented_in_the_report(api: MemoryAPI) -> None:
    _fake_backups(api, [0])
    report = prune(api.paths)
    assert report["policy"]["keep_count"] == DEFAULT_KEEP_COUNT


# ---------------------------------------------------------------- export


def test_export_jsonl_to_a_file(api: MemoryAPI) -> None:
    seed_corpus(api)
    out = api.paths.root / "export.jsonl"
    result = export_jsonl(api, out=out)
    assert result.memories == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["record"] == "memory" for line in lines)


def test_export_jsonl_inline(api: MemoryAPI) -> None:
    seed_corpus(api)
    result = export_jsonl(api)
    assert result.path is None
    assert len(result.lines) == 2
    json.dumps(result.to_json())


def test_export_includes_tombstones_by_default(api: MemoryAPI) -> None:
    memories = seed_corpus(api)
    api.delete(memories[1].id, permanent=True)
    result = export_jsonl(api)
    kinds = [json.loads(line)["record"] for line in result.lines]
    assert "tombstone" in kinds


def test_export_markdown_dir_copies_files_verbatim(api: MemoryAPI) -> None:
    """The output *is* the format: no importer needed."""
    memories = seed_corpus(api)
    out = api.paths.root / "md-export"
    result = export_markdown_dir(api, out)
    assert result.memories == 2
    original = api.paths.memory_file(memories[0].type.value, memories[0].id).read_bytes()
    exported = (out / memories[0].type.value / f"{memories[0].id}.md").read_bytes()
    assert exported == original


# ---------------------------------------------------------------- round trip


def test_jsonl_round_trip_is_lossless(api: MemoryAPI, tmp_path: Path) -> None:
    """Every durable field survives export and import, field by field."""
    seed_corpus(api)
    before = field_snapshot(api)
    export = export_jsonl(api)

    target = MemoryAPI.initialize(tmp_path / "restored")
    try:
        result = import_jsonl(target, "\n".join(export.lines))
        assert len(result.created) == 2, result.to_json()
        assert result.errors == []
        after = field_snapshot(target)
    finally:
        target.close()

    assert set(before) == set(after), "ids must be preserved"
    for mid, payload in before.items():
        assert after[mid] == payload, f"{mid} differs after a round trip"


def test_markdown_round_trip_is_lossless(api: MemoryAPI, tmp_path: Path) -> None:
    seed_corpus(api)
    before = field_snapshot(api)
    out = api.paths.root / "md-export"
    export_markdown_dir(api, out)

    target = MemoryAPI.initialize(tmp_path / "restored-md")
    try:
        result = import_markdown_dir(target, out)
        assert len(result.created) == 2, result.to_json()
        after = field_snapshot(target)
    finally:
        target.close()
    assert after == before


def test_round_trip_preserves_unknown_extra_keys(api: MemoryAPI, tmp_path: Path) -> None:
    """Forward compatibility: a newer export's extra key is not dropped."""
    api.create(
        type=MemoryType.FACT,
        title="A claim",
        body="b",
        extra={"future_field": {"nested": [1, 2]}},
    )
    export = export_jsonl(api)
    target = MemoryAPI.initialize(tmp_path / "fc")
    try:
        import_jsonl(target, "\n".join(export.lines))
        restored = next(iter(target.store.iter_all()))
        assert restored.extra["future_field"] == {"nested": [1, 2]}
    finally:
        target.close()


def test_imported_root_rebuilds_to_the_same_index(api: MemoryAPI, tmp_path: Path) -> None:
    seed_corpus(api)
    export = export_jsonl(api)
    target = MemoryAPI.initialize(tmp_path / "rebuilt")
    try:
        import_jsonl(target, "\n".join(export.lines))
        before = _index_dump(target)
        target.rebuild_index()
        assert _index_dump(target) == before
    finally:
        target.close()


def _index_dump(api: MemoryAPI) -> dict[str, list[tuple[Any, ...]]]:
    out: dict[str, list[tuple[Any, ...]]] = {}
    for table, columns in (
        ("memories", "id, type, title, status, confidence, salience, profile, content_hash"),
        ("memory_tags", "memory_id, tag"),
        ("memory_entities", "memory_id, entity"),
        ("memory_sources", "memory_id, source_id, locator"),
        ("relations", "src_id, dst_id, type, implied"),
        ("tombstones", "memory_id, permanent, content_hash"),
    ):
        rows = api.conn.execute(f"SELECT {columns} FROM {table}").fetchall()
        out[table] = sorted(tuple(row) for row in rows)
    return out


# ---------------------------------------------------------------- import rules


def test_import_is_additive_and_skips_existing(api: MemoryAPI) -> None:
    seed_corpus(api)
    export = export_jsonl(api)
    result = import_jsonl(api, "\n".join(export.lines))
    assert result.created == []
    assert len(result.skipped) == 2
    assert all(entry["reason"] == "already present" for entry in result.skipped)


def test_import_cannot_resurrect_a_permanently_deleted_memory(
    api: MemoryAPI, tmp_path: Path
) -> None:
    """Otherwise the retention guarantee is bypassable by exporting first."""
    memory = api.create(
        type=MemoryType.FACT,
        title="A removed claim",
        body="Gone for good.",
        sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")],
    )
    export = export_jsonl(api, include_tombstones=False)
    api.delete(memory.id, permanent=True, reason="user asked")

    result = import_jsonl(api, "\n".join(export.lines))
    assert result.created == []
    assert result.skipped[0]["reason"] == "permanently deleted in this root"
    assert api.get(memory.id) is None


def test_tombstones_are_restored_before_memories(api: MemoryAPI, tmp_path: Path) -> None:
    """Line order must not decide whether a deletion survives a round trip."""
    memory = api.create(
        type=MemoryType.FACT,
        title="A removed claim",
        body="Gone for good.",
        sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")],
    )
    export_lines = export_jsonl(api, include_tombstones=False).lines
    api.delete(memory.id, permanent=True)
    tomb = api.tombstone(memory.id)
    assert tomb is not None
    tomb_line = json.dumps({"record": "tombstone", **tomb.to_json()})

    # Memory first, tombstone last: the deletion must still win.
    stream = "\n".join([*export_lines, tomb_line])
    target = MemoryAPI.initialize(tmp_path / "ordered")
    try:
        result = import_jsonl(target, stream)
        assert result.tombstones_restored == [memory.id]
        assert result.created == []
        assert target.get(memory.id) is None
    finally:
        target.close()


def test_import_reports_a_malformed_line_and_continues(api: MemoryAPI, tmp_path: Path) -> None:
    good = json.dumps({"record": "memory", "type": "fact", "title": "A good claim", "body": "b"})
    stream = "\n".join(["{not json", good, "[1,2,3]"])
    target = MemoryAPI.initialize(tmp_path / "partial")
    try:
        result = import_jsonl(target, stream)
        assert len(result.created) == 1
        assert len(result.errors) == 2
    finally:
        target.close()


def test_import_skips_an_unknown_record_type(api: MemoryAPI, tmp_path: Path) -> None:
    stream = json.dumps({"record": "something-new", "payload": 1})
    target = MemoryAPI.initialize(tmp_path / "unknown")
    try:
        result = import_jsonl(target, stream)
        assert result.skipped[0]["reason"] == "unknown record type"
    finally:
        target.close()


def test_import_rejects_an_unknown_memory_type(api: MemoryAPI, tmp_path: Path) -> None:
    stream = json.dumps({"record": "memory", "type": "nonsense", "title": "x"})
    target = MemoryAPI.initialize(tmp_path / "badtype")
    try:
        result = import_jsonl(target, stream)
        assert result.errors
    finally:
        target.close()


def test_import_result_is_json_serializable(api: MemoryAPI) -> None:
    seed_corpus(api)
    json.dumps(import_jsonl(api, "\n".join(export_jsonl(api).lines)).to_json())


def test_superseded_memories_survive_a_round_trip(api: MemoryAPI, tmp_path: Path) -> None:
    """History is part of the data, so an export has to carry it."""
    original = api.create(type=MemoryType.FACT, title="Old claim", body="Outdated.")
    api.replace(original.id, title="New claim", body="Current.")

    export = export_jsonl(api)
    target = MemoryAPI.initialize(tmp_path / "history")
    try:
        import_jsonl(target, "\n".join(export.lines))
        restored = target.require(original.id)
        assert restored.status is MemoryStatus.SUPERSEDED
        assert restored.superseded_by is not None
    finally:
        target.close()


# ---------------------------------------------------------------- restore


def test_restore_extracts_into_an_empty_root(api: MemoryAPI, tmp_path: Path) -> None:
    seed_corpus(api)
    before = field_snapshot(api)
    archive = Path(backup(api).path)

    target_root = tmp_path / "restored-root"
    report = restore(archive, target_root)
    assert report["extracted_files"] > 0

    restored = MemoryAPI(target_root)
    restored.open()
    try:
        restored.rebuild_index()
        assert field_snapshot(restored) == before
    finally:
        restored.close()


def test_restore_refuses_a_populated_root(api: MemoryAPI, tmp_path: Path) -> None:
    """Merging an archive into a live tree is an import, not a restore."""
    seed_corpus(api)
    archive = Path(backup(api).path)
    with pytest.raises(BackupError, match="not empty"):
        restore(archive, api.paths.root)


def test_restore_rejects_a_path_traversing_archive(tmp_path: Path) -> None:
    """Never trust an archive's member names."""
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        backup_mod._add_bytes(
            tar, "manifest.json", json.dumps({"archive_version": ARCHIVE_VERSION}).encode()
        )
        backup_mod._add_bytes(tar, "../escaped.txt", b"pwned")

    with pytest.raises(BackupError, match="escapes"):
        restore(archive, tmp_path / "target")
    assert not (tmp_path / "escaped.txt").exists()


def test_restored_root_has_no_index_until_rebuilt(api: MemoryAPI, tmp_path: Path) -> None:
    seed_corpus(api)
    archive = Path(backup(api).path)
    target_root = tmp_path / "no-index"
    restore(archive, target_root)
    # ensure_tree creates the directory, but nothing has populated the db.
    assert not (target_root / "data" / "index" / "memory.sqlite").exists()


def test_restore_reports_the_next_step(api: MemoryAPI, tmp_path: Path) -> None:
    seed_corpus(api)
    report = restore(Path(backup(api).path), tmp_path / "hinted")
    assert "rebuild" in report["next_step"]
