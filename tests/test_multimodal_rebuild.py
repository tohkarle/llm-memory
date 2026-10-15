"""Rebuild equivalence for derived multimodal content.

The load-bearing claim of the whole design is that ``data/`` is the truth and
the index is disposable. Phase 3 adds derived files, segments with locators,
and a work queue, so each of those has to survive ``rm memory.sqlite`` --
otherwise deleting the index silently loses a transcription.

The queue is the deliberate exception: a pending task is not reconstructible
from the filesystem, so rebuild must *preserve* it rather than clear it.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from memory import db, workqueue
from memory.api import MemoryAPI

#: Every table a rebuild is expected to reproduce exactly.
DERIVED_TABLES = (
    "memories",
    "assets",
    "sources",
    "derivations",
    "segments",
    "segments_fts",
    "tombstones",
    "relations",
)


def dump(conn: Any, tables: tuple[str, ...] = DERIVED_TABLES) -> dict[str, list[str]]:
    """A canonical, order-independent snapshot of the index."""
    out: dict[str, list[str]] = {}
    for table in tables:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        # sqlite3.Row is not a mapping: `in` would test values, so .keys() is
        # the only way to enumerate columns.
        out[table] = sorted(
            json.dumps(
                {key: row[key] for key in row.keys()},  # noqa: SIM118
                sort_keys=True,
                default=str,
            )
            for row in rows
        )
    return out


def drop_index(paths: Any) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.index_db) + suffix).unlink(missing_ok=True)


def make_png(path: Path, width: int = 24, height: int = 12) -> Path:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def make_wav(path: Path) -> Path:
    data = b"\x00\x00" * 2000
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return path


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    """A root holding several ingested media types and a memory."""
    root = tmp_path / "root"
    source = tmp_path / "src"
    source.mkdir()

    (source / "notes.md").write_text(
        "\n\n".join(f"Paragraph {i} about uv packaging and pytest fixtures." for i in range(12)),
        encoding="utf-8",
    )
    (source / "long.txt").write_text(
        "\n\n".join(f"Section {i}. " + "filler words " * 40 for i in range(6)),
        encoding="utf-8",
    )
    make_png(source / "shot.png")
    make_wav(source / "clip.wav")

    api = MemoryAPI.initialize(root)
    try:
        api.create(type="fact", title="Project uses uv", body="For every Python project.")
        for name in ("notes.md", "long.txt", "shot.png"):
            api.ingest(source / name, tags=["imported"])
        # Inline so there is a real derivation, not just a queued task.
        api.ingest(source / "clip.wav", defer=False)
        # Plus one genuinely queued task, to prove the queue is preserved.
        api.ingest(source / "notes.md", defer=True)
    finally:
        api.close()
    return root


class TestRebuildEquivalence:
    def test_index_delete_then_rebuild_is_identical(self, populated: Path) -> None:
        with MemoryAPI(populated) as api:
            before = dump(api.conn)
            assert before["derivations"], "the fixture must produce derivations"
            assert before["segments"], "the fixture must produce segments"

        drop_index(MemoryAPI(populated).paths)

        with MemoryAPI(populated) as rebuilt:
            rebuilt.rebuild_index()
            after = dump(rebuilt.conn)

        for table in DERIVED_TABLES:
            assert after[table] == before[table], f"{table} diverged across rebuild"

    def test_rebuild_reports_what_it_reconstructed(self, populated: Path) -> None:
        with MemoryAPI(populated) as api:
            expected_derivations = api.conn.execute(
                "SELECT count(*) c FROM derivations"
            ).fetchone()["c"]
            expected_segments = api.conn.execute("SELECT count(*) c FROM segments").fetchone()["c"]

        drop_index(MemoryAPI(populated).paths)
        with MemoryAPI(populated) as rebuilt:
            report = rebuilt.rebuild_index(verify_after=True)
            assert report["derivations"] == expected_derivations
            assert report["segments"] == expected_segments
            assert report["verified"] is True
            assert report["errors"] == []

    def test_verify_is_clean_after_rebuild(self, populated: Path) -> None:
        drop_index(MemoryAPI(populated).paths)
        with MemoryAPI(populated) as rebuilt:
            rebuilt.rebuild_index()
            assert rebuilt.verify()["clean"] is True

    def test_segment_search_survives_rebuild(self, populated: Path) -> None:
        """The FTS index is contentful, so it must be repopulated, not assumed."""
        with MemoryAPI(populated) as api:
            before = [h["segment_id"] for h in api.search_segments("packaging")]
            assert before

        drop_index(MemoryAPI(populated).paths)
        with MemoryAPI(populated) as rebuilt:
            rebuilt.rebuild_index()
            after = [h["segment_id"] for h in rebuilt.search_segments("packaging")]
        assert sorted(after) == sorted(before)

    def test_rebuild_is_idempotent(self, populated: Path) -> None:
        with MemoryAPI(populated) as api:
            api.rebuild_index()
            once = dump(api.conn)
            api.rebuild_index()
            twice = dump(api.conn)
        assert once == twice

    def test_segment_locators_survive_rebuild(self, populated: Path) -> None:
        """Timestamps and page numbers are provenance; losing them loses the
        ability to cite."""
        with MemoryAPI(populated) as api:
            api.conn.execute(
                "UPDATE segments SET start_ms = 1500, end_ms = 4200, page = 7, "
                "speaker = 'alex' WHERE ordinal = 0"
            )
            api.conn.commit()
            # The index is not the truth, so write the locators into the
            # derived file too -- that is what a rebuild reads.
            touched = 0
            for path in api.paths.derived_glob():
                payload = json.loads(path.read_text(encoding="utf-8"))
                segments = payload.get("segments") or []
                if not segments:
                    continue
                segments[0].update({"start_ms": 1500, "end_ms": 4200, "page": 7, "speaker": "alex"})
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                touched += 1
            assert touched, "the fixture must have derived files to amend"

        with MemoryAPI(populated) as rebuilt:
            rebuilt.rebuild_index()
            rows = rebuilt.conn.execute(
                "SELECT start_ms, end_ms, page, speaker FROM segments WHERE ordinal = 0"
            ).fetchall()
        assert rows
        assert any(
            r["start_ms"] == 1500
            and r["end_ms"] == 4200
            and r["page"] == 7
            and r["speaker"] == "alex"
            for r in rows
        )


class TestQueueIsNotDerivedState:
    def test_rebuild_preserves_pending_tasks(self, populated: Path) -> None:
        """A queued transcription is not in the filesystem, so a rebuild that
        cleared it would silently drop the work."""
        with MemoryAPI(populated) as api:
            pending_before = api.queue_stats()["pending"]
            assert pending_before >= 1
            ids_before = {t.id for t in workqueue.listing(api.conn)}
            api.rebuild_index()
            assert api.queue_stats()["pending"] == pending_before
            assert {t.id for t in workqueue.listing(api.conn)} == ids_before

    def test_a_queued_task_still_runs_after_a_rebuild(self, populated: Path) -> None:
        with MemoryAPI(populated) as api:
            api.rebuild_index()
            report = api.queue_run()
            assert report["processed"] >= 1
            assert not report["errors"]

    def test_deleting_the_index_does_lose_the_queue(self, populated: Path) -> None:
        """Documenting the real boundary: the queue lives only in the index,
        so a full delete loses it. Rebuild preserves it; `rm` cannot."""
        drop_index(MemoryAPI(populated).paths)
        with MemoryAPI(populated) as rebuilt:
            rebuilt.rebuild_index()
            assert rebuilt.queue_stats()["pending"] == 0


class TestDerivedFilesAreSelfDescribing:
    def test_every_derived_file_embeds_its_cache_key(self, populated: Path) -> None:
        with MemoryAPI(populated) as api:
            paths = list(api.paths.derived_glob())
            assert paths
            for path in paths:
                if path.suffix != ".json":
                    continue
                header = json.loads(path.read_text(encoding="utf-8"))["_derivation"]
                for field in (
                    "asset_id",
                    "kind",
                    "processor",
                    "processor_version",
                    "input_hash",
                    "params_hash",
                    "id",
                ):
                    assert header.get(field), f"{path.name} is missing {field}"

    def test_the_filename_is_the_derivation_id(self, populated: Path) -> None:
        """What lets the rebuild key rows without a lookup table."""
        with MemoryAPI(populated) as api:
            for path in api.paths.derived_glob():
                if path.suffix != ".json":
                    continue
                header = json.loads(path.read_text(encoding="utf-8"))["_derivation"]
                assert header["id"] == path.stem

    def test_originals_are_never_modified(self, populated: Path) -> None:
        """Derived output goes under derived/; blobs keep their content hash."""
        with MemoryAPI(populated) as api:
            for row in api.conn.execute("SELECT sha256 FROM assets"):
                blob = api.paths.asset_blob(row["sha256"])
                assert blob.exists()
                import hashlib

                assert hashlib.sha256(blob.read_bytes()).hexdigest() == row["sha256"]


class TestSchemaMigration:
    def test_a_v1_index_migrates_in_place(self, tmp_path: Path) -> None:
        """v1 -> v2 is additive, so it must not demand a full rebuild."""
        root = tmp_path / "root"
        api = MemoryAPI.initialize(root)
        api.create(type="fact", title="Kept across migration", body="body")
        conn = api.conn
        # Simulate the older index: drop what v2 added, declare v1.
        conn.execute("DROP TABLE IF EXISTS segments_fts")
        conn.execute("DROP TABLE IF EXISTS work_queue")
        db.set_meta(conn, "schema_version", "1")
        conn.commit()
        api.close()

        with MemoryAPI(root) as reopened:
            assert db.schema_version(reopened.conn) == 2
            assert not db.missing_tables(reopened.conn)
            assert reopened.get_asset("nope") is None
            found = reopened.search("Kept")
            assert [h.title for h in found] == ["Kept across migration"]

    def test_migration_repopulates_the_segment_index(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        source = tmp_path / "n.md"
        source.write_text(
            "\n\n".join(f"Para {i} about uv packaging." for i in range(6)), encoding="utf-8"
        )
        api = MemoryAPI.initialize(root)
        api.ingest(source)
        api.conn.execute("DROP TABLE segments_fts")
        db.set_meta(api.conn, "schema_version", "1")
        api.conn.commit()
        api.close()

        with MemoryAPI(root) as reopened:
            assert reopened.search_segments("packaging")

    def test_a_future_schema_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        api = MemoryAPI.initialize(root)
        db.set_meta(api.conn, "schema_version", "99")
        api.conn.commit()
        api.close()
        with pytest.raises(db.SchemaError, match="newer than this build"):
            MemoryAPI(root).open()
