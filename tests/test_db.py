"""Schema, connection policy, and index synchronization."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory import db
from memory.models import (
    Asset,
    Memory,
    MemoryStatus,
    MemoryType,
    Relation,
    RelationType,
    Segment,
    Sensitivity,
    SourceKind,
    SourceRef,
    new_id,
    tombstone_for,
)
from memory.paths import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    p = Paths(tmp_path)
    p.ensure_tree()
    return p


@pytest.fixture
def conn(paths: Paths) -> Iterator[sqlite3.Connection]:
    c = db.connect(paths)
    db.init_schema(c)
    yield c
    c.close()


def make_memory(**kw: object) -> Memory:
    defaults: dict[str, object] = {
        "type": MemoryType.FACT,
        "title": "Test memory",
        "body": "Body text about python tooling.",
    }
    defaults.update(kw)
    return Memory(**defaults)  # type: ignore[arg-type]


def add(conn: sqlite3.Connection, m: Memory) -> Memory:
    db.index_memory(conn, m, path=f"memories/{m.type.value}s/{m.id}.md")
    return m


def fts_ids(conn: sqlite3.Connection, query: str) -> list[str]:
    rows = conn.execute(
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
        "WHERE memories_fts MATCH ? ORDER BY rank",
        (query,),
    )
    return [r["id"] for r in rows]


class TestConnection:
    def test_fts5_is_available(self, conn: sqlite3.Connection) -> None:
        assert db.has_fts5(conn)

    def test_pragmas_applied(self, conn: sqlite3.Connection) -> None:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_creates_index_directory(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        c = db.connect(p)
        try:
            assert p.index_dir.is_dir()
        finally:
            c.close()

    def test_read_only_rejects_writes(self, paths: Paths) -> None:
        c = db.connect(paths)
        db.init_schema(c)
        c.close()
        ro = db.connect(paths, read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                ro.execute("INSERT INTO meta(key, value) VALUES('x','y')")
        finally:
            ro.close()

    def test_row_factory_gives_mapping_access(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "k", "v")
        assert conn.execute("SELECT value FROM meta WHERE key='k'").fetchone()["value"] == "v"


class TestSchema:
    def test_all_expected_tables_exist(self, conn: sqlite3.Connection) -> None:
        assert db.missing_tables(conn) == set()

    def test_version_recorded(self, conn: sqlite3.Connection) -> None:
        assert db.schema_version(conn) == db.SCHEMA_VERSION

    def test_init_is_idempotent(self, conn: sqlite3.Connection) -> None:
        db.init_schema(conn)
        db.init_schema(conn)
        assert db.missing_tables(conn) == set()

    def test_requires_rebuild_is_none_when_healthy(self, conn: sqlite3.Connection) -> None:
        assert db.requires_rebuild(conn) is None

    def test_requires_rebuild_detects_missing_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE memory_tags")
        reason = db.requires_rebuild(conn)
        assert reason is not None and "memory_tags" in reason

    def test_requires_rebuild_detects_version_mismatch(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "schema_version", "99")
        reason = db.requires_rebuild(conn)
        assert reason is not None and "version" in reason

    def test_migrate_from_empty_initializes(self, paths: Paths) -> None:
        c = db.connect(paths)
        try:
            db.migrate(c)
            assert db.schema_version(c) == db.SCHEMA_VERSION
        finally:
            c.close()

    def test_migrate_refuses_unknown_future_version(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "schema_version", "7")
        with pytest.raises(db.SchemaError, match="rebuild"):
            db.migrate(conn)

    @staticmethod
    def _insert_row(conn: sqlite3.Connection, **overrides: object) -> None:
        row: dict[str, object] = {
            "id": new_id(),
            "type": "fact",
            "title": "t",
            "body": "b",
            "status": "active",
            "sensitivity": "normal",
            "confidence": 0.8,
            "salience": 0.5,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "path": new_id(),
            "content_hash": "h",
            "indexed_at": "2026-01-01T00:00:00Z",
        }
        row.update(overrides)
        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        conn.execute(
            f"INSERT INTO memories({columns}) VALUES({placeholders})",
            row,
        )

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("type", "nonsense"),
            ("status", "zombie"),
            ("sensitivity", "cosmic"),
            ("profile", 2),
        ],
    )
    def test_enum_check_constraints(
        self, conn: sqlite3.Connection, column: str, value: object
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            self._insert_row(conn, **{column: value})

    def test_valid_row_is_accepted(self, conn: sqlite3.Connection) -> None:
        self._insert_row(conn)
        assert conn.execute("SELECT count(*) c FROM memories").fetchone()["c"] == 1

    @pytest.mark.parametrize("column", ["confidence", "salience"])
    @pytest.mark.parametrize("value", [5.0, -0.5])
    def test_unit_range_constraints(
        self, conn: sqlite3.Connection, column: str, value: float
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            self._insert_row(conn, **{column: value})

    def test_path_is_unique(self, conn: sqlite3.Connection) -> None:
        a, b = make_memory(), make_memory()
        db.index_memory(conn, a, path="memories/facts/same.md")
        with pytest.raises(sqlite3.IntegrityError):
            db.index_memory(conn, b, path="memories/facts/same.md")

    def test_cascade_removes_child_rows(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(tags=["python"], entities=["uv"]))
        conn.execute("DELETE FROM memories WHERE id = ?", (m.id,))
        for table in ("memory_tags", "memory_entities"):
            count = conn.execute(
                f"SELECT count(*) c FROM {table} WHERE memory_id = ?",
                (m.id,),
            ).fetchone()["c"]
            assert count == 0, table


class TestIndexMemory:
    def test_inserts_base_row(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Prefers uv"))
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (m.id,)).fetchone()
        assert row["title"] == "Prefers uv"
        assert row["status"] == "active"
        assert row["content_hash"] == m.content_hash

    def test_is_searchable(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Prefers uv over pip"))
        assert fts_ids(conn, "uv") == [m.id]

    def test_tags_and_entities_are_searchable_as_text(self, conn: sqlite3.Connection) -> None:
        """The reason FTS sync is manual: JSON columns must index flattened."""
        m = add(conn, make_memory(tags=["tooling"], entities=["Poetry"]))
        assert fts_ids(conn, "tooling") == [m.id]
        assert fts_ids(conn, "Poetry") == [m.id]

    def test_update_replaces_fts_row(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Original wording"))
        assert fts_ids(conn, "Original") == [m.id]

        updated = m.with_updates(title="Rewritten wording")
        add(conn, updated)

        assert fts_ids(conn, "Original") == []
        assert fts_ids(conn, "Rewritten") == [m.id]

    def test_update_preserves_rowid(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory())
        before = conn.execute("SELECT rowid FROM memories WHERE id=?", (m.id,)).fetchone()[0]
        add(conn, m.with_updates(body="different body"))
        after = conn.execute("SELECT rowid FROM memories WHERE id=?", (m.id,)).fetchone()[0]
        assert before == after

    def test_update_does_not_duplicate_rows(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory())
        for i in range(5):
            add(conn, m.with_updates(body=f"body {i}"))
        assert conn.execute("SELECT count(*) c FROM memories").fetchone()["c"] == 1
        assert conn.execute("SELECT count(*) c FROM memories_fts").fetchone()["c"] == 1

    def test_tags_land_in_side_table(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(tags=["python", "tooling"]))
        rows = conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id=? ORDER BY tag", (m.id,)
        ).fetchall()
        assert [r["tag"] for r in rows] == ["python", "tooling"]

    def test_removed_tags_disappear(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(tags=["python", "tooling"]))
        add(conn, m.with_updates(tags=["python"]))
        rows = conn.execute("SELECT tag FROM memory_tags WHERE memory_id=?", (m.id,)).fetchall()
        assert [r["tag"] for r in rows] == ["python"]

    def test_sources_create_rows_and_links(self, conn: sqlite3.Connection) -> None:
        src = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md", locator="L1-L5")
        m = add(conn, make_memory(sources=[src]))
        assert (
            conn.execute("SELECT count(*) c FROM sources WHERE id=?", (src.source_id,)).fetchone()[
                "c"
            ]
            == 1
        )
        link = conn.execute(
            "SELECT locator FROM memory_sources WHERE memory_id=?", (m.id,)
        ).fetchone()
        assert link["locator"] == "L1-L5"

    def test_shared_source_is_not_duplicated(self, conn: sqlite3.Connection) -> None:
        src = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/shared.md")
        add(conn, make_memory(sources=[src]))
        add(conn, make_memory(sources=[src]))
        assert conn.execute("SELECT count(*) c FROM sources").fetchone()["c"] == 1

    def test_relations_get_forward_and_implied_edges(self, conn: sqlite3.Connection) -> None:
        target = new_id()
        m = add(
            conn,
            make_memory(relations=[Relation(type=RelationType.CAUSES, target=target)]),
        )
        forward = conn.execute(
            "SELECT * FROM relations WHERE src_id=? AND dst_id=? AND implied=0",
            (m.id, target),
        ).fetchone()
        implied = conn.execute(
            "SELECT * FROM relations WHERE src_id=? AND dst_id=? AND implied=1",
            (target, m.id),
        ).fetchone()
        assert forward is not None and implied is not None
        assert forward["type"] == "causes"

    def test_supersedes_materializes_edges(self, conn: sqlite3.Connection) -> None:
        older = new_id()
        m = add(conn, make_memory(supersedes=[older]))
        row = conn.execute(
            "SELECT type FROM relations WHERE src_id=? AND dst_id=? AND implied=0",
            (m.id, older),
        ).fetchone()
        assert row["type"] == "supersedes"

    def test_dangling_relation_target_is_allowed(self, conn: sqlite3.Connection) -> None:
        """A purged target must not block the write; verify reports it instead."""
        add(conn, make_memory(relations=[Relation(type=RelationType.RELATES_TO, target=new_id())]))
        assert db.integrity_check(conn) == []

    def test_assets_are_linked(self, conn: sqlite3.Connection) -> None:
        aid = new_id()
        m = add(conn, make_memory(assets=[aid]))
        row = conn.execute(
            "SELECT asset_id FROM memory_assets WHERE memory_id=?", (m.id,)
        ).fetchone()
        assert row["asset_id"] == aid

    def test_json_columns_preserve_order(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(tags=["alpha", "beta"]))
        row = conn.execute("SELECT tags FROM memories WHERE id=?", (m.id,)).fetchone()
        assert json.loads(row["tags"]) == ["alpha", "beta"]


class TestUnindex:
    def test_removes_base_and_fts_rows(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Ephemeral"))
        db.unindex_memory(conn, m.id)
        assert conn.execute("SELECT count(*) c FROM memories").fetchone()["c"] == 0
        assert fts_ids(conn, "Ephemeral") == []

    def test_removes_outgoing_edges_keeps_incoming(self, conn: sqlite3.Connection) -> None:
        """A purge must not silently rewrite other memories' history."""
        other = add(conn, make_memory(title="Survivor"))
        doomed = add(
            conn,
            make_memory(relations=[Relation(type=RelationType.CAUSES, target=other.id)]),
        )
        add(
            conn,
            other.with_updates(
                relations=[Relation(type=RelationType.RELATES_TO, target=doomed.id)]
            ),
        )

        db.unindex_memory(conn, doomed.id)

        # The purged memory's own declarations go.
        declared = conn.execute(
            "SELECT count(*) c FROM relations WHERE src_id=? AND implied=0", (doomed.id,)
        ).fetchone()["c"]
        assert declared == 0

        # The survivor's declaration about it stays, reported as dangling by
        # verify rather than silently rewritten.
        incoming = conn.execute(
            "SELECT count(*) c FROM relations WHERE dst_id=? AND implied=0", (doomed.id,)
        ).fetchone()["c"]
        assert incoming == 1, "the survivor's own edge must remain"

        # And so does that edge's mirror, which is derived from it: the
        # survivor owns both, so only the survivor's reindex may remove them.
        mirror = conn.execute(
            "SELECT count(*) c FROM relations WHERE src_id=? AND implied=1", (doomed.id,)
        ).fetchone()["c"]
        assert mirror == 1

    def test_implied_edges_are_owned_by_the_declaring_memory(
        self, conn: sqlite3.Connection
    ) -> None:
        """Reindexing one memory must not drop an edge another memory owns.

        Regression: a broad delete cleared implied edges whose src_id matched,
        including ones created by a different memory's frontmatter. Only that
        memory's own reindex would restore them, so the table depended on
        write order and diverged from a rebuild.
        """
        a = add(conn, make_memory(title="A"))
        b = add(conn, make_memory(title="B"))
        # B declares an edge to A, creating the mirror A -> B (implied).
        add(conn, b.with_updates(relations=[Relation(type=RelationType.CAUSES, target=a.id)]))
        mirror_sql = "SELECT count(*) c FROM relations WHERE src_id=? AND dst_id=? AND implied=1"
        assert conn.execute(mirror_sql, (a.id, b.id)).fetchone()["c"] == 1

        # Reindexing A (which declares nothing) must leave B's mirror alone.
        add(conn, a.with_updates(body="unrelated change"))
        assert conn.execute(mirror_sql, (a.id, b.id)).fetchone()["c"] == 1

    def test_is_safe_on_missing_id(self, conn: sqlite3.Connection) -> None:
        db.unindex_memory(conn, new_id())

    def test_integrity_holds_after_unindex(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory())
        db.unindex_memory(conn, m.id)
        assert db.integrity_check(conn) == []


class TestSoftDeleteVisibility:
    """Deleted memories are excluded by status filter, not by index surgery.

    With external content, omitting an FTS row would itself be reported as
    divergence, so the index always mirrors the base table.
    """

    def test_deleted_memory_stays_in_index_but_filters_out(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Soft deleted item"))
        add(conn, m.with_updates(status=MemoryStatus.DELETED))

        assert fts_ids(conn, "Soft") == [m.id], "still indexed"
        rows = conn.execute(
            "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
            "WHERE memories_fts MATCH 'Soft' AND m.status = 'active'"
        ).fetchall()
        assert rows == [], "excluded by the status filter search.py applies"

    def test_integrity_holds(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory())
        add(conn, m.with_updates(status=MemoryStatus.DELETED))
        assert db.integrity_check(conn) == []


class TestRebuildFts:
    def test_restores_searchability(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Findable text", tags=["tagword"]))
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('delete-all')")
        assert fts_ids(conn, "Findable") == []

        db.rebuild_fts(conn)

        assert fts_ids(conn, "Findable") == [m.id]
        assert fts_ids(conn, "tagword") == [m.id], "tags must index flattened, not as JSON"

    def test_integrity_holds_after_rebuild(self, conn: sqlite3.Connection) -> None:
        for i in range(10):
            add(conn, make_memory(title=f"Item {i}", tags=[f"t{i}"]))
        db.rebuild_fts(conn)
        assert db.integrity_check(conn) == []

    def test_json_tags_are_searchable_as_bare_terms(self, conn: sqlite3.Connection) -> None:
        """The tokenizer strips JSON punctuation, so no flattening is needed."""
        m = add(conn, make_memory(tags=["alpha", "beta"]))
        db.rebuild_fts(conn)
        assert fts_ids(conn, "alpha") == [m.id]
        assert fts_ids(conn, "beta") == [m.id]

    def test_rebuild_keeps_index_in_sync_with_base(self, conn: sqlite3.Connection) -> None:
        for i in range(5):
            add(conn, make_memory(title=f"Item {i}", tags=[f"t{i}"], entities=[f"E{i}"]))
        db.rebuild_fts(conn)
        assert db.integrity_check(conn) == []


class TestIntegrityCheck:
    def test_clean_database(self, conn: sqlite3.Connection) -> None:
        add(conn, make_memory())
        assert db.integrity_check(conn) == []

    def test_detects_fts_divergence(self, conn: sqlite3.Connection) -> None:
        """The check that keeps the manual-sync decision honest."""
        add(conn, make_memory(title="Original"))
        conn.execute("UPDATE memories SET title = 'Tampered' WHERE 1=1")
        assert db.integrity_check(conn) != []


class TestMetaAndDirty:
    def test_set_and_get(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "built_at", "2026-10-15")
        assert db.get_meta(conn, "built_at") == "2026-10-15"

    def test_overwrite(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "k", "one")
        db.set_meta(conn, "k", "two")
        assert db.get_meta(conn, "k") == "two"

    def test_missing_key_is_none(self, conn: sqlite3.Connection) -> None:
        assert db.get_meta(conn, "absent") is None

    def test_dirty_ids_round_trip(self, conn: sqlite3.Connection) -> None:
        db.add_dirty(conn, ["b", "a"])
        db.add_dirty(conn, ["a", "c"])
        assert db.take_dirty(conn) == ["a", "b", "c"]
        assert db.take_dirty(conn) == []


class TestOtherRows:
    def test_asset_round_trip(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="a" * 64, media_type="application/pdf", bytes=10)
        db.index_asset(conn, a, blob_path="assets/blobs/aa/aa/x", meta_path="assets/meta/x.json")
        row = conn.execute("SELECT * FROM assets WHERE id=?", (a.id,)).fetchone()
        assert row["sha256"] == "a" * 64
        assert row["version"] == 1

    def test_asset_replace_bumps_version(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="a" * 64, media_type="text/plain", bytes=1)
        db.index_asset(conn, a, blob_path="p", meta_path="m")
        a2 = Asset(
            id=a.id,
            sha256="b" * 64,
            media_type="text/plain",
            bytes=2,
            prev_sha256="a" * 64,
            version=2,
        )
        db.index_asset(conn, a2, blob_path="p2", meta_path="m")
        row = conn.execute("SELECT * FROM assets WHERE id=?", (a.id,)).fetchone()
        assert row["version"] == 2
        assert row["prev_sha256"] == "a" * 64
        assert conn.execute("SELECT count(*) c FROM assets").fetchone()["c"] == 1

    def test_unindex_asset(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="c" * 64, media_type="text/plain", bytes=1)
        db.index_asset(conn, a, blob_path="p", meta_path="m")
        db.unindex_asset(conn, a.id)
        assert conn.execute("SELECT count(*) c FROM assets").fetchone()["c"] == 0

    def test_segment_round_trip(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory())
        s = Segment(text="a span of text", ordinal=0, memory_id=m.id, start_ms=1000, end_ms=2000)
        db.index_segment(conn, s)
        row = conn.execute("SELECT * FROM segments WHERE id=?", (s.id,)).fetchone()
        assert row["text"] == "a span of text"
        assert row["start_ms"] == 1000

    def test_segment_requires_an_owner(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            db.index_segment(conn, Segment(text="orphan", ordinal=0))

    def test_tombstone_round_trip(self, conn: sqlite3.Connection) -> None:
        m = make_memory(sources=[SourceRef(kind=SourceKind.MANUAL, ref="manual")])
        t = tombstone_for(m, permanent=True, reason="user asked")
        db.index_tombstone(conn, t, path=f".tombstones/{m.id}.json")
        row = conn.execute("SELECT * FROM tombstones WHERE memory_id=?", (m.id,)).fetchone()
        assert row["permanent"] == 1
        assert row["content_hash"] == m.content_hash
        assert json.loads(row["source_refs"])[0]["ref"] == "manual"

    def test_derivation_cache_key_is_unique(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="d" * 64, media_type="audio/wav", bytes=5)
        db.index_asset(conn, a, blob_path="p", meta_path="m")
        base = {
            "id": new_id(),
            "asset_id": a.id,
            "kind": "transcript",
            "processor": "whisper",
            "processor_version": "1.0",
            "input_hash": "h1",
            "params_hash": "p1",
            "created_at": "2026-10-15T00:00:00Z",
        }
        db.index_derivation(conn, base)
        with pytest.raises(sqlite3.IntegrityError):
            db.index_derivation(conn, {**base, "id": new_id()})

    def test_derivation_version_bump_is_a_new_row(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="e" * 64, media_type="audio/wav", bytes=5)
        db.index_asset(conn, a, blob_path="p", meta_path="m")
        base = {
            "id": new_id(),
            "asset_id": a.id,
            "kind": "transcript",
            "processor": "whisper",
            "processor_version": "1.0",
            "input_hash": "h1",
            "params_hash": "p1",
            "created_at": "2026-10-15T00:00:00Z",
        }
        db.index_derivation(conn, base)
        db.index_derivation(conn, {**base, "id": new_id(), "processor_version": "2.0"})
        assert conn.execute("SELECT count(*) c FROM derivations").fetchone()["c"] == 2

    def test_derivations_cascade_with_asset(self, conn: sqlite3.Connection) -> None:
        a = Asset(sha256="f" * 64, media_type="audio/wav", bytes=5)
        db.index_asset(conn, a, blob_path="p", meta_path="m")
        db.index_derivation(
            conn,
            {
                "id": new_id(),
                "asset_id": a.id,
                "kind": "transcript",
                "processor": "whisper",
                "processor_version": "1.0",
                "input_hash": "h",
                "params_hash": "p",
                "created_at": "2026-10-15T00:00:00Z",
            },
        )
        conn.execute("DELETE FROM assets WHERE id=?", (a.id,))
        assert conn.execute("SELECT count(*) c FROM derivations").fetchone()["c"] == 0


class TestClearAll:
    def test_empties_every_table(self, conn: sqlite3.Connection) -> None:
        m = add(
            conn,
            make_memory(
                tags=["t"],
                entities=["e"],
                assets=[new_id()],
                sources=[SourceRef(kind=SourceKind.MANUAL, ref="r")],
                relations=[Relation(type=RelationType.CAUSES, target=new_id())],
            ),
        )
        db.index_tombstone(conn, tombstone_for(m, permanent=False), path="t.json")
        db.clear_all(conn)
        for table in db.EXPECTED_TABLES - {"meta"}:
            count = conn.execute(f"SELECT count(*) c FROM {table}").fetchone()["c"]
            assert count == 0, table

    def test_preserves_meta(self, conn: sqlite3.Connection) -> None:
        db.set_meta(conn, "schema_version", "1")
        db.clear_all(conn)
        assert db.get_meta(conn, "schema_version") == "1"

    def test_integrity_holds_after_clear(self, conn: sqlite3.Connection) -> None:
        add(conn, make_memory())
        db.clear_all(conn)
        assert db.integrity_check(conn) == []


class TestSearchPrerequisites:
    """Behaviors search.py will depend on, pinned here."""

    def test_diacritics_are_folded(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Visited the café"))
        assert fts_ids(conn, "cafe") == [m.id]

    def test_hyphenated_token_matches_whole(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Uses claude-code daily"))
        assert fts_ids(conn, '"claude-code"') == [m.id]

    def test_prefix_query(self, conn: sqlite3.Connection) -> None:
        m = add(conn, make_memory(title="Python tooling"))
        assert fts_ids(conn, "pyth*") == [m.id]

    def test_bm25_is_negative_and_orders_by_relevance(self, conn: sqlite3.Connection) -> None:
        """Scoring in search.py relies on bm25 returning negatives."""
        add(conn, make_memory(title="uv uv uv", body="uv uv uv uv"))
        add(conn, make_memory(title="mentions uv once", body="a longer body " * 40))
        rows = conn.execute(
            "SELECT bm25(memories_fts) AS score FROM memories_fts "
            "WHERE memories_fts MATCH 'uv' ORDER BY rank"
        ).fetchall()
        scores = [r["score"] for r in rows]
        assert all(s < 0 for s in scores)
        assert scores == sorted(scores), "more negative (better) must come first"

    def test_status_and_profile_filters_are_usable(self, conn: sqlite3.Connection) -> None:
        add(conn, make_memory(title="Profile item", profile=True))
        add(conn, make_memory(title="Normal item"))
        add(conn, make_memory(title="Gone", status=MemoryStatus.DELETED))
        rows = conn.execute(
            "SELECT id FROM memories WHERE profile=1 AND status='active'"
        ).fetchall()
        assert len(rows) == 1

    def test_secret_sensitivity_is_filterable(self, conn: sqlite3.Connection) -> None:
        add(conn, make_memory(title="Secret thing", sensitivity=Sensitivity.SECRET))
        add(conn, make_memory(title="Normal thing"))
        rows = conn.execute("SELECT id FROM memories WHERE sensitivity != 'secret'").fetchall()
        assert len(rows) == 1
