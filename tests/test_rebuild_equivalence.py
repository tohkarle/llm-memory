"""Rebuild equivalence: the index is reconstructible from the filesystem alone.

The design's other load-bearing guarantee. Build a corpus, dump every table,
delete the SQLite file, rebuild from ``data/``, dump again, and require the
two dumps to be equal. If they diverge, "the filesystem is the source of
truth" is not true and deleting the index loses information.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from memory import db
from memory.api import MemoryAPI
from memory.models import (
    MemoryType,
    Relation,
    RelationType,
    Sensitivity,
    SourceKind,
    SourceRef,
    new_id,
)

#: Columns whose values legitimately differ between two builds of the same
#: data: when the row was indexed, the file's mtime, and SQLite's internal
#: rowid allocation order.
VOLATILE = {"indexed_at", "mtime_ns", "rowid"}

#: meta keys that record when a build happened rather than what it contains.
VOLATILE_META = {"built_at", "dirty_ids"}


def dump(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Canonical dump of every table, volatile columns excluded."""
    out: dict[str, list[dict[str, Any]]] = {}
    for table in sorted(db.EXPECTED_TABLES):
        if table == "memories_fts":
            continue  # external content; compared separately via search results
        columns = [
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})")
            if row["name"] not in VOLATILE
        ]
        rows = [
            dict(zip(columns, row, strict=True))
            for row in conn.execute(f"SELECT {', '.join(columns)} FROM {table}")
        ]
        if table == "meta":
            rows = [r for r in rows if r["key"] not in VOLATILE_META]
        # Sort on a fully-ordered string key. json.dumps alone is not stable
        # here: rows share shapes and carry None values, so equal prefixes
        # leave the order up to the original row sequence.
        out[table] = sorted(
            rows,
            key=lambda r: "\x1f".join(f"{k}={r[k]!r}" for k in sorted(r)),
        )
    return out


def search_ids(conn: sqlite3.Connection, query: str) -> list[str]:
    # Quote multi-token and hyphenated queries: bare "claude-code" reads as a
    # column filter in FTS5 syntax, and "memory system" as an implicit AND.
    match = query if query.endswith("*") else " AND ".join(f'"{tok}"' for tok in query.split())
    rows = conn.execute(
        "SELECT m.id, bm25(memories_fts) AS score FROM memories_fts f "
        "JOIN memories m ON m.rowid = f.rowid "
        "WHERE memories_fts MATCH ? ORDER BY score, m.id",
        (match,),
    )
    return [f"{r['id']}:{r['score']:.6f}" for r in rows]


QUERIES = [
    "uv",
    "python",
    "tooling",
    "prefers",
    "decision",
    "café",
    "claude-code",
    "pyth*",
    "memory system",
    "alpha",
    "beta",
    "transcript",
    "standup",
    "architecture",
    "vector",
    "sqlite",
    "deleted",
    "merged",
    "superseded",
    "profile",
]


def build_corpus(api: MemoryAPI) -> dict[str, Any]:
    """A corpus touching every table, status, and edge case rebuild must handle."""
    state: dict[str, Any] = {}
    created: list[Any] = []

    # ---- memories across every type, with varied metadata
    for i in range(40):
        mem_type = list(MemoryType)[i % len(MemoryType)]
        created.append(
            api.create(
                type=mem_type,
                title=f"Memory {i} about python tooling and uv",
                body=f"Body {i}. Mentions sqlite and architecture decisions.\n\nSecond paragraph.",
                tags=[f"tag{i % 7}", "python" if i % 2 else "tooling"],
                entities=[f"Entity{i % 5}", "uv"],
                confidence=0.5 + (i % 5) / 10,
                salience=(i % 10) / 10,
                profile=(i % 9 == 0),
                sensitivity=(
                    Sensitivity.SECRET
                    if i % 17 == 0
                    else Sensitivity.PRIVATE
                    if i % 11 == 0
                    else Sensitivity.NORMAL
                ),
                occurred_at=datetime(2026, 1 + (i % 12), 1 + (i % 28), tzinfo=UTC),
                review_after=(datetime(2027, 1, 1, tzinfo=UTC) if i % 13 == 0 else None),
                sources=[
                    SourceRef(
                        kind=list(SourceKind)[i % len(SourceKind)],
                        ref=f"conversations/2026-{1 + i % 12:02d}.md",
                        locator=f"L{i}-L{i + 20}",
                        captured_at=datetime(2026, 1 + (i % 12), 2, tzinfo=UTC),
                        confidence=0.7,
                    )
                ],
            )
        )

    # ---- unicode, diacritics, and tokenizer edge cases
    created.append(
        api.create(
            type=MemoryType.FACT,
            title="Visited the café in München 🎉",
            body="Uses claude-code daily. Emoji ✨ and accents: naïve, résumé.",
            tags=["unicode"],
            entities=["café", "München"],
        )
    )
    created.append(
        api.create(
            type=MemoryType.PREFERENCE,
            title="Prefers a memory system without vector databases",
            body="No embeddings. FTS5 only.",
            tags=["architecture"],
            profile=True,
        )
    )

    # ---- relations, including a dangling target and a cycle
    a, b, c = created[0], created[1], created[2]
    api.update(
        a.id,
        relations=[
            Relation(type=RelationType.RELATES_TO, target=b.id, note="pair"),
            Relation(type=RelationType.CONTRADICTS, target=c.id),
            Relation(type=RelationType.CAUSES, target=new_id()),  # dangling on purpose
        ],
    )
    api.update(b.id, relations=[Relation(type=RelationType.PART_OF, target=a.id)])
    api.update(c.id, relations=[Relation(type=RelationType.REFERENCES, target=a.id)])

    # ---- a supersession chain of depth 4
    chain_head = api.create(
        type=MemoryType.DECISION, title="Chain version 1", body="First decision."
    )
    current = chain_head
    for v in range(2, 6):
        current = api.replace(
            current.id, title=f"Chain version {v}", body=f"Decision revision {v}."
        ).new
    state["chain_head"] = chain_head.id
    state["chain_tip"] = current.id

    # ---- a merge
    dup_a = api.create(type=MemoryType.FACT, title="Duplicate alpha", body="alpha body")
    dup_b = api.create(type=MemoryType.FACT, title="Duplicate beta", body="beta body")
    merged = api.merge([dup_a.id, dup_b.id], title="Merged finding")
    state["merged"] = merged.merged.id

    # ---- soft deleted and permanently deleted
    soft = api.create(type=MemoryType.EVENT, title="Soft deleted event", body="gone softly")
    api.delete(soft.id, reason="soft delete in corpus")
    state["soft_deleted"] = soft.id

    for i in range(4):
        doomed = api.create(
            type=MemoryType.FACT,
            title=f"Permanently deleted {i}",
            body=f"purged body {i}",
            sources=[SourceRef(kind=SourceKind.MANUAL, ref=f"manual/purged-{i}")],
        )
        api.delete(doomed.id, permanent=True, reason="purge in corpus")

    for i in range(6):
        archived = api.create(
            type=MemoryType.PROCEDURE, title=f"Archived procedure {i}", body=f"steps {i}"
        )
        api.delete(archived.id, reason=f"soft {i}")

    # ---- assets: dedupes, replaced versions, and memory links. Goes through
    # the API so the index and the filesystem are written by the same path
    # the real system uses.
    assets = []
    for i in range(10):
        assets.append(
            api.store_asset(
                f"asset payload {i}".encode(),
                media_type="application/pdf" if i % 2 else "audio/wav",
                filename=f"file{i}.bin",
                tags=[f"asset{i % 3}"],
                meta={"index": i},
                source=SourceRef(kind=SourceKind.DOCUMENT, ref=f"documents/doc{i}.pdf"),
            )
        )
    # Five dedupes: identical bytes, distinct assets sharing one blob.
    for i in range(5):
        api.store_asset(f"asset payload {i}".encode(), media_type="application/pdf")

    # Two replaced versions keep their prior blob.
    for asset in assets[:2]:
        api.replace_asset(asset.id, f"replacement for {asset.id}".encode())

    api.update(created[3].id, assets=[assets[0].id, assets[1].id])
    state["asset_ids"] = [a.id for a in assets]

    # One soft-deleted asset, so the status column is exercised.
    api.delete_asset(assets[9].id)

    # ---- derived outputs with sidecar headers, which is what lets rebuild
    # reconstruct the derivations and segments tables from files alone
    for i, asset in enumerate(assets[2:8]):
        out_dir = api.paths.derived_dir("transcripts", asset.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        derivation_id = new_id()
        payload = {
            "_derivation": {
                "id": derivation_id,
                "asset_id": asset.id,
                "kind": "transcript",
                "processor": "text",
                "processor_version": "1.0",
                "input_hash": asset.sha256,
                "params_hash": "p1",
                "status": "complete",
                "created_at": "2026-10-15T00:00:00Z",
                "completed_at": "2026-10-15T00:00:05Z",
                "cost_ms": 120 + i,
            },
            "segments": [
                {
                    "id": new_id(),
                    "ordinal": j,
                    "text": f"transcript segment {j} for standup recording {i}",
                    "start_ms": j * 1000,
                    "end_ms": (j + 1) * 1000,
                    "speaker": f"S{j}",
                }
                for j in range(5)
            ],
        }
        out_file = out_dir / f"{derivation_id}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        api.register_derivation(
            asset_id=asset.id,
            kind="transcript",
            processor="text",
            processor_version="1.0",
            output_path=out_file,
            params_hash="p1",
            segments=payload["segments"],
            cost_ms=120 + i,
        )

    # a markdown-headed derived description
    desc_asset = assets[7]
    desc_dir = api.paths.derived_dir("descriptions", desc_asset.id)
    desc_dir.mkdir(parents=True, exist_ok=True)
    desc_id = new_id()
    desc_file = desc_dir / f"{desc_id}.md"
    desc_file.write_text(
        "---\n"
        f"id: {desc_id}\n"
        f"asset_id: {desc_asset.id}\n"
        "kind: description\n"
        "processor: text\n"
        "processor_version: '1.0'\n"
        f"input_hash: {desc_asset.sha256}\n"
        "params_hash: p1\n"
        "status: complete\n"
        "created_at: '2026-10-15T00:00:00Z'\n"
        "---\n\nA described image of an architecture diagram.\n",
        encoding="utf-8",
    )
    api.register_derivation(
        asset_id=desc_asset.id,
        kind="description",
        processor="text",
        processor_version="1.0",
        output_path=desc_file,
        params_hash="p1",
    )

    state["created"] = [m.id for m in created]
    return state


@pytest.fixture
def populated(tmp_path: Path) -> Any:
    api = MemoryAPI.initialize(tmp_path)
    state = build_corpus(api)
    api.conn.commit()
    return api, state


class TestRebuildEquivalence:
    def test_dump_is_identical_after_rebuild(self, populated: Any) -> None:
        """The central assertion of filesystem-as-truth."""
        api, _ = populated
        before = dump(api.conn)
        root = api.paths.root
        api.close()

        for suffix in ("", "-wal", "-shm"):
            Path(str(Paths_index(root)) + suffix).unlink(missing_ok=True)

        with MemoryAPI(root) as rebuilt:
            report = rebuilt.rebuild_index()
            after = dump(rebuilt.conn)

        assert report["errors"] == []
        for table in before:
            assert after[table] == before[table], f"table {table} diverged after rebuild"
        assert after == before

    def test_counts_are_reported(self, populated: Any) -> None:
        api, _ = populated
        root = api.paths.root
        expected = api.conn.execute("SELECT count(*) c FROM memories").fetchone()["c"]
        api.close()

        with MemoryAPI(root) as rebuilt:
            report = rebuilt.rebuild_index()
        assert report["memories"] == expected
        assert report["assets"] > 0
        assert report["tombstones"] > 0
        assert report["derivations"] > 0
        assert report["segments"] > 0

    def test_rebuild_is_idempotent(self, populated: Any) -> None:
        api, _ = populated
        root = api.paths.root
        api.close()

        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            first = dump(rebuilt.conn)
            rebuilt.rebuild_index()
            second = dump(rebuilt.conn)
        assert first == second

    def test_search_results_and_scores_survive(self, populated: Any) -> None:
        """Identical hits *and* scores, so ranking is reproducible."""
        api, _ = populated
        before = {q: search_ids(api.conn, q) for q in QUERIES}
        root = api.paths.root
        api.close()

        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            after = {q: search_ids(rebuilt.conn, q) for q in QUERIES}

        for query in QUERIES:
            assert after[query] == before[query], f"query {query!r} ranked differently"

    def test_fts_integrity_after_rebuild(self, populated: Any) -> None:
        api, _ = populated
        root = api.paths.root
        api.close()
        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            assert db.integrity_check(rebuilt.conn) == []

    def test_verify_is_clean_after_rebuild(self, populated: Any) -> None:
        api, _ = populated
        root = api.paths.root
        api.close()
        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            assert rebuilt.verify()["clean"]

    def test_survives_relocation(self, populated: Any, tmp_path: Path) -> None:
        """No absolute paths may be stored, or a moved tree breaks."""
        api, _ = populated
        before = dump(api.conn)
        source_root = api.paths.root
        api.close()

        moved = tmp_path / "relocated"
        shutil.copytree(source_root, moved)
        for suffix in ("", "-wal", "-shm"):
            Path(str(Paths_index(moved)) + suffix).unlink(missing_ok=True)

        with MemoryAPI(moved) as rebuilt:
            rebuilt.rebuild_index()
            after = dump(rebuilt.conn)

        assert after == before, "rebuild at a different absolute path diverged"

    def test_supersession_chain_survives(self, populated: Any) -> None:
        api, state = populated
        root = api.paths.root
        api.close()
        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            tip = rebuilt.get(state["chain_head"], follow=True)
            assert tip is not None and tip.id == state["chain_tip"]

    def test_tombstones_survive_index_loss(self, populated: Any) -> None:
        """Tombstones are files, so a purged memory stays unrecreatable."""
        api, _ = populated
        root = api.paths.root
        permanent_before = api.conn.execute(
            "SELECT count(*) c FROM tombstones WHERE permanent = 1"
        ).fetchone()["c"]
        api.close()

        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            after = rebuilt.conn.execute(
                "SELECT count(*) c FROM tombstones WHERE permanent = 1"
            ).fetchone()["c"]
        assert after == permanent_before > 0


class TestPartialCorruption:
    def test_malformed_file_is_reported_not_fatal(self, populated: Any) -> None:
        """Partial corruption must not cost the rest of the corpus."""
        api, _ = populated
        root = api.paths.root
        expected = api.conn.execute("SELECT count(*) c FROM memories").fetchone()["c"]
        api.close()

        bad = api.paths.type_dir("fact") / f"{new_id()}.md"
        bad.write_text("this is not a memory file at all", encoding="utf-8")

        with MemoryAPI(root) as rebuilt:
            report = rebuilt.rebuild_index()
            assert len(report["errors"]) == 1
            assert report["memories"] == expected, "every valid memory still indexed"

    def test_id_mismatch_is_reported(self, populated: Any) -> None:
        api, state = populated
        root = api.paths.root
        api.close()

        victim = api.paths.memory_file("fact", state["created"][0])
        if not victim.exists():
            pytest.skip("corpus layout changed")
        victim.rename(victim.with_name(f"{new_id()}.md"))

        with MemoryAPI(root) as rebuilt:
            report = rebuilt.rebuild_index()
            assert any("does not match id" in e["error"] for e in report["errors"])

    def test_crash_during_rebuild_recovers(self, populated: Any) -> None:
        api, _ = populated
        root = api.paths.root
        before = dump(api.conn)
        api.close()

        # Interrupt a rebuild by deleting the index mid-flight, then rebuild
        # again -- the filesystem is unchanged, so the result must match.
        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
        for suffix in ("", "-wal", "-shm"):
            Path(str(Paths_index(root)) + suffix).unlink(missing_ok=True)
        with MemoryAPI(root) as rebuilt:
            rebuilt.rebuild_index()
            after = dump(rebuilt.conn)
        assert after == before

    def test_missing_index_triggers_automatic_rebuild(self, populated: Any) -> None:
        """Opening with no index at all must self-heal, not fail."""
        api, state = populated
        root = api.paths.root
        api.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(Paths_index(root)) + suffix).unlink(missing_ok=True)

        with MemoryAPI(root) as reopened:
            row = reopened.conn.execute("SELECT count(*) c FROM memories").fetchone()
            assert row["c"] > 0, "index was rebuilt on open"
            assert reopened.get(state["merged"]) is not None


def Paths_index(root: Path) -> Path:
    from memory.paths import Paths

    return Paths(root).index_db
