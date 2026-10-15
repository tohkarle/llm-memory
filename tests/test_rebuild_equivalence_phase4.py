"""Rebuild equivalence for Phase 4 state.

The load-bearing guarantee the whole design rests on, applied to everything
this phase adds: build a corpus using consolidation, extraction and cascade
deletion, dump every table, delete the SQLite file entirely, rebuild from
``data/`` alone, and require the dumps to be equal.

If they diverge, something Phase 4 writes is index-only state, and "the
filesystem is the source of truth" is false for it. Reuses the Phase 1-2
dump helper on purpose: a second, more forgiving comparison here would be a
way to pass without actually holding the invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.backup import export_jsonl, import_jsonl
from memory.cascade import delete_source
from memory.consolidate import apply_all, detect
from memory.extract import commit, from_suggestions
from memory.models import MemoryType, SourceKind, SourceRef
from memory.paths import Paths

from .test_rebuild_equivalence import dump


@pytest.fixture
def api(tmp_path: Path) -> Any:
    instance = MemoryAPI.initialize(tmp_path / "root")
    try:
        yield instance
    finally:
        instance.close()


def rebuilt_dump(api: MemoryAPI) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dump, destroy the index, rebuild from the filesystem, dump again."""
    before = dump(api.conn)
    root = api.paths.root
    api.close()

    paths = Paths(root)
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.index_db) + suffix).unlink(missing_ok=True)
    assert not paths.index_db.exists()

    api.open()
    api.rebuild_index()
    return before, dump(api.conn)


def conversation(api: MemoryAPI, name: str) -> SourceRef:
    path = api.paths.data / "conversations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("A chat log.\n", encoding="utf-8")
    return SourceRef(kind=SourceKind.CONVERSATION, ref=f"conversations/{name}")


def test_consolidated_merges_survive_index_loss(api: MemoryAPI) -> None:
    source = conversation(api, "a.md")
    for _ in range(2):
        api.create(
            type=MemoryType.PREFERENCE,
            title="Prefers uv over pip",
            body="Uses uv for every Python project.",
            sources=[source],
            force_recreate=True,
        )
    report = apply_all(api, detect(api))
    assert report.applied, "the corpus must actually exercise a merge"

    before, after = rebuilt_dump(api)
    assert after == before


def test_recorded_contradictions_survive_index_loss(api: MemoryAPI) -> None:
    source = conversation(api, "b.md")
    api.create(
        type=MemoryType.FACT,
        title="Deploys on Fridays",
        body="Ships to production on Friday afternoons.",
        sources=[source],
    )
    api.create(
        type=MemoryType.FACT,
        title="Does not deploy on Fridays",
        body="Never ships to production on Friday.",
        sources=[source],
    )
    report = apply_all(api, detect(api), kinds=["contradiction"])
    assert any(entry.get("verb") == "relate" for entry in report.applied)

    before, after = rebuilt_dump(api)
    assert after == before
    # The edge itself must be back, not merely an equal-looking empty table.
    edges = api.conn.execute(
        "SELECT count(*) c FROM relations WHERE type = 'contradicts'"
    ).fetchone()
    assert edges["c"] > 0


def test_extracted_memories_survive_index_loss(api: MemoryAPI) -> None:
    source = conversation(api, "c.md")
    report = from_suggestions(
        [
            {"type": "preference", "title": "Prefers uv over pip", "body": "Uses uv."},
            {"type": "fact", "title": "Works in Pacific time", "body": "US west coast."},
            {"type": "decision", "title": "Chose SQLite for the index", "body": "No server."},
        ],
        source=source,
    )
    commit(api, report)
    assert len(report.created) == 3

    before, after = rebuilt_dump(api)
    assert after == before


def test_orphaned_memories_survive_index_loss(api: MemoryAPI) -> None:
    """The orphan tag and review flag live in frontmatter, so a rebuild keeps them."""
    source = conversation(api, "d.md")
    memory = api.create(
        type=MemoryType.FACT, title="A claim from the chat", body="Stated there.", sources=[source]
    )
    delete_source(api, source.ref)

    before, after = rebuilt_dump(api)
    assert after == before
    assert api.orphaned_memories(), "the review queue must survive"
    assert api.require(memory.id).sources == []


def test_cascade_purge_survives_index_loss(api: MemoryAPI) -> None:
    source = conversation(api, "e.md")
    memory = api.create(
        type=MemoryType.FACT, title="A removed claim", body="Gone.", sources=[source]
    )
    delete_source(api, source.ref, memory_policy="purge")

    before, after = rebuilt_dump(api)
    assert after == before
    # And the deletion still holds after the index was reconstructed.
    assert api.get(memory.id) is None
    tomb = api.tombstone(memory.id)
    assert tomb is not None and tomb.permanent


def test_imported_corpus_survives_index_loss(api: MemoryAPI, tmp_path: Path) -> None:
    source = conversation(api, "f.md")
    api.create(
        type=MemoryType.PREFERENCE,
        title="Prefers ruff",
        body="Uses ruff for linting.",
        tags=["tooling"],
        sources=[source],
    )
    lines = export_jsonl(api).lines

    target = MemoryAPI.initialize(tmp_path / "imported")
    try:
        import_jsonl(target, "\n".join(lines))
        before, after = rebuilt_dump(target)
        assert after == before
    finally:
        target.close()


def test_provider_config_is_not_index_state(api: MemoryAPI) -> None:
    """Provider settings are input, not derived state: a rebuild must not touch them."""
    from memory.providers import ProviderConfig, read_config, write_config

    write_config(api.paths.data, ProviderConfig(provider="echo", model="m1"))
    api.rebuild_index()
    assert read_config(api.paths.data).model == "m1"


def test_everything_at_once_survives_index_loss(api: MemoryAPI) -> None:
    """One corpus exercising every Phase 4 write path together."""
    kept = conversation(api, "kept.md")
    doomed = conversation(api, "doomed.md")

    for _ in range(2):
        api.create(
            type=MemoryType.PREFERENCE,
            title="Prefers uv over pip",
            body="Uses uv everywhere.",
            sources=[kept],
            force_recreate=True,
        )
    api.create(
        type=MemoryType.FACT,
        title="Uses tabs for indentation",
        body="Prefers tab characters in source.",
        sources=[kept],
    )
    api.create(
        type=MemoryType.FACT,
        title="Does not use tabs for indentation",
        body="Never uses tab characters.",
        sources=[kept],
    )
    extraction = from_suggestions(
        [{"type": "fact", "title": "Works in Pacific time", "body": "West coast."}],
        source=kept,
    )
    commit(api, extraction)
    api.create(
        type=MemoryType.FACT, title="A doomed claim", body="Cited elsewhere.", sources=[doomed]
    )

    apply_all(api, detect(api), kinds=["duplicate", "contradiction"])
    delete_source(api, doomed.ref)

    before, after = rebuilt_dump(api)
    assert after == before


def test_verify_is_clean_after_a_phase4_rebuild(api: MemoryAPI) -> None:
    source = conversation(api, "g.md")
    for _ in range(2):
        api.create(
            type=MemoryType.FACT,
            title="Prefers uv",
            body="Uses uv.",
            sources=[source],
            force_recreate=True,
        )
    apply_all(api, detect(api))
    rebuilt_dump(api)

    report = api.verify()
    assert report["integrity"] == []
    assert report["fs_only"] == [] and report["db_only"] == []
    assert report["hash_mismatch"] == []


def test_fts_still_finds_consolidated_memories_after_rebuild(api: MemoryAPI) -> None:
    """A merged memory has to remain searchable, not just present."""
    source = conversation(api, "h.md")
    for _ in range(2):
        api.create(
            type=MemoryType.PREFERENCE,
            title="Prefers uv over pip",
            body="Uses uv for every Python project.",
            sources=[source],
            force_recreate=True,
        )
    apply_all(api, detect(api))
    rebuilt_dump(api)

    hits = api.search("uv")
    assert hits, "the surviving memory must still match a keyword query"


def test_the_helper_really_destroys_the_index(api: MemoryAPI, tmp_path: Path) -> None:
    """Guards every test above: a no-op rebuild would make them all vacuous."""
    api.create(type=MemoryType.FACT, title="A claim", body="Body.")
    index = api.paths.index_db
    assert index.exists()

    api.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(index) + suffix).unlink(missing_ok=True)
    assert not index.exists(), "the index must be gone before the rebuild"

    api.open()
    assert api.conn.execute("SELECT count(*) c FROM memories").fetchone()["c"] == 1


def test_dump_helper_is_the_shared_one() -> None:
    """Comparing with a more forgiving dump would be a way to pass dishonestly."""
    import tests.test_rebuild_equivalence as phase12

    assert dump is phase12.dump
