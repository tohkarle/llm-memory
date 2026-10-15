"""Dependency-aware deletion: the blast radius is visible, and it is complete.

The property that matters most: after deleting a source, nothing derived
from it survives -- not the transcript, not the segment rows, not the
derivation cache entry -- and a rebuild from the filesystem does not bring
any of it back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.cascade import (
    MEMORY_POLICIES,
    ORPHAN_TAG,
    delete_source,
    execute,
    orphaned_memories,
    plan_deletion,
)
from memory.models import MemoryStatus, MemoryType, SourceKind, SourceRef

CONV_REF = "conversations/2026-10-15.md"


@pytest.fixture
def api(tmp_path: Path) -> Any:
    instance = MemoryAPI.initialize(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def conversation(api: MemoryAPI, ref: str = CONV_REF, text: str = "A chat log.\n") -> SourceRef:
    """Write a raw source file and return the ref memories will cite."""
    path = api.paths.data / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return SourceRef(kind=SourceKind.CONVERSATION, ref=ref)


def cited_memory(api: MemoryAPI, source: SourceRef, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "type": MemoryType.FACT,
        "title": "A claim from the chat",
        "body": "Stated in conversation.",
        "sources": [source],
    }
    defaults.update(kw)
    return api.create(**defaults)


def derived_transcript(api: MemoryAPI, source: SourceRef, *, asset_id: str) -> Path:
    """Register a derivation the way a processor would, header and all."""
    derivation_id = "11111111-2222-3333-4444-555555555555"
    out_dir = api.paths.derived_dir("transcripts", asset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{derivation_id}.json"
    out.write_text(
        json.dumps(
            {
                "_derivation": {
                    "id": derivation_id,
                    "asset_id": asset_id,
                    "source_id": source.source_id,
                    "kind": "transcript",
                    "processor": "text",
                    "processor_version": "1.0",
                    "input_hash": "abc",
                    "params_hash": "default",
                    "status": "complete",
                    "created_at": "2026-10-15T00:00:00Z",
                },
                "segments": [
                    {"ordinal": 0, "text": "first span of the transcript"},
                    {"ordinal": 1, "text": "second span of the transcript"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    api.register_derivation(
        asset_id=asset_id,
        kind="transcript",
        processor="text",
        processor_version="1.0",
        output_path=out,
        segments=[
            {"ordinal": 0, "text": "first span of the transcript"},
            {"ordinal": 1, "text": "second span of the transcript"},
        ],
    )
    return out


# ---------------------------------------------------------------- planning


def test_plan_is_read_only(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    before = api.paths.memory_file(memory.type.value, memory.id).read_bytes()
    raw_before = (api.paths.data / CONV_REF).read_bytes()

    plan = plan_deletion(api, CONV_REF)
    assert plan.memories

    assert api.paths.memory_file(memory.type.value, memory.id).read_bytes() == before
    assert (api.paths.data / CONV_REF).read_bytes() == raw_before


def test_plan_finds_the_raw_file_and_citing_memories(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    plan = plan_deletion(api, CONV_REF)
    assert plan.raw_files == [CONV_REF]
    assert [m["memory_id"] for m in plan.memories] == [memory.id]
    assert plan.source_id == source.source_id


def test_plan_walks_derivations_from_the_source(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    derived_transcript(api, source, asset_id=asset.id)

    plan = plan_deletion(api, CONV_REF)
    assert plan.derivations, "a derivation computed from this source must be in the plan"
    assert plan.derived_files
    assert plan.segments == 2


def test_plan_reports_an_unknown_source_rather_than_failing(api: MemoryAPI) -> None:
    conversation(api, ref="conversations/orphan.md")
    plan = plan_deletion(api, "conversations/orphan.md")
    assert plan.warnings
    # The file still exists and is still the right thing to remove.
    assert plan.raw_files == ["conversations/orphan.md"]


def test_plan_matches_a_bare_filename(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    plan = plan_deletion(api, "2026-10-15.md")
    assert plan.source_id == source.source_id


def test_plan_refuses_an_unknown_memory_policy(api: MemoryAPI) -> None:
    with pytest.raises(ValueError, match="unknown memory policy"):
        plan_deletion(api, CONV_REF, memory_policy="incinerate")


def test_plan_will_not_escape_the_data_directory(api: MemoryAPI) -> None:
    """A ref is a relative path by contract; a crafted one must not aim deletion."""
    outside = api.paths.root / "outside.txt"
    outside.write_text("do not delete me", encoding="utf-8")
    plan = plan_deletion(api, "../outside.txt")
    assert plan.raw_files == []
    assert outside.is_file()


def test_plan_is_json_serializable(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    json.dumps(plan_deletion(api, CONV_REF).to_json())


def test_plan_counts_other_sources_on_a_citing_memory(api: MemoryAPI) -> None:
    source = conversation(api)
    other = SourceRef(kind=SourceKind.DOCUMENT, ref="documents/spec.md")
    cited_memory(api, source, sources=[source, other])
    plan = plan_deletion(api, CONV_REF)
    assert plan.memories[0]["other_sources"] == 1


# ---------------------------------------------------------------- execution


def test_deleting_a_source_removes_derived_state(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    transcript = derived_transcript(api, source, asset_id=asset.id)
    assert transcript.is_file()

    result = delete_source(api, CONV_REF)

    assert not transcript.exists(), "the derived transcript must be gone"
    assert not (api.paths.data / CONV_REF).exists(), "the raw file must be gone"
    assert result.removed_derivations
    assert result.removed_segments == 2
    assert api.conn.execute("SELECT count(*) c FROM derivations").fetchone()["c"] == 0
    assert api.conn.execute("SELECT count(*) c FROM segments").fetchone()["c"] == 0


def test_deleted_source_does_not_come_back_on_rebuild(api: MemoryAPI) -> None:
    """The filesystem is authoritative, so a rebuild must agree the source is gone."""
    source = conversation(api)
    cited_memory(api, source)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    derived_transcript(api, source, asset_id=asset.id)

    delete_source(api, CONV_REF)
    api.rebuild_index()

    assert api.conn.execute("SELECT count(*) c FROM derivations").fetchone()["c"] == 0
    assert api.conn.execute("SELECT count(*) c FROM segments").fetchone()["c"] == 0
    rows = api.conn.execute("SELECT ref FROM sources WHERE ref = ?", (CONV_REF,)).fetchall()
    assert rows == []


def test_orphan_is_the_default_memory_policy(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    result = delete_source(api, CONV_REF)

    assert result.orphaned_memories == [memory.id]
    survivor = api.require(memory.id)
    assert survivor.status is MemoryStatus.ACTIVE, "an orphaned memory is kept"
    assert survivor.sources == [], "the dangling reference is dropped"
    assert ORPHAN_TAG in survivor.tags
    assert survivor.review_after is not None, "it needs a human to re-establish provenance"
    assert survivor.extra["provenance_lost"]["source_ref"] == CONV_REF


def test_orphaning_keeps_other_sources(api: MemoryAPI) -> None:
    source = conversation(api)
    other = SourceRef(kind=SourceKind.DOCUMENT, ref="documents/spec.md")
    memory = cited_memory(api, source, sources=[source, other])
    delete_source(api, CONV_REF)
    survivor = api.require(memory.id)
    assert [s.ref for s in survivor.sources] == ["documents/spec.md"]
    # Provenance still exists, so no review flag is forced.
    assert survivor.review_after is None


def test_orphaned_memories_are_discoverable(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    delete_source(api, CONV_REF)
    assert [m["id"] for m in orphaned_memories(api)] == [memory.id]


def test_delete_policy_soft_deletes_citing_memories(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    result = delete_source(api, CONV_REF, memory_policy="delete", reason="user asked")
    assert result.deleted_memories == [memory.id]
    assert api.require(memory.id).status is MemoryStatus.DELETED
    assert api.tombstone(memory.id) is not None


def test_purge_policy_removes_memories_permanently_with_tombstones(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    delete_source(api, CONV_REF, memory_policy="purge", reason="user asked")

    assert api.get(memory.id) is None
    tomb = api.tombstone(memory.id)
    assert tomb is not None and tomb.permanent


def test_purged_memory_cannot_be_recreated_from_a_retained_copy(api: MemoryAPI) -> None:
    """The retention rule: a permanent deletion must not be re-extractable."""
    source = conversation(api)
    memory = cited_memory(api, source, title="A removed claim", body="Gone for good.")
    delete_source(api, CONV_REF, memory_policy="purge", remove_raw=False)

    from memory.store import Tombstoned

    with pytest.raises(Tombstoned):
        api.create(
            type=MemoryType.FACT,
            title="A removed claim",
            body="Gone for good.",
            sources=[source],
        )
    assert memory.id not in {m.id for m in api.store.iter_all(include_deleted=True)}


def test_keep_policy_leaves_memories_untouched(api: MemoryAPI) -> None:
    source = conversation(api)
    memory = cited_memory(api, source)
    before = api.paths.memory_file(memory.type.value, memory.id).read_bytes()
    result = delete_source(api, CONV_REF, memory_policy="keep")
    assert result.kept_memories == [memory.id]
    assert api.paths.memory_file(memory.type.value, memory.id).read_bytes() == before


def test_remove_raw_false_keeps_the_source_file(api: MemoryAPI) -> None:
    """For reclaiming space from derived state without discarding the original."""
    source = conversation(api)
    cited_memory(api, source)
    asset = api.store_asset(b"bytes", media_type="text/plain", source=source)
    transcript = derived_transcript(api, source, asset_id=asset.id)

    delete_source(api, CONV_REF, memory_policy="keep", remove_raw=False)
    assert (api.paths.data / CONV_REF).is_file()
    assert not transcript.exists()


def test_execute_reports_errors_without_aborting(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    plan = plan_deletion(api, CONV_REF)
    # Name a memory that no longer exists, as if it were deleted between
    # planning and execution.
    plan.memories.append({"memory_id": "00000000-0000-4000-8000-000000000000", "title": "ghost"})
    result = execute(api, plan)
    assert result.errors
    assert result.orphaned_memories, "the real memory was still processed"


def test_result_is_json_serializable(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    json.dumps(delete_source(api, CONV_REF).to_json())


def test_deleting_a_source_with_nothing_attached_is_harmless(api: MemoryAPI) -> None:
    plan = plan_deletion(api, "conversations/never-existed.md")
    assert plan.empty
    result = execute(api, plan)
    assert not result.errors


def test_every_policy_is_executable(api: MemoryAPI) -> None:
    """Guards the policy list against drift from the execute() dispatch."""
    for policy in MEMORY_POLICIES:
        ref = f"conversations/{policy}.md"
        source = conversation(api, ref=ref)
        cited_memory(api, source, title=f"Claim for {policy}", force_recreate=True)
        result = delete_source(api, ref, memory_policy=policy)
        assert not result.errors, (policy, result.errors)


def test_cascade_leaves_the_index_consistent(api: MemoryAPI) -> None:
    source = conversation(api)
    cited_memory(api, source)
    asset = api.store_asset(b"bytes", media_type="text/plain", source=source)
    derived_transcript(api, source, asset_id=asset.id)
    delete_source(api, CONV_REF, memory_policy="purge")

    report = api.verify()
    assert report["integrity"] == []
    assert report["fs_only"] == [] and report["db_only"] == []


def test_assets_imported_from_the_source_are_in_the_plan(api: MemoryAPI) -> None:
    """An asset's origin lives in its sidecar, not in the assets index table.

    Deriving this from the index found nothing, and because `rebuild_index`
    re-derives `sources` from asset sidecars, the deleted source row came
    straight back on the next rebuild.
    """
    source = conversation(api)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    plan = plan_deletion(api, CONV_REF)
    assert [a["id"] for a in plan.assets] == [asset.id]


def test_source_deletion_removes_the_assets_that_cite_it(api: MemoryAPI) -> None:
    source = conversation(api)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    result = delete_source(api, CONV_REF)

    assert result.removed_assets == [asset.id]
    assert api.get_asset(asset.id) is None
    assert not api.paths.asset_meta(asset.id).exists()


def test_keep_policy_also_keeps_assets(api: MemoryAPI) -> None:
    """`keep` is for reclaiming derived state, not for discarding originals."""
    source = conversation(api)
    asset = api.store_asset(b"raw bytes", media_type="text/plain", source=source)
    result = delete_source(api, CONV_REF, memory_policy="keep")
    assert result.removed_assets == []
    assert api.get_asset(asset.id) is not None
