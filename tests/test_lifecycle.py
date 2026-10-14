"""Supersession: replace, merge, and the chains they create."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.models import (
    MemoryStatus,
    MemoryType,
    Relation,
    RelationType,
    Sensitivity,
    SourceKind,
    SourceRef,
    new_id,
)
from memory.store import Conflict


@pytest.fixture
def api(tmp_path: Path) -> Any:
    instance = MemoryAPI.initialize(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def seed(api: MemoryAPI, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "type": MemoryType.FACT,
        "title": "Original claim",
        "body": "Original body.",
    }
    defaults.update(kw)
    return api.create(**defaults)


class TestReplace:
    def test_mints_a_new_id(self, api: MemoryAPI) -> None:
        old = seed(api)
        result = api.replace(old.id, title="Revised claim", body="New body.")
        assert result.new.id != old.id

    def test_old_version_survives_with_full_body(self, api: MemoryAPI) -> None:
        """Supersession is versioning, not deletion: history stays readable."""
        old = seed(api, body="The body that must survive.")
        api.replace(old.id, title="Revised", body="New body.")

        back = api.get(old.id)
        assert back is not None
        assert back.status is MemoryStatus.SUPERSEDED
        assert back.body == "The body that must survive."

    def test_sets_both_directions(self, api: MemoryAPI) -> None:
        old = seed(api)
        result = api.replace(old.id, title="Revised", body="b")

        superseded = api.get(old.id)
        assert superseded is not None
        assert superseded.superseded_by == result.new.id
        assert old.id in result.new.supersedes

    def test_sets_valid_until(self, api: MemoryAPI) -> None:
        old = seed(api)
        api.replace(old.id, title="Revised", body="b")
        back = api.get(old.id)
        assert back is not None and back.valid_until is not None

    def test_writes_no_tombstone(self, api: MemoryAPI) -> None:
        """Nothing was deleted, so nothing should be blocked from recreation."""
        old = seed(api)
        api.replace(old.id, title="Revised", body="b")
        assert api.tombstone(old.id) is None

    def test_inherits_context(self, api: MemoryAPI) -> None:
        src = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")
        old = seed(api, tags=["python"], entities=["uv"], sources=[src], profile=True)
        result = api.replace(old.id, title="Revised", body="b")

        assert result.new.tags == ["python"]
        assert result.new.entities == ["uv"]
        assert result.new.profile is True
        assert [s.ref for s in result.new.sources] == ["conversations/a.md"]

    def test_overrides_win_over_inheritance(self, api: MemoryAPI) -> None:
        old = seed(api, tags=["old"], confidence=0.5)
        result = api.replace(old.id, title="Revised", body="b", tags=["new"], confidence=0.95)
        assert result.new.tags == ["new"]
        assert result.new.confidence == pytest.approx(0.95)

    def test_can_change_type(self, api: MemoryAPI) -> None:
        old = seed(api, type=MemoryType.FACT)
        result = api.replace(old.id, title="Now a decision", body="b", type=MemoryType.DECISION)
        assert result.new.type is MemoryType.DECISION
        assert api.get(old.id).type is MemoryType.FACT  # type: ignore[union-attr]

    def test_follow_returns_the_live_tip(self, api: MemoryAPI) -> None:
        first = seed(api)
        second = api.replace(first.id, title="Second", body="b").new
        third = api.replace(second.id, title="Third", body="b").new

        tip = api.get(first.id, follow=True)
        assert tip is not None and tip.id == third.id

    def test_deep_chain_resolves(self, api: MemoryAPI) -> None:
        current = seed(api)
        ids = [current.id]
        for i in range(4):
            current = api.replace(current.id, title=f"Version {i + 2}", body="b").new
            ids.append(current.id)

        tip = api.get(ids[0], follow=True)
        assert tip is not None and tip.id == ids[-1]
        for stale in ids[:-1]:
            memory = api.get(stale)
            assert memory is not None and memory.status is MemoryStatus.SUPERSEDED

    def test_cycle_terminates(self, api: MemoryAPI) -> None:
        """A hand-corrupted cycle must not hang the resolver."""
        a = seed(api, title="A")
        b = seed(api, title="B")
        api.update(a.id, status="active")
        # Force a cycle directly through the store, bypassing replace().
        with api.journal:
            api.store.overwrite(api.require(a.id).with_updates(superseded_by=b.id))
            api.store.overwrite(api.require(b.id).with_updates(superseded_by=a.id))

        tip = api.get(a.id, follow=True)
        assert tip is not None

    def test_index_hides_superseded_by_default(self, api: MemoryAPI) -> None:
        old = seed(api, title="Findable original")
        api.replace(old.id, title="Findable revision", body="b")
        rows = api.conn.execute("SELECT id FROM memories WHERE status = 'active'").fetchall()
        assert len(rows) == 1

    def test_relation_edges_are_recorded(self, api: MemoryAPI) -> None:
        old = seed(api)
        result = api.replace(old.id, title="Revised", body="b")
        forward = api.conn.execute(
            "SELECT * FROM relations WHERE src_id=? AND dst_id=? AND type='supersedes'",
            (result.new.id, old.id),
        ).fetchone()
        assert forward is not None

    def test_deleted_memory_cannot_be_replaced(self, api: MemoryAPI) -> None:
        old = seed(api)
        api.delete(old.id)
        with pytest.raises(Conflict, match="deleted"):
            api.replace(old.id, title="Revised", body="b")

    def test_verify_is_clean_after_replace(self, api: MemoryAPI) -> None:
        old = seed(api)
        api.replace(old.id, title="Revised", body="b")
        assert api.verify()["clean"]


class TestMerge:
    def test_mints_a_new_id_by_default(self, api: MemoryAPI) -> None:
        a = seed(api, title="Duplicate A")
        b = seed(api, title="Duplicate B", body="Other body.")
        result = api.merge([a.id, b.id])
        assert result.merged.id not in (a.id, b.id)

    def test_parents_are_superseded_and_intact(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", body="Body A.")
        b = seed(api, title="B", body="Body B.")
        result = api.merge([a.id, b.id])

        for parent_id in (a.id, b.id):
            parent = api.get(parent_id)
            assert parent is not None
            assert parent.status is MemoryStatus.SUPERSEDED
            assert parent.superseded_by == result.merged.id
            assert parent.body, "parent body must survive the merge"

    def test_source_union_is_complete(self, api: MemoryAPI) -> None:
        """The critical property: provenance must not narrow."""
        src_a = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")
        src_b = SourceRef(kind=SourceKind.DOCUMENT, ref="documents/b.pdf", locator="p.3")
        src_c = SourceRef(kind=SourceKind.MANUAL, ref="manual")

        a = seed(api, title="A", sources=[src_a, src_c])
        b = seed(api, title="B", body="other", sources=[src_b])
        result = api.merge([a.id, b.id])

        refs = {s.ref for s in result.merged.sources}
        assert refs == {"conversations/a.md", "documents/b.pdf", "manual"}

    def test_duplicate_sources_are_deduped(self, api: MemoryAPI) -> None:
        src = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/same.md")
        a = seed(api, title="A", sources=[src])
        b = seed(api, title="B", body="other", sources=[src])
        result = api.merge([a.id, b.id])
        assert len(result.merged.sources) == 1

    def test_tags_and_entities_union(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", tags=["python"], entities=["uv"])
        b = seed(api, title="B", body="other", tags=["tooling"], entities=["pip"])
        result = api.merge([a.id, b.id])
        assert result.merged.tags == ["python", "tooling"]
        assert set(result.merged.entities) == {"uv", "pip"}

    def test_confidence_and_salience_take_the_max(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", confidence=0.4, salience=0.2)
        b = seed(api, title="B", body="other", confidence=0.9, salience=0.8)
        result = api.merge([a.id, b.id])
        assert result.merged.confidence == pytest.approx(0.9)
        assert result.merged.salience == pytest.approx(0.8)

    def test_created_and_occurred_take_the_min(self, api: MemoryAPI) -> None:
        early = datetime(2020, 1, 1, tzinfo=UTC)
        late = datetime(2026, 1, 1, tzinfo=UTC)
        a = seed(api, title="A", occurred_at=late)
        b = seed(api, title="B", body="other", occurred_at=early)
        result = api.merge([a.id, b.id])
        assert result.merged.occurred_at == early

    def test_profile_and_sensitivity_are_the_strongest(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", profile=False, sensitivity=Sensitivity.NORMAL)
        b = seed(api, title="B", body="other", profile=True, sensitivity=Sensitivity.SECRET)
        result = api.merge([a.id, b.id])
        assert result.merged.profile is True
        assert result.merged.sensitivity is Sensitivity.SECRET

    def test_body_carries_parent_sections(self, api: MemoryAPI) -> None:
        a = seed(api, title="First claim", body="Body of first.")
        b = seed(api, title="Second claim", body="Body of second.")
        result = api.merge([a.id, b.id])
        assert "Body of first." in result.merged.body
        assert "Body of second." in result.merged.body
        assert "First claim" in result.merged.body

    def test_explicit_body_overrides(self, api: MemoryAPI) -> None:
        a = seed(api, title="A")
        b = seed(api, title="B", body="other")
        result = api.merge([a.id, b.id], title="Consolidated", body="Hand written.")
        assert result.merged.body == "Hand written."
        assert result.merged.title == "Consolidated"

    def test_into_keeps_the_target_id(self, api: MemoryAPI) -> None:
        a = seed(api, title="Keeper")
        b = seed(api, title="Folded", body="other")
        result = api.merge([a.id, b.id], into=a.id)

        assert result.merged.id == a.id
        folded = api.get(b.id)
        assert folded is not None and folded.superseded_by == a.id
        keeper = api.get(a.id)
        assert keeper is not None and keeper.status is MemoryStatus.ACTIVE

    def test_into_must_be_one_of_the_ids(self, api: MemoryAPI) -> None:
        a = seed(api, title="A")
        b = seed(api, title="B", body="other")
        with pytest.raises(ValueError, match="must be one of"):
            api.merge([a.id, b.id], into=new_id())

    def test_supersedes_lists_every_parent(self, api: MemoryAPI) -> None:
        ids = [seed(api, title=f"Dup {i}", body=f"body {i}").id for i in range(3)]
        result = api.merge(ids)
        assert sorted(result.merged.supersedes) == sorted(ids)

    def test_three_way_merge(self, api: MemoryAPI) -> None:
        ids = [seed(api, title=f"Dup {i}", body=f"body {i}").id for i in range(3)]
        result = api.merge(ids)
        assert len(result.superseded) == 3
        for mid in ids:
            memory = api.get(mid)
            assert memory is not None and memory.status is MemoryStatus.SUPERSEDED

    def test_self_referential_relations_are_dropped(self, api: MemoryAPI) -> None:
        """A relation between two parents is meaningless once they collapse."""
        a = seed(api, title="A")
        b = seed(
            api,
            title="B",
            body="other",
            relations=[Relation(type=RelationType.RELATES_TO, target=a.id)],
        )
        result = api.merge([a.id, b.id])
        assert all(r.target != a.id for r in result.merged.relations)
        assert all(r.target != b.id for r in result.merged.relations)

    def test_external_relations_survive(self, api: MemoryAPI) -> None:
        outside = seed(api, title="Outside", body="unrelated")
        a = seed(
            api,
            title="A",
            relations=[Relation(type=RelationType.CAUSES, target=outside.id)],
        )
        b = seed(api, title="B", body="other")
        result = api.merge([a.id, b.id])
        assert any(r.target == outside.id for r in result.merged.relations)

    def test_needs_at_least_two(self, api: MemoryAPI) -> None:
        a = seed(api)
        with pytest.raises(ValueError, match="at least two"):
            api.merge([a.id])

    def test_rejects_unknown_strategy(self, api: MemoryAPI) -> None:
        a = seed(api, title="A")
        b = seed(api, title="B", body="other")
        with pytest.raises(ValueError, match="unknown merge strategy"):
            api.merge([a.id, b.id], strategy="vibes")

    def test_prefer_newest_leads_with_newest(self, api: MemoryAPI) -> None:
        older = datetime(2020, 1, 1, tzinfo=UTC)
        a = seed(api, title="Older", body="older body", type=MemoryType.FACT)
        b = seed(api, title="Newer", body="newer body", type=MemoryType.DECISION)
        # Timestamps are second-precision, so push one clearly into the past
        # rather than relying on creation order within the same second.
        with api.journal:
            api.store.overwrite(api.require(a.id).with_updates(updated_at=older))

        result = api.merge([a.id, b.id], strategy="prefer-newest")
        assert result.merged.type is MemoryType.DECISION
        assert result.merged.body.index("newer body") < result.merged.body.index("older body")

    def test_merge_order_is_deterministic_on_ties(self, api: MemoryAPI) -> None:
        """Same-second timestamps must still produce a reproducible merge."""
        a = seed(api, title="A", body="alpha body")
        b = seed(api, title="B", body="beta body")
        first = api.merge([a.id, b.id], strategy="prefer-newest")
        bodies = [first.merged.body]

        c = seed(api, title="A", body="alpha body")
        d = seed(api, title="B", body="beta body")
        second = api.merge([c.id, d.id], strategy="prefer-newest")
        bodies.append(second.merged.body)

        import re

        normalized = [re.sub(r"\([0-9a-f]{8}\)", "(id)", b) for b in bodies]
        assert normalized[0] == normalized[1]

    def test_prefer_highest_confidence_leads(self, api: MemoryAPI) -> None:
        a = seed(api, title="Unsure", body="unsure body", confidence=0.3)
        b = seed(api, title="Sure", body="sure body", confidence=0.95)
        result = api.merge([a.id, b.id], strategy="prefer-highest-confidence")
        assert result.merged.body.index("sure body") < result.merged.body.index("unsure body")

    def test_deleted_memory_cannot_be_merged(self, api: MemoryAPI) -> None:
        a = seed(api, title="A")
        b = seed(api, title="B", body="other")
        api.delete(b.id)
        with pytest.raises(Conflict, match="deleted"):
            api.merge([a.id, b.id])

    def test_duplicate_ids_are_collapsed(self, api: MemoryAPI) -> None:
        a = seed(api, title="A")
        b = seed(api, title="B", body="other")
        result = api.merge([a.id, b.id, a.id])
        assert len(result.superseded) == 2

    def test_verify_is_clean_after_merge(self, api: MemoryAPI) -> None:
        ids = [seed(api, title=f"Dup {i}", body=f"b{i}").id for i in range(3)]
        api.merge(ids)
        assert api.verify()["clean"]

    def test_merged_memory_is_searchable(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", body="distinctive alpha text")
        b = seed(api, title="B", body="distinctive beta text")
        result = api.merge([a.id, b.id], title="Consolidated finding")
        row = api.conn.execute(
            "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
            "WHERE memories_fts MATCH 'alpha' AND m.status = 'active'"
        ).fetchone()
        assert row is not None and row["id"] == result.merged.id


class TestGcBlobs:
    def test_reports_orphans_without_deleting(self, api: MemoryAPI) -> None:
        from memory.lifecycle import gc_blobs

        with api.journal:
            blob = api.store.put_blob(b"unreferenced bytes")
        report = gc_blobs(api.paths, api.conn, dry_run=True)
        assert report["removed"] == [api.paths.rel(blob.path)]
        assert blob.path.exists()

    def test_removes_orphans_when_asked(self, api: MemoryAPI) -> None:
        from memory.lifecycle import gc_blobs

        with api.journal:
            blob = api.store.put_blob(b"unreferenced bytes")
        gc_blobs(api.paths, api.conn, dry_run=False)
        assert not blob.path.exists()

    def test_keeps_referenced_blobs(self, api: MemoryAPI) -> None:
        from memory.lifecycle import gc_blobs
        from memory.models import Asset

        with api.journal:
            blob = api.store.put_blob(b"referenced bytes")
            asset = Asset(sha256=blob.sha256, media_type="text/plain", bytes=blob.bytes)
            api.store.write_asset_meta(asset)
        from memory import db as db_module

        db_module.index_asset(
            api.conn,
            asset,
            blob_path=api.paths.rel(blob.path),
            meta_path=api.paths.rel(api.paths.asset_meta(asset.id)),
        )
        api.conn.commit()

        report = gc_blobs(api.paths, api.conn, dry_run=False)
        assert report["removed"] == []
        assert blob.path.exists()
