"""Query sanitation, filters, ranking, and relation traversal."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

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
from memory.search import (
    EDGE_WEIGHT,
    SearchError,
    SearchFilters,
    fts_query,
    normalize_bm25,
    recency_decay,
)


@pytest.fixture
def api(tmp_path: Path) -> Iterator[MemoryAPI]:
    instance = MemoryAPI.initialize(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def seed(api: MemoryAPI, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "type": MemoryType.FACT,
        "title": "A memory",
        "body": "Some body text.",
    }
    defaults.update(kw)
    return api.create(**defaults)


def ids(hits: list[Any]) -> list[str]:
    return [h.memory_id for h in hits]


class TestFtsQuerySanitation:
    @pytest.mark.parametrize(
        "raw",
        [
            '"',
            "*",
            "NEAR(",
            "a OR b",
            "AND",
            "-x",
            '""',
            "(((",
            "a:b",
            "^anchor",
            "{brace}",
            "col^2",
            "what's up?",
            "100% sure",
            "a~b",
            "[bracket]",
        ],
    )
    def test_never_raises_on_hostile_input(self, api: MemoryAPI, raw: str) -> None:
        """FTS5 syntax in user text must not become a SQL error.

        Such a query either matches nothing or matches normally; it never
        raises, so callers need not sanitize before searching.
        """
        seed(api, title="Ordinary memory", body="nothing special")
        assert isinstance(api.search(raw), list)

    def test_hyphenated_term_is_quoted(self, api: MemoryAPI) -> None:
        target = seed(api, title="Uses claude-code daily", body="cli tooling")
        assert ids(api.search("claude-code")) == [target.id]

    def test_multi_term_queries_rank_rather_than_exclude(self, api: MemoryAPI) -> None:
        """A missing term must lower a hit's rank, not remove it.

        Requiring every term made natural-language questions unanswerable:
        "how do I rebuild the index" matched nothing, not even a memory
        titled "How to rebuild the memory index".
        """
        both = seed(api, title="python and uv together", body="x")
        partial = seed(api, title="python only", body="y")
        found = ids(api.search("python uv"))
        assert found[0] == both.id, "matching more terms must rank higher"
        assert partial.id in found, "a partial match stays reachable"

    def test_stopwords_are_dropped(self, api: MemoryAPI) -> None:
        """A question is mostly function words; requiring them matches nothing."""
        target = seed(api, title="How to rebuild the memory index", body="steps")
        assert target.id in ids(api.search("how do I rebuild the index"))

    def test_prefix_matching_covers_simple_inflections(self, api: MemoryAPI) -> None:
        """No stemmer, so multi-term queries prefix-match instead."""
        target = seed(api, title="Runs the test suite often", body="pytest")
        assert target.id in ids(api.search("running tests suite"))

    def test_single_term_is_matched_exactly(self, api: MemoryAPI) -> None:
        seed(api, title="Prefers uv", body="x")
        other = seed(api, title="Unrelated entirely", body="y")
        assert other.id not in ids(api.search("uv"))

    def test_a_query_of_only_stopwords_still_searches(self, api: MemoryAPI) -> None:
        seed(api, title="The thing", body="x")
        assert isinstance(api.search("the"), list)

    def test_prefix_search(self, api: MemoryAPI) -> None:
        target = seed(api, title="Python tooling", body="x")
        assert target.id in ids(api.search("pyth*"))

    def test_diacritics_are_folded(self, api: MemoryAPI) -> None:
        target = seed(api, title="Visited the café", body="x")
        assert ids(api.search("cafe")) == [target.id]

    def test_verbatim_escape_hatch_allows_or(self, api: MemoryAPI) -> None:
        """The documented workaround for the paraphrase gap."""
        a = seed(api, title="Prefers uv", body="x")
        b = seed(api, title="Dislikes pip", body="y")
        found = ids(api.search("fts:uv OR pip"))
        assert set(found) == {a.id, b.id}

    def test_empty_query_raises_in_builder(self) -> None:
        with pytest.raises(SearchError, match="empty query"):
            fts_query("   ")

    def test_empty_verbatim_raises(self) -> None:
        with pytest.raises(SearchError, match="empty fts:"):
            fts_query("fts:   ")

    def test_punctuation_only_raises(self) -> None:
        with pytest.raises(SearchError, match="no searchable terms"):
            fts_query("*** ((( ")

    def test_malformed_verbatim_is_reported_as_search_error(self, api: MemoryAPI) -> None:
        seed(api)
        with pytest.raises(SearchError, match="invalid query"):
            api.search("fts:NEAR(((")


class TestFilters:
    def test_defaults_to_active_only(self, api: MemoryAPI) -> None:
        live = seed(api, title="Live memory", body="findable")
        gone = seed(api, title="Gone memory", body="findable")
        api.delete(gone.id)
        old = seed(api, title="Old memory", body="findable")
        api.replace(old.id, title="New memory", body="findable")

        found = ids(api.search("findable"))
        assert live.id in found
        assert gone.id not in found
        assert old.id not in found

    def test_explicit_status_includes_superseded(self, api: MemoryAPI) -> None:
        old = seed(api, title="Original", body="findable")
        api.replace(old.id, title="Revision", body="findable")
        found = ids(api.search("findable", statuses=["superseded"]))
        assert found == [old.id]

    def test_type_filter(self, api: MemoryAPI) -> None:
        fact = seed(api, type=MemoryType.FACT, title="A fact", body="shared")
        seed(api, type=MemoryType.EVENT, title="An event", body="shared")
        assert ids(api.search("shared", types=["fact"])) == [fact.id]

    def test_unknown_type_raises(self, api: MemoryAPI) -> None:
        with pytest.raises(SearchError, match="unknown memory type"):
            api.search("x", types=["nonsense"])

    def test_tags_any(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", body="x", tags=["python"])
        b = seed(api, title="B", body="x", tags=["rust"])
        seed(api, title="C", body="x", tags=["go"])
        found = set(ids(api.search(None, tags_any=["python", "rust"])))
        assert found == {a.id, b.id}

    def test_tags_all(self, api: MemoryAPI) -> None:
        both = seed(api, title="Both", body="x", tags=["python", "tooling"])
        seed(api, title="One", body="x", tags=["python"])
        assert ids(api.search(None, tags_all=["python", "tooling"])) == [both.id]

    def test_entities_any_is_case_insensitive(self, api: MemoryAPI) -> None:
        target = seed(api, title="A", body="x", entities=["Poetry"])
        assert ids(api.search(None, entities_any=["poetry"])) == [target.id]

    def test_profile_filter(self, api: MemoryAPI) -> None:
        flagged = seed(api, title="Profile item", body="x", profile=True)
        seed(api, title="Normal item", body="x")
        assert ids(api.search(None, profile=True)) == [flagged.id]

    def test_min_confidence(self, api: MemoryAPI) -> None:
        sure = seed(api, title="Sure", body="x", confidence=0.95)
        seed(api, title="Unsure", body="x", confidence=0.2)
        assert ids(api.search(None, min_confidence=0.9)) == [sure.id]

    def test_secret_is_excluded_by_default(self, api: MemoryAPI) -> None:
        """A secret memory must not surface unless explicitly requested."""
        secret = seed(api, title="Secret thing", body="hidden", sensitivity=Sensitivity.SECRET)
        normal = seed(api, title="Normal thing", body="hidden")

        default = ids(api.search("hidden"))
        assert default == [normal.id]

        with_secret = ids(api.search("hidden", sensitivity_max="secret"))
        assert set(with_secret) == {secret.id, normal.id}

    def test_private_is_included_by_default(self, api: MemoryAPI) -> None:
        private = seed(api, title="Private", body="x", sensitivity=Sensitivity.PRIVATE)
        assert private.id in ids(api.search(None))

    def test_unknown_sensitivity_raises(self, api: MemoryAPI) -> None:
        with pytest.raises(SearchError, match="unknown sensitivity"):
            api.search(None, sensitivity_max="cosmic")

    def test_occurred_range(self, api: MemoryAPI) -> None:
        early = seed(api, title="Early", body="x", occurred_at=datetime(2020, 1, 1, tzinfo=UTC))
        late = seed(api, title="Late", body="x", occurred_at=datetime(2026, 1, 1, tzinfo=UTC))
        found = ids(api.search(None, occurred_after=datetime(2023, 1, 1, tzinfo=UTC)))
        assert found == [late.id]
        assert early.id not in found

    def test_created_range(self, api: MemoryAPI) -> None:
        target = seed(api, title="Now", body="x")
        future = datetime.now(UTC) + timedelta(days=1)
        assert ids(api.search(None, created_before=future)) == [target.id]
        assert ids(api.search(None, created_after=future)) == []

    def test_source_ref_substring(self, api: MemoryAPI) -> None:
        target = seed(
            api,
            title="From a chat",
            body="x",
            sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/2026-10-15.md")],
        )
        seed(api, title="No source", body="x")
        assert ids(api.search(None, source_ref="2026-10-15")) == [target.id]

    def test_related_to(self, api: MemoryAPI) -> None:
        anchor = seed(api, title="Anchor", body="x")
        linked = seed(
            api,
            title="Linked",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=anchor.id)],
        )
        found = ids(api.search(None, related_to=anchor.id))
        assert linked.id in found

    def test_exclude_ids(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", body="shared")
        b = seed(api, title="B", body="shared")
        assert ids(api.search("shared", exclude_ids=[a.id])) == [b.id]

    def test_review_due(self, api: MemoryAPI) -> None:
        due = seed(api, title="Due", body="x", review_after=datetime(2020, 1, 1, tzinfo=UTC))
        seed(api, title="Later", body="x", review_after=datetime(2099, 1, 1, tzinfo=UTC))
        seed(api, title="Never", body="x")
        assert ids(api.review_due()) == [due.id]

    def test_combined_filters(self, api: MemoryAPI) -> None:
        target = seed(
            api,
            type=MemoryType.PREFERENCE,
            title="Prefers uv for python",
            body="packaging",
            tags=["tooling"],
            confidence=0.9,
            profile=True,
        )
        seed(api, type=MemoryType.FACT, title="Prefers uv too", body="packaging")
        found = ids(
            api.search(
                "uv",
                types=["preference"],
                tags_any=["tooling"],
                min_confidence=0.8,
                profile=True,
            )
        )
        assert found == [target.id]

    def test_limit_and_offset(self, api: MemoryAPI) -> None:
        for i in range(5):
            seed(api, title=f"Item {i}", body="paged")
        page1 = api.search("paged", limit=2)
        page2 = api.search("paged", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert not set(ids(page1)) & set(ids(page2))

    def test_filters_and_kwargs_are_mutually_exclusive(self, api: MemoryAPI) -> None:
        with pytest.raises(ValueError, match="not both"):
            api.search("x", filters=SearchFilters(), limit=5)


class TestRanking:
    def test_scores_are_stable_across_runs(self, api: MemoryAPI) -> None:
        for i in range(5):
            seed(api, title=f"Memory {i} about uv", body="body text " * (i + 1))
        first = [(h.memory_id, round(h.score, 6)) for h in api.search("uv")]
        second = [(h.memory_id, round(h.score, 6)) for h in api.search("uv")]
        assert first == second

    def test_scores_are_stable_across_pagination(self, api: MemoryAPI) -> None:
        """Normalization must not depend on the result set, or paging shifts scores."""
        for i in range(6):
            seed(api, title=f"Memory {i} about uv", body="body " * (i + 1))
        full = {h.memory_id: round(h.score, 6) for h in api.search("uv", limit=10)}
        paged: dict[str, float] = {}
        for offset in (0, 2, 4):
            for hit in api.search("uv", limit=2, offset=offset):
                paged[hit.memory_id] = round(hit.score, 6)
        assert paged == full

    def test_profile_memories_outrank_peers(self, api: MemoryAPI) -> None:
        plain = seed(api, title="Mentions uv", body="same body")
        flagged = seed(api, title="Mentions uv", body="same body", profile=True)
        order = ids(api.search("uv"))
        assert order.index(flagged.id) < order.index(plain.id)

    def test_higher_salience_ranks_higher(self, api: MemoryAPI) -> None:
        low = seed(api, title="uv mention", body="same", salience=0.1)
        high = seed(api, title="uv mention", body="same", salience=0.9)
        order = ids(api.search("uv"))
        assert order.index(high.id) < order.index(low.id)

    def test_type_prior_breaks_ties(self, api: MemoryAPI) -> None:
        event = seed(api, type=MemoryType.EVENT, title="uv note", body="same")
        pref = seed(api, type=MemoryType.PREFERENCE, title="uv note", body="same")
        order = ids(api.search("uv"))
        assert order.index(pref.id) < order.index(event.id)

    def test_recency_decay_is_monotonic(self) -> None:
        now = datetime(2026, 10, 15, tzinfo=UTC)
        recent = recency_decay(now - timedelta(days=1), now=now)
        older = recency_decay(now - timedelta(days=200), now=now)
        ancient = recency_decay(now - timedelta(days=2000), now=now)
        assert 1.0 >= recent > older > ancient > 0.0

    def test_recency_decay_handles_missing_and_bad_values(self) -> None:
        assert recency_decay(None) == 0.0
        assert recency_decay("not a date") == 0.0

    def test_normalize_bm25_increases_with_match_quality(self) -> None:
        """Regression: the transform was inverted, ranking worse matches first.

        FTS5 returns more negative bm25 for better matches, so the mapping
        must increase in -bm25. Values below are real ones observed from
        SQLite for a strong and a weak match.
        """
        strong = normalize_bm25(-2.340942)
        weak = normalize_bm25(-2.246608)
        assert strong > weak

        assert normalize_bm25(-100.0) > normalize_bm25(-10.0) > normalize_bm25(-1.0)
        assert 0.0 <= normalize_bm25(-1.0) < 1.0
        assert normalize_bm25(None) == 0.0
        assert normalize_bm25(0.0) == 0.0

    def test_best_fts_match_scores_highest_end_to_end(self, api: MemoryAPI) -> None:
        """The inverted transform ranked the weakest bm25 match first."""
        seed(api, title="uv", body="uv")
        seed(api, title="uv packaging", body="uv is great " * 50)
        for i in range(8):
            seed(api, title=f"filler {i}", body="unrelated text")

        hits = api.search("uv")
        assert len(hits) == 2
        # Lowest (most negative) bm25 is the best match; it must also carry
        # the highest composite score and come first.
        best_by_bm25 = min(hits, key=lambda h: h.bm25 or 0.0).memory_id
        assert hits[0].memory_id == best_by_bm25
        assert max(hits, key=lambda h: h.score).memory_id == best_by_bm25

    def test_ordering_by_updated(self, api: MemoryAPI) -> None:
        # updated_at is second-precision, so an update in the same second as
        # the create would tie. Push one clearly into the past instead.
        first = seed(api, title="First", body="ordered")
        second = seed(api, title="Second", body="ordered")
        with api.journal:
            api.store.overwrite(
                api.require(first.id).with_updates(updated_at=datetime(2020, 1, 1, tzinfo=UTC))
            )
        api._reindex_one(first.id)
        api.conn.commit()

        order = ids(api.search(None, order="updated"))
        assert order == [second.id, first.id]

    def test_ordering_by_occurred(self, api: MemoryAPI) -> None:
        early = seed(api, title="Early", body="x", occurred_at=datetime(2020, 1, 1, tzinfo=UTC))
        late = seed(api, title="Late", body="x", occurred_at=datetime(2026, 1, 1, tzinfo=UTC))
        assert ids(api.search(None, order="occurred"))[:2] == [late.id, early.id]

    def test_snippet_is_returned(self, api: MemoryAPI) -> None:
        seed(api, title="Snippet test", body="a distinctive phrase appears here")
        hit = api.search("distinctive")[0]
        assert hit.snippet

    def test_metadata_only_search_has_no_bm25(self, api: MemoryAPI) -> None:
        seed(api, title="A", body="x", tags=["t"])
        hit = api.search(None, tags_any=["t"])[0]
        assert hit.bm25 is None
        assert hit.matched_via in ("filter", "profile")


class TestGetRelated:
    def test_outgoing_edges(self, api: MemoryAPI) -> None:
        target = seed(api, title="Target", body="x")
        source = seed(
            api,
            title="Source",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=target.id)],
        )
        related = api.get_related(source.id, direction="out")
        assert [r.memory_id for r in related] == [target.id]
        assert related[0].relation == "causes"

    def test_incoming_edges(self, api: MemoryAPI) -> None:
        target = seed(api, title="Target", body="x")
        source = seed(
            api,
            title="Source",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=target.id)],
        )
        related = api.get_related(target.id, direction="in")
        assert [r.memory_id for r in related] == [source.id]

    def test_depth_two_traversal(self, api: MemoryAPI) -> None:
        c = seed(api, title="C", body="x")
        b = seed(
            api,
            title="B",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=c.id)],
        )
        a = seed(
            api,
            title="A",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=b.id)],
        )
        one_hop = api.get_related(a.id, depth=1, direction="out")
        two_hop = api.get_related(a.id, depth=2, direction="out")
        assert [r.memory_id for r in one_hop] == [b.id]
        assert {r.memory_id for r in two_hop} == {b.id, c.id}
        assert next(r.hop for r in two_hop if r.memory_id == c.id) == 2

    def test_cycle_terminates(self, api: MemoryAPI) -> None:
        a = seed(api, title="A", body="x")
        b = seed(
            api,
            title="B",
            body="x",
            relations=[Relation(type=RelationType.RELATES_TO, target=a.id)],
        )
        api.update(a.id, relations=[Relation(type=RelationType.RELATES_TO, target=b.id)])
        related = api.get_related(a.id, depth=5)
        assert {r.memory_id for r in related} == {b.id}

    def test_dangling_target_is_skipped(self, api: MemoryAPI) -> None:
        source = seed(
            api,
            title="Source",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=new_id())],
        )
        assert api.get_related(source.id) == []

    def test_relation_type_filter(self, api: MemoryAPI) -> None:
        caused = seed(api, title="Caused", body="x")
        part = seed(api, title="Part", body="x")
        anchor = seed(
            api,
            title="Anchor",
            body="x",
            relations=[
                Relation(type=RelationType.CAUSES, target=caused.id),
                Relation(type=RelationType.PART_OF, target=part.id),
            ],
        )
        found = api.get_related(anchor.id, types=["causes"], direction="out")
        assert [r.memory_id for r in found] == [caused.id]

    def test_unknown_relation_type_raises(self, api: MemoryAPI) -> None:
        anchor = seed(api)
        with pytest.raises(SearchError, match="unknown relation type"):
            api.get_related(anchor.id, types=["telepathy"])

    def test_implied_edges_excluded_by_default(self, api: MemoryAPI) -> None:
        """Including them would report every relationship twice."""
        target = seed(api, title="Target", body="x")
        source = seed(
            api,
            title="Source",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=target.id)],
        )
        default = api.get_related(source.id)
        assert all(not r.implied for r in default)

    def test_supersession_chain_is_traversable(self, api: MemoryAPI) -> None:
        """An agent must be able to read a memory's history."""
        original = seed(api, title="Version 1", body="x")
        revised = api.replace(original.id, title="Version 2", body="y").new
        related = api.get_related(revised.id, types=["supersedes"])
        assert original.id in {r.memory_id for r in related}

    def test_contradicts_is_weighted_highest(self) -> None:
        """A disputed memory must surface before merely related ones."""
        assert EDGE_WEIGHT["contradicts"] == max(EDGE_WEIGHT.values())
        assert EDGE_WEIGHT["contradicts"] > EDGE_WEIGHT["relates_to"]

    def test_limit_is_respected(self, api: MemoryAPI) -> None:
        targets = [seed(api, title=f"T{i}", body="x") for i in range(5)]
        anchor = seed(
            api,
            title="Anchor",
            body="x",
            relations=[Relation(type=RelationType.RELATES_TO, target=t.id) for t in targets],
        )
        assert len(api.get_related(anchor.id, limit=2)) == 2

    def test_status_filter(self, api: MemoryAPI) -> None:
        target = seed(api, title="Target", body="x")
        anchor = seed(
            api,
            title="Anchor",
            body="x",
            relations=[Relation(type=RelationType.CAUSES, target=target.id)],
        )
        api.delete(target.id)
        assert api.get_related(anchor.id, statuses=["active"]) == []
        assert api.get_related(anchor.id, statuses=["deleted"])


class TestProfileMemories:
    def test_returns_only_flagged_active(self, api: MemoryAPI) -> None:
        flagged = seed(api, title="Identity", body="x", profile=True)
        seed(api, title="Ordinary", body="x")
        stale = seed(api, title="Old identity", body="x", profile=True)
        api.delete(stale.id)
        assert ids(api.profile()) == [flagged.id]

    def test_bypasses_keyword_matching(self, api: MemoryAPI) -> None:
        """The mitigation for the paraphrase gap: no query needed."""
        flagged = seed(api, title="Prefers uv over pip", body="packaging choice", profile=True)
        # A query sharing no terms finds nothing...
        assert ids(api.search("javascript bundlers")) == []
        # ...but the profile is always available.
        assert flagged.id in ids(api.profile())
