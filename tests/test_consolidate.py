"""Consolidation: detection is deterministic, and it proposes rather than writes.

Two properties carry the most weight here:

* the whole detector runs with no provider configured, so these tests never
  need a model;
* nothing in :mod:`memory.consolidate` mutates the store unless a caller
  explicitly applies a proposal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from memory import consolidate
from memory.api import MemoryAPI
from memory.consolidate import (
    DUPLICATE_THRESHOLD,
    NEAR_THRESHOLD,
    Proposal,
    apply_all,
    apply_proposal,
    containment,
    detect,
    jaccard,
    polarity_conflict,
    similarity,
    tokens,
)
from memory.models import (
    MemoryStatus,
    MemoryType,
    Relation,
    RelationType,
    SourceKind,
    SourceRef,
)
from memory.providers import EchoProvider


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
        "title": "A claim",
        "body": "Some body text.",
    }
    defaults.update(kw)
    return api.create(**defaults)


#: A pair that genuinely lands between the two thresholds, verified by
#: ``test_borderline_fixture_is_actually_borderline`` below. Paraphrases like
#: this are the band where a provider changes the answer, so the LLM tests
#: need a pair the lexical pass deliberately declines to call.
BORDERLINE_PAIR = (
    ("Runs tests with pytest", "Uses pytest for all Python testing."),
    ("Testing is done with pytest", "All Python tests run under pytest."),
)


def seed_borderline_pair(api: MemoryAPI) -> tuple[Any, Any]:
    (ta, ba), (tb, bb) = BORDERLINE_PAIR
    return seed(api, title=ta, body=ba), seed(api, title=tb, body=bb)


def test_borderline_fixture_is_actually_borderline(api: MemoryAPI) -> None:
    """Guards the LLM tests below: if tuning moves this pair out of the band,
    those tests would silently stop exercising the provider path."""
    a, b = seed_borderline_pair(api)
    score, _ = similarity(a, b)
    assert NEAR_THRESHOLD <= score < DUPLICATE_THRESHOLD, score


# ---------------------------------------------------------------- similarity math


def test_tokens_drops_stopwords_and_short_tokens() -> None:
    assert tokens("The user prefers uv to pip") == {"user", "prefers", "pip"}


def test_tokens_is_case_and_accent_insensitive() -> None:
    assert tokens("Café Tooling") == tokens("CAFE tooling")


def test_jaccard_bounds() -> None:
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_containment_catches_the_short_inside_long_case() -> None:
    """Jaccard collapses here; containment is why the pair stays visible."""
    short = {"uv", "python"}
    long = {"uv", "python", *(f"w{i}" for i in range(20))}
    assert jaccard(short, long) < 0.2
    assert containment(short, long) == 1.0


def test_identical_content_scores_exactly_one(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers uv", body="Uses uv everywhere.")
    b = seed(api, title="Prefers  UV", body="uses UV everywhere.", force_recreate=True)
    score, evidence = similarity(a, b)
    assert score == 1.0
    assert evidence["exact_content_match"] is True


def test_unrelated_memories_score_low(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers uv over pip", body="Python packaging tooling.")
    b = seed(api, title="Allergic to shellfish", body="Avoids prawns and crab.")
    score, _ = similarity(a, b)
    assert score < NEAR_THRESHOLD


def test_title_only_memories_are_not_penalised_for_empty_bodies(api: MemoryAPI) -> None:
    """A memory with no body is all title; scoring must not treat that as missing evidence."""
    a = seed(api, title="Works in Pacific timezone", body="")
    b = seed(api, title="Works in the Pacific timezone", body="")
    score, _ = similarity(a, b)
    assert score >= DUPLICATE_THRESHOLD


# ---------------------------------------------------------------- polarity


def test_polarity_conflict_detects_opposite_claims(api: MemoryAPI) -> None:
    a = seed(api, title="Uses tabs for indentation", body="Prefers tab characters in source.")
    b = seed(api, title="Does not use tabs for indentation", body="Never uses tab characters.")
    conflict = polarity_conflict(a, b)
    assert conflict is not None
    assert conflict["negated_side"] == b.id


def test_polarity_conflict_requires_a_shared_subject(api: MemoryAPI) -> None:
    """Negation about an unrelated topic is not a contradiction."""
    a = seed(api, title="Never uses pip", body="Avoids pip entirely.")
    b = seed(api, title="Enjoys mountain hiking", body="Hikes every weekend.")
    assert polarity_conflict(a, b) is None


def test_polarity_conflict_ignores_pairs_negated_on_both_sides(api: MemoryAPI) -> None:
    a = seed(api, title="Does not use pip for packaging", body="Avoids pip.")
    b = seed(api, title="Never uses pip for packaging", body="Avoids pip too.")
    assert polarity_conflict(a, b) is None


# ------------------------------------------------------- similarity precision

#: Pairs that must never be folded together. Every other similarity test here
#: measures *recall* -- that a known duplicate is found -- which is why a
#: tokenizer that discarded `5s`, `v1` and `2` could score contradictory
#: memories at 1.000 through a green suite. These measure precision: each
#: pair shares nearly all its vocabulary and still asserts different things,
#: so a detector that calls them duplicates is silently destroying a
#: distinction the reader depends on.
NON_DUPLICATE_PAIRS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    (
        "unit suffixes",
        ("Retry delay is 5s", "The retry delay is 5s."),
        ("Retry delay is 5m", "The retry delay is 5m."),
    ),
    (
        "version pins",
        ("Pin the client to v1", "Pin the client to v1."),
        ("Pin the client to v2", "Pin the client to v2."),
    ),
    (
        "bare counts",
        ("Set retries to 2", "Set retries to 2."),
        ("Set retries to 10", "Set retries to 10."),
    ),
    (
        "digit-count asymmetry",
        ("Timeout is 30 seconds", "The timeout is 30 seconds."),
        ("Timeout is 300 seconds", "The timeout is 300 seconds."),
    ),
    (
        "port numbers",
        ("Postgres listens on port 5432", "Connect to port 5432."),
        ("Postgres listens on port 5433", "Connect to port 5433."),
    ),
    (
        "different subjects, same predicate",
        ("Alice prefers vim", "Alice edits everything in vim."),
        ("Bob prefers vim", "Bob edits everything in vim."),
    ),
    (
        "unmarked antonyms",
        ("Enable the cache", "The cache should be enabled."),
        ("Disable the cache", "The cache should be disabled."),
    ),
)


@pytest.mark.parametrize(
    ("case", "left", "right"),
    NON_DUPLICATE_PAIRS,
    ids=[case for case, _, _ in NON_DUPLICATE_PAIRS],
)
def test_near_identical_but_contradictory_pairs_are_not_duplicates(
    api: MemoryAPI, case: str, left: tuple[str, str], right: tuple[str, str]
) -> None:
    """`--apply` merges `duplicate` by default, so this is the auto-merge gate."""
    seed(api, title=left[0], body=left[1])
    seed(api, title=right[0], body=right[1])
    report = detect(api)
    assert [p for p in report.proposals if p.kind == "duplicate"] == [], case


@pytest.mark.parametrize(
    ("case", "left", "right"),
    NON_DUPLICATE_PAIRS,
    ids=[case for case, _, _ in NON_DUPLICATE_PAIRS],
)
def test_near_identical_but_contradictory_pairs_score_below_the_duplicate_cut(
    api: MemoryAPI, case: str, left: tuple[str, str], right: tuple[str, str]
) -> None:
    a = seed(api, title=left[0], body=left[1])
    b = seed(api, title=right[0], body=right[1])
    score, _ = similarity(a, b)
    assert score < DUPLICATE_THRESHOLD, f"{case}: scored {score}"


def test_tokens_keeps_digit_bearing_tokens() -> None:
    """`5s`, `v2` and `10` are the whole difference between two memories."""
    assert tokens("retry delay 5s") == {"retry", "delay", "5s"}
    assert tokens("pin to v1") == {"pin", "v1"}
    assert tokens("set retries to 10") == {"set", "retries", "10"}


#: Pairs where retaining digit-bearing tokens is *not* enough on its own:
#: `containment` is asymmetric, so one side carrying an extra number still
#: reads as "wholly contained". Without the veto each of these scores above
#: `DUPLICATE_THRESHOLD` and auto-merges -- see the regression test below,
#: which asserts exactly that counterfactual.
VETO_REQUIRED_PAIRS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    (
        "containment asymmetry",
        (
            "Upstream request timeout is 30 seconds",
            "The upstream request timeout is set to 30 seconds for every call.",
        ),
        (
            "Upstream request timeout is 300 seconds",
            "The upstream request timeout is set to 300 seconds for every call and retries.",
        ),
    ),
    (
        "long shared phrasing",
        (
            "Pin the vendored client library to v1",
            "Pin the vendored client library to v1 in the lockfile.",
        ),
        (
            "Pin the vendored client library to v2",
            "Pin the vendored client library to v2 in the lockfile.",
        ),
    ),
)


@pytest.mark.parametrize(
    ("case", "left", "right"),
    VETO_REQUIRED_PAIRS,
    ids=[case for case, _, _ in VETO_REQUIRED_PAIRS],
)
def test_the_numeric_veto_is_what_keeps_these_pairs_apart(
    api: MemoryAPI,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> None:
    """Proves the veto is load-bearing, not belt-and-braces.

    Keeping `300` and `v2` as tokens is necessary but insufficient here: with
    the veto disabled these score well above the duplicate cut, so a fix that
    only touched the tokenizer would still auto-merge them.
    """
    a = seed(api, title=left[0], body=left[1])
    b = seed(api, title=right[0], body=right[1])

    monkeypatch.setattr(consolidate, "numeric_conflict", lambda *_: False)
    without_veto, _ = similarity(a, b)
    monkeypatch.undo()
    with_veto, evidence = similarity(a, b)

    assert without_veto >= DUPLICATE_THRESHOLD, f"{case}: the veto has nothing to do"
    assert with_veto < DUPLICATE_THRESHOLD, f"{case}: scored {with_veto}"
    assert evidence["numeric_conflict"] is True
    assert with_veto >= NEAR_THRESHOLD, "the pair must stay visible, not vanish"


def test_numeric_disagreement_does_not_block_a_genuine_paraphrase(
    api: MemoryAPI,
) -> None:
    """The veto needs *both* sides numeric and disagreeing, or it would mute
    the short-inside-long case containment exists to catch."""
    a = seed(api, title="Deploys from the v2 branch", body="Releases cut from v2.")
    b = seed(api, title="Deploys from the v2 branch", body="Releases are cut from v2.")
    score, _ = similarity(a, b)
    assert score >= DUPLICATE_THRESHOLD, score


# ---------------------------------------------------------------- detection


def test_detect_finds_exact_duplicates(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers uv over pip", body="Uses uv for every Python project.")
    b = seed(
        api,
        title="Prefers uv over pip",
        body="Uses uv for every Python project.",
        force_recreate=True,
    )
    report = detect(api)
    duplicates = [p for p in report.proposals if p.kind == "duplicate"]
    assert len(duplicates) == 1
    assert set(duplicates[0].memory_ids) == {a.id, b.id}
    assert duplicates[0].verb == "merge"


def test_detect_runs_with_no_provider_configured(api: MemoryAPI) -> None:
    seed(api, title="Prefers uv", body="Uses uv.")
    seed(api, title="Prefers uv", body="Uses uv.", force_recreate=True)
    report = detect(api)
    assert report.provider is None
    assert report.proposals


def test_detect_proposes_contradictions_as_relate(api: MemoryAPI) -> None:
    seed(api, title="Deploys on Fridays", body="Ships to production on Friday afternoons.")
    seed(api, title="Does not deploy on Fridays", body="Never ships to production on Friday.")
    report = detect(api)
    contradictions = [p for p in report.proposals if p.kind == "contradiction"]
    assert contradictions
    assert contradictions[0].verb == "relate"
    assert contradictions[0].arguments["relation"] == RelationType.CONTRADICTS.value


def test_detect_writes_nothing(api: MemoryAPI) -> None:
    """Detection is read-only. The corpus must be byte-identical afterwards."""
    a = seed(api, title="Prefers uv", body="Uses uv for everything.")
    b = seed(api, title="Prefers uv", body="Uses uv for everything.", force_recreate=True)
    before = {m.id: api.paths.memory_file(m.type.value, m.id).read_bytes() for m in (a, b)}
    report = detect(api)
    assert report.proposals
    after = {m.id: api.paths.memory_file(m.type.value, m.id).read_bytes() for m in (a, b)}
    assert before == after


def test_detect_respects_an_existing_contradicts_edge(api: MemoryAPI) -> None:
    """A declared dispute is the user's answer; do not propose merging it away."""
    a = seed(api, title="Prefers dark mode", body="Uses dark themes in every editor.")
    b = seed(
        api,
        title="Prefers dark mode",
        body="Uses dark themes in every editor.",
        force_recreate=True,
        relations=[Relation(type=RelationType.CONTRADICTS, target=a.id)],
    )
    report = detect(api)
    pairs = [set(p.memory_ids) for p in report.proposals if p.verb == "merge"]
    assert {a.id, b.id} not in pairs


def test_detect_skips_superseded_pairs(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers pip", body="Uses pip for Python packaging work.")
    api.replace(a.id, title="Prefers uv", body="Uses uv for Python packaging work.")
    report = detect(api)
    # The superseded original is not active, so it is not a consolidation
    # candidate at all -- history must not be proposed for merging.
    ids = {mid for p in report.proposals for mid in p.memory_ids}
    assert a.id not in ids


def test_detect_reports_stale_memories_from_declared_dates(api: MemoryAPI) -> None:
    past = datetime.now(UTC) - timedelta(days=30)
    stale = seed(api, title="Contract runs through March", body="x", review_after=past)
    report = detect(api)
    kinds = {p.kind: p for p in report.proposals}
    assert "stale" in kinds
    assert stale.id in kinds["stale"].memory_ids


def test_expired_valid_until_proposes_replace(api: MemoryAPI) -> None:
    past = datetime.now(UTC) - timedelta(days=5)
    seed(api, title="Working at ACME", body="Employed at ACME Corp.", valid_until=past)
    report = detect(api)
    stale = [p for p in report.proposals if p.kind == "stale"]
    assert stale and stale[0].verb == "replace"


def test_stale_detection_does_not_guess_from_age(api: MemoryAPI) -> None:
    """Durable facts do not expire. Only declared dates make a memory stale."""
    old = datetime.now(UTC) - timedelta(days=3650)
    seed(api, title="Born in Singapore", body="Birthplace.", occurred_at=old, created_at=old)
    report = detect(api, include_near=False)
    assert not [p for p in report.proposals if p.kind == "stale"]


def test_detect_filters_by_type_and_tag(api: MemoryAPI) -> None:
    seed(api, type=MemoryType.PREFERENCE, title="Prefers uv", body="Uses uv.", tags=["tooling"])
    seed(
        api,
        type=MemoryType.PREFERENCE,
        title="Prefers uv",
        body="Uses uv.",
        tags=["tooling"],
        force_recreate=True,
    )
    seed(api, type=MemoryType.EVENT, title="Attended a conference", body="PyCon.")
    report = detect(api, types=["preference"], tags=["tooling"])
    assert all("preference" not in p.kind for p in report.proposals)
    assert report.proposals


def test_detect_honours_the_limit(api: MemoryAPI) -> None:
    for _ in range(6):
        seed(api, title="Prefers uv over pip", body="Uses uv.", force_recreate=True)
    report = detect(api, limit=2)
    assert len(report.proposals) <= 2


def test_proposal_key_is_order_independent() -> None:
    left = Proposal(kind="duplicate", verb="merge", memory_ids=["b", "a"], score=1.0, rationale="")
    right = Proposal(kind="duplicate", verb="merge", memory_ids=["a", "b"], score=1.0, rationale="")
    assert left.key() == right.key()


def test_report_is_json_serializable(api: MemoryAPI) -> None:
    seed(api, title="Prefers uv", body="Uses uv.")
    seed(api, title="Prefers uv", body="Uses uv.", force_recreate=True)
    json.dumps(detect(api).to_json())


def test_duplicate_proposal_prefers_the_better_survivor(api: MemoryAPI) -> None:
    """Merge unions sources but only one id survives, so provenance should lead."""
    thin = seed(api, title="Prefers uv over pip", body="Uses uv.")
    rich = seed(
        api,
        title="Prefers uv over pip",
        body="Uses uv.",
        force_recreate=True,
        tags=["tooling", "python"],
        sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")],
    )
    report = detect(api)
    merge_proposal = next(p for p in report.proposals if p.verb == "merge")
    assert merge_proposal.arguments["into"] == rich.id
    assert merge_proposal.memory_ids[0] == rich.id
    assert thin.id in merge_proposal.memory_ids


# ---------------------------------------------------------------- provider path


def test_provider_only_adjudicates_the_borderline_band(api: MemoryAPI) -> None:
    """A model must not change what the lexical pass already decided."""
    seed(api, title="Prefers uv over pip", body="Uses uv for all Python work.")
    seed(
        api,
        title="Prefers uv over pip",
        body="Uses uv for all Python work.",
        force_recreate=True,
    )
    provider = EchoProvider()
    report = detect(api, provider=provider)
    assert [p for p in report.proposals if p.kind == "duplicate"]
    # An exact duplicate is decided lexically, so the model is never asked.
    assert not provider.calls


def test_provider_verdict_promotes_a_borderline_pair(api: MemoryAPI) -> None:
    a, b = seed_borderline_pair(api)
    lexical, _ = similarity(a, b)
    assert NEAR_THRESHOLD <= lexical < DUPLICATE_THRESHOLD, lexical

    provider = EchoProvider(
        replies={
            "memory-task: adjudicate": json.dumps(
                {
                    "adjudicate": [
                        {
                            "pair": 1,
                            "verdict": "duplicate",
                            "confidence": 0.9,
                            "reason": "both state uv is the packaging tool",
                        }
                    ]
                }
            )
        }
    )
    report = detect(api, provider=provider, include_stale=False)
    duplicates = [p for p in report.proposals if p.verb == "merge"]
    assert duplicates
    assert duplicates[0].source == "llm:echo"
    assert duplicates[0].evidence["llm_confidence"] == 0.9
    assert report.provider == "echo"


def test_provider_distinct_verdict_yields_no_proposal(api: MemoryAPI) -> None:
    seed_borderline_pair(api)
    provider = EchoProvider(
        replies={
            "memory-task: adjudicate": json.dumps(
                {"adjudicate": [{"pair": 1, "verdict": "distinct", "reason": "different scope"}]}
            )
        }
    )
    report = detect(api, provider=provider, include_stale=False)
    assert not [p for p in report.proposals if p.verb == "merge"]


def test_provider_failure_degrades_to_a_warning(api: MemoryAPI) -> None:
    """A dead local runtime must cost proposals, not the whole scan."""
    seed_borderline_pair(api)

    class Broken:
        name = "broken"
        model = None

        def complete(self, prompt: str, **kw: Any) -> Any:
            raise RuntimeError("connection refused")

        def describe(self) -> dict[str, Any]:
            return {}

    from memory.providers import ProviderError

    class Failing(Broken):
        def complete(self, prompt: str, **kw: Any) -> Any:
            raise ProviderError("connection refused")

    report = detect(api, provider=Failing())  # type: ignore[arg-type]
    assert report.warnings
    assert "connection refused" in report.warnings[0]


def test_unparseable_provider_reply_degrades_to_a_warning(api: MemoryAPI) -> None:
    seed_borderline_pair(api)
    provider = EchoProvider(replies={"memory-task: adjudicate": "I'm not going to answer that."})
    report = detect(api, provider=provider, include_stale=False)
    assert report.warnings


# ---------------------------------------------------------------- application


def test_apply_proposal_merges_through_the_existing_verb(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers uv", body="Uses uv for everything.")
    b = seed(api, title="Prefers uv", body="Uses uv for everything.", force_recreate=True)
    report = detect(api)
    proposal = next(p for p in report.proposals if p.verb == "merge")
    outcome = apply_proposal(api, proposal)
    assert outcome["verb"] == "merge"
    survivor = outcome["result"]["merged"]["id"]
    assert survivor in {a.id, b.id}
    other = (set(proposal.memory_ids) - {survivor}).pop()
    assert api.require(other).status is MemoryStatus.SUPERSEDED


def test_apply_proposal_records_a_contradiction(api: MemoryAPI) -> None:
    seed(api, title="Deploys on Fridays", body="Ships to production on Friday afternoons.")
    seed(api, title="Does not deploy on Fridays", body="Never ships to production on Friday.")
    report = detect(api)
    proposal = next(p for p in report.proposals if p.verb == "relate")
    apply_proposal(api, proposal)
    source = api.require(proposal.arguments["source_id"])
    assert any(
        r.type is RelationType.CONTRADICTS and r.target == proposal.arguments["target_id"]
        for r in source.relations
    )


def test_applying_a_relation_twice_is_idempotent(api: MemoryAPI) -> None:
    seed(api, title="Deploys on Fridays", body="Ships to production on Friday afternoons.")
    seed(api, title="Does not deploy on Fridays", body="Never ships to production on Friday.")
    proposal = next(p for p in detect(api).proposals if p.verb == "relate")
    apply_proposal(api, proposal)
    second = apply_proposal(api, proposal)
    assert second["result"]["unchanged"] is True


def test_advisory_verbs_refuse_to_apply(api: MemoryAPI) -> None:
    """`replace` needs a new title and body; inventing one is not this tool's call."""
    proposal = Proposal(
        kind="stale", verb="replace", memory_ids=["x"], score=0.8, rationale="expired"
    )
    with pytest.raises(ValueError, match="advisory"):
        apply_proposal(api, proposal)


def test_apply_all_defaults_to_duplicates_only(api: MemoryAPI) -> None:
    seed(api, title="Prefers uv", body="Uses uv for everything.")
    seed(api, title="Prefers uv", body="Uses uv for everything.", force_recreate=True)
    seed(api, title="Deploys on Fridays", body="Ships to production on Friday afternoons.")
    seed(api, title="Does not deploy on Fridays", body="Never ships to production on Friday.")
    report = apply_all(api, detect(api))
    verbs = {entry.get("verb") for entry in report.applied}
    assert verbs == {"merge"}


def test_apply_all_does_not_cascade_within_one_pass(api: MemoryAPI) -> None:
    """A merged parent is history; a later proposal naming it must be skipped."""
    for _ in range(3):
        seed(api, title="Prefers uv over pip", body="Uses uv.", force_recreate=True)
    report = apply_all(api, detect(api))
    merged = [e for e in report.applied if e.get("verb") == "merge"]
    skipped = [e for e in report.applied if "skipped" in e]
    assert len(merged) == 1
    assert skipped


def test_apply_all_survives_one_failing_proposal(api: MemoryAPI) -> None:
    a = seed(api, title="Prefers uv", body="Uses uv for everything.")
    seed(api, title="Prefers uv", body="Uses uv for everything.", force_recreate=True)
    seed(api, title="Likes strong coffee", body="Drinks espresso daily.")
    seed(api, title="Likes strong coffee", body="Drinks espresso daily.", force_recreate=True)
    report = detect(api)
    assert len([p for p in report.proposals if p.verb == "merge"]) == 2

    # Delete one parent out from under the first proposal.
    api.delete(a.id, permanent=True, reason="test")
    applied = apply_all(api, report).applied
    assert any("error" in entry for entry in applied)
    assert any(entry.get("verb") == "merge" for entry in applied)


def test_merge_preserves_the_union_of_sources(api: MemoryAPI) -> None:
    """The provenance guarantee consolidation exists to protect."""
    seed(
        api,
        title="Prefers uv over pip",
        body="Uses uv.",
        sources=[SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/a.md")],
    )
    seed(
        api,
        title="Prefers uv over pip",
        body="Uses uv.",
        force_recreate=True,
        sources=[SourceRef(kind=SourceKind.DOCUMENT, ref="documents/b.md")],
    )
    report = apply_all(api, detect(api))
    merged_id = report.applied[0]["result"]["merged"]["id"]
    refs = {s.ref for s in api.require(merged_id).sources}
    assert refs == {"conversations/a.md", "documents/b.md"}


# ---------------------------------------------------------------- rebuild safety


def test_consolidation_results_survive_an_index_rebuild(api: MemoryAPI) -> None:
    """Everything consolidation writes goes through the existing verbs, so a
    rebuild from the filesystem alone must reproduce it exactly."""
    seed(api, title="Prefers uv", body="Uses uv for everything.")
    seed(api, title="Prefers uv", body="Uses uv for everything.", force_recreate=True)
    seed(api, title="Deploys on Fridays", body="Ships to production on Friday afternoons.")
    seed(api, title="Does not deploy on Fridays", body="Never ships to production on Friday.")
    apply_all(api, detect(api), kinds=["duplicate", "contradiction"])

    before = _dump(api)
    api.rebuild_index()
    assert _dump(api) == before


def _dump(api: MemoryAPI) -> dict[str, list[tuple[Any, ...]]]:
    out: dict[str, list[tuple[Any, ...]]] = {}
    for table, columns in (
        ("memories", "id, status, superseded_by, content_hash"),
        ("relations", "src_id, dst_id, type, implied"),
        ("memory_sources", "memory_id, source_id, locator"),
    ):
        rows = api.conn.execute(f"SELECT {columns} FROM {table}").fetchall()
        out[table] = sorted(tuple(row) for row in rows)
    return out


def test_module_exposes_no_hidden_write_path() -> None:
    """The proposes-not-writes rule, asserted on the module surface itself."""
    public = {name for name in consolidate.__all__}
    assert "detect" in public
    # Only the two explicit application entry points may mutate.
    assert {"apply_proposal", "apply_all"} <= public
