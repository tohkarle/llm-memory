"""Extraction: candidates from source material, provenance always required.

The suite runs entirely without a model. The LLM path is exercised through
the deterministic ``echo`` provider, which is why it ships as a real
provider rather than a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.extract import (
    HEURISTIC_CONFIDENCE,
    MATERIAL_CHAR_LIMIT,
    Candidate,
    ExtractionReport,
    annotate_duplicates,
    commit,
    from_suggestions,
    from_text,
    heuristic_candidates,
    llm_candidates,
)
from memory.models import MemoryType, SourceKind, SourceRef
from memory.processors.base import ProcessResult
from memory.providers import EchoProvider

CONVERSATION = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/2026-10-15.md")


@pytest.fixture
def api(tmp_path: Path) -> Any:
    instance = MemoryAPI.initialize(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


# ---------------------------------------------------------------- the seam


def test_consumes_the_process_result_seam() -> None:
    """``ProcessResult.memory_suggestions`` is documented as the Phase 4 seam."""
    result = ProcessResult(
        text="...",
        memory_suggestions=[
            {"type": "preference", "title": "Prefers uv over pip", "body": "Uses uv."},
        ],
    )
    report = from_suggestions(result.memory_suggestions, source=CONVERSATION)
    assert len(report.candidates) == 1
    assert report.candidates[0].type is MemoryType.PREFERENCE
    assert report.extracted_by == "processor"


def test_suggestion_inherits_the_fallback_source() -> None:
    report = from_suggestions([{"title": "Works in Pacific time"}], source=CONVERSATION)
    assert report.candidates[0].sources[0].ref == CONVERSATION.ref


def test_suggestion_locator_lands_on_the_source() -> None:
    """Provenance should point at a span, not just a file."""
    report = from_suggestions(
        [{"title": "Prefers uv", "locator": "L120-L142"}], source=CONVERSATION
    )
    assert report.candidates[0].sources[0].locator == "L120-L142"


def test_suggestion_may_carry_its_own_source_spec() -> None:
    report = from_suggestions([{"title": "A claim", "source": "document:documents/x.md:p3"}])
    source = report.candidates[0].sources[0]
    assert (source.kind, source.ref, source.locator) == (
        SourceKind.DOCUMENT,
        "documents/x.md",
        "p3",
    )


def test_candidate_without_any_source_is_refused() -> None:
    """A claim whose origin cannot be checked is worse than no memory."""
    report = from_suggestions([{"title": "Unsourced claim"}])
    assert not report.candidates
    assert "no source" in report.warnings[0]


def test_one_bad_suggestion_does_not_lose_the_good_ones() -> None:
    report = from_suggestions(
        [
            {"title": "Prefers uv"},
            {"body": "no title here"},
            {"title": "Bad type", "type": "nonsense"},
            {"title": "Works in Pacific time"},
        ],
        source=CONVERSATION,
    )
    assert [c.title for c in report.candidates] == ["Prefers uv", "Works in Pacific time"]
    assert len(report.warnings) == 2


def test_suggestion_metadata_is_normalized() -> None:
    report = from_suggestions(
        [
            {
                "title": "  Prefers   uv  ",
                "tags": ["Tooling", "TOOLING", "python "],
                "entities": ["uv", "UV"],
                "confidence": 5.0,
            }
        ],
        source=CONVERSATION,
    )
    candidate = report.candidates[0]
    assert candidate.title == "Prefers uv"
    assert candidate.tags == ["python", "tooling"]
    assert candidate.entities == ["uv"]
    assert candidate.confidence == 1.0  # clamped into [0, 1]


def test_non_mapping_suggestion_is_reported() -> None:
    report = from_suggestions(["just a string"], source=CONVERSATION)  # type: ignore[list-item]
    assert not report.candidates
    assert report.warnings


# ---------------------------------------------------------------- heuristics


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("I always run tests before pushing.", MemoryType.PREFERENCE),
        ("I prefer uv over pip.", MemoryType.PREFERENCE),
        ("I'd rather use ruff than flake8.", MemoryType.PREFERENCE),
        ("We decided to drop Python 3.11 support.", MemoryType.DECISION),
        ("I work in Singapore most of the year.", MemoryType.FACT),
        ("My usual setup is zsh with starship.", MemoryType.PREFERENCE),
        ("Remember that the staging key rotates monthly.", MemoryType.FACT),
    ],
)
def test_heuristic_recognises_explicit_statements(text: str, expected_type: MemoryType) -> None:
    candidates = heuristic_candidates(text, source=CONVERSATION)
    assert candidates, text
    assert candidates[0].type is expected_type


def test_heuristic_ignores_ephemera() -> None:
    """Precision over recall: a false memory costs the user more than a missed one."""
    text = (
        "Can you fix the failing test? Thanks! Let me check the logs. "
        "Hmm, that is odd. The build is green now."
    )
    assert heuristic_candidates(text, source=CONVERSATION) == []


def test_heuristic_is_deterministic() -> None:
    text = "I always use ruff. I prefer uv over pip."
    first = [c.title for c in heuristic_candidates(text, source=CONVERSATION)]
    second = [c.title for c in heuristic_candidates(text, source=CONVERSATION)]
    assert first == second and first


def test_heuristic_candidates_carry_a_span_locator() -> None:
    candidates = heuristic_candidates("I prefer uv over pip.", source=CONVERSATION)
    assert candidates[0].sources[0].locator is not None


def test_heuristic_confidence_is_calibrated_as_guessed() -> None:
    candidates = heuristic_candidates("I prefer uv over pip.", source=CONVERSATION)
    assert candidates[0].confidence == HEURISTIC_CONFIDENCE <= 0.5


def test_heuristic_deduplicates_repeated_statements() -> None:
    text = "I prefer uv over pip. I prefer uv over pip."
    assert len(heuristic_candidates(text, source=CONVERSATION)) == 1


def test_heuristic_truncates_an_overlong_title() -> None:
    long_claim = "I prefer " + "x" * 400
    candidates = heuristic_candidates(long_claim[:290], source=CONVERSATION)
    for candidate in candidates:
        assert len(candidate.title) <= 200


def test_heuristic_honours_the_limit() -> None:
    text = " ".join(f"I prefer tool{i} over the alternative." for i in range(30))
    assert len(heuristic_candidates(text, source=CONVERSATION, limit=5)) <= 5


# ---------------------------------------------------------------- no provider


def test_from_text_without_a_provider_uses_the_heuristic() -> None:
    report = from_text("I prefer uv over pip.", source=CONVERSATION)
    assert report.extracted_by == "heuristic"
    assert report.provider is None
    assert report.candidates


def test_from_text_without_a_provider_never_errors_on_empty_input() -> None:
    report = from_text("", source=CONVERSATION)
    assert report.candidates == []
    assert report.extracted_by == "heuristic"


# ---------------------------------------------------------------- llm path


def _reply(items: list[dict[str, Any]]) -> dict[str, str]:
    return {"memory-task: extract": json.dumps({"extract": items})}


def test_llm_candidates_are_parsed_and_normalized() -> None:
    provider = EchoProvider(
        replies=_reply(
            [
                {
                    "type": "preference",
                    "title": "Prefers uv over pip",
                    "body": "Uses uv for every project.",
                    "tags": ["Tooling"],
                    "confidence": 0.9,
                    "quote": "I always use uv",
                }
            ]
        )
    )
    candidates, warnings = llm_candidates(provider, "transcript text", source=CONVERSATION)
    assert not warnings
    assert candidates[0].confidence == 0.9
    assert candidates[0].tags == ["tooling"]
    assert candidates[0].source_kind == "llm:echo"
    assert candidates[0].evidence["quote"] == "I always use uv"


def test_llm_candidates_inherit_the_source() -> None:
    provider = EchoProvider(replies=_reply([{"title": "A durable claim"}]))
    candidates, _ = llm_candidates(provider, "text", source=CONVERSATION)
    assert candidates[0].sources[0].ref == CONVERSATION.ref


def test_empty_llm_answer_is_respected_without_fallback() -> None:
    """A model that honestly found nothing must not be overridden by patterns."""
    provider = EchoProvider()  # its default extract reply is {"extract": []}
    report = from_text(
        "I prefer uv over pip.", source=CONVERSATION, provider=provider, fallback=False
    )
    assert report.candidates == []
    assert report.extracted_by == "llm:echo"


def test_empty_llm_answer_falls_back_when_asked() -> None:
    provider = EchoProvider()
    report = from_text("I prefer uv over pip.", source=CONVERSATION, provider=provider)
    assert report.candidates
    assert report.extracted_by == "heuristic"
    assert any("fell back" in w for w in report.warnings)


def test_provider_failure_is_a_warning_not_an_exception() -> None:
    from memory.providers import ProviderError

    class Failing:
        name = "failing"
        model = None

        def complete(self, prompt: str, **kw: Any) -> Any:
            raise ProviderError("connection refused")

        def describe(self) -> dict[str, Any]:
            return {}

    candidates, warnings = llm_candidates(
        Failing(),  # type: ignore[arg-type]
        "text",
        source=CONVERSATION,
    )
    assert candidates == []
    assert "connection refused" in warnings[0]


def test_unparseable_llm_reply_is_a_warning() -> None:
    provider = EchoProvider(replies={"memory-task: extract": "I'd rather not."})
    candidates, warnings = llm_candidates(provider, "text", source=CONVERSATION)
    assert candidates == []
    assert warnings


def test_oversized_material_is_truncated_and_reported() -> None:
    provider = EchoProvider(replies=_reply([]))
    _, warnings = llm_candidates(provider, "x" * (MATERIAL_CHAR_LIMIT + 100), source=CONVERSATION)
    assert any("truncated" in w for w in warnings)


def test_llm_bad_candidate_does_not_lose_the_good_ones() -> None:
    provider = EchoProvider(
        replies=_reply([{"title": "Good claim"}, {"body": "no title"}, {"title": "Also good"}])
    )
    candidates, warnings = llm_candidates(provider, "text", source=CONVERSATION)
    assert [c.title for c in candidates] == ["Good claim", "Also good"]
    assert warnings


# ---------------------------------------------------------------- duplicates


def test_annotate_duplicates_flags_an_existing_memory(api: MemoryAPI) -> None:
    api.create(
        type=MemoryType.PREFERENCE,
        title="Prefers uv over pip",
        body="Uses uv for every Python project.",
    )
    report = from_suggestions(
        [
            {
                "type": "preference",
                "title": "Prefers uv over pip",
                "body": "Uses uv for every Python project.",
            }
        ],
        source=CONVERSATION,
    )
    annotate_duplicates(api, report)
    assert report.candidates[0].duplicate_of
    assert report.candidates[0].duplicate_of[0]["verdict"] == "duplicate"


def test_annotate_duplicates_leaves_novel_candidates_alone(api: MemoryAPI) -> None:
    api.create(type=MemoryType.FACT, title="Allergic to shellfish", body="Avoids prawns.")
    report = from_suggestions([{"title": "Prefers uv over pip"}], source=CONVERSATION)
    annotate_duplicates(api, report)
    assert report.candidates[0].duplicate_of == []


# ---------------------------------------------------------------- commit


def test_commit_creates_through_the_ordinary_verb(api: MemoryAPI) -> None:
    report = from_suggestions(
        [{"type": "preference", "title": "Prefers uv over pip", "body": "Uses uv."}],
        source=CONVERSATION,
    )
    commit(api, report)
    assert len(report.created) == 1
    memory = api.require(report.created[0]["memory_id"])
    assert memory.title == "Prefers uv over pip"
    # Provenance survived the write path.
    assert memory.sources[0].ref == CONVERSATION.ref


def test_commit_skips_below_min_confidence(api: MemoryAPI) -> None:
    report = from_suggestions(
        [{"title": "A guess", "confidence": 0.3}, {"title": "Stated outright", "confidence": 0.95}],
        source=CONVERSATION,
    )
    commit(api, report, min_confidence=0.6)
    assert [c["title"] for c in report.created] == ["Stated outright"]
    assert report.skipped[0]["title"] == "A guess"


def test_commit_skips_known_duplicates(api: MemoryAPI) -> None:
    api.create(type=MemoryType.FACT, title="Prefers uv over pip", body="Uses uv everywhere.")
    report = from_suggestions(
        [{"title": "Prefers uv over pip", "body": "Uses uv everywhere."}], source=CONVERSATION
    )
    annotate_duplicates(api, report)
    commit(api, report)
    assert report.created == []
    assert report.skipped[0]["reason"] == "already recorded"


def test_commit_respects_a_tombstone(api: MemoryAPI) -> None:
    """Re-extraction from a retained conversation must not resurrect a deletion."""
    memory = api.create(
        type=MemoryType.FACT,
        title="A regrettable fact",
        body="Something the user removed.",
        sources=[CONVERSATION],
    )
    api.delete(memory.id, permanent=True, reason="user asked")

    report = from_suggestions(
        [{"title": "A regrettable fact", "body": "Something the user removed."}],
        source=CONVERSATION,
    )
    commit(api, report)
    assert report.created == []
    assert report.skipped[0]["reason"] == "permanently deleted before"


def test_commit_continues_past_a_tombstoned_candidate(api: MemoryAPI) -> None:
    dead = api.create(
        type=MemoryType.FACT, title="Dead claim", body="Removed.", sources=[CONVERSATION]
    )
    api.delete(dead.id, permanent=True)
    report = from_suggestions(
        [
            {"title": "Dead claim", "body": "Removed."},
            {"title": "A fresh claim", "body": "Still true.", "source": "manual:cli"},
        ],
        source=CONVERSATION,
    )
    commit(api, report)
    assert [c["title"] for c in report.created] == ["A fresh claim"]
    assert len(report.skipped) == 1


def test_commit_is_reportable_as_json(api: MemoryAPI) -> None:
    report = from_suggestions([{"title": "Prefers uv"}], source=CONVERSATION)
    commit(api, report)
    json.dumps(report.to_json())


def test_candidate_create_kwargs_matches_the_api(api: MemoryAPI) -> None:
    """The shape guarantee: no translation layer between the two."""
    candidate = Candidate(
        type=MemoryType.PREFERENCE,
        title="Prefers uv",
        body="Uses uv.",
        sources=[CONVERSATION],
    )
    memory = api.create(**candidate.create_kwargs())
    assert memory.type is MemoryType.PREFERENCE


def test_extracted_memories_survive_an_index_rebuild(api: MemoryAPI) -> None:
    report = from_suggestions(
        [
            {"type": "preference", "title": "Prefers uv over pip", "body": "Uses uv."},
            {"type": "fact", "title": "Works in Pacific time", "body": "US west coast."},
        ],
        source=CONVERSATION,
    )
    commit(api, report)
    before = _dump(api)
    api.rebuild_index()
    assert _dump(api) == before


def _dump(api: MemoryAPI) -> dict[str, list[tuple[Any, ...]]]:
    out: dict[str, list[tuple[Any, ...]]] = {}
    for table, columns in (
        ("memories", "id, type, title, status, content_hash"),
        ("memory_sources", "memory_id, source_id, locator"),
        ("sources", "id, kind, ref"),
    ):
        rows = api.conn.execute(f"SELECT {columns} FROM {table}").fetchall()
        out[table] = sorted(tuple(row) for row in rows)
    return out


def test_report_defaults_are_empty_not_none() -> None:
    report = ExtractionReport(candidates=[])
    assert report.to_json()["total"] == 0
    assert report.created == [] and report.skipped == []
