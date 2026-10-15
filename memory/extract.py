"""Turn source material into candidate memories. Proposes; never writes.

Same contract as :mod:`memory.consolidate`: this module produces
:class:`Candidate` objects and a caller decides. ``mem extract`` prints them;
``mem extract --commit`` creates them through :meth:`MemoryAPI.create`, which
is where tombstones, validation and the journal already live.

Two entry points, and the distinction matters:

:func:`from_suggestions`
    Consumes ``ProcessResult.memory_suggestions`` -- the Phase 4 seam
    documented in :mod:`memory.processors.base`. A processor that already
    knows what a file means (an ingestion pipeline, a structured import) puts
    candidates there, and this validates and normalizes them without
    second-guessing the content.

:func:`from_text`
    Reads raw text. With a provider configured it asks the model for
    candidates; without one it falls back to a deterministic heuristic that
    finds explicit first-person statements. The heuristic is intentionally
    conservative -- precision over recall, because a false memory is worse
    than a missed one and the user is the one who has to live with it.

**Every candidate carries a source.** A memory extracted from a conversation
and not citing it is a claim that cannot be checked later, so extraction
refuses to emit one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import (
    Memory,
    MemoryType,
    SourceKind,
    SourceRef,
    clean_entities,
    clean_tags,
    clean_title,
    parse_ts,
)
from .providers import LLMProvider, ProviderError, extract_json_object
from .store import Tombstoned

if TYPE_CHECKING:  # pragma: no cover
    from .api import MemoryAPI

#: Length bounds for a heuristic candidate. Shorter is usually a fragment,
#: longer is usually a paragraph that should become several memories.
MIN_STATEMENT_CHARS = 12
MAX_STATEMENT_CHARS = 300

#: Confidence assigned by the deterministic path. Deliberately middling: the
#: heuristic matched a sentence pattern, which is weaker evidence than a
#: model (or a human) having understood it. See the skill docs on
#: calibration -- ~0.5 means "guessed".
HEURISTIC_CONFIDENCE = 0.5

#: Confidence floor for an LLM candidate when the model does not say.
LLM_DEFAULT_CONFIDENCE = 0.6


class ExtractionError(ValueError):
    """Source material or a suggestion could not be turned into a candidate."""


@dataclass(slots=True)
class Candidate:
    """A proposed memory. Nothing has been written.

    Shaped so ``api.create(**candidate.create_kwargs())`` just works: a
    caller should never have to translate between this and the write path.
    """

    type: MemoryType
    title: str
    body: str = ""
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    confidence: float = HEURISTIC_CONFIDENCE
    salience: float = 0.5
    sources: list[SourceRef] = field(default_factory=list)
    occurred_at: Any = None
    rationale: str = ""
    source_kind: str = "heuristic"
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Populated by :func:`annotate_duplicates`, so a caller sees that a
    #: candidate restates something already recorded before committing it.
    duplicate_of: list[dict[str, Any]] = field(default_factory=list)

    def create_kwargs(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "type": self.type.value,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "entities": list(self.entities),
            "confidence": self.confidence,
            "salience": self.salience,
            "sources": list(self.sources),
        }
        if self.occurred_at is not None:
            fields["occurred_at"] = self.occurred_at
        return fields

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "entities": self.entities,
            "confidence": round(self.confidence, 4),
            "salience": round(self.salience, 4),
            "sources": [s.to_yaml() for s in self.sources],
            "rationale": self.rationale,
            "extracted_by": self.source_kind,
            "evidence": self.evidence,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(slots=True)
class ExtractionReport:
    candidates: list[Candidate]
    provider: str | None = None
    extracted_by: str = "heuristic"
    warnings: list[str] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "total": len(self.candidates),
            "extracted_by": self.extracted_by,
            "provider": self.provider,
            "candidates": [c.to_json() for c in self.candidates],
            "created": self.created,
            "skipped": self.skipped,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------- suggestions


def from_suggestions(
    suggestions: Sequence[dict[str, Any]],
    *,
    source: SourceRef | None = None,
    default_type: MemoryType = MemoryType.FACT,
) -> ExtractionReport:
    """Normalize ``ProcessResult.memory_suggestions`` into candidates.

    Tolerant of a malformed entry -- it is reported and skipped rather than
    failing the batch, because a processor producing twelve good suggestions
    and one bad one should yield twelve memories.
    """
    candidates: list[Candidate] = []
    warnings: list[str] = []

    for i, raw in enumerate(suggestions):
        try:
            candidates.append(
                _candidate_from_mapping(
                    raw, source=source, default_type=default_type, origin="processor"
                )
            )
        except (ExtractionError, ValueError) as exc:
            warnings.append(f"suggestion[{i}]: {exc}")

    return ExtractionReport(candidates=candidates, extracted_by="processor", warnings=warnings)


def _candidate_from_mapping(
    raw: Any,
    *,
    source: SourceRef | None,
    default_type: MemoryType,
    origin: str,
) -> Candidate:
    if not isinstance(raw, dict):
        raise ExtractionError(f"expected a mapping, got {type(raw).__name__}")

    title = clean_title(str(raw.get("title") or ""))
    if not title:
        raise ExtractionError("missing 'title'")

    type_raw = str(raw.get("type") or default_type.value)
    if type_raw not in set(MemoryType):
        raise ExtractionError(f"unknown memory type {type_raw!r}")

    sources = _collect_sources(raw, fallback=source)
    if not sources:
        # A memory whose origin cannot be checked is worse than no memory.
        raise ExtractionError(
            "no source: a candidate must cite where it came from "
            "(pass --source, or include 'source' in the suggestion)"
        )

    occurred = raw.get("occurred_at")
    return Candidate(
        type=MemoryType(type_raw),
        title=title,
        body=str(raw.get("body") or "").strip(),
        tags=clean_tags([str(t) for t in (raw.get("tags") or [])]),
        entities=clean_entities([str(e) for e in (raw.get("entities") or [])]),
        confidence=_unit(raw.get("confidence"), LLM_DEFAULT_CONFIDENCE),
        salience=_unit(raw.get("salience"), 0.5),
        sources=sources,
        occurred_at=(parse_ts(occurred, field_name="occurred_at") if occurred else None),
        rationale=str(raw.get("rationale") or raw.get("reason") or "").strip(),
        source_kind=origin,
        evidence={k: v for k, v in raw.items() if k in ("quote", "locator", "segment_id")},
    )


def _collect_sources(raw: dict[str, Any], *, fallback: SourceRef | None) -> list[SourceRef]:
    out: list[SourceRef] = []
    problems: list[str] = []

    declared = raw.get("sources")
    if isinstance(declared, list):
        for entry in declared:
            parsed = SourceRef.from_yaml(entry, problems=problems, where="source")
            if parsed is not None:
                out.append(parsed)
    single = raw.get("source")
    if isinstance(single, dict):
        parsed = SourceRef.from_yaml(single, problems=problems, where="source")
        if parsed is not None:
            out.append(parsed)
    elif isinstance(single, str) and single.strip():
        out.append(_parse_source_spec(single.strip()))

    if out:
        # A locator the suggestion carried belongs on the source: it is what
        # makes provenance point at a span rather than a whole file.
        locator = raw.get("locator")
        if locator and out[0].locator is None:
            out[0] = SourceRef(
                kind=out[0].kind,
                ref=out[0].ref,
                locator=str(locator),
                captured_at=out[0].captured_at,
                confidence=out[0].confidence,
            )
        return out

    if fallback is not None:
        locator = raw.get("locator")
        if locator:
            return [
                SourceRef(
                    kind=fallback.kind,
                    ref=fallback.ref,
                    locator=str(locator),
                    captured_at=fallback.captured_at,
                )
            ]
        return [fallback]
    return []


def _parse_source_spec(spec: str) -> SourceRef:
    """``KIND:REF[:LOCATOR]``, matching the CLI's ``--source`` form."""
    parts = spec.split(":", 2)
    if len(parts) < 2 or parts[0] not in set(SourceKind):
        raise ExtractionError(
            f"expected KIND:REF[:LOCATOR] with kind in "
            f"{', '.join(sorted(set(SourceKind)))}, got {spec!r}"
        )
    return SourceRef(
        kind=SourceKind(parts[0]),
        ref=parts[1],
        locator=parts[2] if len(parts) > 2 else None,
    )


def _unit(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


# ---------------------------------------------------------------- heuristics

#: First-person patterns that state something durable. Each group 1 is the
#: claim. Kept small and explicit: a broad pattern set would extract
#: half the conversation, and the cost of a wrong memory is paid by the user.
STATEMENT_PATTERNS: tuple[tuple[str, MemoryType, str], ...] = (
    (r"\bI (?:always|usually|generally|typically) ([^.!?\n]{8,})", MemoryType.PREFERENCE, "habit"),
    (r"\bI (?:prefer|like|love|favour|favor) ([^.!?\n]{4,})", MemoryType.PREFERENCE, "preference"),
    (r"\bI (?:hate|dislike|avoid|never use) ([^.!?\n]{4,})", MemoryType.PREFERENCE, "aversion"),
    (r"\bI'?d rather ([^.!?\n]{4,})", MemoryType.PREFERENCE, "preference"),
    (r"\bmy (?:usual|standard|default|go-to) ([^.!?\n]{4,})", MemoryType.PREFERENCE, "default"),
    (r"\b(?:we|I) decided (?:to |that )?([^.!?\n]{4,})", MemoryType.DECISION, "decision"),
    (r"\bI (?:work|live|am based) (?:in|at|on) ([^.!?\n]{3,})", MemoryType.FACT, "location"),
    (r"\bI use ([^.!?\n]{3,}) for ([^.!?\n]{3,})", MemoryType.PREFERENCE, "tooling"),
    (r"\bremember (?:that )?([^.!?\n]{8,})", MemoryType.FACT, "explicit request"),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), mem_type, label)
    for pattern, mem_type, label in STATEMENT_PATTERNS
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def heuristic_candidates(
    text: str,
    *,
    source: SourceRef,
    limit: int = 20,
) -> list[Candidate]:
    """Find explicit first-person statements. Deterministic, no model.

    This is the no-LLM fallback the design requires, and it is honest about
    being shallow: it matches stated intent ("I always…", "we decided…") and
    nothing else. Anything requiring comprehension is a provider's job.
    """
    out: list[Candidate] = []
    seen: set[str] = set()

    for ordinal, sentence in enumerate(_SENTENCE_SPLIT.split(text)):
        flat = " ".join(sentence.split())
        if not (MIN_STATEMENT_CHARS <= len(flat) <= MAX_STATEMENT_CHARS):
            continue
        for pattern, mem_type, label in _COMPILED:
            match = pattern.search(flat)
            if match is None:
                continue
            title = clean_title(flat.rstrip(".!?"))
            if len(title) > 200:
                title = title[:197].rstrip() + "…"
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Candidate(
                    type=mem_type,
                    title=title,
                    body="",
                    confidence=HEURISTIC_CONFIDENCE,
                    sources=[
                        SourceRef(
                            kind=source.kind,
                            ref=source.ref,
                            locator=source.locator or f"sentence:{ordinal}",
                            captured_at=source.captured_at,
                        )
                    ],
                    rationale=f"matched the {label} pattern; verify before committing",
                    source_kind="heuristic",
                    evidence={"pattern": label, "quote": flat[:300]},
                )
            )
            break
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- llm path

EXTRACT_SYSTEM = (
    "You extract durable memories about one person from source material. "
    "You record only what will still be true next month. Answer only with JSON."
)

EXTRACT_PROMPT = """memory-task: extract

Extract durable memories from the material below. Reply with JSON only:

{{"extract": [{{"type": "fact|preference|event|decision|procedure",
  "title": "a standalone assertion, not a topic",
  "body": "one or two sentences of detail",
  "tags": ["lowercase"], "entities": ["Proper Names"],
  "confidence": 0.0-1.0, "quote": "the span this came from"}}]}}

Rules:
- One claim per memory. Two claims cannot be superseded independently.
- The title must assert something ("Prefers uv over pip"), not name a topic
  ("Python tooling").
- Skip anything ephemeral: a task for today, a passing remark, small talk.
- Skip anything you are not reasonably sure of. Fewer, better memories.
- confidence: 0.9+ stated outright, ~0.7 inferred, ~0.5 guessed.
- Return {{"extract": []}} if the material contains nothing durable.

Material:
{material}
"""

#: Material longer than this is truncated. A local model with a small context
#: window would otherwise fail or silently drop the tail; truncating visibly
#: and reporting it is better than either.
MATERIAL_CHAR_LIMIT = 24_000


def llm_candidates(
    provider: LLMProvider,
    text: str,
    *,
    source: SourceRef,
    limit: int = 20,
) -> tuple[list[Candidate], list[str]]:
    """Ask a configured model for candidates.

    Returns warnings rather than raising on a provider or parse failure, so a
    caller can fall back to the heuristic path and still produce something.
    """
    warnings: list[str] = []
    material = text.strip()
    if not material:
        return [], ["source material is empty"]
    if len(material) > MATERIAL_CHAR_LIMIT:
        warnings.append(
            f"material truncated to {MATERIAL_CHAR_LIMIT} of {len(material)} characters"
        )
        material = material[:MATERIAL_CHAR_LIMIT]

    try:
        completion = provider.complete(
            EXTRACT_PROMPT.format(material=material),
            system=EXTRACT_SYSTEM,
            max_tokens=2000,
        )
        payload = extract_json_object(completion.text)
    except (ProviderError, ValueError) as exc:
        return [], [f"provider extraction failed: {exc}"]

    raw_items = payload.get("extract") or payload.get("items") or []
    if not isinstance(raw_items, list):
        return [], ["provider returned no list of candidates"]

    out: list[Candidate] = []
    for i, raw in enumerate(raw_items[:limit]):
        try:
            candidate = _candidate_from_mapping(
                raw, source=source, default_type=MemoryType.FACT, origin=f"llm:{provider.name}"
            )
        except (ExtractionError, ValueError) as exc:
            warnings.append(f"candidate[{i}]: {exc}")
            continue
        out.append(candidate)
    return out, warnings


def from_text(
    text: str,
    *,
    source: SourceRef,
    provider: LLMProvider | None = None,
    limit: int = 20,
    fallback: bool = True,
) -> ExtractionReport:
    """Extract candidates from raw text, with or without a model.

    With a provider, the model leads and the heuristic fills in only if the
    model produced nothing *and* ``fallback`` is set -- otherwise a working
    model returning an honest empty answer would be overridden by pattern
    matches it had already decided against.
    """
    if provider is None:
        return ExtractionReport(
            candidates=heuristic_candidates(text, source=source, limit=limit),
            extracted_by="heuristic",
        )

    candidates, warnings = llm_candidates(provider, text, source=source, limit=limit)
    extracted_by = f"llm:{provider.name}"

    if not candidates and fallback:
        candidates = heuristic_candidates(text, source=source, limit=limit)
        if candidates:
            extracted_by = "heuristic"
            warnings.append("provider returned nothing; fell back to the deterministic heuristic")

    return ExtractionReport(
        candidates=candidates,
        provider=provider.name,
        extracted_by=extracted_by,
        warnings=warnings,
    )


# ---------------------------------------------------------------- review


def annotate_duplicates(
    api: MemoryAPI, report: ExtractionReport, *, threshold: float | None = None
) -> ExtractionReport:
    """Flag candidates that restate something already recorded.

    Extraction's main failure mode in practice is re-extracting the same
    claim from a conversation that gets processed twice. Checking here, before
    anything is written, is cheaper than merging afterwards -- and it reuses
    the same lexical scorer :mod:`memory.consolidate` uses, so the two agree.
    """
    from .consolidate import DUPLICATE_THRESHOLD, NEAR_THRESHOLD, similarity
    from .search import SearchFilters, search

    cut = threshold if threshold is not None else NEAR_THRESHOLD

    for candidate in report.candidates:
        probe = Memory(
            type=candidate.type,
            title=candidate.title,
            body=candidate.body,
            tags=candidate.tags,
            entities=candidate.entities,
        )
        hits = search(
            api.conn,
            candidate.title,
            SearchFilters(sensitivity_max="secret", limit=10),
        )
        matches: list[dict[str, Any]] = []
        for hit in hits:
            existing = api.get(hit.memory_id)
            if existing is None:
                continue
            score, _ = similarity(probe, existing)
            if score >= cut:
                matches.append(
                    {
                        "memory_id": existing.id,
                        "title": existing.title,
                        "score": round(score, 4),
                        "verdict": "duplicate" if score >= DUPLICATE_THRESHOLD else "similar",
                    }
                )
        candidate.duplicate_of = sorted(matches, key=lambda m: -float(m["score"]))[:3]
    return report


def commit(
    api: MemoryAPI,
    report: ExtractionReport,
    *,
    min_confidence: float = 0.0,
    skip_duplicates: bool = True,
    force_recreate: bool = False,
) -> ExtractionReport:
    """Create the candidates, through the ordinary ``create`` verb.

    Skips rather than raises on a tombstoned candidate: re-extraction from a
    retained conversation hitting a tombstone is the resurrection guard doing
    its job, and it must not abort the other candidates. ``force_recreate``
    is plumbed through but never defaulted -- overriding a deletion the user
    asked for needs an explicit decision.
    """
    for candidate in report.candidates:
        if candidate.confidence < min_confidence:
            report.skipped.append(
                {
                    "title": candidate.title,
                    "reason": f"confidence {candidate.confidence} below {min_confidence}",
                }
            )
            continue
        if skip_duplicates and any(m["verdict"] == "duplicate" for m in candidate.duplicate_of):
            report.skipped.append(
                {
                    "title": candidate.title,
                    "reason": "already recorded",
                    "duplicate_of": candidate.duplicate_of,
                }
            )
            continue
        try:
            memory = api.create(force_recreate=force_recreate, **candidate.create_kwargs())
        except Tombstoned as exc:
            report.skipped.append(
                {
                    "title": candidate.title,
                    "reason": "permanently deleted before",
                    "tombstone": exc.tombstone.memory_id,
                    "matched_on": exc.matched_on,
                }
            )
            continue
        except (ValueError, RuntimeError) as exc:
            report.skipped.append(
                {"title": candidate.title, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        report.created.append({"memory_id": memory.id, "title": memory.title})
    return report


__all__ = [
    "HEURISTIC_CONFIDENCE",
    "MATERIAL_CHAR_LIMIT",
    "Candidate",
    "ExtractionError",
    "ExtractionReport",
    "annotate_duplicates",
    "commit",
    "from_suggestions",
    "from_text",
    "heuristic_candidates",
    "llm_candidates",
]
