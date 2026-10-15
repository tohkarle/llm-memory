"""Duplicate, contradiction and staleness detection. Proposes; never writes.

The separation this module is built around: **consolidation proposes, the
existing verbs execute.** Nothing here calls ``merge``, ``replace`` or
``delete``. It emits :class:`Proposal` objects naming the verb and its
arguments, and a caller -- an agent reading JSON, or ``mem consolidate
--apply`` -- decides. Three reasons that line is where it is:

* merging the wrong pair destroys distinctions the user cares about, and only
  the user (or a model with the conversation in front of it) can judge;
* the journal, tombstones and provenance rules already live in
  :class:`~memory.api.MemoryAPI`, and a second write path would diverge;
* a proposal is reviewable. A silent automatic merge is not.

**Candidate generation is lexical and deterministic.** FTS5 finds memories
sharing vocabulary, then scoring is token overlap, title similarity and
metadata agreement -- no embeddings, no vector store, and no model required.
A provider, when configured, only *adjudicates* candidates the lexical pass
already found; it never widens the search. So the detector's recall is the
same whether or not a model is present, which is what makes the test suite
meaningful.

The known limit, stated plainly: two memories that say the same thing in
entirely different words will not be proposed ("prefers uv" vs "pip is
banned here"). That is the same limit :mod:`memory.search` documents, and
the mitigation is the same -- the model doing the recall sees both in a
context pack and can call ``merge`` itself.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .models import (
    Memory,
    MemoryStatus,
    RelationType,
    fmt_ts,
    normalize_text,
)
from .providers import LLMProvider, ProviderError, extract_json_object
from .search import STOPWORDS, SearchFilters, search

if TYPE_CHECKING:  # pragma: no cover
    from .api import MemoryAPI

#: What a proposal asks the caller to do. Each maps onto an existing verb,
#: which is why there is no new mutation code in this module.
Verb = Literal["merge", "replace", "relate", "review", "delete"]

#: Why a pair was surfaced.
Kind = Literal["duplicate", "contradiction", "stale", "near-duplicate"]

#: Scoring weights for lexical duplicate detection. One dict so tuning is a
#: single visible edit, as in :mod:`memory.search`.
WEIGHTS: dict[str, float] = {
    "body_overlap": 0.45,
    "title_overlap": 0.35,
    "tag_overlap": 0.10,
    "entity_overlap": 0.10,
}

#: At or above this, two memories are proposed as duplicates (``merge``).
DUPLICATE_THRESHOLD = 0.72

#: Between this and ``DUPLICATE_THRESHOLD``, they are proposed as related
#: and left for a human: close enough to notice, not close enough to fold.
NEAR_THRESHOLD = 0.45

#: Identical normalized content. Not a judgement call -- the same memory was
#: written twice, so the proposal is high-confidence regardless of weights.
EXACT_SCORE = 1.0

#: How many FTS candidates to consider per memory. A cap keeps the pass
#: linear-ish on a large corpus; candidates are FTS-ranked, so the ones
#: dropped are the least lexically similar.
CANDIDATES_PER_MEMORY = 25

#: Tokens shorter than this carry no discriminating signal and inflate
#: overlap between unrelated memories.
MIN_TOKEN_LEN = 3

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")

#: Lexical negation/polarity markers. Two memories that share vocabulary but
#: disagree on one of these are the classic contradiction the deterministic
#: pass *can* catch: "uses tabs" vs "does not use tabs".
NEGATIONS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "isnt",
        "arent",
        "stopped",
        "quit",
        "avoid",
        "avoids",
        "dislikes",
        "hates",
        "refuses",
        "against",
        "without",
    }
)


@dataclass(slots=True)
class Proposal:
    """One suggested consolidation. Data only -- nothing has been written."""

    kind: Kind
    verb: Verb
    memory_ids: list[str]
    score: float
    rationale: str
    titles: list[str] = field(default_factory=list)
    #: Ready-to-run arguments for the named verb, so a caller never has to
    #: reconstruct them from prose.
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = "lexical"
    evidence: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, ...]:
        """Identity of the proposal, independent of pair ordering."""
        return (self.kind, *sorted(self.memory_ids))

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verb": self.verb,
            "memory_ids": self.memory_ids,
            "titles": self.titles,
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "arguments": self.arguments,
            "source": self.source,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class ConsolidationReport:
    scanned: int
    compared: int
    proposals: list[Proposal]
    provider: str | None = None
    warnings: list[str] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "compared": self.compared,
            "total": len(self.proposals),
            "by_kind": self.counts(),
            "provider": self.provider,
            "proposals": [p.to_json() for p in self.proposals],
            "applied": self.applied,
            "warnings": self.warnings,
        }

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for proposal in self.proposals:
            out[proposal.kind] = out.get(proposal.kind, 0) + 1
        return out


# ---------------------------------------------------------------- similarity


def fold_diacritics(text: str) -> str:
    """Strip combining marks, matching the FTS5 tokenizer.

    The index is built with ``remove_diacritics 2``, so FTS5 already treats
    ``café`` and ``cafe`` as one term. Scoring has to agree: otherwise a pair
    the candidate query *found* by shared vocabulary would score as sharing
    none of it, and the two halves of the detector would disagree about what
    a word is.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def tokens(text: str) -> set[str]:
    """Content tokens used for overlap.

    Reuses :data:`memory.search.STOPWORDS` on purpose: the detector and the
    search ranker should disagree about what carries signal as rarely as
    possible, since a user comparing ``mem search`` output to a duplicate
    proposal is comparing the same notion of "shares vocabulary".
    """
    return {
        token
        for token in _TOKEN_RE.findall(fold_diacritics(normalize_text(text)))
        if len(token) >= MIN_TOKEN_LEN and token not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def containment(left: set[str], right: set[str]) -> float:
    """Overlap relative to the *smaller* set.

    Jaccard alone misjudges the common real case of a one-line memory and a
    paragraph-long one saying the same thing: the union is dominated by the
    longer body, so the score collapses even when the short one is wholly
    contained. Taking the max of the two keeps that pair visible.
    """
    smaller = min(len(left), len(right))
    if smaller == 0:
        return 0.0
    return len(left & right) / smaller


def similarity(a: Memory, b: Memory) -> tuple[float, dict[str, Any]]:
    """Lexical similarity in [0, 1], with the component scores as evidence.

    Evidence is returned, not logged: a proposal a user cannot inspect is a
    proposal they have to take on faith.
    """
    if a.content_hash == b.content_hash:
        return EXACT_SCORE, {"exact_content_match": True}

    a_body, b_body = tokens(a.body), tokens(b.body)
    a_title, b_title = tokens(a.title), tokens(b.title)
    a_tags, b_tags = set(a.tags), set(b.tags)
    a_ents = {e.casefold() for e in a.entities}
    b_ents = {e.casefold() for e in b.entities}

    body = max(jaccard(a_body, b_body), containment(a_body, b_body))
    title = max(jaccard(a_title, b_title), containment(a_title, b_title))

    components = {
        "body_overlap": round(body, 4),
        "title_overlap": round(title, 4),
        "tag_overlap": round(jaccard(a_tags, b_tags), 4),
        "entity_overlap": round(jaccard(a_ents, b_ents), 4),
    }

    # Renormalize over the components that actually carry evidence. Most
    # memories have no tags and no entities, so scoring their absence as a
    # zero would cap every text-only pair at 80% of the possible score and
    # push ordinary paraphrases below the threshold -- the detector would
    # miss exactly the duplicates it exists to find. Absent metadata is *no
    # evidence*, not *evidence against*.
    present = {"body_overlap", "title_overlap"}
    if a_tags or b_tags:
        present.add("tag_overlap")
    if a_ents or b_ents:
        present.add("entity_overlap")
    # A memory with no body is all title; comparing bodies that are both
    # empty says nothing either way.
    if not a_body and not b_body:
        present.discard("body_overlap")

    weight = sum(WEIGHTS[name] for name in present)
    score = sum(WEIGHTS[name] * components[name] for name in present) / weight

    return min(score, 1.0), components


def polarity_conflict(a: Memory, b: Memory) -> dict[str, Any] | None:
    """Detect a shared-subject pair that disagrees on negation.

    Deliberately narrow. This fires on "uses X" vs "does not use X", which is
    the contradiction a lexical pass can actually establish; anything subtler
    is left to a configured model, which is why the LLM path exists at all.
    """
    a_text = f"{a.title} {a.body}"
    b_text = f"{b.title} {b.body}"
    a_neg = tokens(a_text) & NEGATIONS
    b_neg = tokens(b_text) & NEGATIONS
    # Both negated or neither negated is not a polarity difference.
    if bool(a_neg) == bool(b_neg):
        return None

    # The subject has to actually be shared, or "never use pip" and "likes
    # hiking" would read as a contradiction about nothing.
    shared = (tokens(a_text) - NEGATIONS) & (tokens(b_text) - NEGATIONS)
    subject = (tokens(a.title) | tokens(b.title)) - NEGATIONS
    if not subject or len(shared & subject) < max(1, len(subject) // 3):
        return None

    return {
        "negated": sorted(a_neg | b_neg),
        "shared_terms": sorted(shared)[:12],
        "negated_side": a.id if a_neg else b.id,
    }


# ---------------------------------------------------------------- candidates


def candidate_pairs(
    api: MemoryAPI,
    *,
    types: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    memory_ids: Sequence[str] | None = None,
    limit_memories: int = 500,
    candidates_per_memory: int = CANDIDATES_PER_MEMORY,
) -> Iterator[tuple[Memory, Memory]]:
    """Yield each comparable pair once, using FTS5 to avoid the full cross product.

    For every memory, FTS is queried with that memory's own distinctive terms;
    anything it returns shares vocabulary and is worth scoring. A quadratic
    scan over a few hundred memories would also be affordable, but it is not
    affordable at ten thousand, and this keeps the same code path at both
    sizes.
    """
    anchors = _anchor_memories(
        api, types=types, tags=tags, memory_ids=memory_ids, limit=limit_memories
    )
    by_id = {m.id: m for m in anchors}
    seen: set[tuple[str, str]] = set()

    for anchor in anchors:
        for other_id in _fts_neighbours(api, anchor, limit=candidates_per_memory):
            if other_id == anchor.id:
                continue
            first, second = sorted((anchor.id, other_id))
            if (first, second) in seen:
                continue
            seen.add((first, second))
            other = by_id.get(other_id)
            if other is None:
                fetched = api.get(other_id)
                if fetched is None or fetched.status is not MemoryStatus.ACTIVE:
                    continue
                # A memory outside the filtered anchor set is still a valid
                # partner: a duplicate does not have to carry the same tag.
                other = fetched
                by_id[other_id] = other
            yield anchor, other


def _anchor_memories(
    api: MemoryAPI,
    *,
    types: Sequence[str] | None,
    tags: Sequence[str] | None,
    memory_ids: Sequence[str] | None,
    limit: int,
) -> list[Memory]:
    if memory_ids:
        found = [api.get(mid) for mid in memory_ids]
        return [m for m in found if m is not None]
    hits = search(
        api.conn,
        None,
        SearchFilters(
            types=list(types) if types else None,
            tags_any=list(tags) if tags else None,
            statuses=[MemoryStatus.ACTIVE.value],
            sensitivity_max="secret",
            limit=limit,
            order="updated",
        ),
    )
    out: list[Memory] = []
    for hit in hits:
        memory = api.get(hit.memory_id)
        if memory is not None:
            out.append(memory)
    return out


def _fts_neighbours(api: MemoryAPI, memory: Memory, *, limit: int) -> list[str]:
    """Ids sharing vocabulary with this memory, FTS-ranked."""
    terms = sorted(tokens(f"{memory.title} {memory.body}"))
    if not terms:
        return []
    # Title terms first: they are the memory's assertion, so they are the
    # terms most likely to be shared by a genuine duplicate. The body tail is
    # truncated because an FTS5 query of several hundred ORed terms is slow
    # and matches almost everything.
    title_terms = sorted(tokens(memory.title))
    ordered = title_terms + [t for t in terms if t not in set(title_terms)]
    query = "fts:" + " OR ".join(f'"{t}"' for t in ordered[:40])
    try:
        hits = search(
            api.conn,
            query,
            SearchFilters(
                statuses=[MemoryStatus.ACTIVE.value],
                sensitivity_max="secret",
                exclude_ids=[memory.id],
                limit=limit,
                order="score",
            ),
        )
    except Exception:
        # A query that FTS5 rejects must degrade to "no candidates for this
        # memory", never abort a whole consolidation pass.
        return []
    return [h.memory_id for h in hits]


# ---------------------------------------------------------------- detection


def detect(
    api: MemoryAPI,
    *,
    types: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    memory_ids: Sequence[str] | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
    near_threshold: float = NEAR_THRESHOLD,
    include_near: bool = True,
    include_stale: bool = True,
    limit: int = 50,
    limit_memories: int = 500,
    provider: LLMProvider | None = None,
) -> ConsolidationReport:
    """Scan for duplicates, contradictions and staleness. Writes nothing.

    Runs fully without a provider. When one is given it adjudicates the
    borderline band only -- pairs the lexical pass scored between
    ``near_threshold`` and ``threshold`` -- because that is where a model
    changes the answer and where a wrong automatic merge would do damage.
    """
    proposals: dict[tuple[str, ...], Proposal] = {}
    warnings: list[str] = []
    compared = 0
    scanned_ids: set[str] = set()

    borderline: list[tuple[Memory, Memory, float, dict[str, Any]]] = []

    for left, right in candidate_pairs(
        api,
        types=types,
        tags=tags,
        memory_ids=memory_ids,
        limit_memories=limit_memories,
    ):
        compared += 1
        scanned_ids.update((left.id, right.id))

        # An existing edge is the user's answer already: proposing a merge
        # for a pair someone deliberately marked `contradicts` would be the
        # tool arguing with its own records.
        existing = _existing_edge(api.conn, left.id, right.id)

        score, evidence = similarity(left, right)
        conflict = polarity_conflict(left, right)

        if conflict is not None and score >= near_threshold:
            if existing != RelationType.CONTRADICTS.value:
                proposal = _contradiction_proposal(left, right, score, evidence, conflict)
                proposals[proposal.key()] = proposal
            continue

        if score >= threshold:
            if existing not in (
                RelationType.SUPERSEDES.value,
                RelationType.CONTRADICTS.value,
            ):
                proposal = _duplicate_proposal(left, right, score, evidence)
                proposals[proposal.key()] = proposal
            continue

        if score >= near_threshold:
            borderline.append((left, right, score, evidence))

    if provider is not None and borderline:
        adjudicated, llm_warnings = _adjudicate(provider, borderline)
        warnings.extend(llm_warnings)
        for proposal in adjudicated:
            proposals.setdefault(proposal.key(), proposal)
    elif include_near:
        for left, right, score, evidence in borderline:
            if _existing_edge(api.conn, left.id, right.id) is None:
                proposal = _near_proposal(left, right, score, evidence)
                proposals.setdefault(proposal.key(), proposal)

    if include_stale:
        for proposal in _stale_proposals(api, limit=limit):
            proposals.setdefault(proposal.key(), proposal)
            scanned_ids.update(proposal.memory_ids)

    ordered = sorted(proposals.values(), key=lambda p: (-p.score, p.kind, p.memory_ids))
    return ConsolidationReport(
        scanned=len(scanned_ids),
        compared=compared,
        proposals=ordered[:limit],
        provider=(provider.name if provider is not None else None),
        warnings=warnings,
    )


def _existing_edge(conn: sqlite3.Connection, left: str, right: str) -> str | None:
    """The declared relation between two memories, in either direction."""
    row = conn.execute(
        "SELECT type FROM relations WHERE (src_id = ? AND dst_id = ?) "
        "OR (src_id = ? AND dst_id = ?) "
        "ORDER BY CASE type WHEN 'contradicts' THEN 0 WHEN 'supersedes' THEN 1 ELSE 2 END "
        "LIMIT 1",
        (left, right, right, left),
    ).fetchone()
    return str(row["type"]) if row is not None else None


def _duplicate_proposal(
    left: Memory, right: Memory, score: float, evidence: dict[str, Any]
) -> Proposal:
    # Order matters for the caller: `--into` folds into the first id, and the
    # memory with more provenance is the better survivor because merge unions
    # sources but only one id lives on.
    ordered = sorted((left, right), key=_survivor_rank, reverse=True)
    exact = bool(evidence.get("exact_content_match"))
    return Proposal(
        kind="duplicate" if exact or score >= 0.85 else "near-duplicate",
        verb="merge",
        memory_ids=[m.id for m in ordered],
        score=score,
        titles=[m.title for m in ordered],
        rationale=(
            "identical normalized content"
            if exact
            else f"high lexical overlap (body {evidence.get('body_overlap')}, "
            f"title {evidence.get('title_overlap')})"
        ),
        arguments={
            "ids": [m.id for m in ordered],
            "into": ordered[0].id,
            "strategy": "union",
        },
        evidence=evidence,
    )


def _survivor_rank(memory: Memory) -> tuple[int, int, float, str]:
    """Prefer the memory with more provenance, then more metadata, then newer."""
    return (
        len(memory.sources),
        len(memory.tags) + len(memory.entities) + len(memory.assets),
        memory.confidence,
        fmt_ts(memory.updated_at) or "",
    )


def _contradiction_proposal(
    left: Memory,
    right: Memory,
    score: float,
    evidence: dict[str, Any],
    conflict: dict[str, Any],
) -> Proposal:
    # Newest first: if one of these supersedes the other, it is the recent
    # one, and `replace` takes the *old* id as its argument.
    ordered = sorted((left, right), key=lambda m: fmt_ts(m.updated_at) or "", reverse=True)
    return Proposal(
        kind="contradiction",
        verb="relate",
        memory_ids=[m.id for m in ordered],
        score=score,
        titles=[m.title for m in ordered],
        rationale=(
            "shared subject with opposite polarity "
            f"({', '.join(conflict['negated'])}); if the newer one replaces the "
            "older, use `replace`, otherwise record the dispute"
        ),
        arguments={
            "relation": RelationType.CONTRADICTS.value,
            "source_id": ordered[0].id,
            "target_id": ordered[1].id,
            "alternative": {"verb": "replace", "memory_id": ordered[1].id},
        },
        evidence={**evidence, **conflict},
    )


def _near_proposal(left: Memory, right: Memory, score: float, evidence: dict[str, Any]) -> Proposal:
    ordered = sorted((left, right), key=_survivor_rank, reverse=True)
    return Proposal(
        kind="near-duplicate",
        verb="review",
        memory_ids=[m.id for m in ordered],
        score=score,
        titles=[m.title for m in ordered],
        rationale=(
            "moderate lexical overlap -- related, but too different to merge "
            "automatically; read both and decide"
        ),
        arguments={"ids": [m.id for m in ordered]},
        evidence=evidence,
    )


def _stale_proposals(api: MemoryAPI, *, limit: int) -> list[Proposal]:
    """Memories the data itself says are out of date.

    No heuristics about content: this reports what the user already declared
    by setting ``review_after`` or ``valid_until``. Guessing staleness from
    age would quietly bury durable facts, which do not expire.
    """
    out: list[Proposal] = []
    now = fmt_ts(_now())

    for row in api.conn.execute(
        "SELECT id, title, review_after, valid_until FROM memories "
        "WHERE status = 'active' AND ("
        "  (review_after IS NOT NULL AND review_after <= ?) OR "
        "  (valid_until  IS NOT NULL AND valid_until  <= ?)) "
        "ORDER BY COALESCE(valid_until, review_after), id LIMIT ?",
        (now, now, limit),
    ):
        expired = row["valid_until"] is not None and row["valid_until"] <= (now or "")
        out.append(
            Proposal(
                kind="stale",
                verb="review" if not expired else "replace",
                memory_ids=[row["id"]],
                score=0.6 if not expired else 0.8,
                titles=[row["title"]],
                rationale=(
                    f"valid_until {row['valid_until']} has passed; confirm, replace, or archive it"
                    if expired
                    else f"review_after {row['review_after']} has passed; re-confirm or update it"
                ),
                arguments={"memory_id": row["id"]},
                evidence={
                    "review_after": row["review_after"],
                    "valid_until": row["valid_until"],
                },
            )
        )
    return out


def _now() -> Any:
    from .models import now_utc

    return now_utc()


# ---------------------------------------------------------------- adjudication

ADJUDICATE_SYSTEM = (
    "You compare pairs of recorded memories about one person and decide "
    "whether each pair is a duplicate, a contradiction, or genuinely "
    "distinct. Answer only with JSON."
)

ADJUDICATE_PROMPT = """memory-task: adjudicate

For each numbered pair below, decide the relationship. Reply with JSON only:

{{"adjudicate": [{{"pair": 1, "verdict": "duplicate|contradiction|distinct",
  "confidence": 0.0-1.0, "reason": "one short sentence"}}]}}

Rules:
- "duplicate" means both state the same fact; one could be deleted without
  losing information.
- "contradiction" means both cannot be true at once.
- "distinct" means they are merely related. Prefer "distinct" when unsure.

Pairs:
{pairs}
"""


def _adjudicate(
    provider: LLMProvider,
    borderline: list[tuple[Memory, Memory, float, dict[str, Any]]],
    *,
    batch: int = 10,
) -> tuple[list[Proposal], list[str]]:
    """Ask a model about pairs the lexical pass could not call.

    A provider failure degrades to zero proposals from this band plus a
    warning, never an exception: a consolidation scan that aborts because a
    local runtime was down would be worse than one that reports less.
    """
    proposals: list[Proposal] = []
    warnings: list[str] = []

    for start in range(0, len(borderline), batch):
        chunk = borderline[start : start + batch]
        prompt = ADJUDICATE_PROMPT.format(pairs=_render_pairs(chunk))
        try:
            completion = provider.complete(prompt, system=ADJUDICATE_SYSTEM, max_tokens=1200)
            payload = extract_json_object(completion.text)
        except (ProviderError, ValueError) as exc:
            warnings.append(f"provider adjudication failed: {exc}")
            continue

        verdicts = payload.get("adjudicate") or payload.get("items") or []
        if not isinstance(verdicts, list):
            warnings.append("provider adjudication returned no list of verdicts")
            continue

        for raw in verdicts:
            if not isinstance(raw, dict):
                continue
            index = _opt_int(raw.get("pair"))
            if index is None or not 1 <= index <= len(chunk):
                continue
            left, right, score, evidence = chunk[index - 1]
            verdict = str(raw.get("verdict") or "").strip().casefold()
            reason = str(raw.get("reason") or "").strip()
            confidence = _opt_float(raw.get("confidence"))

            if verdict == "duplicate":
                proposal = _duplicate_proposal(left, right, score, evidence)
            elif verdict == "contradiction":
                proposal = _contradiction_proposal(
                    left, right, score, evidence, {"negated": [], "shared_terms": []}
                )
            else:
                continue
            proposal.source = f"llm:{provider.name}"
            proposal.rationale = reason or proposal.rationale
            if confidence is not None:
                proposal.evidence["llm_confidence"] = confidence
            proposals.append(proposal)

    return proposals, warnings


def _render_pairs(chunk: list[tuple[Memory, Memory, float, dict[str, Any]]]) -> str:
    blocks: list[str] = []
    for i, (left, right, score, _) in enumerate(chunk, start=1):
        blocks.append(
            f"[{i}] lexical_overlap={score:.2f}\n"
            f"  A ({left.type.value}): {left.title}\n"
            f"     {_clip(left.body)}\n"
            f"  B ({right.type.value}): {right.title}\n"
            f"     {_clip(right.body)}"
        )
    return "\n\n".join(blocks)


def _clip(text: str, limit: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- application


def apply_proposal(api: MemoryAPI, proposal: Proposal) -> dict[str, Any]:
    """Execute one proposal through the existing verbs.

    Only ``merge`` and ``relate`` are executed. ``replace``, ``delete`` and
    ``review`` need content or a judgement this module does not have -- a
    replacement needs a new title and body, and inventing one would be the
    tool deciding what the user now believes.
    """
    if proposal.verb == "merge":
        ids = list(proposal.arguments.get("ids") or proposal.memory_ids)
        if len(ids) < 2:
            raise ValueError("a merge proposal needs at least two ids")
        result = api.merge(
            ids,
            into=proposal.arguments.get("into"),
            strategy=str(proposal.arguments.get("strategy") or "union"),
        )
        return {"verb": "merge", "result": result.to_json()}

    if proposal.verb == "relate":
        source_id = str(proposal.arguments["source_id"])
        target_id = str(proposal.arguments["target_id"])
        rel_type = RelationType(
            str(proposal.arguments.get("relation") or RelationType.CONTRADICTS.value)
        )
        memory = api.require(source_id)
        if any(r.type is rel_type and r.target == target_id for r in memory.relations):
            return {"verb": "relate", "result": {"unchanged": True, "memory_id": source_id}}
        from .models import Relation

        updated = api.update(
            source_id,
            relations=[
                *memory.relations,
                Relation(type=rel_type, target=target_id, note=proposal.rationale[:200]),
            ],
        )
        return {
            "verb": "relate",
            "result": {
                "memory_id": updated.id,
                "relation": rel_type.value,
                "target": target_id,
            },
        }

    raise ValueError(
        f"{proposal.verb} proposals are advisory: they need content or a judgement "
        "`consolidate --apply` cannot supply. Run the verb yourself."
    )


def apply_all(
    api: MemoryAPI, report: ConsolidationReport, *, kinds: Sequence[str] | None = None
) -> ConsolidationReport:
    """Apply every automatically-applicable proposal, reporting each outcome.

    One proposal failing does not stop the rest: a merge can legitimately
    fail (a parent was deleted between detection and application) and the
    remaining proposals are still good.
    """
    allowed = set(kinds) if kinds else {"duplicate"}
    applied: list[dict[str, Any]] = []
    # A merge supersedes its parents, so a later proposal naming one of them
    # would operate on history. Tracking consumed ids keeps `--apply` from
    # cascading in one pass.
    consumed: set[str] = set()

    for proposal in report.proposals:
        if proposal.kind not in allowed or proposal.verb not in ("merge", "relate"):
            continue
        if consumed & set(proposal.memory_ids):
            applied.append(
                {
                    "proposal": proposal.key(),
                    "skipped": "a memory in this proposal was already consolidated in this pass",
                }
            )
            continue
        try:
            outcome = apply_proposal(api, proposal)
        except Exception as exc:
            applied.append({"proposal": proposal.key(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        consumed.update(proposal.memory_ids)
        applied.append({"proposal": proposal.key(), **outcome})

    report.applied = applied
    return report


__all__ = [
    "DUPLICATE_THRESHOLD",
    "EXACT_SCORE",
    "NEAR_THRESHOLD",
    "WEIGHTS",
    "ConsolidationReport",
    "Proposal",
    "apply_all",
    "apply_proposal",
    "candidate_pairs",
    "containment",
    "detect",
    "fold_diacritics",
    "jaccard",
    "polarity_conflict",
    "similarity",
    "tokens",
]
