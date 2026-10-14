"""Keyword search, metadata filters, and relation traversal.

No embeddings. Retrieval is FTS5 for terms, SQL for structure, and the graph
for context; the calling model does the final relevance reasoning over a
budgeted pack assembled in :mod:`memory.context`.

The known limit of this approach: a query sharing no terms with a memory
will not match it ("python packaging" misses "prefers uv over pip"). Three
things soften that -- tags and entities are indexed alongside the body,
profile memories bypass search entirely in context assembly, and a query
prefixed ``fts:`` is passed to FTS5 verbatim so a caller can OR synonyms
together.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from .models import (
    SENSITIVITY_ORDER,
    MemoryStatus,
    MemoryType,
    RelationType,
    fmt_ts,
)

#: Ranking weights. One dict so tuning is a single visible edit.
WEIGHTS: dict[str, float] = {
    "bm25": 1.0,
    "recency": 0.35,
    "salience": 0.25,
    "confidence": 0.15,
    "profile": 0.40,
    "type": 0.10,
}

#: Mild priors: durable statements about the user outrank one-off events.
TYPE_PRIOR: dict[str, float] = {
    "preference": 1.0,
    "fact": 0.9,
    "procedure": 0.8,
    "decision": 0.7,
    "event": 0.5,
}

RECENCY_HALF_LIFE_DAYS = 180.0

#: Relation strengths for context expansion. ``contradicts`` leads on
#: purpose: if a retrieved memory is disputed, the caller must see the
#: dispute or it will state something stale with confidence.
EDGE_WEIGHT: dict[str, float] = {
    "contradicts": 1.0,
    "supersedes": 0.9,
    "causes": 0.8,
    "part_of": 0.7,
    "derived_from": 0.6,
    "relates_to": 0.5,
    "references": 0.4,
}

VERBATIM_PREFIX = "fts:"

_FTS_SPECIALS = re.compile(r'[*"():^{}\[\]~]')
_TOKEN_SPLIT = re.compile(r"\s+")

Order = Literal["score", "updated", "created", "occurred"]


@dataclass(slots=True)
class SearchFilters:
    types: list[str] | None = None
    statuses: list[str] | None = None
    tags_any: list[str] | None = None
    tags_all: list[str] | None = None
    entities_any: list[str] | None = None
    profile: bool | None = None
    min_confidence: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    review_due: bool = False
    source_ref: str | None = None
    related_to: str | None = None
    sensitivity_max: str = "private"
    exclude_ids: list[str] = field(default_factory=list)
    limit: int = 20
    offset: int = 0
    order: Order = "score"

    def effective_statuses(self) -> list[str]:
        """Default to active only.

        Superseded and deleted memories stay queryable by id and by explicit
        request, but must not surface in ordinary search -- otherwise every
        correction a user makes keeps resurfacing beside its replacement.
        """
        return self.statuses or [MemoryStatus.ACTIVE.value]


@dataclass(slots=True)
class Hit:
    memory_id: str
    title: str
    type: str
    status: str
    score: float
    bm25: float | None
    snippet: str
    tags: list[str]
    updated_at: str | None
    path: str
    matched_via: str

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "title": self.title,
            "type": self.type,
            "status": self.status,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4) if self.bm25 is not None else None,
            "snippet": self.snippet,
            "tags": self.tags,
            "updated_at": self.updated_at,
            "path": self.path,
            "matched_via": self.matched_via,
        }


@dataclass(slots=True)
class RelatedHit:
    memory_id: str
    title: str
    type: str
    status: str
    relation: str
    direction: Literal["out", "in"]
    hop: int
    implied: bool
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "title": self.title,
            "type": self.type,
            "status": self.status,
            "relation": self.relation,
            "direction": self.direction,
            "hop": self.hop,
            "implied": self.implied,
            "note": self.note,
        }


class SearchError(ValueError):
    """The query could not be parsed."""


def fts_query(raw: str) -> str:
    """Turn user text into a safe FTS5 MATCH expression.

    FTS5's query language treats bare punctuation as syntax, so an
    unsanitized query is a syntax error rather than a no-match -- ``what's
    up?`` or ``claude-code`` would raise. Each token is quoted and ANDed.

    A query starting ``fts:`` is passed through untouched, which is the
    escape hatch for OR-expansion and NEAR.
    """
    text = raw.strip()
    if not text:
        raise SearchError("empty query")
    if text.startswith(VERBATIM_PREFIX):
        verbatim = text[len(VERBATIM_PREFIX) :].strip()
        if not verbatim:
            raise SearchError("empty fts: query")
        return verbatim

    tokens: list[str] = []
    for token in _TOKEN_SPLIT.split(text):
        prefix = token.endswith("*")
        cleaned = _FTS_SPECIALS.sub(" ", token).strip()
        if not cleaned:
            continue
        quoted = '"' + cleaned.replace('"', "") + '"'
        tokens.append(quoted + "*" if prefix else quoted)
    if not tokens:
        raise SearchError(f"no searchable terms in {raw!r}")
    return " AND ".join(tokens)


def recency_decay(
    then: datetime | str | None,
    *,
    now: datetime | None = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay in (0, 1]. Recent memories score higher."""
    if then is None:
        return 0.0
    if isinstance(then, str):
        try:
            then = datetime.fromisoformat(then.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    reference = now or datetime.now(UTC)
    age_days = max((reference - then).total_seconds() / 86400.0, 0.0)
    return math.pow(0.5, age_days / half_life_days)


def normalize_bm25(bm25: float | None) -> float:
    """Map FTS5's bm25 onto [0, 1), increasing with match quality.

    bm25() returns negative values where *more negative is a better match*,
    so the transform has to be increasing in ``-bm25``. Normalizing against
    the result-set maximum instead would make a row's score depend on which
    page it appeared on; this mapping does not, so scores are stable across
    pagination and reproducible after an index rebuild.
    """
    if bm25 is None:
        return 0.0
    quality = max(0.0, -bm25)
    return quality / (1.0 + quality)


def _filter_sql(filters: SearchFilters, params: list[Any]) -> list[str]:
    """Build WHERE fragments shared by keyword and metadata-only search."""
    clauses: list[str] = []

    statuses = filters.effective_statuses()
    clauses.append(f"m.status IN ({', '.join('?' * len(statuses))})")
    params.extend(statuses)

    if filters.types:
        for value in filters.types:
            if value not in set(MemoryType):
                raise SearchError(f"unknown memory type: {value}")
        clauses.append(f"m.type IN ({', '.join('?' * len(filters.types))})")
        params.extend(filters.types)

    cap = SENSITIVITY_ORDER.get(filters.sensitivity_max)
    if cap is None:
        raise SearchError(f"unknown sensitivity: {filters.sensitivity_max}")
    allowed = [name for name, rank in SENSITIVITY_ORDER.items() if rank <= cap]
    clauses.append(f"m.sensitivity IN ({', '.join('?' * len(allowed))})")
    params.extend(allowed)

    if filters.profile is not None:
        clauses.append("m.profile = ?")
        params.append(1 if filters.profile else 0)

    if filters.min_confidence is not None:
        clauses.append("m.confidence >= ?")
        params.append(filters.min_confidence)

    bounds: tuple[tuple[str, datetime | None, str], ...] = (
        ("created_at", filters.created_after, ">="),
        ("created_at", filters.created_before, "<="),
        ("occurred_at", filters.occurred_after, ">="),
        ("occurred_at", filters.occurred_before, "<="),
    )
    for column, bound_at, op in bounds:
        if bound_at is not None:
            clauses.append(f"m.{column} {op} ?")
            params.append(fmt_ts(bound_at))

    if filters.review_due:
        clauses.append("m.review_after IS NOT NULL AND m.review_after <= ?")
        params.append(fmt_ts(datetime.now(UTC)))

    if filters.tags_any:
        placeholders = ", ".join("?" * len(filters.tags_any))
        clauses.append(f"m.id IN (SELECT memory_id FROM memory_tags WHERE tag IN ({placeholders}))")
        params.extend(filters.tags_any)

    if filters.tags_all:
        placeholders = ", ".join("?" * len(filters.tags_all))
        clauses.append(
            "m.id IN (SELECT memory_id FROM memory_tags "
            f"WHERE tag IN ({placeholders}) GROUP BY memory_id HAVING count(*) = ?)"
        )
        params.extend(filters.tags_all)
        params.append(len(set(filters.tags_all)))

    if filters.entities_any:
        placeholders = ", ".join("?" * len(filters.entities_any))
        clauses.append(
            "m.id IN (SELECT memory_id FROM memory_entities "
            f"WHERE entity COLLATE NOCASE IN ({placeholders}))"
        )
        params.extend(filters.entities_any)

    if filters.source_ref:
        clauses.append(
            "m.id IN (SELECT ms.memory_id FROM memory_sources ms "
            "JOIN sources s ON s.id = ms.source_id WHERE s.ref LIKE ?)"
        )
        params.append(f"%{filters.source_ref}%")

    if filters.related_to:
        clauses.append(
            "m.id IN (SELECT dst_id FROM relations WHERE src_id = ? "
            "UNION SELECT src_id FROM relations WHERE dst_id = ?)"
        )
        params.extend([filters.related_to, filters.related_to])

    if filters.exclude_ids:
        placeholders = ", ".join("?" * len(filters.exclude_ids))
        clauses.append(f"m.id NOT IN ({placeholders})")
        params.extend(filters.exclude_ids)

    return clauses


_ORDER_SQL: dict[str, str] = {
    "updated": "m.updated_at DESC, m.id",
    "created": "m.created_at DESC, m.id",
    "occurred": "m.occurred_at IS NULL, m.occurred_at DESC, m.id",
}


def search(
    conn: sqlite3.Connection,
    query: str | None = None,
    filters: SearchFilters | None = None,
    *,
    now: datetime | None = None,
) -> list[Hit]:
    """Keyword and/or metadata search, ranked.

    With no query this is a pure metadata listing ordered by recency, which
    is what makes ``mem search --tag x`` and ``mem review`` work.
    """
    filters = filters or SearchFilters()
    params: list[Any] = []
    clauses = _filter_sql(filters, params)

    if query and query.strip():
        try:
            match = fts_query(query)
        except SearchError:
            # The query was all punctuation or operators, so no term can
            # match. That is an empty result, not a failure -- raising here
            # would make every caller guard ordinary user text.
            return []
        sql = (
            "SELECT m.id, m.title, m.type, m.status, m.tags, m.updated_at, m.path, "
            "m.salience, m.confidence, m.profile, "
            "bm25(memories_fts) AS bm25, "
            "snippet(memories_fts, 1, '', '', '…', 24) AS snip "
            "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
            f"WHERE memories_fts MATCH ? AND {' AND '.join(clauses)}"
        )
        bound = [match, *params]
        matched_via = "fts"
    else:
        sql = (
            "SELECT m.id, m.title, m.type, m.status, m.tags, m.updated_at, m.path, "
            "m.salience, m.confidence, m.profile, "
            "NULL AS bm25, substr(m.body, 1, 160) AS snip "
            "FROM memories m "
            f"WHERE {' AND '.join(clauses)}"
        )
        bound = list(params)
        matched_via = "filter"

    try:
        rows = conn.execute(sql, bound).fetchall()
    except sqlite3.OperationalError as exc:
        raise SearchError(f"invalid query: {exc}") from None

    hits: list[Hit] = []
    for row in rows:
        score = _score(row, now=now)
        hits.append(
            Hit(
                memory_id=row["id"],
                title=row["title"],
                type=row["type"],
                status=row["status"],
                score=score,
                bm25=row["bm25"],
                snippet=(row["snip"] or "").strip(),
                tags=json.loads(row["tags"] or "[]"),
                updated_at=row["updated_at"],
                path=row["path"],
                matched_via="profile" if row["profile"] else matched_via,
            )
        )

    if filters.order == "score":
        hits.sort(key=lambda h: (-h.score, h.updated_at or "", h.memory_id))
    elif filters.order == "updated":
        hits.sort(key=lambda h: (h.updated_at or "", h.memory_id), reverse=True)
    else:
        # created_at/occurred_at are not carried on Hit, so let SQLite supply
        # the ordering for exactly the ids we matched.
        position = {
            mid: i
            for i, mid in enumerate(_ordered_ids(conn, [h.memory_id for h in hits], filters.order))
        }
        hits.sort(key=lambda h: position.get(h.memory_id, len(position)))

    return hits[filters.offset : filters.offset + filters.limit]


def _ordered_ids(conn: sqlite3.Connection, ids: list[str], key: str) -> list[str]:
    if not ids:
        return []
    placeholders = ", ".join("?" * len(ids))
    column = "created_at" if key == "created" else "occurred_at"
    rows = conn.execute(
        f"SELECT id FROM memories WHERE id IN ({placeholders}) "
        f"ORDER BY {column} IS NULL, {column} DESC, id",
        ids,
    )
    return [row["id"] for row in rows]


def _score(row: sqlite3.Row, *, now: datetime | None = None) -> float:
    return (
        WEIGHTS["bm25"] * normalize_bm25(row["bm25"])
        + WEIGHTS["recency"] * recency_decay(row["updated_at"], now=now)
        + WEIGHTS["salience"] * float(row["salience"])
        + WEIGHTS["confidence"] * float(row["confidence"])
        + WEIGHTS["profile"] * (1.0 if row["profile"] else 0.0)
        + WEIGHTS["type"] * TYPE_PRIOR.get(row["type"], 0.5)
    )


def get_related(
    conn: sqlite3.Connection,
    mid: str,
    *,
    depth: int = 1,
    types: list[str] | None = None,
    direction: Literal["out", "in", "both"] = "both",
    include_implied: bool = False,
    statuses: list[str] | None = None,
    limit: int = 50,
) -> list[RelatedHit]:
    """Walk the relation graph outward from one memory.

    Breadth-first with a visited set, so a cycle terminates. Implied (inverse)
    edges are excluded by default: including them would report every edge
    twice.
    """
    if types:
        for value in types:
            if value not in set(RelationType):
                raise SearchError(f"unknown relation type: {value}")

    allowed_status = statuses or [s.value for s in MemoryStatus]
    seen = {mid}
    frontier = [mid]
    out: list[RelatedHit] = []

    for hop in range(1, max(depth, 1) + 1):
        if not frontier or len(out) >= limit:
            break
        next_frontier: list[str] = []
        placeholders = ", ".join("?" * len(frontier))
        status_ph = ", ".join("?" * len(allowed_status))

        queries: list[tuple[str, str]] = []
        if direction in ("out", "both"):
            queries.append(
                (
                    "out",
                    "SELECT r.src_id AS anchor, r.dst_id AS other, r.type, r.note, r.implied, "
                    "m.title, m.type AS mtype, m.status "
                    "FROM relations r JOIN memories m ON m.id = r.dst_id "
                    f"WHERE r.src_id IN ({placeholders}) AND m.status IN ({status_ph})",
                )
            )
        if direction in ("in", "both"):
            queries.append(
                (
                    "in",
                    "SELECT r.dst_id AS anchor, r.src_id AS other, r.type, r.note, r.implied, "
                    "m.title, m.type AS mtype, m.status "
                    "FROM relations r JOIN memories m ON m.id = r.src_id "
                    f"WHERE r.dst_id IN ({placeholders}) AND m.status IN ({status_ph})",
                )
            )

        for label, sql in queries:
            if types:
                sql += f" AND r.type IN ({', '.join('?' * len(types))})"
            if not include_implied:
                sql += " AND r.implied = 0"
            bound = [*frontier, *allowed_status, *(types or [])]
            for row in conn.execute(sql, bound):
                other = row["other"]
                if other in seen:
                    continue
                seen.add(other)
                next_frontier.append(other)
                out.append(
                    RelatedHit(
                        memory_id=other,
                        title=row["title"],
                        type=row["mtype"],
                        status=row["status"],
                        relation=row["type"],
                        direction=label,  # type: ignore[arg-type]
                        hop=hop,
                        implied=bool(row["implied"]),
                        note=row["note"],
                    )
                )
                if len(out) >= limit:
                    break
        frontier = next_frontier

    out.sort(key=lambda r: (r.hop, -EDGE_WEIGHT.get(r.relation, 0.3), r.memory_id))
    return out[:limit]


def dangling_relations(conn: sqlite3.Connection, mid: str) -> list[dict[str, Any]]:
    """Edges from this memory whose target no longer exists."""
    return [
        {"target": row["dst_id"], "type": row["type"]}
        for row in conn.execute(
            "SELECT dst_id, type FROM relations WHERE src_id = ? AND implied = 0 "
            "AND dst_id NOT IN (SELECT id FROM memories)",
            (mid,),
        )
    ]


def profile_memories(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    sensitivity_max: str = "private",
) -> list[Hit]:
    """Always-relevant memories, bypassing keyword matching entirely.

    This is what keeps identity and standing preferences available when a
    query shares no vocabulary with them.
    """
    filters = SearchFilters(
        profile=True,
        limit=limit,
        sensitivity_max=sensitivity_max,
        order="score",
    )
    return search(conn, None, filters)
