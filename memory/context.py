"""Assemble a token-budgeted context pack for a prompt.

Three stages, in priority order: always-on profile memories, keyword hits,
then one hop of relation expansion. The result is rendered Markdown with an
id comment on every block, so the consuming model can cite a memory and then
fetch its untruncated text by id -- the pack is a navigable index rather than
a dead end.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal

from .search import (
    EDGE_WEIGHT,
    Hit,
    SearchFilters,
    get_related,
    profile_memories,
    search,
)

#: Leaves headroom for the estimator being imperfect. Under-filling by a few
#: percent is cheap; overflowing a context window is not.
SAFETY_FACTOR = 0.92

#: Below this, a block is noise rather than information -- a three-word
#: fragment of a memory is worse than its absence.
MIN_BLOCK_TOKENS = 40

#: Fraction of a truncated body to keep before marking it.
TRUNCATE_TO = 0.6

PROFILE_TYPE_ORDER = ("preference", "fact", "procedure", "decision", "event")

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CJK_RE = re.compile(r"[㐀-鿿가-힯぀-ヿ]")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

Stage = Literal["profile", "fts", "relation", "recent"]


def _default_token_counter(text: str) -> int:
    """Estimate tokens without a tokenizer dependency.

    ``len(text) / 4`` underestimates Markdown and code by 15-30%, because
    punctuation and indentation tokenize densely -- exactly the failure that
    overflows a context window. Counting words and non-word characters
    separately lands within roughly 8% on mixed content and is biased high,
    which is the safe direction.
    """
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    word_chars = sum(len(w) for w in words)
    other_chars = len(text) - word_chars
    cjk = len(_CJK_RE.findall(text))
    return max(1, ceil(len(words) * 1.33 + other_chars * 0.5 + cjk * 0.7))


_token_counter: Callable[[str], int] = _default_token_counter


def set_token_counter(counter: Callable[[str], int] | None) -> None:
    """Install an exact tokenizer, or restore the built-in estimate.

    Keeps the engine model-agnostic: a caller that knows its tokenizer can
    supply it, and nothing here depends on one being available.
    """
    global _token_counter
    _token_counter = counter or _default_token_counter


def estimate_tokens(text: str) -> int:
    return _token_counter(text)


@dataclass(slots=True)
class ContextBlock:
    memory_id: str
    type: str
    title: str
    text: str
    tokens: int
    score: float
    stage: Stage
    reason: str
    truncated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "type": self.type,
            "title": self.title,
            "text": self.text,
            "tokens": self.tokens,
            "score": round(self.score, 4),
            "stage": self.stage,
            "reason": self.reason,
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class ContextResult:
    query: str
    token_budget: int
    tokens_used: int
    blocks: list[ContextBlock] = field(default_factory=list)
    markdown: str = ""
    dropped: int = 0
    truncated: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "dropped": self.dropped,
            "truncated": self.truncated,
            "blocks": [b.to_json() for b in self.blocks],
            "markdown": self.markdown,
        }


def _block_text(title: str, body: str) -> str:
    return f"{title}\n\n{body}".strip() if body.strip() else title


#: Tokens the renderer adds per block beyond its text: the "### [type] title"
#: heading, the id/score comment, and the blank lines around them. Budgeting
#: on text alone would overflow once a pack held several blocks.
def _render_overhead(block: ContextBlock) -> int:
    scaffold = (
        f"### [{block.type}] {block.title}\n"
        f"<!-- id:{block.memory_id} score:{block.score:.2f} {block.reason} -->\n\n\n"
    )
    return estimate_tokens(scaffold) - estimate_tokens(block.title)


def _rendered_cost(block: ContextBlock) -> int:
    return block.tokens + _render_overhead(block)


def _truncate_body(body: str) -> str:
    """Cut a body at a paragraph boundary, marking the cut.

    Cutting mid-sentence produces text a model may quote as if complete, so
    the boundary and the marker both matter.
    """
    target = max(int(len(body) * TRUNCATE_TO), 1)
    paragraphs = _PARAGRAPH_RE.split(body)
    kept: list[str] = []
    length = 0
    for para in paragraphs:
        if kept and length + len(para) > target:
            break
        kept.append(para)
        length += len(para)
    if not kept:
        kept = [body[:target]]
    return "\n\n".join(kept).rstrip() + "\n\n…[truncated]"


def _make_block(hit: Hit, body: str, *, score: float, stage: Stage, reason: str) -> ContextBlock:
    text = _block_text(hit.title, body)
    return ContextBlock(
        memory_id=hit.memory_id,
        type=hit.type,
        title=hit.title,
        text=text,
        tokens=estimate_tokens(text),
        score=score,
        stage=stage,
        reason=reason,
    )


#: What :func:`get_context` accepts to repair rows that lag their files. It is
#: handed ``(id, path, mtime_ns)`` triples -- paths exactly as the index stores
#: them, relative to ``data/`` -- and returns fresh bodies for whichever it
#: could re-read. Resolving those paths needs a root, which is why this is a
#: callback from :meth:`memory.api.MemoryAPI.get_context` rather than a stat
#: call here: this module holds a connection and nothing else.
BodyRefresher = Callable[[list[tuple[str, str, int]]], dict[str, str]]


def _bodies(
    conn: sqlite3.Connection,
    ids: list[str],
    *,
    refresh: BodyRefresher | None = None,
) -> dict[str, str]:
    """Fetch block bodies from the index, repairing any that lag their files.

    Bodies come from the index because that is what makes assembly one query
    rather than one file read per candidate. The cost is that an edit made
    outside the API -- a hand edit, a ``git pull``, a sync tool -- leaves the
    row stale with nothing on the read path to notice, and a context pack is
    where a stale body does the most damage: the model reads it as current and
    states it confidently.

    Every row carries the ``mtime_ns`` it was indexed at, so ``refresh`` is
    given the candidates whose stored mtime is worth re-checking and returns
    the bodies it could re-read. mtime is a filter, not proof -- ``verify``
    compares content hashes, which is the authority -- but it is one ``stat``
    per block and it catches every realistic outside edit.
    """
    if not ids:
        return {}
    placeholders = ", ".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, body, path, mtime_ns FROM memories WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    bodies = {row["id"]: row["body"] for row in rows}

    if refresh is None:
        return bodies

    # A stored 0 means the mtime was never recorded (the column's default, or
    # an unreadable file at index time). There is nothing to compare against,
    # so the row is served as-is and `verify` is left to report it; checking
    # anyway would re-read those files on every single call.
    candidates = [(row["id"], row["path"], row["mtime_ns"]) for row in rows if row["mtime_ns"]]
    if candidates:
        bodies.update(refresh(candidates))
    return bodies


def get_context(
    conn: sqlite3.Connection,
    query: str = "",
    *,
    token_budget: int = 2000,
    filters: SearchFilters | None = None,
    profile_share: float = 0.25,
    relation_share: float = 0.20,
    expand_depth: int = 1,
    include_profile: bool = True,
    seed_count: int = 5,
    candidate_limit: int = 60,
    refresh: BodyRefresher | None = None,
) -> ContextResult:
    """Build a context pack for ``query`` within ``token_budget``.

    Unused profile budget rolls over into the keyword stage, never the
    reverse: identity is worth including but should not crowd out the answer
    to the actual question.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    base_filters = filters or SearchFilters()
    header_tokens = estimate_tokens("## Profile\n## Relevant memories\n## Related\n")
    budget = max(int(token_budget * SAFETY_FACTOR) - header_tokens, 1)

    chosen: dict[str, ContextBlock] = {}
    spent = 0
    dropped = 0

    # ---- stage 1: profile memories, which bypass keyword matching entirely
    profile_cap = int(budget * profile_share) if include_profile else 0
    if profile_cap:
        profile_hits = profile_memories(
            conn, limit=20, sensitivity_max=base_filters.sensitivity_max
        )
        bodies = _bodies(conn, [h.memory_id for h in profile_hits], refresh=refresh)
        for hit in profile_hits:
            block = _make_block(
                hit,
                bodies.get(hit.memory_id, ""),
                score=hit.score,
                stage="profile",
                reason="profile",
            )
            cost = _rendered_cost(block)
            if spent + cost > profile_cap:
                # At a small budget the share may not fit even one block, but
                # dropping identity entirely is the wrong failure: include the
                # top profile memory regardless, truncated if need be.
                if chosen or cost > budget:
                    continue
                _shorten_to(block, min(profile_cap, budget))
                cost = _rendered_cost(block)
                if spent + cost > budget:
                    continue
            chosen[hit.memory_id] = block
            spent += cost

    # ---- stage 2: keyword hits (or most-recent on a cold start)
    relation_reserve = int(budget * relation_share) if expand_depth else 0
    fts_ceiling = budget - relation_reserve

    hit_filters = _with(base_filters, limit=candidate_limit, exclude_ids=list(chosen))
    if query.strip():
        hits = search(conn, query, hit_filters)
        stage: Stage = "fts"
    else:
        # Cold start: no query means "what should I know", so fall back to
        # the most recently touched memories.
        hits = search(conn, None, _with(hit_filters, order="updated"))
        stage = "recent"

    hits = _dedupe_to_tips(conn, hits, chosen)
    bodies = _bodies(conn, [h.memory_id for h in hits], refresh=refresh)
    seeds: list[tuple[str, float]] = []
    for hit in hits:
        block = _make_block(
            hit,
            bodies.get(hit.memory_id, ""),
            score=hit.score,
            stage=stage,
            reason=f"{stage}:{hit.score:.2f}",
        )
        seeds.append((hit.memory_id, hit.score))
        cost = _rendered_cost(block)
        if spent + cost > fts_ceiling:
            dropped += 1
            continue
        chosen[hit.memory_id] = block
        spent += cost

    # ---- stage 3: one hop out from the strongest seeds
    if expand_depth:
        for seed_id, parent_score in seeds[:seed_count]:
            for rel in get_related(
                conn,
                seed_id,
                depth=expand_depth,
                direction="both",
                statuses=[s for s in base_filters.effective_statuses()],
                limit=10,
            ):
                if rel.memory_id in chosen:
                    continue
                weight = EDGE_WEIGHT.get(rel.relation, 0.3)
                score = 0.6 * parent_score * weight / max(rel.hop, 1)
                body = _bodies(conn, [rel.memory_id], refresh=refresh).get(rel.memory_id, "")
                text = _block_text(rel.title, body)
                block = ContextBlock(
                    memory_id=rel.memory_id,
                    type=rel.type,
                    title=rel.title,
                    text=text,
                    tokens=estimate_tokens(text),
                    score=score,
                    stage="relation",
                    reason=f"relation:{rel.relation}:{rel.hop}hop",
                )
                cost = _rendered_cost(block)
                if spent + cost > budget:
                    dropped += 1
                    continue
                chosen[rel.memory_id] = block
                spent += cost

    blocks = _fit(list(chosen.values()), budget)
    fitted_ids = {b.memory_id for b in blocks}
    dropped += len(chosen) - len(fitted_ids)

    ordered = _order(blocks)
    markdown = render_markdown(ordered)
    return ContextResult(
        query=query,
        token_budget=token_budget,
        tokens_used=estimate_tokens(markdown),
        blocks=ordered,
        markdown=markdown,
        dropped=dropped,
        truncated=[b.memory_id for b in ordered if b.truncated],
    )


def _with(filters: SearchFilters, **changes: Any) -> SearchFilters:
    from dataclasses import replace

    return replace(filters, **changes)


def _dedupe_to_tips(
    conn: sqlite3.Connection, hits: list[Hit], chosen: dict[str, ContextBlock]
) -> list[Hit]:
    """Collapse hits that point at the same live memory.

    A superseded memory and its replacement would otherwise both appear,
    presenting the model with a stale claim beside its correction.
    """
    out: list[Hit] = []
    seen: set[str] = set(chosen)
    for hit in hits:
        row = conn.execute(
            "SELECT superseded_by FROM memories WHERE id = ?", (hit.memory_id,)
        ).fetchone()
        mid = hit.memory_id
        guard = 0
        while row and row["superseded_by"] and guard < 16:
            mid = row["superseded_by"]
            row = conn.execute("SELECT superseded_by FROM memories WHERE id = ?", (mid,)).fetchone()
            guard += 1
        if mid in seen:
            continue
        seen.add(mid)
        out.append(hit)
    return out


def _shorten_to(block: ContextBlock, allowance: int) -> None:
    """Trim a block's body until its rendered cost fits ``allowance``."""
    for _ in range(8):
        if _rendered_cost(block) <= allowance:
            return
        body = block.text.split("\n\n", 1)
        if len(body) < 2 or not body[1].strip():
            return
        shortened = _block_text(block.title, _truncate_body(body[1]))
        new_tokens = estimate_tokens(shortened)
        if new_tokens >= block.tokens:
            return
        block.text = shortened
        block.tokens = new_tokens
        block.truncated = True


def _fit(blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
    """Bring the pack under budget by truncating, then dropping.

    Accounts for each block's rendered cost, not just its text: the heading
    and id comment are tokens too, and ignoring them overflows the budget
    once a pack holds more than a couple of blocks.
    """
    total = sum(_rendered_cost(b) for b in blocks)
    if total <= budget:
        return blocks

    working = sorted(blocks, key=lambda b: (-b.score, b.memory_id))

    for block in reversed(working):
        if total <= budget:
            break
        body = block.text.split("\n\n", 1)
        if len(body) < 2:
            continue
        shortened = _block_text(block.title, _truncate_body(body[1]))
        new_tokens = estimate_tokens(shortened)
        if new_tokens < MIN_BLOCK_TOKENS or new_tokens >= block.tokens:
            continue
        total -= block.tokens - new_tokens
        block.text = shortened
        block.tokens = new_tokens
        block.truncated = True

    while total > budget and working:
        total -= _rendered_cost(working.pop())

    return working


def _order(blocks: list[ContextBlock]) -> list[ContextBlock]:
    """Profile first by type, then everything else by score."""

    def profile_key(block: ContextBlock) -> tuple[int, float, str]:
        try:
            rank = PROFILE_TYPE_ORDER.index(block.type)
        except ValueError:
            rank = len(PROFILE_TYPE_ORDER)
        return (rank, -block.score, block.memory_id)

    profile = sorted((b for b in blocks if b.stage == "profile"), key=profile_key)
    rest = sorted(
        (b for b in blocks if b.stage != "profile"),
        key=lambda b: (-b.score, b.memory_id),
    )
    return [*profile, *rest]


def render_markdown(blocks: list[ContextBlock]) -> str:
    """Render the pack, with an id comment per block for follow-up lookups."""
    if not blocks:
        return ""

    sections: list[tuple[str, list[ContextBlock]]] = [
        ("Profile", [b for b in blocks if b.stage == "profile"]),
        ("Relevant memories", [b for b in blocks if b.stage in ("fts", "recent")]),
        ("Related", [b for b in blocks if b.stage == "relation"]),
    ]

    lines: list[str] = []
    for heading, group in sections:
        if not group:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for block in group:
            lines.append(f"### [{block.type}] {block.title}")
            lines.append(f"<!-- id:{block.memory_id} score:{block.score:.2f} {block.reason} -->")
            body = block.text.split("\n\n", 1)
            if len(body) > 1 and body[1].strip():
                lines.append("")
                lines.append(body[1].strip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
