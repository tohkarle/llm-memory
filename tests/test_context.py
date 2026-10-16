"""Context assembly: budgeting, staging, truncation, and rendering."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memory.api import MemoryAPI
from memory.context import (
    MIN_BLOCK_TOKENS,
    SAFETY_FACTOR,
    estimate_tokens,
    render_markdown,
    set_token_counter,
)
from memory.models import MemoryType, Relation, RelationType, Sensitivity


@pytest.fixture
def api(tmp_path: Path) -> Iterator[MemoryAPI]:
    instance = MemoryAPI.initialize(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(autouse=True)
def _reset_counter() -> Iterator[None]:
    yield
    set_token_counter(None)


def seed(api: MemoryAPI, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "type": MemoryType.FACT,
        "title": "A memory",
        "body": "Some body text.",
    }
    defaults.update(kw)
    return api.create(**defaults)


def populate(api: MemoryAPI, count: int = 20) -> list[Any]:
    return [
        seed(
            api,
            title=f"Memory {i} about python tooling",
            body=f"Body paragraph {i} about uv and packaging.\n\nSecond paragraph here.",
            tags=["python"],
        )
        for i in range(count)
    ]


class TestTokenEstimate:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_grows_with_length(self) -> None:
        short = estimate_tokens("one two three")
        longer = estimate_tokens("one two three " * 20)
        assert longer > short

    def test_biased_high_versus_char_quarter(self) -> None:
        """The whole point: len/4 under-counts markdown and code.

        Under-counting is what overflows a context window, so the estimator
        must not fall below that naive figure on punctuation-dense text.
        """
        markdown = "## Heading\n\n- `item_one` = 1\n- `item_two` = 2\n\n```py\nx = {'a': 1}\n```"
        assert estimate_tokens(markdown) >= len(markdown) / 4

    def test_handles_cjk(self) -> None:
        assert estimate_tokens("日本語のテキストです") > 0

    def test_counter_is_replaceable(self) -> None:
        set_token_counter(lambda text: len(text))
        assert estimate_tokens("abcd") == 4
        set_token_counter(None)
        assert estimate_tokens("abcd") != 4

    def test_within_reasonable_range_of_tiktoken(self) -> None:
        """Calibration check, skipped when the optional extra is absent."""
        tiktoken = pytest.importorskip("tiktoken")
        encoder = tiktoken.get_encoding("cl100k_base")
        corpora = [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "## Heading\n\n- bullet one\n- bullet two\n\nSome *emphasis* here.\n",
            "def f(x):\n    return {'a': [1, 2, 3], 'b': x * 2}\n",
            "https://example.com/path?query=value&other=thing",
            "Mixed: café, naïve, 日本語, emoji 🎉 and numbers 12345.",
        ]
        for text in corpora:
            exact = len(encoder.encode(text))
            estimate = estimate_tokens(text)
            assert 0.6 * exact <= estimate <= 1.8 * exact, text[:40]


class TestBudget:
    @pytest.mark.parametrize("budget", [50, 200, 500, 2000, 8000])
    def test_never_exceeds_budget(self, api: MemoryAPI, budget: int) -> None:
        """The hard guarantee: the pack always fits."""
        populate(api, 25)
        result = api.get_context("python tooling", token_budget=budget)
        assert result.tokens_used <= budget

    @pytest.mark.parametrize("budget", [50, 200, 500, 2000])
    def test_block_tokens_also_fit(self, api: MemoryAPI, budget: int) -> None:
        populate(api, 25)
        result = api.get_context("python tooling", token_budget=budget)
        assert sum(b.tokens for b in result.blocks) <= budget

    def test_rejects_non_positive_budget(self, api: MemoryAPI) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            api.get_context("x", token_budget=0)

    def test_reports_dropped_blocks(self, api: MemoryAPI) -> None:
        populate(api, 30)
        result = api.get_context("python tooling", token_budget=120)
        assert result.dropped > 0

    def test_larger_budget_includes_more(self, api: MemoryAPI) -> None:
        populate(api, 25)
        small = api.get_context("python tooling", token_budget=200)
        large = api.get_context("python tooling", token_budget=4000)
        assert len(large.blocks) > len(small.blocks)

    def test_safety_factor_leaves_headroom(self, api: MemoryAPI) -> None:
        populate(api, 30)
        result = api.get_context("python tooling", token_budget=1000)
        assert result.tokens_used <= int(1000 * SAFETY_FACTOR) + 1


class TestProfileStage:
    def test_profile_appears_even_at_a_tiny_budget(self, api: MemoryAPI) -> None:
        flagged = seed(api, title="Core identity fact", body="Short.", profile=True)
        populate(api, 10)
        result = api.get_context("python tooling", token_budget=100)
        assert flagged.id in {b.memory_id for b in result.blocks}

    def test_profile_is_included_without_matching_the_query(self, api: MemoryAPI) -> None:
        """The paraphrase mitigation, end to end."""
        flagged = seed(
            api,
            title="Prefers uv over pip",
            body="Packaging preference.",
            profile=True,
        )
        result = api.get_context("javascript bundler choices", token_budget=1000)
        assert flagged.id in {b.memory_id for b in result.blocks}

    def test_can_be_disabled(self, api: MemoryAPI) -> None:
        flagged = seed(api, title="Identity", body="x", profile=True)
        result = api.get_context("", token_budget=1000, include_profile=False)
        assert all(b.stage != "profile" for b in result.blocks)
        assert flagged.id not in {b.memory_id for b in result.blocks if b.stage == "profile"}

    def test_unused_profile_budget_rolls_over(self, api: MemoryAPI) -> None:
        """Identity should not waste budget it does not need."""
        populate(api, 20)
        with_profile = api.get_context("python tooling", token_budget=800)
        seed(api, title="One tiny profile item", body="x", profile=True)
        still_full = api.get_context("python tooling", token_budget=800)
        # Adding a small profile memory must not halve the pack.
        assert len(still_full.blocks) >= len(with_profile.blocks) - 1

    def test_profile_share_is_capped(self, api: MemoryAPI) -> None:
        for i in range(10):
            seed(
                api,
                title=f"Profile item {i}",
                body="A fairly long body paragraph to consume budget. " * 5,
                profile=True,
            )
        populate(api, 10)
        result = api.get_context("python tooling", token_budget=1000)
        profile_tokens = sum(b.tokens for b in result.blocks if b.stage == "profile")
        assert profile_tokens <= int(1000 * SAFETY_FACTOR * 0.25) + MIN_BLOCK_TOKENS


class TestStages:
    def test_keyword_hits_are_included(self, api: MemoryAPI) -> None:
        target = seed(api, title="A distinctive keyword here", body="body")
        result = api.get_context("distinctive", token_budget=1000)
        assert target.id in {b.memory_id for b in result.blocks}

    def test_empty_query_falls_back_to_recent(self, api: MemoryAPI) -> None:
        """A cold start should still say something useful."""
        populate(api, 5)
        result = api.get_context("", token_budget=1000)
        assert result.blocks
        assert any(b.stage == "recent" for b in result.blocks)

    def test_relation_expansion_pulls_in_neighbours(self, api: MemoryAPI) -> None:
        neighbour = seed(api, title="Neighbour memory", body="unrelated words")
        anchor = seed(
            api,
            title="Anchor with distinctive term",
            body="body",
            relations=[Relation(type=RelationType.CAUSES, target=neighbour.id)],
        )
        result = api.get_context("distinctive", token_budget=2000)
        found = {b.memory_id: b.stage for b in result.blocks}
        assert found.get(anchor.id) == "fts"
        assert found.get(neighbour.id) == "relation"

    def test_contradiction_surfaces(self, api: MemoryAPI) -> None:
        """A disputed memory must reach the model, or it states stale facts."""
        dispute = seed(api, title="Actually the opposite is true", body="counter")
        claim = seed(
            api,
            title="A distinctive claim",
            body="body",
            relations=[Relation(type=RelationType.CONTRADICTS, target=dispute.id)],
        )
        result = api.get_context("distinctive", token_budget=2000)
        ids = {b.memory_id for b in result.blocks}
        assert claim.id in ids
        assert dispute.id in ids

    def test_expansion_can_be_disabled(self, api: MemoryAPI) -> None:
        neighbour = seed(api, title="Neighbour", body="x")
        seed(
            api,
            title="Anchor distinctive",
            body="b",
            relations=[Relation(type=RelationType.CAUSES, target=neighbour.id)],
        )
        result = api.get_context("distinctive", token_budget=2000, expand_depth=0)
        assert all(b.stage != "relation" for b in result.blocks)

    def test_no_duplicate_blocks(self, api: MemoryAPI) -> None:
        flagged = seed(
            api, title="Profile and keyword match", body="distinctive body", profile=True
        )
        result = api.get_context("distinctive", token_budget=2000)
        ids = [b.memory_id for b in result.blocks]
        assert len(ids) == len(set(ids))
        assert ids.count(flagged.id) == 1

    def test_superseded_alias_does_not_duplicate(self, api: MemoryAPI) -> None:
        """A stale claim must not appear beside the correction that replaced it."""
        old = seed(api, title="Original distinctive claim", body="old body")
        new = api.replace(old.id, title="Revised distinctive claim", body="new body").new
        result = api.get_context("distinctive", token_budget=2000)
        ids = {b.memory_id for b in result.blocks}
        assert new.id in ids
        assert old.id not in ids

    def test_secret_memories_are_excluded(self, api: MemoryAPI) -> None:
        secret = seed(api, title="Distinctive secret", body="x", sensitivity=Sensitivity.SECRET)
        result = api.get_context("distinctive", token_budget=1000)
        assert secret.id not in {b.memory_id for b in result.blocks}


class TestTruncation:
    def test_marks_truncated_blocks(self, api: MemoryAPI) -> None:
        seed(
            api,
            title="Long distinctive memory",
            body="\n\n".join(f"Paragraph {i} with a good deal of text." for i in range(30)),
        )
        result = api.get_context("distinctive", token_budget=200)
        if result.truncated:
            assert any(b.truncated for b in result.blocks)
            assert "…[truncated]" in "".join(b.text for b in result.blocks if b.truncated)

    def test_truncates_at_a_paragraph_boundary(self, api: MemoryAPI) -> None:
        body = "\n\n".join(f"Paragraph number {i} here." for i in range(40))
        seed(api, title="Boundary distinctive test", body=body)
        result = api.get_context("distinctive", token_budget=220)
        for block in result.blocks:
            if block.truncated:
                kept = block.text.replace("\n\n…[truncated]", "")
                assert not kept.endswith("Paragraph number")

    def test_never_keeps_a_block_below_the_floor(self, api: MemoryAPI) -> None:
        """A three-word fragment is worse than no block at all."""
        populate(api, 20)
        result = api.get_context("python tooling", token_budget=300)
        for block in result.blocks:
            if block.truncated:
                assert block.tokens >= MIN_BLOCK_TOKENS


class TestRendering:
    def test_sections_appear_in_priority_order(self, api: MemoryAPI) -> None:
        seed(api, title="Identity item", body="x", profile=True)
        neighbour = seed(api, title="Neighbour", body="y")
        seed(
            api,
            title="Anchor distinctive",
            body="z",
            relations=[Relation(type=RelationType.CAUSES, target=neighbour.id)],
        )
        markdown = api.get_context("distinctive", token_budget=3000).markdown
        assert markdown.index("## Profile") < markdown.index("## Relevant memories")
        assert markdown.index("## Relevant memories") < markdown.index("## Related")

    def test_each_block_carries_an_id_comment(self, api: MemoryAPI) -> None:
        """This is what makes the pack navigable: cite, then fetch by id."""
        target = seed(api, title="Distinctive memory", body="body")
        markdown = api.get_context("distinctive", token_budget=1000).markdown
        assert f"<!-- id:{target.id}" in markdown

    def test_id_comments_are_parseable(self, api: MemoryAPI) -> None:
        populate(api, 5)
        markdown = api.get_context("python", token_budget=2000).markdown
        found = re.findall(r"<!-- id:([0-9a-f-]{36}) score:([0-9.]+) ([^>]+) -->", markdown)
        assert found
        for mid, _score, _reason in found:
            assert api.get(mid) is not None

    def test_type_is_shown(self, api: MemoryAPI) -> None:
        seed(api, type=MemoryType.PREFERENCE, title="Distinctive preference", body="x")
        markdown = api.get_context("distinctive", token_budget=1000).markdown
        assert "### [preference]" in markdown

    def test_empty_result_renders_empty(self, api: MemoryAPI) -> None:
        result = api.get_context("nothingmatchesthis", token_budget=1000)
        assert result.blocks == []
        assert result.markdown == ""
        assert render_markdown([]) == ""

    def test_markdown_matches_reported_usage(self, api: MemoryAPI) -> None:
        populate(api, 10)
        result = api.get_context("python", token_budget=1500)
        assert result.tokens_used == estimate_tokens(result.markdown)


class TestDeterminism:
    def test_identical_calls_produce_identical_packs(self, api: MemoryAPI) -> None:
        populate(api, 15)
        first = api.get_context("python tooling", token_budget=900)
        second = api.get_context("python tooling", token_budget=900)
        assert first.markdown == second.markdown
        assert [b.memory_id for b in first.blocks] == [b.memory_id for b in second.blocks]

    def test_order_is_stable_across_many_runs(self, api: MemoryAPI) -> None:
        populate(api, 12)
        packs = {api.get_context("uv packaging", token_budget=700).markdown for _ in range(5)}
        assert len(packs) == 1


class TestJson:
    def test_result_serializes(self, api: MemoryAPI) -> None:
        import json

        populate(api, 5)
        payload = api.get_context("python", token_budget=1000).to_json()
        json.dumps(payload)
        assert payload["token_budget"] == 1000
        assert "blocks" in payload and "markdown" in payload
        assert all("memory_id" in b for b in payload["blocks"])


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(budget=st.integers(min_value=30, max_value=6000))
def test_budget_is_never_exceeded_property(tmp_path_factory: Any, budget: int) -> None:
    """Property: for any budget, the assembled pack fits inside it."""
    root = tmp_path_factory.mktemp("ctx")
    api = MemoryAPI.initialize(root)
    try:
        for i in range(12):
            api.create(
                type=MemoryType.FACT,
                title=f"Memory {i} about tooling",
                body=f"Paragraph {i}.\n\nAnother paragraph with more words here.",
                profile=(i % 4 == 0),
            )
        result = api.get_context("tooling", token_budget=budget)
        assert result.tokens_used <= budget
        assert sum(b.tokens for b in result.blocks) <= budget
    finally:
        api.close()


class TestStaleBodyRefresh:
    """An edit made outside the API must not surface as a stale context block.

    Bodies are served from the index, so these tests go behind it: they write
    the file directly, exactly as a hand edit, a `git pull` or a sync tool
    would, and assert the read path notices.
    """

    @staticmethod
    def _rewrite_file(api: MemoryAPI, mid: str, body: str) -> Path:
        """Replace a memory's body on disk, leaving the index untouched."""
        path = api.store.locate(mid)
        assert path is not None
        text = path.read_text(encoding="utf-8")
        head, sep, _ = text.partition("\n---\n")
        assert sep, "expected frontmatter"
        path.write_text(f"{head}{sep}\n{body}\n", encoding="utf-8")
        # Guarantee a distinguishable mtime even on a coarse clock.
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return path

    def test_outside_edit_reaches_the_context_pack(self, api: MemoryAPI) -> None:
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        self._rewrite_file(api, memory.id, "Rewritten body about uv.")

        result = api.get_context("packaging tool")

        assert "Rewritten body about uv." in result.markdown
        assert "Original body about pip." not in result.markdown

    def test_the_repair_is_persisted_to_the_index(self, api: MemoryAPI) -> None:
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        self._rewrite_file(api, memory.id, "Rewritten body about uv.")

        api.get_context("packaging tool")

        row = api.conn.execute("SELECT body FROM memories WHERE id = ?", (memory.id,)).fetchone()
        assert row["body"] == "Rewritten body about uv."

    def test_search_sees_the_refreshed_text_afterwards(self, api: MemoryAPI) -> None:
        """The reindex rebuilds FTS too, so a term only in the new body matches.

        No trailing punctuation on the probe term: the tokenizer's ``tokenchars``
        include ``.``, so "sqlalchemy." and "sqlalchemy" are different tokens.
        """
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        self._rewrite_file(api, memory.id, "Rewritten body mentioning sqlalchemy everywhere")

        assert api.search("sqlalchemy") == []
        api.get_context("packaging tool")

        assert [h.memory_id for h in api.search("sqlalchemy")] == [memory.id]

    def test_untouched_memories_are_not_rewritten(self, api: MemoryAPI) -> None:
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        before = api.conn.execute(
            "SELECT mtime_ns, indexed_at FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()

        api.get_context("packaging tool")

        after = api.conn.execute(
            "SELECT mtime_ns, indexed_at FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()
        assert after["mtime_ns"] == before["mtime_ns"]
        assert after["indexed_at"] == before["indexed_at"]

    def test_a_deleted_file_does_not_break_the_read(self, api: MemoryAPI) -> None:
        """A missing file is a verify-level problem, not a reason to fail a read."""
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        path = api.store.locate(memory.id)
        assert path is not None
        path.unlink()

        result = api.get_context("packaging tool")

        assert "Original body about pip." in result.markdown

    def test_an_unparseable_file_falls_back_to_the_indexed_body(self, api: MemoryAPI) -> None:
        memory = seed(api, title="Packaging tool", body="Original body about pip.")
        path = api.store.locate(memory.id)
        assert path is not None
        path.write_text("this is not a memory file", encoding="utf-8")

        result = api.get_context("packaging tool")

        assert "Original body about pip." in result.markdown

    def test_a_read_only_api_still_serves_fresh_text(self, tmp_path: Path) -> None:
        writable = MemoryAPI.initialize(tmp_path)
        try:
            memory = seed(writable, title="Packaging tool", body="Original body about pip.")
            self._rewrite_file(writable, memory.id, "Rewritten body about uv.")
        finally:
            writable.close()

        with MemoryAPI(tmp_path, read_only=True) as reader:
            result = reader.get_context("packaging tool")
            assert "Rewritten body about uv." in result.markdown
