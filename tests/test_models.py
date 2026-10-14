"""Frontmatter codec, validation, and path math."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from memory.models import (
    Asset,
    Memory,
    MemoryStatus,
    MemoryType,
    Relation,
    RelationType,
    Sensitivity,
    SourceKind,
    SourceRef,
    Tombstone,
    ValidationError,
    clean_tags,
    clean_title,
    content_hash,
    fmt_ts,
    new_id,
    parse_memory_markdown,
    parse_ts,
    render_memory_markdown,
    tombstone_for,
    validate,
)
from memory.paths import Paths, plural

TS = datetime(2026, 10, 15, 4, 12, 33, tzinfo=UTC)


def full_memory() -> Memory:
    """A memory exercising every field, including unknown-key passthrough."""
    return Memory(
        id="9b1f2c7e-4a30-4b19-9b7c-1d2e3f4a5b6c",
        type=MemoryType.PREFERENCE,
        status=MemoryStatus.ACTIVE,
        title="Prefers uv over pip for Python projects",
        body="Uses uv everywhere.\n\nSecond paragraph with *markdown*.",
        created_at=TS,
        updated_at=TS,
        occurred_at=TS,
        valid_from=TS,
        valid_until=None,
        review_after=TS,
        confidence=0.9,
        salience=0.6,
        profile=True,
        sensitivity=Sensitivity.PRIVATE,
        tags=["python", "tooling"],
        entities=["pip", "uv"],
        assets=["3f2e1d0c-4b5a-6978-8a9b-0c1d2e3f4a5b"],
        superseded_by=None,
        supersedes=["11111111-2222-3333-4444-555555555555"],
        sources=[
            SourceRef(
                kind=SourceKind.CONVERSATION,
                ref="conversations/2026-10-15-setup.md",
                locator="L120-L142",
                captured_at=TS,
                confidence=0.9,
            )
        ],
        relations=[
            Relation(
                type=RelationType.RELATES_TO,
                target="22222222-3333-4444-5555-666666666666",
                note="same toolchain decision",
            )
        ],
        extra={"phase3_hint": {"nested": [1, 2, 3]}},
    )


class TestRoundTrip:
    def test_full_memory_is_byte_identical(self) -> None:
        original = full_memory()
        once = render_memory_markdown(original)
        parsed = parse_memory_markdown(once)
        twice = render_memory_markdown(parsed)
        assert once == twice

    def test_every_field_survives(self) -> None:
        original = full_memory()
        parsed = parse_memory_markdown(render_memory_markdown(original))
        for attr in (
            "id",
            "type",
            "status",
            "title",
            "body",
            "created_at",
            "updated_at",
            "occurred_at",
            "valid_from",
            "valid_until",
            "review_after",
            "confidence",
            "salience",
            "profile",
            "sensitivity",
            "tags",
            "entities",
            "assets",
            "superseded_by",
            "supersedes",
            "schema_version",
        ):
            assert getattr(parsed, attr) == getattr(original, attr), attr
        assert parsed.sources == original.sources
        assert parsed.relations == original.relations

    def test_unknown_keys_round_trip(self) -> None:
        """The forward-compat seam: Phase 3 fields must survive a Phase 1 writer."""
        parsed = parse_memory_markdown(render_memory_markdown(full_memory()))
        assert parsed.extra == {"phase3_hint": {"nested": [1, 2, 3]}}
        assert "phase3_hint" in render_memory_markdown(parsed)

    def test_minimal_memory(self) -> None:
        m = Memory(type=MemoryType.FACT, title="Minimal")
        parsed = parse_memory_markdown(render_memory_markdown(m))
        assert parsed.body == ""
        assert parsed.tags == []
        assert parsed.sources == []


class TestTitleHandling:
    def test_frontmatter_title_beats_divergent_h1(self) -> None:
        text = (
            "---\n"
            f"id: {new_id()}\n"
            "schema_version: 1\n"
            "type: fact\n"
            "status: active\n"
            "title: Authoritative title\n"
            "created_at: 2026-10-15T00:00:00Z\n"
            "updated_at: 2026-10-15T00:00:00Z\n"
            "---\n\n"
            "# Stale heading\n\n"
            "Body text.\n"
        )
        parsed = parse_memory_markdown(text)
        assert parsed.title == "Authoritative title"
        assert parsed.body == "Body text."
        # The H1 is regenerated from frontmatter on write.
        assert "# Authoritative title" in render_memory_markdown(parsed)
        assert "Stale heading" not in render_memory_markdown(parsed)

    def test_h1_used_when_frontmatter_title_absent(self) -> None:
        text = (
            "---\n"
            f"id: {new_id()}\n"
            "type: fact\n"
            "created_at: 2026-10-15T00:00:00Z\n"
            "updated_at: 2026-10-15T00:00:00Z\n"
            "---\n\n"
            "# Heading is the title\n\n"
            "Body.\n"
        )
        assert parse_memory_markdown(text).title == "Heading is the title"

    @pytest.mark.parametrize(
        "title",
        [
            "Title with --- triple dashes",
            "Title with: a colon",
            "Title with 🎉 emoji and ünïcode",
            "Title with 'quotes' and \"doubles\"",
            "Title with #hash and `backticks`",
            "Title with [brackets](link)",
        ],
    )
    def test_awkward_titles_round_trip(self, title: str) -> None:
        m = Memory(type=MemoryType.FACT, title=title)
        assert parse_memory_markdown(render_memory_markdown(m)).title == title

    def test_body_containing_frontmatter_delimiter(self) -> None:
        m = Memory(
            type=MemoryType.FACT,
            title="Has a delimiter in the body",
            body="Before\n\n---\n\nAfter",
        )
        assert parse_memory_markdown(render_memory_markdown(m)).body == m.body

    def test_body_with_its_own_h1(self) -> None:
        """Only the leading H1 is consumed; later headings belong to the body."""
        m = Memory(type=MemoryType.FACT, title="T", body="Intro\n\n# Section\n\nMore")
        assert parse_memory_markdown(render_memory_markdown(m)).body == m.body

    def test_large_body(self) -> None:
        body = "\n\n".join(f"Paragraph {i} with some words." for i in range(2000))
        m = Memory(type=MemoryType.FACT, title="Big", body=body)
        assert parse_memory_markdown(render_memory_markdown(m)).body == body

    @pytest.mark.parametrize("raw", ["Title\rwith CR", "Title\twith tab", "Title  double"])
    def test_control_whitespace_in_title_is_collapsed(self, raw: str) -> None:
        """Regression: a bare CR survived into YAML and broke round-trip stability."""
        m = Memory(type=MemoryType.FACT, title=raw)
        once = render_memory_markdown(m)
        parsed = parse_memory_markdown(once)
        assert "\r" not in parsed.title
        assert parsed.title == clean_title(raw)
        assert render_memory_markdown(parsed) == once

    def test_crlf_input_is_normalized(self) -> None:
        text = render_memory_markdown(full_memory()).replace("\n", "\r\n")
        parsed = parse_memory_markdown(text)
        assert "\r" not in parsed.body
        assert parsed.title == full_memory().title


class TestValidation:
    def test_missing_frontmatter(self) -> None:
        with pytest.raises(ValidationError, match="missing YAML frontmatter"):
            parse_memory_markdown("# Just a heading\n\nBody.\n")

    def test_unknown_type_is_fatal(self) -> None:
        text = f"---\nid: {new_id()}\ntype: nonsense\ntitle: X\n---\n\nBody\n"
        with pytest.raises(ValidationError, match="type: unknown"):
            parse_memory_markdown(text)

    def test_unknown_status_is_fatal(self) -> None:
        text = f"---\nid: {new_id()}\ntype: fact\nstatus: zombie\ntitle: X\n---\n\nB\n"
        with pytest.raises(ValidationError, match="status: unknown"):
            parse_memory_markdown(text)

    def test_non_uuid_id_is_reported(self) -> None:
        text = "---\nid: not-a-uuid\ntype: fact\ntitle: X\n---\n\nBody\n"
        with pytest.raises(ValidationError, match="not a uuid"):
            parse_memory_markdown(text)

    def test_invalid_yaml(self) -> None:
        with pytest.raises(ValidationError, match="invalid YAML"):
            parse_memory_markdown("---\n: : :\n  bad\n---\n\nBody\n")

    def test_empty_title_rejected(self) -> None:
        assert any("title" in p for p in validate(Memory(type=MemoryType.FACT, title="  ")))

    def test_long_title_rejected(self) -> None:
        assert any("exceeds" in p for p in validate(Memory(type=MemoryType.FACT, title="x" * 201)))

    def test_out_of_range_confidence(self) -> None:
        m = Memory(type=MemoryType.FACT, title="T", confidence=1.5)
        assert any("confidence" in p for p in validate(m))

    def test_self_reference_rejected(self) -> None:
        mid = new_id()
        m = Memory(id=mid, type=MemoryType.FACT, title="T", superseded_by=mid)
        assert any("itself" in p for p in validate(m))

        m2 = Memory(id=mid, type=MemoryType.FACT, title="T", supersedes=[mid])
        assert any("itself" in p for p in validate(m2))

        m3 = Memory(
            id=mid,
            type=MemoryType.FACT,
            title="T",
            relations=[Relation(type=RelationType.CAUSES, target=mid)],
        )
        assert any("itself" in p for p in validate(m3))

    def test_valid_until_before_valid_from(self) -> None:
        m = Memory(
            type=MemoryType.FACT,
            title="T",
            valid_from=datetime(2026, 5, 1, tzinfo=UTC),
            valid_until=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert any("precedes" in p for p in validate(m))

    def test_bad_relation_target_reported(self) -> None:
        text = (
            f"---\nid: {new_id()}\ntype: fact\ntitle: T\n"
            "relations:\n  - type: causes\n    target: nope\n---\n\nBody\n"
        )
        with pytest.raises(ValidationError, match="target must be a uuid"):
            parse_memory_markdown(text)

    def test_bad_source_kind_reported(self) -> None:
        text = (
            f"---\nid: {new_id()}\ntype: fact\ntitle: T\n"
            "sources:\n  - kind: telepathy\n    ref: x\n---\n\nBody\n"
        )
        with pytest.raises(ValidationError, match="unknown source kind"):
            parse_memory_markdown(text)

    def test_multiple_problems_all_reported(self) -> None:
        m = Memory(type=MemoryType.FACT, title="", confidence=5.0, salience=-1.0)
        assert len(validate(m)) >= 3

    def test_valid_memory_has_no_problems(self) -> None:
        assert validate(full_memory()) == []


class TestTimestamps:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2026-10-15T04:12:33Z", datetime(2026, 10, 15, 4, 12, 33, tzinfo=UTC)),
            ("2026-10-15T04:12:33+00:00", datetime(2026, 10, 15, 4, 12, 33, tzinfo=UTC)),
            ("2026-10-15", datetime(2026, 10, 15, 0, 0, tzinfo=UTC)),
            ("1792210353", datetime.fromtimestamp(1792210353, UTC)),
            (1792210353, datetime.fromtimestamp(1792210353, UTC)),
        ],
    )
    def test_accepted_forms(self, raw: object, expected: datetime) -> None:
        assert parse_ts(raw) == expected

    def test_naive_is_assumed_utc(self) -> None:
        assert parse_ts("2026-10-15T04:12:33").tzinfo == UTC

    def test_offset_is_converted_to_utc(self) -> None:
        assert parse_ts("2026-10-15T06:12:33+02:00") == datetime(
            2026, 10, 15, 4, 12, 33, tzinfo=UTC
        )

    @pytest.mark.parametrize("raw", ["last tuesday", "", "   ", "not-a-date", None, [], {}])
    def test_rejected_forms(self, raw: object) -> None:
        with pytest.raises(ValidationError):
            parse_ts(raw)

    def test_fmt_is_sortable_z_form(self) -> None:
        assert fmt_ts(TS) == "2026-10-15T04:12:33Z"
        assert fmt_ts(None) is None

    def test_fmt_sorts_lexicographically(self) -> None:
        stamps = [
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 10, 15, tzinfo=UTC),
            datetime(2025, 12, 31, tzinfo=UTC),
        ]
        rendered = [fmt_ts(s) for s in stamps]
        assert sorted(rendered) == [fmt_ts(s) for s in sorted(stamps)]


class TestNormalization:
    def test_tags_lowercased_deduped_sorted(self) -> None:
        assert clean_tags(["Zebra", "apple", "APPLE", "  mid dle  "]) == [
            "apple",
            "mid-dle",
            "zebra",
        ]

    def test_empty_tags_dropped(self) -> None:
        assert clean_tags(["", "   ", "ok"]) == ["ok"]

    def test_content_hash_ignores_case_and_whitespace(self) -> None:
        assert content_hash("Hello  World", "Body") == content_hash("hello world", "body")

    def test_content_hash_distinguishes_content(self) -> None:
        assert content_hash("A", "B") != content_hash("A", "C")

    def test_entities_keep_case_but_dedupe(self) -> None:
        parsed = parse_memory_markdown(
            render_memory_markdown(
                Memory(type=MemoryType.FACT, title="T", entities=["uv", "UV", "Pip"])
            )
        )
        assert len(parsed.entities) == 2


class TestSourceRef:
    def test_source_id_is_deterministic(self) -> None:
        a = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/x.md")
        b = SourceRef(kind=SourceKind.CONVERSATION, ref="conversations/x.md", locator="L1")
        assert a.source_id == b.source_id  # locator is not part of source identity
        assert len(a.source_id) == 32

    def test_source_id_varies_with_kind_and_ref(self) -> None:
        a = SourceRef(kind=SourceKind.CONVERSATION, ref="x")
        assert a.source_id != SourceRef(kind=SourceKind.DOCUMENT, ref="x").source_id
        assert a.source_id != SourceRef(kind=SourceKind.CONVERSATION, ref="y").source_id


class TestTombstone:
    def test_round_trip(self) -> None:
        t = tombstone_for(full_memory(), permanent=True, reason="user asked")
        back = Tombstone.from_json(t.to_json())
        assert back.memory_id == t.memory_id
        assert back.permanent is True
        assert back.reason == "user asked"
        assert back.content_hash == full_memory().content_hash

    def test_source_keys_match_memory(self) -> None:
        m = full_memory()
        assert tombstone_for(m, permanent=True).source_keys() == m.source_keys()

    def test_rejects_bad_id(self) -> None:
        with pytest.raises(ValidationError):
            Tombstone.from_json({"memory_id": "nope"})


class TestAsset:
    def test_round_trip(self) -> None:
        a = Asset(
            sha256="a" * 64,
            media_type="application/pdf",
            bytes=1234,
            original_filename="paper.pdf",
            tags=["research"],
            source=SourceRef(kind=SourceKind.DOCUMENT, ref="documents/paper.pdf"),
        )
        back = Asset.from_json(a.to_json())
        assert back.id == a.id
        assert back.sha256 == a.sha256
        assert back.bytes == 1234
        assert back.source is not None
        assert back.source.ref == "documents/paper.pdf"

    def test_rejects_short_sha(self) -> None:
        with pytest.raises(ValidationError, match="sha256"):
            Asset.from_json({"id": new_id(), "sha256": "abc", "bytes": 1})

    def test_rejects_bad_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            Asset.from_json({"id": new_id(), "sha256": "a" * 64, "bytes": 1, "status": "haunted"})


class TestPaths:
    def test_plural_mapping(self) -> None:
        assert plural("fact") == "facts"
        assert plural("preference") == "preferences"
        with pytest.raises(ValueError, match="unknown memory type"):
            plural("nonsense")

    def test_blob_sharding(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        sha = "abcdef" + "0" * 58
        assert p.asset_blob(sha) == p.blobs / "ab" / "cd" / sha

    def test_asset_meta_sharding(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        aid = "7f3a9c21-1111-2222-3333-444444444444"
        assert p.asset_meta(aid) == p.asset_meta_dir / "7f" / f"{aid}.json"

    def test_memory_file_uses_plural_dir(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        mid = new_id()
        assert p.memory_file("preference", mid) == p.memories / "preferences" / f"{mid}.md"

    def test_rel_and_abs_are_inverse(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        p.ensure_tree()
        target = p.memory_file("fact", new_id())
        assert p.abs(p.rel(target)) == target

    def test_rel_is_relative_to_data(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        p.ensure_tree()
        rel = p.rel(p.memory_file("fact", "x" * 36))
        assert rel.startswith("memories/facts/")
        assert not Path(rel).is_absolute()

    def test_ensure_tree_is_idempotent(self, tmp_path: Path) -> None:
        p = Paths(tmp_path)
        p.ensure_tree()
        p.ensure_tree()
        assert p.marker.exists()
        assert all(d.is_dir() for d in p.all_dirs())

    def test_discover_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_ROOT", str(tmp_path))
        assert Paths.discover().root == tmp_path.resolve()

    def test_discover_via_marker_in_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MEMORY_ROOT", raising=False)
        root = tmp_path / "proj"
        Paths(root).ensure_tree()
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        assert Paths.discover(nested).root == root.resolve()

    def test_discover_raises_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MEMORY_ROOT", raising=False)
        from memory.paths import RootNotFound

        with pytest.raises(RootNotFound):
            Paths.discover(tmp_path)

    def test_bad_sha_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sha256"):
            Paths(tmp_path).asset_blob("ab")

    def test_unknown_derived_kind_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="derived kind"):
            Paths(tmp_path).derived_dir("hallucinations", new_id())


# ------------------------------------------------------------------ property test

_titles = st.text(min_size=1, max_size=120).filter(lambda s: s.strip() and "\n" not in s)
_bodies = st.text(max_size=400)
_tag_lists = st.lists(st.from_regex(r"[a-z][a-z0-9_-]{0,15}", fullmatch=True), max_size=5)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    title=_titles,
    body=_bodies,
    tags=_tag_lists,
    mem_type=st.sampled_from(list(MemoryType)),
    status=st.sampled_from(list(MemoryStatus)),
    confidence=st.floats(0, 1),
    profile=st.booleans(),
)
def test_round_trip_is_identity(
    title: str,
    body: str,
    tags: list[str],
    mem_type: MemoryType,
    status: MemoryStatus,
    confidence: float,
    profile: bool,
) -> None:
    """Property: render -> parse -> render is a fixed point for any content."""
    m = Memory(
        type=mem_type,
        status=status,
        title=title,
        body=body,
        tags=clean_tags(tags),
        confidence=confidence,
        profile=profile,
    )
    once = render_memory_markdown(m)
    twice = render_memory_markdown(parse_memory_markdown(once))
    assert once == twice
