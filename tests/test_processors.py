"""Processor registry, text extraction, and the derivation cache key."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memory.processors import (
    REGISTRY,
    ProcessContext,
    Processor,
    ProcessorError,
    ProcessResult,
    TextProcessor,
    cache_key,
    for_media,
    register,
    segments_from_text,
    unregister,
)
from memory.processors.base import Cost, _specificity


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text(
        "# Notes\n\nFirst paragraph about uv.\n\nSecond paragraph about pip.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Restore the default registry after a test adds to it."""
    snapshot = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)


class StubProcessor:
    """An 'expensive' processor, for exercising the scheduler hint and cache."""

    name: str = "stub"
    version: str = "1.0"
    cost: Cost = "expensive"
    media_types: tuple[str, ...] = ("audio/wav",)
    extensions: tuple[str, ...] = (".wav",)
    calls: int = 0

    def can_handle(self, media_type: str, path: Path) -> bool:
        return media_type in self.media_types or path.suffix.lower() in self.extensions

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        type(self).calls += 1
        return ProcessResult(text="transcribed words", meta={"processor": self.name})


class TestRegistry:
    def test_text_processor_is_registered_by_default(self) -> None:
        assert "text" in REGISTRY

    def test_registers_and_unregisters(self, clean_registry: None) -> None:
        register(StubProcessor())
        assert "stub" in REGISTRY
        unregister("stub")
        assert "stub" not in REGISTRY

    def test_duplicate_registration_raises(self, clean_registry: None) -> None:
        register(StubProcessor())
        with pytest.raises(ProcessorError, match="already registered"):
            register(StubProcessor())

    def test_replace_allows_override(self, clean_registry: None) -> None:
        register(StubProcessor())
        register(StubProcessor(), replace=True)
        assert "stub" in REGISTRY

    def test_protocol_is_satisfied(self) -> None:
        assert isinstance(TextProcessor(), Processor)
        assert isinstance(StubProcessor(), Processor)

    def test_unregister_is_safe_when_absent(self) -> None:
        unregister("never-existed")


class TestResolution:
    def test_resolves_by_media_type(self, tmp_path: Path) -> None:
        found = for_media("text/plain", tmp_path / "a.txt")
        assert found is not None and found.name == "text"

    def test_resolves_by_extension_when_type_is_generic(self, tmp_path: Path) -> None:
        found = for_media("application/octet-stream", tmp_path / "a.md")
        assert found is not None and found.name == "text"

    def test_unknown_type_resolves_to_nothing(self, tmp_path: Path) -> None:
        """A type no processor claims must resolve to nothing, not a guess.

        Phase 3 registered image/audio/video/document, so this uses a type
        outside every registered processor's range.
        """
        assert for_media("application/x-sqlite3", tmp_path / "a.db") is None
        assert for_media("application/zip", tmp_path / "a.zip") is None

    @pytest.mark.parametrize(
        ("media_type", "suffix", "expected"),
        [
            ("image/png", ".png", "image"),
            ("audio/mpeg", ".mp3", "audio"),
            ("video/mp4", ".mp4", "video"),
            ("application/pdf", ".pdf", "document"),
            (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document",
                ".docx",
                "document",
            ),
            ("text/markdown", ".md", "text"),
        ],
    )
    def test_media_processors_are_resolvable(
        self, tmp_path: Path, media_type: str, suffix: str, expected: str
    ) -> None:
        found = for_media(media_type, tmp_path / f"input{suffix}")
        assert found is not None and found.name == expected

    def test_exact_match_beats_wildcard(self, clean_registry: None, tmp_path: Path) -> None:
        """A future audio/wav processor must win over a generic audio/* one."""

        class Wildcard:
            name: str = "wildcard"
            version: str = "1.0"
            cost: Cost = "cheap"
            media_types: tuple[str, ...] = ("audio/*",)
            extensions: tuple[str, ...] = ()

            def can_handle(self, media_type: str, path: Path) -> bool:
                return media_type.startswith("audio/")

            def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
                return ProcessResult()

        # Isolated: the real audio processor also claims audio/wav exactly,
        # and this test is about ranking, not about which one ships.
        REGISTRY.clear()
        register(Wildcard())
        register(StubProcessor())
        found = for_media("audio/wav", tmp_path / "a.wav")
        assert found is not None and found.name == "stub"

    def test_specificity_ranking(self) -> None:
        stub = StubProcessor()
        assert _specificity(stub, "audio/wav") == 3
        assert _specificity(stub, "audio/mpeg") == 0
        assert _specificity(TextProcessor(), "text/plain") == 3
        assert _specificity(TextProcessor(), "text/anything") == 2


class TestTextExtraction:
    def test_extracts_text(self, text_file: Path) -> None:
        result = TextProcessor().process(text_file, ProcessContext(media_type="text/markdown"))
        assert result.text is not None
        assert "First paragraph about uv." in result.text

    def test_reports_metadata(self, text_file: Path) -> None:
        result = TextProcessor().process(text_file, ProcessContext())
        assert result.meta["encoding"] == "utf-8"
        assert result.meta["bytes"] > 0
        assert result.meta["segments"] == len(result.segments)

    @pytest.mark.parametrize(
        ("encoding", "text"),
        [
            ("utf-8", "plain ascii text"),
            ("utf-8", "unicode: café, 日本語, 🎉"),
            ("utf-16", "utf sixteen content"),
            ("latin-1", "latin text: café"),
        ],
    )
    def test_decodes_common_encodings(self, tmp_path: Path, encoding: str, text: str) -> None:
        path = tmp_path / "input.txt"
        path.write_bytes(text.encode(encoding))
        result = TextProcessor().process(path, ProcessContext())
        assert result.text is not None and result.text.strip()

    def test_undecodable_bytes_do_not_raise(self, tmp_path: Path) -> None:
        """Latin-1 is the last resort precisely so decoding cannot fail."""
        path = tmp_path / "binary.txt"
        path.write_bytes(bytes(range(256)))
        result = TextProcessor().process(path, ProcessContext())
        assert result.text is not None

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        result = TextProcessor().process(path, ProcessContext())
        assert result.segments == []
        assert result.text == ""

    def test_file_with_no_newlines(self, tmp_path: Path) -> None:
        path = tmp_path / "oneline.txt"
        path.write_text("word " * 500, encoding="utf-8")
        result = TextProcessor().process(path, ProcessContext())
        assert len(result.segments) >= 1

    def test_crlf_is_normalized(self, tmp_path: Path) -> None:
        path = tmp_path / "crlf.txt"
        path.write_bytes(b"first para\r\n\r\nsecond para\r\n")
        result = TextProcessor().process(path, ProcessContext())
        assert result.text is not None and "\r" not in result.text


class TestSegmentation:
    def test_is_deterministic(self) -> None:
        text = "\n\n".join(f"Paragraph {i} with some content." for i in range(20))
        first = [s.text for s in segments_from_text(text)]
        second = [s.text for s in segments_from_text(text)]
        assert first == second

    def test_ordinals_are_sequential(self) -> None:
        text = "\n\n".join(f"Para {i}. " + "filler " * 60 for i in range(10))
        segments = segments_from_text(text, target_chars=400)
        assert [s.ordinal for s in segments] == list(range(len(segments)))

    def test_splits_long_text(self) -> None:
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
        assert len(segments_from_text(text, target_chars=500)) > 1

    def test_short_text_is_one_segment(self) -> None:
        assert len(segments_from_text("Just one short paragraph.")) == 1

    def test_overlap_links_adjacent_segments(self) -> None:
        """A claim spanning a boundary must be findable in one of the halves."""
        text = "\n\n".join(f"Paragraph {i} unique{i}." for i in range(30))
        segments = segments_from_text(text, target_chars=200, overlap_chars=80)
        assert len(segments) > 1
        tail = segments[0].text[-40:]
        assert any(tail[-20:] in s.text for s in segments[1:2])

    def test_zero_overlap_is_allowed(self) -> None:
        text = "\n\n".join(f"Para {i}." for i in range(20))
        segments = segments_from_text(text, target_chars=50, overlap_chars=0)
        assert len(segments) > 1

    def test_empty_input(self) -> None:
        assert segments_from_text("") == []
        assert segments_from_text("   \n\n  ") == []

    def test_carries_owner_ids(self) -> None:
        segments = segments_from_text("content here", asset_id="a", source_id="s")
        assert segments[0].asset_id == "a"
        assert segments[0].source_id == "s"


class TestDerivationCache:
    def test_writes_a_derived_file_with_its_cache_key(self, text_file: Path) -> None:
        """The header is what lets a rebuild reconstruct the cache."""
        out_dir = text_file.parent / "derived"
        ctx = ProcessContext(asset_id="asset-1", derived_dir=out_dir)
        result = TextProcessor().process(text_file, ctx)

        assert len(result.derived) == 1
        artifact = result.derived[0]
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
        header = payload["_derivation"]
        assert header["processor"] == "text"
        assert header["processor_version"] == "1.0"
        assert header["asset_id"] == "asset-1"
        assert header["input_hash"]
        assert header["params_hash"]

    def test_derivation_id_is_stable_for_identical_input(self, text_file: Path) -> None:
        out_dir = text_file.parent / "derived"
        first = TextProcessor().process(
            text_file, ProcessContext(asset_id="a", derived_dir=out_dir)
        )
        second = TextProcessor().process(
            text_file, ProcessContext(asset_id="a", derived_dir=out_dir)
        )
        assert first.derived[0].path == second.derived[0].path

    def test_changing_params_changes_the_key(self, text_file: Path) -> None:
        out_dir = text_file.parent / "derived"
        default = TextProcessor().process(
            text_file, ProcessContext(asset_id="a", derived_dir=out_dir)
        )
        tuned = TextProcessor().process(
            text_file,
            ProcessContext(asset_id="a", derived_dir=out_dir, params={"target_chars": 300}),
        )
        assert default.derived[0].path != tuned.derived[0].path

    def test_params_hash_is_order_independent(self) -> None:
        a = ProcessContext(params={"x": 1, "y": 2})
        b = ProcessContext(params={"y": 2, "x": 1})
        assert a.params_hash() == b.params_hash()

    def test_no_derived_dir_means_no_files(self, text_file: Path) -> None:
        result = TextProcessor().process(text_file, ProcessContext())
        assert result.derived == []

    def test_cache_key_contains_every_field(self) -> None:
        key = cache_key(
            asset_id="a",
            kind="transcript",
            processor="text",
            processor_version="1.0",
            input_hash="h",
            params_hash="p",
        )
        assert set(key) == {
            "asset_id",
            "kind",
            "processor",
            "processor_version",
            "input_hash",
            "params_hash",
        }

    def test_segments_reference_the_derivation(self, text_file: Path) -> None:
        out_dir = text_file.parent / "derived"
        result = TextProcessor().process(
            text_file, ProcessContext(asset_id="a", derived_dir=out_dir)
        )
        assert result.segments
        assert all(s.derivation_id for s in result.segments)


class TestVersionBumpInvalidates:
    def test_a_new_processor_version_is_a_new_derivation(self, text_file: Path) -> None:
        """Bumping the version must not silently reuse the old output."""
        out_dir = text_file.parent / "derived"
        processor = TextProcessor()
        first = processor.process(text_file, ProcessContext(asset_id="a", derived_dir=out_dir))

        processor.version = "2.0"
        second = processor.process(text_file, ProcessContext(asset_id="a", derived_dir=out_dir))

        assert first.derived[0].path != second.derived[0].path
        assert first.derived[0].path.exists(), "the old output is retained, not overwritten"


class TestIndexIntegration:
    def test_processed_output_registers_and_survives_rebuild(self, tmp_path: Path) -> None:
        """End to end: process, register, delete the index, rebuild."""
        from memory.api import MemoryAPI

        source = tmp_path / "input.md"
        source.write_text(
            "\n\n".join(f"Paragraph {i} about uv packaging." for i in range(12)),
            encoding="utf-8",
        )

        root = tmp_path / "root"
        api = MemoryAPI.initialize(root)
        try:
            asset = api.store_asset(source, media_type="text/markdown")
            ctx = ProcessContext(
                asset_id=asset.id,
                media_type="text/markdown",
                derived_dir=api.paths.derived_dir("transcripts", asset.id),
            )
            result = TextProcessor().process(source, ctx)
            api.register_derivation(
                asset_id=asset.id,
                kind="transcript",
                processor="text",
                processor_version="1.0",
                output_path=result.derived[0].path,
                params_hash=ctx.params_hash(),
                segments=[
                    {"id": s.id, "ordinal": s.ordinal, "text": s.text} for s in result.segments
                ],
            )
            before = api.conn.execute("SELECT count(*) c FROM derivations").fetchone()["c"]
            segs_before = api.conn.execute("SELECT count(*) c FROM segments").fetchone()["c"]
            assert before == 1
            assert segs_before == len(result.segments)
        finally:
            api.close()

        for suffix in ("", "-wal", "-shm"):
            Path(str(api.paths.index_db) + suffix).unlink(missing_ok=True)

        with MemoryAPI(root) as rebuilt:
            report = rebuilt.rebuild_index()
            assert report["derivations"] == 1
            assert report["segments"] == segs_before
            assert rebuilt.verify()["clean"]


class TestEntryPoints:
    def test_loader_tolerates_a_broken_plugin(
        self, monkeypatch: pytest.MonkeyPatch, clean_registry: None
    ) -> None:
        """A bad third-party processor must not break startup."""
        from memory.processors import base

        class Boom:
            name = "boom"

            def __call__(self) -> Any:
                raise RuntimeError("plugin is broken")

        class FakeEntry:
            name = "boom"

            def load(self) -> Any:
                return Boom()

        monkeypatch.setattr(base, "entry_points", lambda **_: [FakeEntry()], raising=False)
        monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [FakeEntry()])
        assert base.load_entry_point_processors() == []

    def test_loader_registers_a_working_plugin(
        self, monkeypatch: pytest.MonkeyPatch, clean_registry: None
    ) -> None:
        class FakeEntry:
            name = "stub"

            def load(self) -> Any:
                return StubProcessor

        monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: [FakeEntry()])
        from memory.processors import base

        assert base.load_entry_point_processors() == ["stub"]
        assert "stub" in REGISTRY
