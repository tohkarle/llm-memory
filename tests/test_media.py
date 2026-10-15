"""Optional-dependency probing and graceful degradation.

The contract these tests hold: the core install has no media extras, and
every processor must still run, report what it could extract, and say what
was missing. A test that only passes with Pillow installed would hide
exactly the regression that matters, so absence is simulated explicitly
rather than depending on what this machine happens to have.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from memory import media
from memory.processors import ProcessContext
from memory.processors.audio import AudioProcessor
from memory.processors.document import DocumentProcessor
from memory.processors.image import ImageProcessor
from memory.processors.video import VideoProcessor


@pytest.fixture(autouse=True)
def clear_probe_cache() -> Any:
    """Probes are cached for the process; a test that fakes absence must not
    leak that answer into the next one."""
    media.reset_probes()
    yield
    media.reset_probes()


def make_png(path: Path, width: int = 40, height: int = 24) -> Path:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def make_wav(path: Path, samples: int = 8000) -> Path:
    data = b"\x00\x00" * samples
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return path


class TestCapabilityProbing:
    def test_missing_module_is_reported_not_raised(self) -> None:
        cap = media.python_module("definitely_not_installed_xyz")
        assert cap.available is False
        assert "not installed" in cap.detail
        assert "uv sync --extra media" in cap.hint

    def test_present_module_is_available(self) -> None:
        assert media.python_module("json").available is True

    def test_load_returns_none_for_a_missing_module(self) -> None:
        assert media.load("definitely_not_installed_xyz") is None

    def test_load_returns_the_module_when_present(self) -> None:
        assert media.load("json") is not None

    def test_load_degrades_when_import_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A half-installed native extension must degrade like absence."""
        monkeypatch.setattr(media, "python_module", lambda *a, **k: media.Capability("x", True))

        def boom(name: str) -> Any:
            raise RuntimeError("libjpeg missing")

        monkeypatch.setattr(media.importlib, "import_module", boom)
        assert media.load("anything") is None

    def test_missing_binary_is_reported(self) -> None:
        cap = media.binary("definitely-not-a-real-binary-xyz")
        assert cap.available is False
        assert "not on PATH" in cap.detail

    def test_capability_report_lists_everything(self) -> None:
        report = media.capability_report()
        assert set(report) == {"available", "missing", "detail"}
        names = set(report["available"]) | set(report["missing"])
        assert {"PIL", "pypdf", "docx", "faster_whisper", "ffmpeg"} <= names

    def test_warning_names_the_fixing_extra(self) -> None:
        cap = media.Capability("pypdf", False, "not installed")
        assert "uv sync --extra media" in cap.warning(doing="extracting PDF text")


class TestDegradation:
    def test_collects_warnings_and_missing_names(self) -> None:
        degraded = media.Degradation()
        degraded.note(media.Capability("PIL", False, "not installed"), doing="reading metadata")
        degraded.note(media.Capability("pypdf", False, "not installed"), doing="extracting text")
        meta = degraded.to_meta()
        assert meta["missing_capabilities"] == ["PIL", "pypdf"]
        assert len(meta["warnings"]) == 2

    def test_available_capability_adds_nothing(self) -> None:
        degraded = media.Degradation()
        degraded.note(media.Capability("PIL", True), doing="reading metadata")
        assert degraded.to_meta() == {}
        assert not degraded

    def test_does_not_duplicate_a_missing_name(self) -> None:
        degraded = media.Degradation()
        cap = media.Capability("PIL", False, "not installed")
        degraded.note(cap, doing="one")
        degraded.note(cap, doing="two")
        assert degraded.to_meta()["missing_capabilities"] == ["PIL"]


class TestImageWithoutPillow:
    """Dimensions must survive with no Pillow, which is why the header parser
    exists."""

    @pytest.fixture(autouse=True)
    def no_pillow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = media.load

        def fake(module: str) -> Any:
            if module.startswith("PIL") or module == "pytesseract":
                return None
            return real(module)

        monkeypatch.setattr(media, "load", fake)
        monkeypatch.setattr(
            media, "python_module", lambda m, **k: media.Capability(m, False, "not installed")
        )

    def test_png_dimensions_come_from_the_header(self, tmp_path: Path) -> None:
        result = ImageProcessor().process(
            make_png(tmp_path / "a.png", 40, 24), ProcessContext(media_type="image/png")
        )
        assert result.meta["width"] == 40
        assert result.meta["height"] == 24
        assert result.meta["format"] == "PNG"

    def test_reports_what_was_missing(self, tmp_path: Path) -> None:
        result = ImageProcessor().process(make_png(tmp_path / "a.png"), ProcessContext())
        assert "PIL" in result.meta["missing_capabilities"]
        assert result.meta["ocr"] == "unavailable"

    def test_yields_no_segments_and_that_is_not_a_failure(self, tmp_path: Path) -> None:
        result = ImageProcessor().process(make_png(tmp_path / "a.png"), ProcessContext())
        assert result.segments == []
        assert result.text is None

    def test_a_truncated_image_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)
        result = ImageProcessor().process(path, ProcessContext(media_type="image/png"))
        assert result.meta["bytes"] > 0

    def test_an_empty_file_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        assert ImageProcessor().process(path, ProcessContext()).meta["bytes"] == 0

    def test_ocr_can_be_disabled(self, tmp_path: Path) -> None:
        result = ImageProcessor().process(
            make_png(tmp_path / "a.png"), ProcessContext(params={"ocr": False})
        )
        assert result.meta["ocr"] == "disabled"


class TestHeaderDimensions:
    """The dependency-free parser, one case per container layout."""

    def test_gif(self, tmp_path: Path) -> None:
        from memory.processors.image import _header_dimensions

        path = tmp_path / "a.gif"
        path.write_bytes(b"GIF89a" + struct.pack("<HH", 64, 32) + b"\x00" * 16)
        assert _header_dimensions(path) == {"width": 64, "height": 32, "format": "GIF"}

    def test_bmp(self, tmp_path: Path) -> None:
        from memory.processors.image import _header_dimensions

        path = tmp_path / "a.bmp"
        path.write_bytes(b"BM" + b"\x00" * 16 + struct.pack("<ii", 20, -10) + b"\x00" * 8)
        out = _header_dimensions(path)
        # Height is signed: a negative value means top-down row order, not a
        # negative size.
        assert out["width"] == 20 and out["height"] == 10

    def test_jpeg(self, tmp_path: Path) -> None:
        from memory.processors.image import _header_dimensions

        path = tmp_path / "a.jpg"
        path.write_bytes(
            b"\xff\xd8"
            + b"\xff\xe0"
            + struct.pack(">H", 16)
            + b"\x00" * 14
            + b"\xff\xc0"
            + struct.pack(">H", 17)
            + b"\x08"
            + struct.pack(">HH", 48, 96)
            + b"\x00" * 8
        )
        out = _header_dimensions(path)
        assert out == {"width": 96, "height": 48, "format": "JPEG"}

    def test_unrecognized_reports_nothing(self, tmp_path: Path) -> None:
        from memory.processors.image import _header_dimensions

        path = tmp_path / "a.bin"
        path.write_bytes(b"not an image at all")
        assert _header_dimensions(path) == {}


class TestAudioWithoutWhisper:
    @pytest.fixture(autouse=True)
    def no_whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = media.load
        monkeypatch.setattr(media, "load", lambda m: None if m == "faster_whisper" else real(m))

    def test_degrades_to_no_transcript(self, tmp_path: Path) -> None:
        result = AudioProcessor().process(
            make_wav(tmp_path / "a.wav"), ProcessContext(media_type="audio/wav")
        )
        assert result.meta["transcription"] == "unavailable"
        assert result.segments == []

    def test_names_the_missing_dependency(self, tmp_path: Path) -> None:
        result = AudioProcessor().process(make_wav(tmp_path / "a.wav"), ProcessContext())
        assert "faster_whisper" in result.meta["missing_capabilities"]

    def test_is_declared_expensive(self) -> None:
        """The queue depends on this hint to defer transcription."""
        assert AudioProcessor().cost == "expensive"


class TestVideoWithoutFfmpeg:
    @pytest.fixture(autouse=True)
    def no_ffmpeg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media, "ffmpeg", lambda: media.Capability("ffmpeg", False, "not on PATH", None)
        )
        monkeypatch.setattr(
            media, "ffprobe", lambda: media.Capability("ffprobe", False, "not on PATH", None)
        )

    def test_degrades_without_crashing(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        path.write_bytes(b"\x00" * 64)
        result = VideoProcessor().process(path, ProcessContext(media_type="video/mp4"))
        assert result.meta["transcription"] == "unavailable"
        assert result.segments == []
        assert "ffmpeg" in result.meta["missing_capabilities"]

    def test_writes_no_keyframes(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        path.write_bytes(b"\x00" * 64)
        result = VideoProcessor().process(
            path, ProcessContext(media_type="video/mp4", derived_dir=tmp_path / "d")
        )
        assert "keyframes" not in result.meta


class TestDocumentWithoutLibraries:
    @pytest.fixture(autouse=True)
    def no_libs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = media.load
        monkeypatch.setattr(media, "load", lambda m: None if m in ("pypdf", "docx") else real(m))
        monkeypatch.setattr(
            media, "python_module", lambda m, **k: media.Capability(m, False, "not installed")
        )

    def test_pdf_degrades(self, tmp_path: Path) -> None:
        path = tmp_path / "a.pdf"
        path.write_bytes(b"%PDF-1.4\ntrailer<</Root 1 0 R>>")
        result = DocumentProcessor().process(path, ProcessContext(media_type="application/pdf"))
        assert result.meta["extraction"] == "unavailable"
        assert result.meta["format"] == "pdf"
        assert "pypdf" in result.meta["missing_capabilities"]

    def test_docx_degrades(self, tmp_path: Path) -> None:
        path = tmp_path / "a.docx"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        result = DocumentProcessor().process(path, ProcessContext())
        assert result.meta["extraction"] == "unavailable"
        assert result.meta["format"] == "docx"


class TestProcessorsNeverRaise:
    """The absolute rule: no input crashes a processor on a bare install."""

    @pytest.mark.parametrize(
        "processor",
        [ImageProcessor(), AudioProcessor(), VideoProcessor(), DocumentProcessor()],
        ids=lambda p: str(p.name),
    )
    @pytest.mark.parametrize(
        ("name", "payload"),
        [
            ("empty", b""),
            ("garbage", b"\x00\x01\x02\xff\xfe"),
            ("text", b"this is plain text, not media"),
            ("truncated-png", b"\x89PNG\r\n\x1a\n\x00\x00"),
        ],
    )
    def test_survives_bad_input(
        self, tmp_path: Path, processor: Any, name: str, payload: bytes
    ) -> None:
        path = tmp_path / f"{name}-{processor.name}.bin"
        path.write_bytes(payload)
        result = processor.process(path, ProcessContext())
        assert result.meta["bytes"] == len(payload)
