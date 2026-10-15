"""Image processor: dimensions, embedded metadata, and optional OCR.

Two layers of optionality, which is why this file is longer than it looks
like it should be:

* Pillow absent -> dimensions still come from the file header, which this
  module parses itself for PNG/JPEG/GIF/BMP/WEBP. A dependency-free fallback
  is worth it because "how big is this image" is the one question that is
  always answerable.
* Pillow present, tesseract absent -> metadata and dimensions, no text.

An image with no OCR text yields no segments, and that is a correct result,
not a failure: the asset is still stored, indexed, and findable by filename
and tags.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from .. import media
from ..models import Segment
from .base import Cost, ProcessContext, ProcessResult, segments_from_text
from .derive import file_hash, write_derived

#: EXIF tags worth keeping. The full dump is large, mostly camera internals,
#: and would bloat every asset's metadata for no retrieval value.
EXIF_KEEP = {
    "DateTime",
    "DateTimeOriginal",
    "Make",
    "Model",
    "LensModel",
    "Orientation",
    "ImageDescription",
    "Artist",
    "Copyright",
    "Software",
    "GPSInfo",
}


class ImageProcessor:
    name: str = "image"
    version: str = "1.0"
    #: Cheap even with OCR: tesseract on one page is sub-second, far from the
    #: 40s transcription the deferred lane exists for.
    cost: Cost = "cheap"
    media_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "image/*",
    )
    extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    )

    def can_handle(self, media_type: str, path: Path) -> bool:
        if media_type.startswith("image/"):
            return True
        return path.suffix.lower() in self.extensions

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        degraded = media.Degradation()
        meta: dict[str, Any] = {"bytes": path.stat().st_size}

        pil = media.load("PIL.Image")
        if pil is None:
            degraded.note(media.python_module("PIL"), doing="reading image metadata with Pillow")
            meta.update(_header_dimensions(path))
        else:
            meta.update(self._pillow_metadata(path, pil, degraded))

        text, ocr_meta = self._maybe_ocr(path, ctx, degraded)
        meta.update(ocr_meta)

        segments: list[Segment] = []
        if text and text.strip():
            segments = segments_from_text(
                text,
                asset_id=ctx.asset_id,
                source_id=ctx.source_id,
                target_chars=int(ctx.params.get("target_chars", 1200)),
                overlap_chars=int(ctx.params.get("overlap_chars", 100)),
            )
        meta["segments"] = len(segments)
        meta.update(degraded.to_meta())

        derived = []
        # Still write a derivation with zero segments: it records that this
        # image was processed by this version with these params, which is
        # what stops the queue from retrying it forever.
        artifact, _ = write_derived(
            ctx=ctx,
            kind="description",
            processor=self.name,
            processor_version=self.version,
            input_hash=file_hash(path),
            segments=segments,
            meta=meta,
        )
        if artifact is not None:
            derived.append(artifact)

        return ProcessResult(text=text or None, segments=segments, derived=derived, meta=meta)

    # ------------------------------------------------------------------ internals

    def _pillow_metadata(self, path: Path, pil: Any, degraded: media.Degradation) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            with pil.open(path) as img:
                out["width"], out["height"] = int(img.width), int(img.height)
                out["mode"] = str(img.mode)
                out["format"] = str(img.format or "")
                if getattr(img, "n_frames", 1) > 1:
                    out["frames"] = int(img.n_frames)
                exif = self._exif(img)
                if exif:
                    out["exif"] = exif
        except Exception as exc:  # a corrupt image is data, not a bug
            degraded.add(f"Pillow could not read the image ({exc}); falling back to the header")
            out.update(_header_dimensions(path))
        return out

    def _exif(self, img: Any) -> dict[str, Any]:
        getexif = getattr(img, "getexif", None)
        if getexif is None:
            return {}
        try:
            raw = getexif()
        except Exception:
            return {}
        if not raw:
            return {}
        try:
            from PIL.ExifTags import TAGS  # type: ignore[import-not-found]
        except Exception:
            return {}

        out: dict[str, Any] = {}
        for tag_id, value in raw.items():
            name = TAGS.get(tag_id, str(tag_id))
            if name not in EXIF_KEEP:
                continue
            # EXIF values include bytes and rationals that do not survive a
            # JSON round-trip, and the derived file must round-trip exactly
            # for the rebuild to match. Stringify anything exotic.
            if isinstance(value, bytes):
                out[name] = value.decode("utf-8", errors="replace").strip("\x00")
            elif isinstance(value, str | int | float):
                out[name] = value
            else:
                out[name] = str(value)
        return out

    def _maybe_ocr(
        self, path: Path, ctx: ProcessContext, degraded: media.Degradation
    ) -> tuple[str, dict[str, Any]]:
        """Run OCR if asked for and possible. Absence is not an error."""
        if not ctx.params.get("ocr", True):
            return "", {"ocr": "disabled"}

        pytesseract = media.load("pytesseract")
        if pytesseract is None:
            degraded.note(media.python_module("pytesseract"), doing="OCR")
            return "", {"ocr": "unavailable"}
        tess = media.tesseract()
        if not tess.available:
            degraded.note(tess, doing="OCR")
            return "", {"ocr": "unavailable"}
        pil = media.load("PIL.Image")
        if pil is None:
            return "", {"ocr": "unavailable"}

        try:
            with pil.open(path) as img:
                text = str(pytesseract.image_to_string(img, lang=ctx.params.get("ocr_lang", "eng")))
        except Exception as exc:  # OCR failure degrades, never crashes
            degraded.add(f"OCR failed ({exc}); the image is stored without extracted text")
            return "", {"ocr": "failed"}
        cleaned = text.strip()
        return cleaned, {"ocr": "complete" if cleaned else "empty"}


# ---------------------------------------------------------------- header parsing


def _header_dimensions(path: Path) -> dict[str, Any]:
    """Read dimensions straight from the container header, without Pillow.

    Covers the formats that carry width and height at a fixed early offset.
    Anything else reports nothing rather than guessing.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        return {}

    try:
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            width, height = struct.unpack(">II", head[16:24])
            return {"width": int(width), "height": int(height), "format": "PNG"}
        if head.startswith(b"GIF8") and len(head) >= 10:
            width, height = struct.unpack("<HH", head[6:10])
            return {"width": int(width), "height": int(height), "format": "GIF"}
        if head.startswith(b"BM") and len(head) >= 26:
            width, height = struct.unpack("<ii", head[18:26])
            return {"width": abs(int(width)), "height": abs(int(height)), "format": "BMP"}
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            return _webp_dimensions(path)
        if head.startswith(b"\xff\xd8"):
            return _jpeg_dimensions(path)
    except (struct.error, OSError, ValueError):
        return {}
    return {}


def _jpeg_dimensions(path: Path) -> dict[str, Any]:
    """Walk JPEG markers to the start-of-frame, which carries the size."""
    with path.open("rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return {}
        while True:
            byte = fh.read(1)
            if not byte:
                return {}
            if byte != b"\xff":
                continue
            marker = fh.read(1)
            while marker == b"\xff":  # fill bytes are legal padding
                marker = fh.read(1)
            if not marker:
                return {}
            code = marker[0]
            # SOF0..SOF15, excluding the non-frame markers DHT/JPG/DAC.
            if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                fh.read(3)  # segment length + sample precision
                raw = fh.read(4)
                if len(raw) < 4:
                    return {}
                height, width = struct.unpack(">HH", raw)
                return {"width": int(width), "height": int(height), "format": "JPEG"}
            if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                continue  # markers without a payload
            raw = fh.read(2)
            if len(raw) < 2:
                return {}
            (length,) = struct.unpack(">H", raw)
            if length < 2:
                return {}
            fh.seek(length - 2, 1)


def _webp_dimensions(path: Path) -> dict[str, Any]:
    """WEBP keeps size in a chunk whose layout depends on the variant."""
    with path.open("rb") as fh:
        data = fh.read(32)
    if len(data) < 30:
        return {}
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return {"width": int(width) & 0x3FFF, "height": int(height) & 0x3FFF, "format": "WEBP"}
    if chunk == b"VP8L":
        (bits,) = struct.unpack("<I", data[21:25])
        return {
            "width": int(bits & 0x3FFF) + 1,
            "height": int((bits >> 14) & 0x3FFF) + 1,
            "format": "WEBP",
        }
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return {"width": width, "height": height, "format": "WEBP"}
    return {"format": "WEBP"}


__all__ = ["ImageProcessor"]
