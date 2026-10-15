"""Optional-dependency probing for the media processors.

Every heavy dependency in Phase 3 -- Pillow, faster-whisper, pypdf,
python-docx, tesseract, ffmpeg -- is an extra. The core install has none of
them, and that configuration is the one the test suite runs in, so the rule
is absolute: a missing dependency degrades a processor to the metadata it can
still extract and attaches a warning. It never raises on import, and it never
crashes a pipeline.

Probes are cached because they are not free (``ffmpeg -version`` forks a
process) and the answer cannot change inside one run.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import cache
from typing import Any

#: ``extra`` is the name in ``pyproject.toml``, so a warning can tell the
#: user the exact command that fixes it.
MEDIA_EXTRA = "media"

#: Which extra actually ships each optional module. A hint that names the
#: wrong extra is worse than no hint, because the user runs it and the
#: dependency is still missing.
MODULE_EXTRAS: dict[str, str] = {
    "PIL": "media",
    "pypdf": "media",
    "docx": "media",
    "faster_whisper": "transcription",
    "pytesseract": "ocr",
}


@dataclass(frozen=True, slots=True)
class Capability:
    """Whether one optional dependency is usable, and what to do if not."""

    name: str
    available: bool
    detail: str = ""
    extra: str | None = MEDIA_EXTRA

    @property
    def hint(self) -> str:
        if self.available:
            return ""
        if self.extra:
            return f"install it with `uv sync --extra {self.extra}`"
        return "install it and retry"

    def warning(self, *, doing: str) -> str:
        return f"{doing}: {self.name} is unavailable ({self.detail}); {self.hint}"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "extra": self.extra,
        }


@cache
def python_module(module: str, *, extra: str | None = None) -> Capability:
    """Probe an importable module without importing it.

    ``find_spec`` rather than ``import``: importing a large module the caller
    may never use costs real startup time, and a module that raises at import
    would otherwise take the probe with it.
    """
    # Resolve the owning extra from the table unless the caller overrode it,
    # so the remediation hint is a command that actually works.
    owning = extra or MODULE_EXTRAS.get(module.split(".", 1)[0], MEDIA_EXTRA)
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return Capability(module, False, "not installed", owning)
    return Capability(module, True, "", owning)


def load(module: str) -> Any | None:
    """Import a probed module, or return ``None`` if it is unusable.

    Catches ``Exception``, not just ``ImportError``: a half-installed native
    extension (Pillow without its libjpeg, say) raises from inside its own
    import, and that must degrade exactly like absence.
    """
    if not python_module(module).available:
        return None
    try:
        return importlib.import_module(module)
    except Exception:  # any import failure is unavailability
        return None


@cache
def binary(name: str, *, extra: str | None = None) -> Capability:
    """Probe an external executable on ``PATH``."""
    found = shutil.which(name)
    if found is None:
        return Capability(name, False, "not on PATH", extra)
    return Capability(name, True, found, extra)


@cache
def ffmpeg() -> Capability:
    return binary("ffmpeg")


@cache
def ffprobe() -> Capability:
    return binary("ffprobe")


@cache
def tesseract() -> Capability:
    """OCR is optional on top of an already-optional Pillow."""
    return binary("tesseract")


def run_binary(
    argv: list[str],
    *,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[bytes]:
    """Run a media tool, capturing output.

    ``check=False``: these tools exit non-zero on inputs that are merely
    unusual, and the caller decides whether that is fatal or a warning.
    ``stdin`` is closed so a prompting ffmpeg cannot hang the pipeline.
    """
    return subprocess.run(  # argv is built from probed binaries
        argv,
        capture_output=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


@dataclass(slots=True)
class Degradation:
    """Accumulates the warnings a processor produced while degrading.

    Carried into ``ProcessResult.meta`` so the CLI can surface exactly which
    capability was missing, rather than the user seeing an empty transcript
    and guessing.
    """

    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def note(self, cap: Capability, *, doing: str) -> None:
        if cap.available:
            return
        self.warnings.append(cap.warning(doing=doing))
        if cap.name not in self.missing:
            self.missing.append(cap.name)

    def add(self, message: str) -> None:
        self.warnings.append(message)

    def to_meta(self) -> dict[str, Any]:
        if not self.warnings and not self.missing:
            return {}
        return {"warnings": list(self.warnings), "missing_capabilities": list(self.missing)}

    def __bool__(self) -> bool:
        return bool(self.warnings)


def capability_report() -> dict[str, Any]:
    """Everything optional, and whether it is present. Feeds ``mem doctor``."""
    caps = [
        python_module("PIL"),
        python_module("pypdf"),
        python_module("docx"),
        python_module("faster_whisper"),
        python_module("pytesseract"),
        ffmpeg(),
        ffprobe(),
        tesseract(),
    ]
    return {
        "available": sorted(c.name for c in caps if c.available),
        "missing": sorted(c.name for c in caps if not c.available),
        "detail": {c.name: c.to_json() for c in caps},
    }


def reset_probes() -> None:
    """Clear the probe caches. Tests use this to simulate absence."""
    python_module.cache_clear()
    binary.cache_clear()
    ffmpeg.cache_clear()
    ffprobe.cache_clear()
    tesseract.cache_clear()


__all__ = [
    "MEDIA_EXTRA",
    "MODULE_EXTRAS",
    "Capability",
    "Degradation",
    "binary",
    "capability_report",
    "ffmpeg",
    "ffprobe",
    "load",
    "python_module",
    "reset_probes",
    "run_binary",
    "tesseract",
]
