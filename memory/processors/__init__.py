"""Content processors: the extensible ingestion seam.

Text, image, audio, video and document processors are registered by default.
Third-party processors are published by other packages through the
``memory.processors`` entry point group and loaded by
:func:`load_entry_point_processors`.

Importing this package is cheap and never fails on a machine without the
media extras: a processor's heavy dependencies are probed lazily inside
``process()``, never at import time. See :mod:`memory.media`.
"""

from __future__ import annotations

from .audio import AudioProcessor
from .base import (
    REGISTRY,
    Cost,
    DerivedArtifact,
    ProcessContext,
    Processor,
    ProcessorError,
    ProcessResult,
    cache_key,
    clear_registry,
    for_media,
    iter_registered,
    load_entry_point_processors,
    register,
    segments_from_text,
    unregister,
)
from .derive import write_derived
from .document import DocumentProcessor
from .image import ImageProcessor
from .text import TextProcessor
from .video import VideoProcessor

for _processor in (
    TextProcessor(),
    ImageProcessor(),
    AudioProcessor(),
    VideoProcessor(),
    DocumentProcessor(),
):
    register(_processor, replace=True)

__all__ = [
    "REGISTRY",
    "AudioProcessor",
    "Cost",
    "DerivedArtifact",
    "DocumentProcessor",
    "ImageProcessor",
    "ProcessContext",
    "ProcessResult",
    "Processor",
    "ProcessorError",
    "TextProcessor",
    "VideoProcessor",
    "cache_key",
    "clear_registry",
    "for_media",
    "iter_registered",
    "load_entry_point_processors",
    "register",
    "segments_from_text",
    "unregister",
    "write_derived",
]
