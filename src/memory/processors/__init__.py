"""Content processors: the extensible ingestion seam.

The text processor is registered by default. Media processors arrive in
Phase 3, either imported and registered directly or published by another
package through the ``memory.processors`` entry point group.
"""

from __future__ import annotations

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
from .text import TextProcessor

register(TextProcessor(), replace=True)

__all__ = [
    "REGISTRY",
    "Cost",
    "DerivedArtifact",
    "ProcessContext",
    "ProcessResult",
    "Processor",
    "ProcessorError",
    "TextProcessor",
    "cache_key",
    "clear_registry",
    "for_media",
    "iter_registered",
    "load_entry_point_processors",
    "register",
    "segments_from_text",
    "unregister",
]
