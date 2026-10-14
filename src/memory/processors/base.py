"""The processor interface: how content becomes searchable text.

This is the Phase 3 seam. Image, audio, video and document processors drop
in behind this protocol without touching storage, because the tables they
need (``segments``, ``derivations``) and the directories they write to
(``data/derived/``) already exist and are exercised by the text processor.

A processor never modifies its input. It returns extracted text, ordered
segments, and a manifest of files it wrote; the caller indexes the result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from ..models import Segment

#: Scheduler hint. Phase 3's queue runs cheap processors inline and defers
#: expensive ones (a 40s transcription) to background work.
Cost = Literal["cheap", "expensive"]

ENTRY_POINT_GROUP = "memory.processors"


@dataclass(slots=True)
class DerivedArtifact:
    """A file a processor wrote under ``data/derived/``.

    ``cache_key`` is embedded in the file itself, which is what lets
    ``rebuild_index`` reconstruct the derivations table from the filesystem
    alone. Without it the processor cache would be index-only state and
    deleting the index would silently lose it.
    """

    kind: str
    path: Path
    content_hash: str
    cache_key: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessContext:
    """What a processor is told about its input."""

    asset_id: str | None = None
    source_id: str | None = None
    media_type: str = "application/octet-stream"
    derived_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def params_hash(self) -> str:
        payload = json.dumps(self.params, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(slots=True)
class ProcessResult:
    text: str | None = None
    segments: list[Segment] = field(default_factory=list)
    derived: list[DerivedArtifact] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    #: Phase 4 fills this with candidate memories to extract. Phase 1-2
    #: ignores it, but the field exists so adding extraction needs no
    #: signature change.
    memory_suggestions: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class Processor(Protocol):
    name: str
    media_types: tuple[str, ...]
    extensions: tuple[str, ...]
    cost: Cost
    version: str

    def can_handle(self, media_type: str, path: Path) -> bool: ...

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult: ...


REGISTRY: dict[str, Processor] = {}


class ProcessorError(RuntimeError):
    """A processor could not handle its input."""


def register(processor: Processor, *, replace: bool = False) -> None:
    if processor.name in REGISTRY and not replace:
        raise ProcessorError(f"processor {processor.name!r} is already registered")
    REGISTRY[processor.name] = processor


def unregister(name: str) -> None:
    REGISTRY.pop(name, None)


def clear_registry() -> None:
    REGISTRY.clear()


def _specificity(processor: Processor, media_type: str) -> int:
    """Rank how precisely a processor claims a media type.

    An exact match beats a wildcard subtype, which beats an extension-only
    match, so a future "audio/wav" processor wins over a generic "audio/*".
    """
    if media_type in processor.media_types:
        return 3
    major = media_type.split("/", 1)[0]
    if f"{major}/*" in processor.media_types:
        return 2
    if "*/*" in processor.media_types:
        return 1
    return 0


def for_media(media_type: str, path: Path) -> Processor | None:
    """Pick the most specific processor that claims this input."""
    best: tuple[int, Processor] | None = None
    for processor in REGISTRY.values():
        if not processor.can_handle(media_type, path):
            continue
        rank = _specificity(processor, media_type)
        if rank == 0 and path.suffix.lower() in processor.extensions:
            rank = 1
        if rank == 0:
            continue
        if best is None or rank > best[0] or (rank == best[0] and processor.name < best[1].name):
            best = (rank, processor)
    return best[1] if best else None


def load_entry_point_processors() -> list[str]:
    """Discover processors published by other installed packages.

    Lets a media processor ship as a separate distribution without this
    package knowing about it.
    """
    from importlib.metadata import entry_points

    loaded: list[str] = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            candidate = entry.load()
            processor = candidate() if callable(candidate) else candidate
            register(processor, replace=True)
            loaded.append(processor.name)
        except Exception:
            continue
    return loaded


def cache_key(
    *,
    asset_id: str | None,
    kind: str,
    processor: str,
    processor_version: str,
    input_hash: str,
    params_hash: str,
) -> dict[str, Any]:
    """The memoization key a derived file embeds and the index stores."""
    return {
        "asset_id": asset_id,
        "kind": kind,
        "processor": processor,
        "processor_version": processor_version,
        "input_hash": input_hash,
        "params_hash": params_hash,
    }


def segments_from_text(
    text: str,
    *,
    asset_id: str | None = None,
    source_id: str | None = None,
    derivation_id: str | None = None,
    target_chars: int = 1200,
    overlap_chars: int = 100,
) -> list[Segment]:
    """Split text into overlapping, paragraph-aligned segments.

    Overlap matters for retrieval: a claim spanning a boundary would
    otherwise be findable in neither half.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current)

    return [
        Segment(
            text=chunk,
            ordinal=i,
            asset_id=asset_id,
            source_id=source_id,
            derivation_id=derivation_id,
        )
        for i, chunk in enumerate(chunks)
    ]


def iter_registered() -> Iterable[Processor]:
    return tuple(REGISTRY.values())
