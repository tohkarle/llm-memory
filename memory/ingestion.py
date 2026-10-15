"""The ingestion pipeline: a file on disk becomes searchable content.

One path, six steps:

1. Import the original into managed storage (content-addressed blob).
2. Register its metadata, hash and asset id.
3. Detect the media type.
4. Select a processor from the registry.
5. Generate derived content -- inline if the processor is ``cheap``, deferred
   onto the work queue if it is ``expensive``.
6. Index the extracted text and segments, each referencing the originating
   asset and derivation.

Storage reuses :class:`memory.api.MemoryAPI` wholesale (``store_asset``,
``register_derivation``): this module decides *what* to process and *when*,
never *how* to store it.

The original is never modified. Processors read it and write only under
``data/derived/``.
"""

from __future__ import annotations

import mimetypes
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import db, workqueue
from .models import Asset, SourceRef
from .processors import base as processors
from .processors.base import ProcessContext, Processor, ProcessResult

if TYPE_CHECKING:
    from .api import MemoryAPI

#: Which ``data/derived/`` subtree a processor's output belongs in. The
#: processor names its output kind; this maps it onto the three directories
#: ``paths.derived_dir`` accepts.
KIND_DIRS: dict[str, str] = {
    "transcript": "transcripts",
    "description": "descriptions",
    "keyframe": "keyframes",
}

TASK_PROCESS = "process_asset"


class IngestionError(RuntimeError):
    """Ingestion could not proceed."""


@dataclass(slots=True)
class IngestResult:
    """What one ingestion did, in enough detail to act on."""

    asset_id: str
    media_type: str
    sha256: str
    bytes: int
    path: str
    processor: str | None = None
    cost: str | None = None
    status: str = "complete"
    derivation_id: str | None = None
    segments: int = 0
    characters: int = 0
    queued_task: str | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "path": self.path,
            "processor": self.processor,
            "cost": self.cost,
            "status": self.status,
            "derivation_id": self.derivation_id,
            "segments": self.segments,
            "characters": self.characters,
            "queued_task": self.queued_task,
            "warnings": self.warnings,
            "meta": self.meta,
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------- type detection


def detect_media_type(path: Path, *, declared: str | None = None) -> str:
    """Decide a media type, preferring the caller then content then extension.

    Sniffing beats ``mimetypes`` for the formats that matter here because an
    extension is a claim, not evidence: a ``.txt`` that is really a PDF would
    otherwise reach the text processor and produce garbage segments.
    """
    if declared:
        return declared
    sniffed = sniff(path)
    if sniffed:
        return sniffed
    guess, _ = mimetypes.guess_type(path.name)
    return guess or "application/octet-stream"


#: Magic numbers, longest-first where prefixes overlap.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"\x1a\x45\xdf\xa3", "video/x-matroska"),
    (b"ID3", "audio/mpeg"),
)


def sniff(path: Path) -> str | None:
    """Identify a file by its leading bytes. ``None`` when unrecognized."""
    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return None
    if not head:
        return None

    for magic, media_type in _MAGIC:
        if head.startswith(magic):
            return media_type

    # Container formats whose marker sits at a fixed offset rather than 0.
    if head[:4] == b"RIFF" and len(head) >= 12:
        if head[8:12] == b"WAVE":
            return "audio/wav"
        if head[8:12] == b"AVI ":
            return "video/x-msvideo"
        if head[8:12] == b"WEBP":
            return "image/webp"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4"
        if brand.startswith(b"qt"):
            return "video/quicktime"
        return "video/mp4"
    # Office formats are zip containers; the extension is the only cheap
    # discriminator, so defer to it rather than reporting application/zip.
    if head[:2] == b"PK\x03\x04":
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return None
    return None


# ---------------------------------------------------------------- pipeline


def ingest(
    api: MemoryAPI,
    path: Path,
    *,
    media_type: str | None = None,
    tags: list[str] | None = None,
    source: SourceRef | None = None,
    meta: dict[str, Any] | None = None,
    link_to: list[str] | None = None,
    params: dict[str, Any] | None = None,
    process: bool = True,
    force: bool = False,
    defer: bool | None = None,
) -> IngestResult:
    """Import one file and derive searchable content from it.

    ``defer`` overrides the cost-based lane: ``True`` always queues, ``False``
    always runs inline (which is how ``mem queue run`` executes an expensive
    task), and ``None`` -- the default -- follows the processor's ``cost``.
    """
    started = time.monotonic()
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise IngestionError(f"not a file: {source_path}")

    detected = detect_media_type(source_path, declared=media_type)
    asset = api.store_asset(
        source_path,
        media_type=detected,
        tags=tags,
        source=source,
        meta=meta,
        link_to=link_to,
    )

    result = IngestResult(
        asset_id=asset.id,
        media_type=asset.media_type,
        sha256=asset.sha256,
        bytes=asset.bytes,
        path=api.paths.rel(api.paths.asset_blob(asset.sha256)),
    )

    if not process:
        result.status = "stored"
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    outcome = process_asset(
        api,
        asset=asset,
        params=params,
        force=force,
        defer=defer,
    )
    _apply(result, outcome)
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


def process_asset(
    api: MemoryAPI,
    *,
    asset: Asset | str,
    params: dict[str, Any] | None = None,
    force: bool = False,
    defer: bool | None = None,
) -> dict[str, Any]:
    """Derive content for an already-stored asset.

    Separate from :func:`ingest` because the queue runs exactly this: by the
    time a deferred task executes, the asset is long since stored.
    """
    stored = api.store.require_asset(asset) if isinstance(asset, str) else asset
    blob = api.paths.asset_blob(stored.sha256)
    if not blob.exists():
        raise IngestionError(f"asset {stored.id} has no blob on disk ({stored.sha256})")

    processor = processors.for_media(stored.media_type, Path(stored.original_filename or blob.name))
    if processor is None:
        return {
            "status": "unsupported",
            "processor": None,
            "warnings": [
                f"no processor handles {stored.media_type!r}; "
                "the asset is stored and findable by filename and tags"
            ],
        }

    should_defer = processor.cost == "expensive" if defer is None else defer
    if should_defer:
        task = workqueue.enqueue(
            api.conn,
            kind=TASK_PROCESS,
            asset_id=stored.id,
            payload={
                "asset_id": stored.id,
                "processor": processor.name,
                "params": dict(params or {}),
                "force": force,
            },
            # One live task per asset+processor+version: re-ingesting the same
            # recording must not queue a second transcription of it.
            dedupe_key=f"{TASK_PROCESS}:{stored.id}:{processor.name}:{processor.version}",
            priority=200,
        )
        return {
            "status": "queued",
            "processor": processor.name,
            "cost": processor.cost,
            "queued_task": task.id,
        }

    return run_processor(api, stored, processor, params=params, force=force)


def run_processor(
    api: MemoryAPI,
    asset: Asset,
    processor: Processor,
    *,
    params: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one processor and index everything it produced."""
    blob = api.paths.asset_blob(asset.sha256)
    # Name the context by the original filename: a processor that falls back
    # to an extension would otherwise see the hex blob name and no suffix.
    ctx = ProcessContext(
        asset_id=asset.id,
        media_type=asset.media_type,
        derived_dir=_derived_dir(api, processor, asset),
        params=dict(params or {}),
    )

    cached = None if force else _cached(api.conn, asset, processor, ctx)
    if cached is not None:
        return {
            "status": "cached",
            "processor": processor.name,
            "cost": processor.cost,
            "derivation_id": cached["id"],
            "segments": cached["segments"],
            "warnings": [],
        }

    started = time.monotonic()
    result = processor.process(blob, ctx)
    cost_ms = int((time.monotonic() - started) * 1000)
    return index_result(api, asset, processor, result, ctx, cost_ms=cost_ms)


def index_result(
    api: MemoryAPI,
    asset: Asset,
    processor: Processor,
    result: ProcessResult,
    ctx: ProcessContext,
    *,
    cost_ms: int = 0,
) -> dict[str, Any]:
    """Register the derivation and index its segments and text."""
    warnings = list(result.meta.get("warnings") or [])
    text = result.text or ""

    if not result.derived:
        # Nothing was written, so there is nothing to register or rebuild
        # from. Happens when a processor degraded to metadata only.
        return {
            "status": "degraded" if warnings else "empty",
            "processor": processor.name,
            "cost": processor.cost,
            "segments": 0,
            "characters": len(text),
            "warnings": warnings,
            "meta": result.meta,
        }

    artifact = result.derived[0]
    kind = artifact.kind
    # The filename is the derivation id by construction (see derive.py), and
    # register_derivation re-reads it from the file's own header anyway.
    derivation_id = artifact.path.stem

    # Drop the previous generation's segments before indexing this one:
    # reprocessing can yield fewer segments, and an upsert alone would leave
    # the surplus rows behind as phantom search hits.
    db.unindex_segments_for(api.conn, derivation_id=derivation_id)

    row = api.register_derivation(
        asset_id=asset.id,
        kind=kind,
        processor=processor.name,
        processor_version=processor.version,
        output_path=artifact.path,
        params_hash=ctx.params_hash(),
        segments=[_segment_row(s) for s in result.segments],
        cost_ms=cost_ms,
    )

    return {
        "status": "degraded" if warnings else "complete",
        "processor": processor.name,
        "cost": processor.cost,
        "derivation_id": row["id"],
        "segments": len(result.segments),
        "characters": len(text),
        "warnings": warnings,
        "meta": result.meta,
        "output_path": row["output_path"],
    }


def _segment_row(segment: Any) -> dict[str, Any]:
    return {
        "id": segment.id,
        "ordinal": segment.ordinal,
        "text": segment.text,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "page": segment.page,
        "bbox": segment.bbox,
        "speaker": segment.speaker,
        "meta": segment.meta,
    }


def _derived_dir(api: MemoryAPI, processor: Processor, asset: Asset) -> Path:
    """Where this processor's output goes.

    Keyed off the processor rather than the result, because the directory has
    to be decided before the processor runs.
    """
    kind = "descriptions" if processor.name == "image" else "transcripts"
    return api.paths.derived_dir(kind, asset.id)


def _cached(
    conn: sqlite3.Connection,
    asset: Asset,
    processor: Processor,
    ctx: ProcessContext,
) -> dict[str, Any] | None:
    """Find a complete derivation matching this exact cache key.

    This is what keeps a 40-minute transcription from being re-run by a second
    ``mem ingest`` of the same file. ``input_hash`` is the asset's sha256, so
    changed bytes miss the cache by construction.
    """
    row = conn.execute(
        """
        SELECT d.id, d.output_path,
               (SELECT count(*) FROM segments s WHERE s.derivation_id = d.id) segments
          FROM derivations d
         WHERE d.asset_id = ? AND d.processor = ? AND d.processor_version = ?
           AND d.input_hash = ? AND d.params_hash = ? AND d.status = 'complete'
         LIMIT 1
        """,
        (asset.id, processor.name, processor.version, asset.sha256, ctx.params_hash()),
    ).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "segments": row["segments"], "output_path": row["output_path"]}


def _apply(result: IngestResult, outcome: dict[str, Any]) -> None:
    result.status = str(outcome.get("status", result.status))
    result.processor = outcome.get("processor")
    result.cost = outcome.get("cost")
    result.derivation_id = outcome.get("derivation_id")
    result.segments = int(outcome.get("segments") or 0)
    result.characters = int(outcome.get("characters") or 0)
    result.queued_task = outcome.get("queued_task")
    result.warnings = list(outcome.get("warnings") or [])
    result.meta = dict(outcome.get("meta") or {})


# ---------------------------------------------------------------- queue worker


def run_queued_task(api: MemoryAPI, task: workqueue.Task) -> dict[str, Any]:
    """Execute one queued task. The handler ``mem queue run`` drains with.

    Raises on failure so the queue records the error and schedules a retry;
    a degraded result (a missing optional dependency) is a *success* with
    warnings, because retrying it would fail identically until the user
    installs the extra.
    """
    if task.kind != TASK_PROCESS:
        raise IngestionError(f"unknown task kind: {task.kind!r}")
    asset_id = str(task.payload.get("asset_id") or task.asset_id or "")
    if not asset_id:
        raise IngestionError("task carries no asset_id")

    outcome = process_asset(
        api,
        asset=asset_id,
        params=task.payload.get("params") or {},
        force=bool(task.payload.get("force")),
        defer=False,  # already deferred; running it is the point
    )
    return outcome


def ingest_directory(
    api: MemoryAPI,
    directory: Path,
    *,
    recursive: bool = True,
    skip_hidden: bool = True,
    **kwargs: Any,
) -> list[IngestResult]:
    """Ingest every file in a directory.

    One file's failure is reported and skipped rather than aborting the batch:
    a folder of 300 photos must not be lost to one corrupt JPEG.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise IngestionError(f"not a directory: {root}")

    paths = sorted(root.rglob("*") if recursive else root.glob("*"))
    out: list[IngestResult] = []
    for candidate in paths:
        if not candidate.is_file():
            continue
        if skip_hidden and any(part.startswith(".") for part in candidate.relative_to(root).parts):
            continue
        try:
            out.append(ingest(api, candidate, **kwargs))
        except Exception as exc:
            out.append(
                IngestResult(
                    asset_id="",
                    media_type="",
                    sha256="",
                    bytes=0,
                    path=str(candidate),
                    status="error",
                    warnings=[f"{type(exc).__name__}: {exc}"],
                )
            )
    return out


__all__ = [
    "KIND_DIRS",
    "TASK_PROCESS",
    "IngestResult",
    "IngestionError",
    "detect_media_type",
    "ingest",
    "ingest_directory",
    "process_asset",
    "run_processor",
    "run_queued_task",
    "sniff",
]
