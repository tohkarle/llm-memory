"""Shared derived-file writing for the media processors.

The text processor spells this out inline because it is the worked example.
Four more processors doing the same thing by hand would be four chances to
write a header the rebuild cannot read, so the logic lives here once and
every processor goes through it.

The invariant it enforces: a derived file always embeds its own full cache
key under ``_derivation``, because ``lifecycle.rebuild_index`` reads exactly
that to reconstruct the ``derivations`` and ``segments`` rows from the
filesystem alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import Segment, fmt_ts, now_utc
from .base import DerivedArtifact, ProcessContext, cache_key


def derivation_id(key: dict[str, Any]) -> str:
    """A stable uuid-shaped id derived from the cache key.

    Deterministic, so reprocessing identical input with identical parameters
    resolves to the same derivation instead of accumulating duplicates.
    """
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def segment_payload(segment: Segment) -> dict[str, Any]:
    """Serialize a segment, omitting the locators that do not apply.

    Omission rather than nulls: the rebuild reads these back, and a key that
    is absent on write must be absent on read for the two paths to agree.
    """
    out: dict[str, Any] = {
        "id": segment.id,
        "ordinal": segment.ordinal,
        "text": segment.text,
    }
    for field_name in ("start_ms", "end_ms", "page", "bbox", "speaker"):
        value = getattr(segment, field_name)
        if value is not None:
            out[field_name] = value
    if segment.meta:
        out["meta"] = segment.meta
    return out


def write_derived(
    *,
    ctx: ProcessContext,
    kind: str,
    processor: str,
    processor_version: str,
    input_hash: str,
    segments: list[Segment],
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
    status: str = "complete",
    cost_ms: int | None = None,
) -> tuple[DerivedArtifact | None, str | None]:
    """Write one derived JSON file and return it with its derivation id.

    Returns ``(None, None)`` when there is nowhere to write, which is the
    normal case for a bare ``process()`` call in a test: a processor must be
    usable without a configured root.
    """
    if ctx.derived_dir is None:
        return None, None

    key = cache_key(
        asset_id=ctx.asset_id,
        kind=kind,
        processor=processor,
        processor_version=processor_version,
        input_hash=input_hash,
        params_hash=ctx.params_hash(),
    )
    did = derivation_id(key)
    for segment in segments:
        segment.derivation_id = did

    stamp = fmt_ts(now_utc())
    header: dict[str, Any] = {
        **key,
        "id": did,
        "status": status,
        "created_at": stamp,
        "completed_at": stamp if status == "complete" else None,
    }
    # Only when measured. Every field the index holds must come from this
    # header, or a rebuild cannot reproduce the row -- which is precisely the
    # index-only state the derived-file header exists to prevent.
    if cost_ms is not None:
        header["cost_ms"] = cost_ms
    payload: dict[str, Any] = {
        "_derivation": header,
        "meta": meta,
        "segments": [segment_payload(s) for s in segments],
    }
    if extra:
        payload.update(extra)

    ctx.derived_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.derived_dir / f"{did}.json"
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    out.write_text(body, encoding="utf-8")

    artifact = DerivedArtifact(
        kind=kind,
        path=out,
        content_hash="sha256:" + hashlib.sha256(body.encode()).hexdigest(),
        cache_key=key,
    )
    return artifact, did


def file_hash(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a 2GB video does not load."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "derivation_id",
    "file_hash",
    "segment_payload",
    "write_derived",
]
