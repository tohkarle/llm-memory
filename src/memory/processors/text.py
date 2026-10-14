"""Plain-text and Markdown processor.

Also the worked example for Phase 3: it decodes, extracts text, segments it,
and writes a derived file carrying its own cache key. Media processors follow
the same shape with heavier extraction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..models import fmt_ts, now_utc
from .base import (
    Cost,
    DerivedArtifact,
    ProcessContext,
    ProcessResult,
    cache_key,
    segments_from_text,
)

#: Tried in order. UTF-8 first because it is almost always right; latin-1
#: last because it cannot fail, so decoding always succeeds rather than
#: losing a file to a mis-declared encoding.
ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "latin-1")


class TextProcessor:
    name: str = "text"
    version: str = "1.0"
    cost: Cost = "cheap"
    media_types: tuple[str, ...] = (
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "application/json",
        "text/*",
    )
    extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".rst",
        ".log",
    )

    def can_handle(self, media_type: str, path: Path) -> bool:
        if media_type in self.media_types:
            return True
        if media_type.startswith("text/"):
            return True
        return path.suffix.lower() in self.extensions

    def decode(self, raw: bytes) -> tuple[str, str]:
        for encoding in ENCODINGS:
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        return raw.decode("latin-1", errors="replace"), "latin-1"

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        raw = path.read_bytes()
        decoded, encoding = self.decode(raw)
        # Normalize at the boundary so everything downstream -- extracted
        # text, segments, derived files -- sees one line-ending convention.
        text = decoded.replace("\r\n", "\n").replace("\r", "\n")
        input_hash = hashlib.sha256(raw).hexdigest()

        segments = segments_from_text(
            text,
            asset_id=ctx.asset_id,
            source_id=ctx.source_id,
            target_chars=int(ctx.params.get("target_chars", 1200)),
            overlap_chars=int(ctx.params.get("overlap_chars", 100)),
        )

        meta = {
            "encoding": encoding,
            "bytes": len(raw),
            "characters": len(text),
            "lines": text.count("\n") + 1 if text else 0,
            "segments": len(segments),
        }

        derived: list[DerivedArtifact] = []
        if ctx.derived_dir is not None and segments:
            key = cache_key(
                asset_id=ctx.asset_id,
                kind="transcript",
                processor=self.name,
                processor_version=self.version,
                input_hash=input_hash,
                params_hash=ctx.params_hash(),
            )
            derivation_id = _derivation_id(key)
            for segment in segments:
                segment.derivation_id = derivation_id

            stamp = fmt_ts(now_utc())
            payload = {
                "_derivation": {
                    **key,
                    "id": derivation_id,
                    "status": "complete",
                    "created_at": stamp,
                    "completed_at": stamp,
                },
                "meta": meta,
                "segments": [{"id": s.id, "ordinal": s.ordinal, "text": s.text} for s in segments],
            }
            ctx.derived_dir.mkdir(parents=True, exist_ok=True)
            out = ctx.derived_dir / f"{derivation_id}.json"
            body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            out.write_text(body, encoding="utf-8")
            derived.append(
                DerivedArtifact(
                    kind="transcript",
                    path=out,
                    content_hash="sha256:" + hashlib.sha256(body.encode()).hexdigest(),
                    cache_key=key,
                )
            )

        return ProcessResult(text=text, segments=segments, derived=derived, meta=meta)


def _derivation_id(key: dict[str, object]) -> str:
    """Derive a stable id from the cache key.

    Deterministic so re-processing identical input with identical parameters
    resolves to the same derivation rather than accumulating duplicates.
    """
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
