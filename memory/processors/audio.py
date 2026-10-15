"""Audio processor: timestamped transcripts via faster-whisper.

The one genuinely expensive processor, hence ``cost = "expensive"``: a
40-minute recording is minutes of CPU, which is exactly what the deferred
queue lane exists for. Nothing here runs inline during ``mem ingest`` unless
the caller asks for it.

Segments carry ``start_ms``/``end_ms``, so a retrieval hit can cite the
moment it came from and a player can seek there. Whisper's own segment
boundaries are kept rather than re-chunked on character count, because they
follow speech pauses and a timestamp is only meaningful if it bounds the text
it was computed for.

Without faster-whisper the processor reports duration (from ffprobe, itself
optional) and extracts no text.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from .. import media
from ..models import Segment
from .base import Cost, ProcessContext, ProcessResult
from .derive import file_hash, write_derived

#: Chosen for a laptop: "base" transcribes faster than real time on CPU and
#: is accurate enough for retrieval, which needs the right words, not
#: publication-quality punctuation.
DEFAULT_MODEL = "base"


class AudioProcessor:
    name: str = "audio"
    version: str = "1.0"
    cost: Cost = "expensive"
    media_types: tuple[str, ...] = (
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/flac",
        "audio/ogg",
        "audio/opus",
        "audio/aac",
        "audio/*",
    )
    extensions: tuple[str, ...] = (
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
        ".aiff",
    )

    def can_handle(self, media_type: str, path: Path) -> bool:
        if media_type.startswith("audio/"):
            return True
        return path.suffix.lower() in self.extensions

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        degraded = media.Degradation()
        meta: dict[str, Any] = {"bytes": path.stat().st_size}
        meta.update(probe_media(path, degraded))

        text, segments, transcript_meta = transcribe(path, ctx, degraded)
        meta.update(transcript_meta)
        meta["segments"] = len(segments)
        meta.update(degraded.to_meta())

        derived = []
        artifact, _ = write_derived(
            ctx=ctx,
            kind="transcript",
            processor=self.name,
            processor_version=self.version,
            input_hash=file_hash(path),
            segments=segments,
            meta=meta,
        )
        if artifact is not None:
            derived.append(artifact)

        return ProcessResult(text=text or None, segments=segments, derived=derived, meta=meta)


# ---------------------------------------------------------------- probing


def probe_media(path: Path, degraded: media.Degradation) -> dict[str, Any]:
    """Duration, codec and channels via ffprobe. Optional, like everything."""
    cap = media.ffprobe()
    if not cap.available:
        degraded.note(cap, doing="reading audio duration")
        return {}

    argv = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_name,codec_type,channels,sample_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = media.run_binary(argv, timeout=60.0)
    except Exception as exc:  # includes TimeoutExpired
        degraded.add(f"ffprobe failed ({exc})")
        return {}
    if proc.returncode != 0:
        degraded.add(f"ffprobe exited {proc.returncode}")
        return {}

    try:
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}

    out: dict[str, Any] = {}
    fmt = payload.get("format") or {}
    if fmt.get("duration"):
        with contextlib.suppress(TypeError, ValueError):
            out["duration_ms"] = int(float(fmt["duration"]) * 1000)
    if fmt.get("bit_rate"):
        with contextlib.suppress(TypeError, ValueError):
            out["bit_rate"] = int(fmt["bit_rate"])
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") == "audio":
            out["codec"] = str(stream.get("codec_name") or "")
            if stream.get("channels") is not None:
                out["channels"] = int(stream["channels"])
            if stream.get("sample_rate"):
                with contextlib.suppress(TypeError, ValueError):
                    out["sample_rate"] = int(stream["sample_rate"])
            break
    return out


# ---------------------------------------------------------------- transcription


def transcribe(
    path: Path,
    ctx: ProcessContext,
    degraded: media.Degradation,
    *,
    asset_id: str | None = None,
) -> tuple[str, list[Segment], dict[str, Any]]:
    """Transcribe to timestamped segments.

    Shared with the video processor, which extracts an audio track and calls
    straight in here rather than duplicating the model handling.
    """
    owner = asset_id or ctx.asset_id
    whisper = media.load("faster_whisper")
    if whisper is None:
        degraded.note(media.python_module("faster_whisper"), doing="transcribing audio")
        return "", [], {"transcription": "unavailable"}

    model_name = str(ctx.params.get("model", DEFAULT_MODEL))
    try:
        model = whisper.WhisperModel(
            model_name,
            device=str(ctx.params.get("device", "cpu")),
            compute_type=str(ctx.params.get("compute_type", "int8")),
        )
    except Exception as exc:  # a missing model download lands here
        degraded.add(f"whisper model {model_name!r} could not be loaded ({exc})")
        return "", [], {"transcription": "failed", "model": model_name}

    language = ctx.params.get("language")
    try:
        raw_segments, info = model.transcribe(
            str(path),
            language=language if isinstance(language, str) else None,
            vad_filter=bool(ctx.params.get("vad_filter", True)),
            beam_size=int(ctx.params.get("beam_size", 5)),
        )
        # The generator does the actual work, so it must be drained inside
        # the try: a decode error surfaces here, not at transcribe().
        collected = list(raw_segments)
    except Exception as exc:
        degraded.add(f"transcription failed ({exc})")
        return "", [], {"transcription": "failed", "model": model_name}

    segments: list[Segment] = []
    parts: list[str] = []
    for item in collected:
        body = str(getattr(item, "text", "") or "").strip()
        if not body:
            continue
        parts.append(body)
        segments.append(
            Segment(
                text=body,
                ordinal=len(segments),
                asset_id=owner,
                source_id=ctx.source_id,
                start_ms=_ms(getattr(item, "start", None)),
                end_ms=_ms(getattr(item, "end", None)),
                meta=_segment_meta(item),
            )
        )

    meta: dict[str, Any] = {"transcription": "complete", "model": model_name}
    detected = getattr(info, "language", None)
    if detected:
        meta["language"] = str(detected)
    probability = getattr(info, "language_probability", None)
    if isinstance(probability, int | float):
        meta["language_probability"] = round(float(probability), 4)
    duration = getattr(info, "duration", None)
    if isinstance(duration, int | float):
        meta["duration_ms"] = int(float(duration) * 1000)
    if not segments:
        meta["transcription"] = "empty"

    return " ".join(parts), segments, meta


def _ms(value: Any) -> int | None:
    if not isinstance(value, int | float):
        return None
    return int(float(value) * 1000)


def _segment_meta(item: Any) -> dict[str, Any]:
    """Keep whisper's own confidence signals; they are useful when hedging."""
    out: dict[str, Any] = {}
    for attr, key in (
        ("avg_logprob", "avg_logprob"),
        ("no_speech_prob", "no_speech_prob"),
        ("compression_ratio", "compression_ratio"),
    ):
        value = getattr(item, attr, None)
        if isinstance(value, int | float):
            out[key] = round(float(value), 4)
    return out


__all__ = ["DEFAULT_MODEL", "AudioProcessor", "probe_media", "transcribe"]
