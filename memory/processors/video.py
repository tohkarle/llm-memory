"""Video processor: audio track to transcript, plus keyframes.

Pipeline: ffmpeg extracts a mono 16kHz WAV to a temporary file, the audio
processor's transcriber turns it into timestamped segments, and ffmpeg writes
keyframe JPEGs under ``data/derived/keyframes/``. The temporary WAV is always
deleted -- it is reconstructible from the original, so keeping it would be
hundreds of megabytes of redundant storage.

The original video is never touched: every ffmpeg invocation reads it and
writes somewhere else.

Keyframes exist to make a video answerable visually ("the slide about
pricing") once Phase 4 can describe them. They are extracted here because
ffmpeg is already a dependency at this point and doing it later would mean a
second pass over the file.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .. import media
from ..models import Segment
from .audio import transcribe
from .base import Cost, ProcessContext, ProcessResult
from .derive import file_hash, write_derived

#: One frame every 30s. Dense enough to catch a slide change, sparse enough
#: that an hour-long video yields 120 images rather than 90,000.
DEFAULT_KEYFRAME_INTERVAL_S = 30.0

#: Guards against a pathological input filling the disk.
MAX_KEYFRAMES = 240


class VideoProcessor:
    name: str = "video"
    version: str = "1.0"
    cost: Cost = "expensive"
    media_types: tuple[str, ...] = (
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "video/x-msvideo",
        "video/mpeg",
        "video/*",
    )
    extensions: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
        ".mpeg",
        ".mpg",
    )

    def can_handle(self, media_type: str, path: Path) -> bool:
        if media_type.startswith("video/"):
            return True
        return path.suffix.lower() in self.extensions

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        degraded = media.Degradation()
        meta: dict[str, Any] = {"bytes": path.stat().st_size}
        meta.update(probe_video(path, degraded))

        text, segments, transcript_meta = self._transcribe_audio_track(path, ctx, degraded)
        meta.update(transcript_meta)

        keyframes = self._keyframes(path, ctx, degraded, duration_ms=meta.get("duration_ms"))
        if keyframes:
            meta["keyframes"] = keyframes

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

    # ------------------------------------------------------------------ audio

    def _transcribe_audio_track(
        self, path: Path, ctx: ProcessContext, degraded: media.Degradation
    ) -> tuple[str, list[Segment], dict[str, Any]]:
        cap = media.ffmpeg()
        if not cap.available:
            degraded.note(cap, doing="extracting the audio track")
            return "", [], {"transcription": "unavailable", "audio_extraction": "unavailable"}

        with tempfile.TemporaryDirectory(prefix="mem-video-") as tmp:
            wav = Path(tmp) / "audio.wav"
            argv = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(path),
                "-vn",  # drop video: whisper wants audio only
                "-ac",
                "1",  # mono
                "-ar",
                "16000",  # whisper's native rate, so it does not resample
                "-f",
                "wav",
                str(wav),
            ]
            try:
                proc = media.run_binary(argv, timeout=float(ctx.params.get("ffmpeg_timeout", 1800)))
            except Exception as exc:  # includes TimeoutExpired
                degraded.add(f"audio extraction failed ({exc})")
                return "", [], {"transcription": "failed", "audio_extraction": "failed"}

            if proc.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
                tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
                reason = tail[0] if tail else f"exit {proc.returncode}"
                # A video with no audio stream is a legitimate input, not a
                # failure: it simply has no transcript.
                if "does not contain any stream" in reason or "Output file #0 does not" in reason:
                    return "", [], {"transcription": "empty", "audio_extraction": "no_audio"}
                degraded.add(f"audio extraction failed ({reason})")
                return "", [], {"transcription": "failed", "audio_extraction": "failed"}

            text, segments, meta = transcribe(wav, ctx, degraded, asset_id=ctx.asset_id)
            meta["audio_extraction"] = "complete"
            return text, segments, meta

    # ------------------------------------------------------------------ keyframes

    def _keyframes(
        self,
        path: Path,
        ctx: ProcessContext,
        degraded: media.Degradation,
        *,
        duration_ms: Any = None,
    ) -> list[dict[str, Any]]:
        if not ctx.params.get("keyframes", True):
            return []
        cap = media.ffmpeg()
        if not cap.available:
            degraded.note(cap, doing="extracting keyframes")
            return []
        if ctx.derived_dir is None or ctx.asset_id is None:
            return []

        # Keyframes are images, so they belong under derived/keyframes/,
        # not beside the transcript JSON.
        out_dir = _keyframe_dir(ctx)
        if out_dir is None:
            return []
        # Rewrite rather than merge: a partial previous run would otherwise
        # leave frames that no longer correspond to the current parameters.
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        interval = float(ctx.params.get("keyframe_interval_s", DEFAULT_KEYFRAME_INTERVAL_S))
        if interval <= 0:
            interval = DEFAULT_KEYFRAME_INTERVAL_S
        argv = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(path),
            "-vf",
            f"fps=1/{interval}",
            "-frames:v",
            str(int(ctx.params.get("max_keyframes", MAX_KEYFRAMES))),
            "-q:v",
            "4",
            str(out_dir / "frame-%04d.jpg"),
        ]
        try:
            proc = media.run_binary(argv, timeout=float(ctx.params.get("ffmpeg_timeout", 1800)))
        except Exception as exc:
            degraded.add(f"keyframe extraction failed ({exc})")
            return []
        if proc.returncode != 0:
            degraded.add(f"keyframe extraction exited {proc.returncode}")

        frames: list[dict[str, Any]] = []
        for index, frame in enumerate(sorted(out_dir.glob("frame-*.jpg"))):
            frames.append(
                {
                    "ordinal": index,
                    "filename": frame.name,
                    # ffmpeg's fps filter samples the midpoint of each
                    # interval, so frame N is at N * interval seconds.
                    "at_ms": int(index * interval * 1000),
                    "bytes": frame.stat().st_size,
                }
            )
        if not frames:
            shutil.rmtree(out_dir, ignore_errors=True)
        return frames


# ---------------------------------------------------------------- probing


def probe_video(path: Path, degraded: media.Degradation) -> dict[str, Any]:
    """Duration, dimensions and codecs. Needs ffprobe, which is optional."""
    cap = media.ffprobe()
    if not cap.available:
        degraded.note(cap, doing="reading video metadata")
        return {}

    argv = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=codec_name,codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = media.run_binary(argv, timeout=120.0)
    except Exception as exc:
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
    has_audio = False
    for stream in payload.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and "width" not in out:
            if stream.get("width"):
                out["width"] = int(stream["width"])
            if stream.get("height"):
                out["height"] = int(stream["height"])
            out["video_codec"] = str(stream.get("codec_name") or "")
            fps = _parse_fps(stream.get("r_frame_rate"))
            if fps is not None:
                out["fps"] = fps
        elif kind == "audio":
            has_audio = True
            out["audio_codec"] = str(stream.get("codec_name") or "")
    out["has_audio"] = has_audio
    return out


def _parse_fps(raw: Any) -> float | None:
    """ffprobe reports frame rate as the fraction ``"30000/1001"``."""
    if not isinstance(raw, str) or "/" not in raw:
        return None
    num, _, den = raw.partition("/")
    try:
        denominator = float(den)
        if denominator == 0:
            return None
        return round(float(num) / denominator, 3)
    except ValueError:
        return None


def _keyframe_dir(ctx: ProcessContext) -> Path | None:
    """Sibling keyframes directory for this asset.

    Derived from the transcript directory rather than taking another
    constructor argument, so the processor keeps the single-``derived_dir``
    contract the protocol defines. ``paths.derived_dir`` builds
    ``derived/<kind>/<ab>/<asset_id>``, so swapping the kind component
    reaches the matching keyframes directory.
    """
    if ctx.derived_dir is None:
        return None
    parts = list(ctx.derived_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in ("transcripts", "descriptions", "keyframes"):
            parts[index] = "keyframes"
            return Path(*parts)
    return ctx.derived_dir / "keyframes"


__all__ = [
    "DEFAULT_KEYFRAME_INTERVAL_S",
    "MAX_KEYFRAMES",
    "VideoProcessor",
    "probe_video",
]
