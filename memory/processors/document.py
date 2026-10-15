"""Document processor: PDF and docx text with structure preserved.

Structure is the point. A PDF segment carries its ``page``, so a retrieval
hit cites "page 7" rather than an offset into a wall of text, and a docx
segment carries the heading it fell under. Both are cheap to capture at
extraction time and impossible to recover afterwards.

pypdf and python-docx are optional. Without them the processor reports the
format it recognized and extracts nothing, which keeps the asset stored and
re-processable once the extra is installed.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .. import media
from ..models import Segment
from .base import Cost, ProcessContext, ProcessResult, segments_from_text
from .derive import file_hash, write_derived

PDF_TYPES = ("application/pdf",)
DOCX_TYPES = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
)


class DocumentProcessor:
    name: str = "document"
    version: str = "1.0"
    #: A 400-page PDF is seconds, not minutes, and the result is needed
    #: immediately for the document to be searchable at all.
    cost: Cost = "cheap"
    media_types: tuple[str, ...] = (*PDF_TYPES, *DOCX_TYPES)
    extensions: tuple[str, ...] = (".pdf", ".docx")

    def can_handle(self, media_type: str, path: Path) -> bool:
        if media_type in self.media_types:
            return True
        return path.suffix.lower() in self.extensions

    def process(self, path: Path, ctx: ProcessContext) -> ProcessResult:
        degraded = media.Degradation()
        suffix = path.suffix.lower()
        is_pdf = ctx.media_type in PDF_TYPES or suffix == ".pdf"

        if is_pdf:
            text, segments, meta = self._pdf(path, ctx, degraded)
            meta["format"] = "pdf"
        else:
            text, segments, meta = self._docx(path, ctx, degraded)
            meta["format"] = "docx"

        meta["bytes"] = path.stat().st_size
        meta["segments"] = len(segments)
        meta["characters"] = len(text)
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

    # ------------------------------------------------------------------ pdf

    def _pdf(
        self, path: Path, ctx: ProcessContext, degraded: media.Degradation
    ) -> tuple[str, list[Segment], dict[str, Any]]:
        pypdf = media.load("pypdf")
        if pypdf is None:
            degraded.note(media.python_module("pypdf"), doing="extracting PDF text")
            return "", [], {"extraction": "unavailable"}

        meta: dict[str, Any] = {}
        try:
            reader = pypdf.PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                # An empty password opens the common "protected from editing"
                # case; a real password is not ours to guess.
                with contextlib.suppress(Exception):
                    reader.decrypt("")
            pages = list(reader.pages)
        except Exception as exc:  # a malformed PDF is data
            degraded.add(f"PDF could not be read ({exc}); stored without extracted text")
            return "", [], {"extraction": "failed"}

        meta["pages"] = len(pages)
        meta.update(self._pdf_info(reader))

        page_texts: list[tuple[int, str]] = []
        for number, page in enumerate(pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:  # one bad page must not lose the rest
                degraded.add(f"page {number} could not be extracted")
                continue
            cleaned = _normalize(raw)
            if cleaned.strip():
                page_texts.append((number, cleaned))

        # Segment per page rather than across the whole document, so every
        # segment has exactly one page number to cite.
        segments: list[Segment] = []
        target = int(ctx.params.get("target_chars", 1200))
        overlap = int(ctx.params.get("overlap_chars", 100))
        for number, body in page_texts:
            for piece in segments_from_text(
                body,
                asset_id=ctx.asset_id,
                source_id=ctx.source_id,
                target_chars=target,
                overlap_chars=overlap,
            ):
                piece.ordinal = len(segments)
                piece.page = number
                segments.append(piece)

        meta["pages_with_text"] = len(page_texts)
        if pages and not page_texts:
            # The overwhelmingly common cause, and the actionable one.
            degraded.add("no text layer found; this PDF is probably scanned images and needs OCR")
            meta["extraction"] = "empty"
        else:
            meta["extraction"] = "complete"

        return "\n\n".join(body for _, body in page_texts), segments, meta

    def _pdf_info(self, reader: Any) -> dict[str, Any]:
        try:
            info = reader.metadata
        except Exception:
            return {}
        if not info:
            return {}
        out: dict[str, Any] = {}
        for key, target in (
            ("/Title", "title"),
            ("/Author", "author"),
            ("/Subject", "subject"),
            ("/Creator", "creator"),
            ("/Producer", "producer"),
            ("/CreationDate", "created_raw"),
        ):
            try:
                value = info.get(key)
            except Exception:
                continue
            if value:
                out[target] = str(value)
        return out

    # ------------------------------------------------------------------ docx

    def _docx(
        self, path: Path, ctx: ProcessContext, degraded: media.Degradation
    ) -> tuple[str, list[Segment], dict[str, Any]]:
        docx = media.load("docx")
        if docx is None:
            degraded.note(media.python_module("docx"), doing="extracting docx text")
            return "", [], {"extraction": "unavailable"}

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            degraded.add(f"docx could not be read ({exc}); stored without extracted text")
            return "", [], {"extraction": "failed"}

        meta: dict[str, Any] = {}
        blocks: list[str] = []
        headings: list[str] = []
        current_heading: str | None = None
        #: Paragraph text keyed by the heading in force, so each segment can
        #: name its section.
        sections: list[tuple[str | None, str]] = []

        for para in document.paragraphs:
            body = _normalize(para.text).strip()
            if not body:
                continue
            style = str(getattr(getattr(para, "style", None), "name", "") or "")
            if style.startswith("Heading") or style == "Title":
                current_heading = body
                headings.append(body)
                blocks.append(f"# {body}" if style == "Title" else f"## {body}")
                sections.append((current_heading, body))
                continue
            blocks.append(body)
            sections.append((current_heading, body))

        table_count = 0
        for table in getattr(document, "tables", []):
            table_count += 1
            for row in table.rows:
                cells = [_normalize(c.text).strip() for c in row.cells]
                line = " | ".join(c for c in cells if c)
                if line:
                    blocks.append(line)
                    sections.append((current_heading, line))

        meta["paragraphs"] = len(document.paragraphs)
        meta["tables"] = table_count
        if headings:
            meta["headings"] = headings
        meta.update(self._docx_props(document))
        meta["extraction"] = "complete" if blocks else "empty"

        # Group consecutive paragraphs under one heading, then segment each
        # group, so a segment never straddles two sections.
        segments: list[Segment] = []
        target = int(ctx.params.get("target_chars", 1200))
        overlap = int(ctx.params.get("overlap_chars", 100))
        for heading, body in _group_sections(sections):
            for piece in segments_from_text(
                body,
                asset_id=ctx.asset_id,
                source_id=ctx.source_id,
                target_chars=target,
                overlap_chars=overlap,
            ):
                piece.ordinal = len(segments)
                if heading:
                    piece.meta = {**piece.meta, "heading": heading}
                segments.append(piece)

        return "\n\n".join(blocks), segments, meta

    def _docx_props(self, document: Any) -> dict[str, Any]:
        try:
            props = document.core_properties
        except Exception:
            return {}
        out: dict[str, Any] = {}
        for attr, target in (
            ("title", "title"),
            ("author", "author"),
            ("subject", "subject"),
            ("last_modified_by", "last_modified_by"),
        ):
            try:
                value = getattr(props, attr, None)
            except Exception:
                continue
            if value:
                out[target] = str(value)
        for attr, target in (("created", "created"), ("modified", "modified")):
            try:
                value = getattr(props, attr, None)
            except Exception:
                continue
            if value is not None:
                out[target] = str(value)
        return out


def _group_sections(sections: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    grouped: list[tuple[str | None, list[str]]] = []
    for heading, body in sections:
        if grouped and grouped[-1][0] == heading:
            grouped[-1][1].append(body)
        else:
            grouped.append((heading, [body]))
    return [(heading, "\n\n".join(bodies)) for heading, bodies in grouped]


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")


__all__ = ["DocumentProcessor"]
