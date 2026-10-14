"""Domain types and the Markdown+YAML frontmatter codec.

Validation is hand-written rather than delegated to pydantic: there is one
validation surface, the field set is small and fully owned, and the
forward-compatible ``extra`` passthrough is easier without a model system
trying to own unknown keys.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

TITLE_MAX = 200


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    DECISION = "decision"
    PROCEDURE = "procedure"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class RelationType(StrEnum):
    SUPERSEDES = "supersedes"
    RELATES_TO = "relates_to"
    CAUSES = "causes"
    PART_OF = "part_of"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    REFERENCES = "references"


class SourceKind(StrEnum):
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    ASSET = "asset"
    MANUAL = "manual"
    IMPORT = "import"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    PRIVATE = "private"
    SECRET = "secret"


#: Ordering used for ``sensitivity_max`` comparisons.
SENSITIVITY_ORDER: dict[str, int] = {"normal": 0, "private": 1, "secret": 2}


class ValidationError(ValueError):
    """A memory (or asset) failed validation. Carries every problem found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


# ---------------------------------------------------------------- time handling

_EPOCH_RE = re.compile(r"^\d{9,11}(\.\d+)?$")


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def parse_ts(value: Any, *, field_name: str = "timestamp") -> datetime:
    """Parse RFC3339, date-only, or epoch seconds. Reject anything else loudly.

    Deliberately strict: timestamps feed ranking, so a silently misparsed value
    is worse than an error. This replaces a python-dateutil dependency.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), UTC)
    if not isinstance(value, str):
        raise ValidationError([f"{field_name}: expected a timestamp, got {type(value).__name__}"])

    raw = value.strip()
    if not raw:
        raise ValidationError([f"{field_name}: empty timestamp"])
    if _EPOCH_RE.match(raw):
        return datetime.fromtimestamp(float(raw), UTC)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise ValidationError(
            [f"{field_name}: not RFC3339, a date, or epoch seconds: {value!r}"]
        ) from None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def fmt_ts(value: datetime | None) -> str | None:
    """Render as RFC3339 UTC with a trailing Z -- lexicographically sortable."""
    if value is None:
        return None
    aware = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------- normalization

_WS_RE = re.compile(r"\s+")
_TAG_OK_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def clean_title(title: str) -> str:
    """Collapse a title to a single line.

    Titles are single-line by contract. A stray ``\r`` or newline would
    otherwise survive into YAML and re-quote differently on the next write,
    breaking round-trip stability.
    """
    return _WS_RE.sub(" ", title).strip()


def normalize_newlines(text: str) -> str:
    """Canonicalize line endings to LF.

    The parser normalizes CRLF and lone CR document-wide, so the writer must
    emit the same form or render -> parse -> render would not be a fixed point.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_tag(tag: str) -> str:
    return _WS_RE.sub("-", tag.strip().casefold()).strip("-")


def clean_tags(tags: list[str] | tuple[str, ...]) -> list[str]:
    """Lowercase, dedupe, sort. Sorted order keeps file diffs stable."""
    return sorted({normalize_tag(t) for t in tags if normalize_tag(t)})


def clean_entities(entities: list[str] | tuple[str, ...]) -> list[str]:
    """Entities keep their case (they are names) but are trimmed and deduped."""
    seen: dict[str, str] = {}
    for e in entities:
        stripped = _WS_RE.sub(" ", e.strip())
        if stripped:
            seen.setdefault(stripped.casefold(), stripped)
    return [seen[k] for k in sorted(seen)]


def new_id() -> str:
    return str(uuid.uuid4())


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def normalize_text(text: str) -> str:
    """Canonical form for content hashing: NFC, casefolded, whitespace collapsed."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def content_hash(title: str, body: str) -> str:
    """Resurrection guard #2: identity of *meaning*, not of bytes."""
    payload = f"{normalize_text(title)}\n{normalize_text(body)}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def title_hash(title: str) -> str:
    return "sha256:" + hashlib.sha256(normalize_text(title).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


# ---------------------------------------------------------------- nested records


@dataclass(slots=True)
class SourceRef:
    kind: SourceKind
    ref: str
    locator: str | None = None
    captured_at: datetime | None = None
    confidence: float | None = None

    @property
    def source_id(self) -> str:
        """Deterministic id so a rebuild regenerates identical rows."""
        payload = f"{self.kind.value}\x1f{self.ref}".encode()
        return hashlib.sha256(payload).hexdigest()[:32]

    def key(self) -> tuple[str, str, str]:
        return (self.kind.value, self.ref, self.locator or "")

    def to_yaml(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value, "ref": self.ref}
        if self.locator is not None:
            out["locator"] = self.locator
        if self.captured_at is not None:
            out["captured_at"] = fmt_ts(self.captured_at)
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out

    @classmethod
    def from_yaml(cls, raw: Any, *, problems: list[str], where: str) -> SourceRef | None:
        if not isinstance(raw, dict):
            problems.append(f"{where}: expected a mapping, got {type(raw).__name__}")
            return None
        kind_raw = raw.get("kind")
        ref = raw.get("ref")
        if kind_raw not in set(SourceKind):
            problems.append(f"{where}: unknown source kind {kind_raw!r}")
            return None
        if not isinstance(ref, str) or not ref.strip():
            problems.append(f"{where}: missing 'ref'")
            return None
        captured = raw.get("captured_at")
        conf = raw.get("confidence")
        if conf is not None and not isinstance(conf, int | float):
            problems.append(f"{where}: confidence must be a number")
            conf = None
        return cls(
            kind=SourceKind(kind_raw),
            ref=ref.strip(),
            locator=(str(raw["locator"]) if raw.get("locator") is not None else None),
            captured_at=(
                parse_ts(captured, field_name=f"{where}.captured_at") if captured else None
            ),
            confidence=(float(conf) if conf is not None else None),
        )


@dataclass(slots=True)
class Relation:
    type: RelationType
    target: str
    note: str | None = None

    def to_yaml(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type.value, "target": self.target}
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_yaml(cls, raw: Any, *, problems: list[str], where: str) -> Relation | None:
        if not isinstance(raw, dict):
            problems.append(f"{where}: expected a mapping, got {type(raw).__name__}")
            return None
        rtype = raw.get("type")
        target = raw.get("target")
        if rtype not in set(RelationType):
            problems.append(f"{where}: unknown relation type {rtype!r}")
            return None
        if not isinstance(target, str) or not is_uuid(target):
            problems.append(f"{where}: target must be a uuid, got {target!r}")
            return None
        return cls(
            type=RelationType(rtype),
            target=target,
            note=(str(raw["note"]) if raw.get("note") else None),
        )


# ---------------------------------------------------------------- memory

#: Frontmatter key order. Explicit so diffs stay stable across writes.
FRONTMATTER_ORDER: tuple[str, ...] = (
    "id",
    "schema_version",
    "type",
    "status",
    "title",
    "created_at",
    "updated_at",
    "occurred_at",
    "valid_from",
    "valid_until",
    "review_after",
    "confidence",
    "salience",
    "profile",
    "sensitivity",
    "tags",
    "entities",
    "assets",
    "superseded_by",
    "supersedes",
    "sources",
    "relations",
)

KNOWN_KEYS = frozenset(FRONTMATTER_ORDER)


@dataclass(slots=True)
class Memory:
    type: MemoryType
    title: str
    body: str = ""
    id: str = field(default_factory=new_id)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    occurred_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    review_after: datetime | None = None
    confidence: float = 0.8
    salience: float = 0.5
    profile: bool = False
    sensitivity: Sensitivity = Sensitivity.NORMAL
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # -------------------------------------------------------------- derived

    @property
    def content_hash(self) -> str:
        return content_hash(self.title, self.body)

    @property
    def title_hash(self) -> str:
        return title_hash(self.title)

    def source_keys(self) -> set[tuple[str, str, str]]:
        return {s.key() for s in self.sources}

    def with_updates(self, **changes: Any) -> Memory:
        return replace(self, **changes)

    def to_public(self) -> dict[str, Any]:
        """JSON-serializable form used by the CLI and MCP server."""
        out = dict(frontmatter_dict(self))
        out["body"] = self.body
        out["content_hash"] = self.content_hash
        return out


def _coerce_unit(value: Any, name: str, default: float, problems: list[str]) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        problems.append(f"{name}: expected a number in [0,1], got {value!r}")
        return default
    f = float(value)
    if not 0.0 <= f <= 1.0:
        problems.append(f"{name}: {f} is outside [0,1]")
        return min(max(f, 0.0), 1.0)
    return f


def validate(m: Memory) -> list[str]:
    """Return every problem found. Callers raise; ``verify`` reports."""
    problems: list[str] = []

    if not is_uuid(m.id):
        problems.append(f"id: not a uuid: {m.id!r}")
    if m.type not in set(MemoryType):
        problems.append(f"type: unknown {m.type!r}")
    if m.status not in set(MemoryStatus):
        problems.append(f"status: unknown {m.status!r}")
    if m.sensitivity not in set(Sensitivity):
        problems.append(f"sensitivity: unknown {m.sensitivity!r}")

    title = m.title.strip()
    if not title:
        problems.append("title: must not be empty")
    elif len(title) > TITLE_MAX:
        problems.append(f"title: {len(title)} chars exceeds {TITLE_MAX}")
    if "\n" in m.title:
        problems.append("title: must be a single line")

    if not 0.0 <= m.confidence <= 1.0:
        problems.append(f"confidence: {m.confidence} outside [0,1]")
    if not 0.0 <= m.salience <= 1.0:
        problems.append(f"salience: {m.salience} outside [0,1]")

    if m.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version: expected {SCHEMA_VERSION}, got {m.schema_version}")

    for t in m.tags:
        if not _TAG_OK_RE.match(t):
            problems.append(f"tags: {t!r} is not [a-z0-9][a-z0-9_-]*")

    if m.superseded_by is not None:
        if not is_uuid(m.superseded_by):
            problems.append(f"superseded_by: not a uuid: {m.superseded_by!r}")
        elif m.superseded_by == m.id:
            problems.append("superseded_by: must not point at itself")
    for sid in m.supersedes:
        if not is_uuid(sid):
            problems.append(f"supersedes: not a uuid: {sid!r}")
        elif sid == m.id:
            problems.append("supersedes: must not contain itself")
    for aid in m.assets:
        if not is_uuid(aid):
            problems.append(f"assets: not a uuid: {aid!r}")

    if m.valid_from and m.valid_until and m.valid_until < m.valid_from:
        problems.append("valid_until: precedes valid_from")

    for rel in m.relations:
        if rel.target == m.id:
            problems.append(f"relations: {rel.type.value} points at itself")

    return problems


def ensure_valid(m: Memory) -> Memory:
    problems = validate(m)
    if problems:
        raise ValidationError(problems)
    return m


# ---------------------------------------------------------------- codec

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(?P<yaml>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_H1_RE = re.compile(r"\A#[ \t]+(?P<title>[^\n]*)(?:\r?\n|\Z)")


def frontmatter_dict(m: Memory) -> dict[str, Any]:
    """Frontmatter mapping in canonical key order, unknown keys appended."""
    data: dict[str, Any] = {
        "id": m.id,
        "schema_version": m.schema_version,
        "type": m.type.value,
        "status": m.status.value,
        "title": clean_title(m.title),
        "created_at": fmt_ts(m.created_at),
        "updated_at": fmt_ts(m.updated_at),
        "occurred_at": fmt_ts(m.occurred_at),
        "valid_from": fmt_ts(m.valid_from),
        "valid_until": fmt_ts(m.valid_until),
        "review_after": fmt_ts(m.review_after),
        "confidence": round(m.confidence, 4),
        "salience": round(m.salience, 4),
        "profile": m.profile,
        "sensitivity": m.sensitivity.value,
        "tags": list(m.tags),
        "entities": list(m.entities),
        "assets": list(m.assets),
        "superseded_by": m.superseded_by,
        "supersedes": list(m.supersedes),
        "sources": [s.to_yaml() for s in m.sources],
        "relations": [r.to_yaml() for r in m.relations],
    }
    for key in sorted(m.extra):
        if key not in KNOWN_KEYS:
            data[key] = m.extra[key]
    return data


def render_memory_markdown(m: Memory) -> str:
    """Serialize to ``---\\nyaml\\n---\\n\\n# title\\n\\nbody``.

    The H1 is regenerated from the frontmatter title on every write so the
    file stays greppable; frontmatter remains authoritative on read.
    """
    front = yaml.safe_dump(
        frontmatter_dict(m),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=10_000,
    ).rstrip("\n")
    body = normalize_newlines(m.body).strip("\n")
    parts = ["---", front, "---", "", f"# {clean_title(m.title)}", ""]
    if body:
        parts += [body, ""]
    return "\n".join(parts)


def parse_memory_markdown(text: str, *, path: Path | None = None) -> Memory:
    """Parse a memory file. Raises ``ValidationError`` listing every problem."""
    where = f" in {path}" if path else ""
    problems: list[str] = []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONTMATTER_RE.match(normalized)
    if not match:
        raise ValidationError([f"missing YAML frontmatter{where}"])

    try:
        loaded = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:
        raise ValidationError([f"invalid YAML{where}: {exc}"]) from None
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValidationError([f"frontmatter must be a mapping{where}"])

    rest = normalized[match.end() :]
    h1 = _H1_RE.match(rest.lstrip("\n"))
    if h1:
        stripped = rest.lstrip("\n")
        rest = stripped[h1.end() :]
    body = rest.strip("\n")

    # ---- required scalars
    mid = loaded.get("id")
    if not isinstance(mid, str) or not is_uuid(mid):
        problems.append(f"id: not a uuid{where}: {mid!r}")
        mid = new_id()

    type_raw = loaded.get("type")
    if type_raw not in set(MemoryType):
        # A bad enum is fatal: we cannot place the file or rank it.
        raise ValidationError([f"type: unknown {type_raw!r}{where}"])
    mem_type = MemoryType(type_raw)

    status_raw = loaded.get("status", MemoryStatus.ACTIVE.value)
    if status_raw not in set(MemoryStatus):
        raise ValidationError([f"status: unknown {status_raw!r}{where}"])
    status = MemoryStatus(status_raw)

    sens_raw = loaded.get("sensitivity", Sensitivity.NORMAL.value)
    if sens_raw not in set(Sensitivity):
        raise ValidationError([f"sensitivity: unknown {sens_raw!r}{where}"])
    sensitivity = Sensitivity(sens_raw)

    # Frontmatter title wins over a divergent H1.
    title_raw = loaded.get("title")
    if isinstance(title_raw, str) and title_raw.strip():
        title = clean_title(title_raw)
    elif h1:
        title = clean_title(h1.group("title"))
    else:
        problems.append(f"title: missing{where}")
        title = ""

    def _ts(key: str, *, default: datetime | None = None) -> datetime | None:
        raw = loaded.get(key)
        if raw is None:
            return default
        try:
            return parse_ts(raw, field_name=key)
        except ValidationError as exc:
            problems.extend(f"{p}{where}" for p in exc.problems)
            return default

    created = _ts("created_at", default=now_utc()) or now_utc()
    updated = _ts("updated_at", default=created) or created

    def _str_list(key: str) -> list[str]:
        raw = loaded.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            problems.append(f"{key}: expected a list{where}")
            return []
        return [str(x) for x in raw]

    sources: list[SourceRef] = []
    raw_sources = loaded.get("sources") or []
    if not isinstance(raw_sources, list):
        problems.append(f"sources: expected a list{where}")
    else:
        for i, raw in enumerate(raw_sources):
            parsed = SourceRef.from_yaml(raw, problems=problems, where=f"sources[{i}]{where}")
            if parsed:
                sources.append(parsed)

    relations: list[Relation] = []
    raw_relations = loaded.get("relations") or []
    if not isinstance(raw_relations, list):
        problems.append(f"relations: expected a list{where}")
    else:
        for i, raw in enumerate(raw_relations):
            parsed_rel = Relation.from_yaml(raw, problems=problems, where=f"relations[{i}]{where}")
            if parsed_rel:
                relations.append(parsed_rel)

    superseded_by = loaded.get("superseded_by")
    if superseded_by is not None and not is_uuid(str(superseded_by)):
        problems.append(f"superseded_by: not a uuid{where}: {superseded_by!r}")
        superseded_by = None

    schema_version = loaded.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        problems.append(f"schema_version: expected an int{where}")
        schema_version = SCHEMA_VERSION

    profile_raw = loaded.get("profile", False)
    if not isinstance(profile_raw, bool):
        problems.append(f"profile: expected a boolean{where}")
        profile_raw = bool(profile_raw)

    extra = {k: v for k, v in loaded.items() if k not in KNOWN_KEYS}

    memory = Memory(
        id=mid,
        type=mem_type,
        status=status,
        title=title,
        body=body,
        created_at=created,
        updated_at=updated,
        occurred_at=_ts("occurred_at"),
        valid_from=_ts("valid_from"),
        valid_until=_ts("valid_until"),
        review_after=_ts("review_after"),
        confidence=_coerce_unit(loaded.get("confidence"), f"confidence{where}", 0.8, problems),
        salience=_coerce_unit(loaded.get("salience"), f"salience{where}", 0.5, problems),
        profile=profile_raw,
        sensitivity=sensitivity,
        tags=clean_tags(_str_list("tags")),
        entities=clean_entities(_str_list("entities")),
        assets=_str_list("assets"),
        superseded_by=(str(superseded_by) if superseded_by else None),
        supersedes=_str_list("supersedes"),
        sources=sources,
        relations=relations,
        extra=extra,
        schema_version=schema_version,
    )

    if problems:
        raise ValidationError(problems)
    return memory


# ---------------------------------------------------------------- assets


@dataclass(slots=True)
class Asset:
    """A thing in the user's world.

    ``id`` identifies the thing and is stable across ``replace_asset``;
    ``sha256`` identifies the current bytes and changes when content does.
    Memories reference ``id`` only, so replacing content never breaks a link.
    """

    sha256: str
    media_type: str
    bytes: int
    id: str = field(default_factory=new_id)
    original_filename: str | None = None
    created_at: datetime | None = None
    imported_at: datetime = field(default_factory=now_utc)
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source: SourceRef | None = None
    prev_sha256: str | None = None
    version: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": SCHEMA_VERSION,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "bytes": self.bytes,
            "original_filename": self.original_filename,
            "created_at": fmt_ts(self.created_at),
            "imported_at": fmt_ts(self.imported_at),
            "status": self.status,
            "tags": list(self.tags),
            "meta": self.meta,
            "source": self.source.to_yaml() if self.source else None,
            "prev_sha256": self.prev_sha256,
            "version": self.version,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, path: Path | None = None) -> Asset:
        where = f" in {path}" if path else ""
        problems: list[str] = []
        aid = raw.get("id")
        if not isinstance(aid, str) or not is_uuid(aid):
            problems.append(f"id: not a uuid{where}: {aid!r}")
        sha = raw.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            problems.append(f"sha256: expected 64 hex chars{where}: {sha!r}")
        size = raw.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            problems.append(f"bytes: expected a non-negative int{where}")
        status = raw.get("status", "active")
        if status not in ("active", "deleted"):
            problems.append(f"status: unknown {status!r}{where}")
        if problems:
            raise ValidationError(problems)

        source = None
        if raw.get("source"):
            source = SourceRef.from_yaml(raw["source"], problems=problems, where=f"source{where}")
        if problems:
            raise ValidationError(problems)

        return cls(
            id=str(aid),
            sha256=str(sha),
            media_type=str(raw.get("media_type") or "application/octet-stream"),
            bytes=int(size),  # type: ignore[arg-type]
            original_filename=(
                str(raw["original_filename"]) if raw.get("original_filename") else None
            ),
            created_at=(parse_ts(raw["created_at"]) if raw.get("created_at") else None),
            imported_at=(parse_ts(raw["imported_at"]) if raw.get("imported_at") else now_utc()),
            status=str(status),
            tags=clean_tags(list(raw.get("tags") or [])),
            meta=dict(raw.get("meta") or {}),
            source=source,
            prev_sha256=(str(raw["prev_sha256"]) if raw.get("prev_sha256") else None),
            version=int(raw.get("version") or 1),
        )


@dataclass(slots=True)
class Segment:
    """A retrievable span of a source or asset. Populated by processors."""

    text: str
    ordinal: int
    id: str = field(default_factory=new_id)
    asset_id: str | None = None
    source_id: str | None = None
    derivation_id: str | None = None
    memory_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    page: int | None = None
    bbox: str | None = None
    speaker: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Tombstone:
    """Deletion record. Never removed -- the only defense against resurrection."""

    memory_id: str
    deleted_at: datetime
    permanent: bool
    reason: str | None = None
    type: str | None = None
    title_hash: str | None = None
    content_hash: str | None = None
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "schema_version": self.schema_version,
            "deleted_at": fmt_ts(self.deleted_at),
            "permanent": self.permanent,
            "reason": self.reason,
            "type": self.type,
            "title_hash": self.title_hash,
            "content_hash": self.content_hash,
            "source_refs": self.source_refs,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, path: Path | None = None) -> Tombstone:
        where = f" in {path}" if path else ""
        mid = raw.get("memory_id")
        if not isinstance(mid, str) or not is_uuid(mid):
            raise ValidationError([f"memory_id: not a uuid{where}: {mid!r}"])
        return cls(
            memory_id=mid,
            deleted_at=parse_ts(raw.get("deleted_at") or now_utc()),
            permanent=bool(raw.get("permanent", False)),
            reason=(str(raw["reason"]) if raw.get("reason") else None),
            type=(str(raw["type"]) if raw.get("type") else None),
            title_hash=(str(raw["title_hash"]) if raw.get("title_hash") else None),
            content_hash=(str(raw["content_hash"]) if raw.get("content_hash") else None),
            source_refs=list(raw.get("source_refs") or []),
            schema_version=int(raw.get("schema_version") or SCHEMA_VERSION),
        )

    def source_keys(self) -> set[tuple[str, str, str]]:
        keys: set[tuple[str, str, str]] = set()
        for ref in self.source_refs:
            if isinstance(ref, dict) and ref.get("ref"):
                keys.add(
                    (
                        str(ref.get("kind") or ""),
                        str(ref["ref"]),
                        str(ref.get("locator") or ""),
                    )
                )
        return keys


def tombstone_for(m: Memory, *, permanent: bool, reason: str | None = None) -> Tombstone:
    return Tombstone(
        memory_id=m.id,
        deleted_at=now_utc(),
        permanent=permanent,
        reason=reason,
        type=m.type.value,
        title_hash=m.title_hash,
        content_hash=m.content_hash,
        source_refs=[s.to_yaml() for s in m.sources],
    )
