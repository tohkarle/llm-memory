"""Backup, export, import, and retention. The filesystem stays authoritative.

An export is a tar archive of ``data/`` minus derived state, so restoring one
needs nothing but ``tar`` and a rebuild. That constraint drives every choice
here:

* **The index is never archived.** ``data/index/`` is rebuildable by
  definition, and shipping it would make the archive claim a truth it does
  not have -- a restored index could disagree with the restored files.
* **Tombstones always are.** They are the only defense against resurrecting
  a permanently deleted memory, so an archive without them would silently
  re-enable re-extraction on restore. Nothing in this module can exclude
  them.
* **Import is additive and refuses to resurrect.** Every memory goes through
  :meth:`MemoryAPI.create`, so the tombstone check that protects a normal
  write protects a restore too.

Retention is explicit, not implicit: :func:`prune` removes backups by age and
count, and it never deletes the newest one regardless of policy. A retention
rule that can empty the backup directory is a data-loss bug, not a policy.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import (
    Memory,
    MemoryType,
    Relation,
    SourceRef,
    Tombstone,
    fmt_ts,
    now_utc,
    parse_memory_markdown,
    parse_ts,
)
from .paths import Paths
from .store import Tombstoned

if TYPE_CHECKING:  # pragma: no cover
    from .api import MemoryAPI

#: Archive format version, written into the manifest. An importer that does
#: not recognise it refuses rather than guessing at a layout change.
ARCHIVE_VERSION = 1

MANIFEST_NAME = "manifest.json"

#: Directories under ``data/`` an archive carries. Deliberately a allowlist:
#: a new directory added later is excluded until someone decides it belongs,
#: which is the safe default for both size and privacy.
ARCHIVED_DIRS: tuple[str, ...] = (
    "memories",
    "conversations",
    "documents",
    "assets",
    ".tombstones",
    "config",
)

#: Never archived. The index is derived; the journal and lock describe a
#: moment in one process's life and would be actively misleading on restore.
EXCLUDED_DIRS: tuple[str, ...] = ("index", ".journal")
EXCLUDED_NAMES: tuple[str, ...] = (".lock",)

#: Default retention. Both bounds apply: a backup must be young enough *and*
#: recent enough in the ordering to survive.
DEFAULT_KEEP_DAYS = 90
DEFAULT_KEEP_COUNT = 10

BACKUP_PREFIX = "memory-backup-"
BACKUP_SUFFIX = ".tar.gz"


class BackupError(RuntimeError):
    """A backup or restore could not be completed."""


class ArchiveFormatError(BackupError):
    """An archive is not one this version can read."""


@dataclass(slots=True)
class BackupResult:
    path: str
    bytes: int
    files: int
    memories: int
    tombstones: int
    created_at: str
    compressed: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "files": self.files,
            "memories": self.memories,
            "tombstones": self.tombstones,
            "created_at": self.created_at,
            "compressed": self.compressed,
        }


@dataclass(slots=True)
class ExportResult:
    path: str | None
    format: str
    memories: int
    tombstones: int
    lines: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "format": self.format,
            "memories": self.memories,
            "tombstones": self.tombstones,
        }
        if self.path is None:
            payload["lines"] = self.lines
        return payload


@dataclass(slots=True)
class ImportResult:
    created: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    tombstones_restored: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "created_count": len(self.created),
            "skipped": self.skipped,
            "skipped_count": len(self.skipped),
            "tombstones_restored": self.tombstones_restored,
            "errors": self.errors,
        }


# ---------------------------------------------------------------- backup


def backup_dir(paths: Paths) -> Path:
    """Where backups live by default: beside ``data/``, not inside it.

    Inside would mean the next backup archives the previous one, and the tree
    would grow geometrically.
    """
    return paths.root / "backups"


def backup(
    api: MemoryAPI,
    *,
    dest: Path | None = None,
    compress: bool = True,
    label: str | None = None,
) -> BackupResult:
    """Archive ``data/`` to a tarball, excluding derived state.

    Takes the write lock: a backup read while a write is in flight could
    capture a half-written memory file, and the point of a backup is that it
    restores to a consistent state.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = BACKUP_SUFFIX if compress else ".tar"
    name = f"{BACKUP_PREFIX}{stamp}{'-' + _slug(label) if label else ''}{suffix}"

    target = Path(dest) if dest is not None else backup_dir(api.paths) / name
    if target.is_dir():
        target = target / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise BackupError(f"{target} already exists")

    memories = 0
    tombstones = 0
    files = 0

    with api.journal:
        manifest = _manifest(api, label=label)
        tmp = target.with_name(target.name + ".partial")
        try:
            mode = "w:gz" if compress else "w"
            with tarfile.open(tmp, mode) as tar:  # type: ignore[call-overload]
                for path in _archive_members(api.paths):
                    arcname = "data/" + api.paths.rel(path)
                    tar.add(path, arcname=arcname, recursive=False)
                    files += 1
                    if arcname.startswith("data/memories/") and path.suffix == ".md":
                        memories += 1
                    elif arcname.startswith("data/.tombstones/"):
                        tombstones += 1

                manifest["memories"] = memories
                manifest["tombstones"] = tombstones
                manifest["files"] = files
                _add_bytes(
                    tar,
                    MANIFEST_NAME,
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                )
            # Only now is the archive complete; a crash before this leaves a
            # `.partial` nobody will mistake for a usable backup.
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    return BackupResult(
        path=str(target),
        bytes=target.stat().st_size,
        files=files,
        memories=memories,
        tombstones=tombstones,
        created_at=str(manifest["created_at"]),
        compressed=compress,
    )


def _manifest(api: MemoryAPI, *, label: str | None) -> dict[str, Any]:
    from . import __version__

    return {
        "archive_version": ARCHIVE_VERSION,
        "created_at": fmt_ts(now_utc()),
        "tool_version": __version__,
        "root": str(api.paths.root),
        "label": label,
        "excludes": list(EXCLUDED_DIRS),
        "note": (
            "data/index is intentionally absent: it is derived state. "
            "Restore with `mem import` or by extracting over data/, then run `mem rebuild`."
        ),
    }


def _archive_members(paths: Paths) -> Iterator[Path]:
    """Every file an archive should carry, in deterministic order."""
    if paths.marker.is_file():
        yield paths.marker
    for name in ARCHIVED_DIRS:
        base = paths.data / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir() or path.name in EXCLUDED_NAMES:
                continue
            if any(part in EXCLUDED_DIRS for part in path.relative_to(paths.data).parts):
                continue
            # A crash can leave an atomic-write temp file behind; it is not
            # part of the durable state and must not restore as a memory.
            if ".tmp-" in path.name or path.name.endswith(".partial"):
                continue
            yield path


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    import io

    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(datetime.now(UTC).timestamp())
    tar.addfile(info, io.BytesIO(payload))


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9-]+", "-", text.casefold()).strip("-")[:40]


# ---------------------------------------------------------------- listing


def list_backups(paths: Paths, *, directory: Path | None = None) -> list[dict[str, Any]]:
    """Backups found on disk, newest first."""
    base = Path(directory) if directory is not None else backup_dir(paths)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in base.iterdir():
        if not path.is_file() or not path.name.startswith(BACKUP_PREFIX):
            continue
        if not (path.name.endswith(".tar") or path.name.endswith(BACKUP_SUFFIX)):
            continue
        stat = path.stat()
        out.append(
            {
                "path": str(path),
                "name": path.name,
                "bytes": stat.st_size,
                "modified_at": fmt_ts(datetime.fromtimestamp(stat.st_mtime, UTC)),
            }
        )
    return sorted(out, key=lambda entry: str(entry["name"]), reverse=True)


def read_manifest(archive: Path) -> dict[str, Any]:
    """Read an archive's manifest without extracting it."""
    try:
        with tarfile.open(archive) as tar:
            # extractfile raises KeyError for a member that is not there, and
            # returns None only for a non-regular member -- so a tarball that
            # simply is not one of ours needs both paths handled, or the user
            # gets a bare KeyError instead of being told what is wrong.
            try:
                member = tar.extractfile(MANIFEST_NAME)
            except KeyError:
                raise ArchiveFormatError(
                    f"{archive} has no {MANIFEST_NAME}; it was not written by `mem backup`"
                ) from None
            if member is None:
                raise ArchiveFormatError(f"{archive}: {MANIFEST_NAME} is not a regular file")
            payload = json.loads(member.read().decode("utf-8"))
    except tarfile.TarError as exc:
        raise ArchiveFormatError(f"{archive} is not a readable tar archive: {exc}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArchiveFormatError(f"{archive}: unreadable manifest: {exc}") from None
    if not isinstance(payload, dict):
        raise ArchiveFormatError(f"{archive}: manifest is not an object")
    version = payload.get("archive_version")
    if version != ARCHIVE_VERSION:
        raise ArchiveFormatError(
            f"{archive}: archive version {version!r} is not supported "
            f"(this build reads v{ARCHIVE_VERSION})"
        )
    return payload


# ---------------------------------------------------------------- retention


def prune(
    paths: Paths,
    *,
    keep_days: int = DEFAULT_KEEP_DAYS,
    keep_count: int = DEFAULT_KEEP_COUNT,
    directory: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply the retention policy to the backup directory.

    Both bounds apply, and the newest backup is always kept whatever the
    policy says. A retention rule that can empty the backup directory is a
    data-loss bug dressed as configuration, so this refuses to express one.
    """
    if keep_days < 0 or keep_count < 1:
        raise ValueError("keep_days must be >= 0 and keep_count >= 1")

    entries = list_backups(paths, directory=directory)
    cutoff = datetime.now(UTC) - timedelta(days=keep_days)

    removed: list[str] = []
    kept: list[str] = []
    freed = 0

    for position, entry in enumerate(entries):
        modified = parse_ts(entry["modified_at"])
        too_old = modified < cutoff
        too_many = position >= keep_count
        # position 0 is the newest; it survives unconditionally.
        if position > 0 and (too_old or too_many):
            freed += int(entry["bytes"])
            removed.append(str(entry["path"]))
            if not dry_run:
                Path(str(entry["path"])).unlink(missing_ok=True)
        else:
            kept.append(str(entry["path"]))

    return {
        "policy": {"keep_days": keep_days, "keep_count": keep_count},
        "removed": removed,
        "kept": kept,
        "bytes_freed": freed,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------- export


def export_jsonl(
    api: MemoryAPI,
    *,
    out: Path | None = None,
    include_deleted: bool = True,
    include_tombstones: bool = True,
) -> ExportResult:
    """Export memories (and tombstones) as JSON Lines.

    One self-describing record per line, each tagged with ``record``, so the
    stream can carry more than one kind of thing and an importer can skip
    what it does not understand. Tombstones travel by default: an export that
    dropped them would restore into a tree where a deleted memory could be
    recreated.
    """
    lines: list[str] = []
    memories = 0
    tombstones = 0

    for memory in api.store.iter_all(include_deleted=include_deleted):
        lines.append(
            json.dumps(
                {"record": "memory", **memory.to_public()},
                default=str,
                ensure_ascii=False,
            )
        )
        memories += 1

    if include_tombstones:
        for tomb in api.store.iter_tombstones():
            lines.append(
                json.dumps(
                    {"record": "tombstone", **tomb.to_json()},
                    default=str,
                    ensure_ascii=False,
                )
            )
            tombstones += 1

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return ExportResult(path=str(out), format="jsonl", memories=memories, tombstones=tombstones)
    return ExportResult(
        path=None, format="jsonl", memories=memories, tombstones=tombstones, lines=lines
    )


def export_markdown_dir(api: MemoryAPI, out: Path, *, include_deleted: bool = True) -> ExportResult:
    """Copy memory files verbatim into a plain directory tree.

    The most durable export there is: the files are already the source of
    truth, so this needs no importer at all -- the output *is* the format.
    """
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for memory in api.store.iter_all(include_deleted=include_deleted):
        source = api.paths.memory_file(memory.type.value, memory.id)
        target = out / memory.type.value / f"{memory.id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        count += 1

    tombs = 0
    tomb_dir = out / "tombstones"
    for tomb in api.store.iter_tombstones():
        tomb_dir.mkdir(parents=True, exist_ok=True)
        (tomb_dir / f"{tomb.memory_id}.json").write_text(
            json.dumps(tomb.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tombs += 1

    return ExportResult(path=str(out), format="markdown", memories=count, tombstones=tombs)


# ---------------------------------------------------------------- import


def import_jsonl(
    api: MemoryAPI,
    source: Path | str,
    *,
    force_recreate: bool = False,
    skip_existing: bool = True,
    restore_tombstones: bool = True,
) -> ImportResult:
    """Import a JSON Lines export. Additive, and it will not resurrect.

    Tombstones are restored **first**, so a stream carrying both a tombstone
    and the memory it killed ends with the deletion winning regardless of
    line order. Getting that backwards would make a round trip through an
    export quietly undo a permanent deletion.
    """
    result = ImportResult()
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source

    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            result.errors.append({"line": str(lineno), "error": f"invalid JSON: {exc}"})
            continue
        if not isinstance(payload, dict):
            result.errors.append({"line": str(lineno), "error": "record is not an object"})
            continue
        records.append(payload)

    if restore_tombstones:
        for payload in records:
            if payload.get("record") != "tombstone":
                continue
            try:
                tomb = Tombstone.from_json(payload)
            except Exception as exc:
                result.errors.append(
                    {"memory_id": str(payload.get("memory_id")), "error": str(exc)}
                )
                continue
            if api.store.read_tombstone(tomb.memory_id) is None:
                _restore_tombstone(api, tomb)
                result.tombstones_restored.append(tomb.memory_id)

    for payload in records:
        record = payload.get("record", "memory")
        if record == "tombstone":
            continue
        if record != "memory":
            result.skipped.append({"record": str(record), "reason": "unknown record type"})
            continue
        try:
            _import_memory(
                api, payload, result, force_recreate=force_recreate, skip_existing=skip_existing
            )
        except Exception as exc:
            result.errors.append(
                {"memory_id": str(payload.get("id")), "error": f"{type(exc).__name__}: {exc}"}
            )

    return result


def _import_memory(
    api: MemoryAPI,
    payload: dict[str, Any],
    result: ImportResult,
    *,
    force_recreate: bool,
    skip_existing: bool,
) -> None:
    mid = str(payload.get("id") or "")
    if mid and api.store.exists(mid):
        if skip_existing:
            result.skipped.append({"memory_id": mid, "reason": "already present"})
            return
        result.skipped.append({"memory_id": mid, "reason": "already present (overwrite refused)"})
        return

    fields = _memory_fields(payload)
    try:
        memory = api.create(force_recreate=force_recreate, **fields)
    except Tombstoned as exc:
        # The import is trying to restore something the user permanently
        # deleted. That is exactly what tombstones exist to prevent.
        result.skipped.append(
            {
                "memory_id": mid,
                "reason": "permanently deleted in this root",
                "matched_on": exc.matched_on,
                "tombstone": exc.tombstone.memory_id,
            }
        )
        return
    result.created.append(memory.id)


def _memory_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an export record into ``create`` keyword arguments."""
    type_raw = str(payload.get("type") or MemoryType.FACT.value)
    if type_raw not in set(MemoryType):
        raise ValueError(f"unknown memory type {type_raw!r}")

    fields: dict[str, Any] = {
        "type": type_raw,
        "title": str(payload.get("title") or ""),
        "body": str(payload.get("body") or ""),
    }
    if payload.get("id"):
        fields["id"] = str(payload["id"])

    for key in ("confidence", "salience"):
        if payload.get(key) is not None:
            fields[key] = float(payload[key])
    if payload.get("profile") is not None:
        fields["profile"] = bool(payload["profile"])
    for key in ("sensitivity", "status"):
        if payload.get(key):
            fields[key] = str(payload[key])
    for key in ("tags", "entities", "assets", "supersedes"):
        if payload.get(key):
            fields[key] = [str(v) for v in payload[key]]
    if payload.get("superseded_by"):
        fields["superseded_by"] = str(payload["superseded_by"])
    for key in (
        "created_at",
        "updated_at",
        "occurred_at",
        "valid_from",
        "valid_until",
        "review_after",
    ):
        if payload.get(key):
            fields[key] = parse_ts(payload[key], field_name=key)

    problems: list[str] = []
    sources: list[SourceRef] = []
    for raw in payload.get("sources") or []:
        parsed = SourceRef.from_yaml(raw, problems=problems, where="sources")
        if parsed is not None:
            sources.append(parsed)
    if sources:
        fields["sources"] = sources

    relations: list[Relation] = []
    for raw in payload.get("relations") or []:
        parsed_rel = Relation.from_yaml(raw, problems=problems, where="relations")
        if parsed_rel is not None:
            relations.append(parsed_rel)
    if relations:
        fields["relations"] = relations

    known = {
        "record",
        "id",
        "schema_version",
        "type",
        "status",
        "title",
        "body",
        "content_hash",
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
    }
    extra = {k: v for k, v in payload.items() if k not in known}
    if extra:
        # Forward compatibility: an unknown key from a newer export is
        # preserved rather than dropped, which is what `extra` is for.
        fields["extra"] = extra
    return fields


def _restore_tombstone(api: MemoryAPI, tomb: Tombstone) -> None:
    from . import db

    path = api.store.write_tombstone(tomb)
    db.index_tombstone(api.conn, tomb, path=api.paths.rel(path))
    api.conn.commit()


def import_markdown_dir(
    api: MemoryAPI, source: Path, *, force_recreate: bool = False
) -> ImportResult:
    """Import a directory of memory Markdown files.

    Reads the same format the store writes, so a hand-edited tree, a git
    checkout, or the output of :func:`export_markdown_dir` all import
    identically.
    """
    result = ImportResult()
    tomb_dir = source / "tombstones"
    if tomb_dir.is_dir():
        for path in sorted(tomb_dir.glob("*.json")):
            try:
                tomb = Tombstone.from_json(json.loads(path.read_text(encoding="utf-8")), path=path)
            except Exception as exc:
                result.errors.append({"path": str(path), "error": str(exc)})
                continue
            if api.store.read_tombstone(tomb.memory_id) is None:
                _restore_tombstone(api, tomb)
                result.tombstones_restored.append(tomb.memory_id)

    for path in sorted(source.rglob("*.md")):
        try:
            memory = parse_memory_markdown(path.read_text(encoding="utf-8"), path=path)
        except Exception as exc:
            result.errors.append({"path": str(path), "error": str(exc)})
            continue
        if api.store.exists(memory.id):
            result.skipped.append({"memory_id": memory.id, "reason": "already present"})
            continue
        try:
            created = api.create(force_recreate=force_recreate, **_memory_create_kwargs(memory))
        except Tombstoned as exc:
            result.skipped.append(
                {
                    "memory_id": memory.id,
                    "reason": "permanently deleted in this root",
                    "matched_on": exc.matched_on,
                }
            )
            continue
        except Exception as exc:
            result.errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        result.created.append(created.id)

    return result


def _memory_create_kwargs(memory: Memory) -> dict[str, Any]:
    return {
        "type": memory.type.value,
        "title": memory.title,
        "body": memory.body,
        "id": memory.id,
        "status": memory.status.value,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "occurred_at": memory.occurred_at,
        "valid_from": memory.valid_from,
        "valid_until": memory.valid_until,
        "review_after": memory.review_after,
        "confidence": memory.confidence,
        "salience": memory.salience,
        "profile": memory.profile,
        "sensitivity": memory.sensitivity.value,
        "tags": list(memory.tags),
        "entities": list(memory.entities),
        "assets": list(memory.assets),
        "superseded_by": memory.superseded_by,
        "supersedes": list(memory.supersedes),
        "sources": list(memory.sources),
        "relations": list(memory.relations),
        "extra": dict(memory.extra),
    }


# ---------------------------------------------------------------- restore


def restore(
    archive: Path,
    target_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract an archive into a root, then leave the index to be rebuilt.

    Refuses a non-empty ``data/`` unless ``overwrite`` is set: merging an
    archive into a live tree is an import, and conflating the two would let
    a restore silently clobber memories the archive never knew about.
    """
    manifest = read_manifest(archive)
    paths = Paths(target_root)
    if paths.data.exists() and any(paths.data.iterdir()) and not overwrite:
        raise BackupError(
            f"{paths.data} is not empty; pass overwrite to replace it, "
            "or use `mem import` to merge instead"
        )

    paths.root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name == MANIFEST_NAME:
                continue
            if not member.isfile():
                continue
            # A tar member naming an absolute or parent path would write
            # outside the target. Never trust an archive's member names.
            destination = (paths.root / member.name).resolve()
            if not destination.is_relative_to(paths.root.resolve()):
                raise BackupError(f"archive member escapes the target root: {member.name!r}")
            handle = tar.extractfile(member)
            if handle is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read())
            extracted += 1

    paths.ensure_tree()
    return {
        "archive": str(archive),
        "root": str(paths.root),
        "extracted_files": extracted,
        "manifest": manifest,
        "next_step": "run `mem rebuild` to reconstruct the index",
    }


__all__ = [
    "ARCHIVED_DIRS",
    "ARCHIVE_VERSION",
    "DEFAULT_KEEP_COUNT",
    "DEFAULT_KEEP_DAYS",
    "EXCLUDED_DIRS",
    "ArchiveFormatError",
    "BackupError",
    "BackupResult",
    "ExportResult",
    "ImportResult",
    "backup",
    "backup_dir",
    "export_jsonl",
    "export_markdown_dir",
    "import_jsonl",
    "import_markdown_dir",
    "list_backups",
    "prune",
    "read_manifest",
    "restore",
]
