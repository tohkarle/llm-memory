"""Filesystem CRUD. The authoritative layer.

This module knows nothing about SQLite. Everything it writes is recoverable
on its own, which is what makes the index safe to delete and rebuild.

Writes go through the journal so an interrupted operation is recoverable;
``Store`` builds the plans and performs the file actions, while
:mod:`memory.journal` owns durability and rollback.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .journal import Journal, OpKind, Step, atomic_write, hash_file
from .models import (
    Asset,
    Memory,
    MemoryStatus,
    MemoryType,
    Tombstone,
    ValidationError,
    content_hash,
    ensure_valid,
    new_id,
    now_utc,
    parse_memory_markdown,
    render_memory_markdown,
    sha256_file,
    title_hash,
    tombstone_for,
)
from .paths import Paths


class StoreError(RuntimeError):
    """A filesystem operation could not be completed."""


class Conflict(StoreError):
    """The target already exists, or has changed underneath us."""


class NotFound(StoreError):
    """No such memory or asset."""


class Tombstoned(StoreError):
    """Creation refused: this content was permanently deleted.

    Carries the dead memory's details so the caller can explain, and so an
    agent can decide whether to pass ``force_recreate``.
    """

    def __init__(self, tombstone: Tombstone, matched_on: str) -> None:
        self.tombstone = tombstone
        self.matched_on = matched_on
        super().__init__(
            f"content matches permanently deleted memory {tombstone.memory_id} "
            f"(matched on {matched_on}); pass force_recreate to override"
        )


@dataclass(slots=True)
class WriteResult:
    """A completed file write, with the journal op the caller must settle."""

    path: Path
    op_id: str
    content_hash: str


@dataclass(slots=True)
class BlobResult:
    sha256: str
    bytes: int
    deduped: bool
    path: Path


class Store:
    """Read and write the authoritative files under ``data/``."""

    def __init__(self, paths: Paths, journal: Journal) -> None:
        self.paths = paths
        self.journal = journal
        self._path_cache: dict[str, Path] = {}

    # ------------------------------------------------------------------ reads

    def locate(self, mid: str) -> Path | None:
        """Find a memory's file without parsing it.

        Memories live in a directory chosen by type, so a lookup by id alone
        has to probe the five type directories. Cached because callers often
        read the same id repeatedly within one operation.
        """
        cached = self._path_cache.get(mid)
        if cached is not None and cached.exists():
            return cached
        for mem_type in MemoryType:
            candidate = self.paths.memory_file(mem_type.value, mid)
            if candidate.exists():
                self._path_cache[mid] = candidate
                return candidate
        self._path_cache.pop(mid, None)
        return None

    def read(self, mid: str) -> Memory | None:
        path = self.locate(mid)
        if path is None:
            return None
        return self.read_path(path)

    def read_path(self, path: Path) -> Memory:
        """Parse one memory file.

        Enforces the filename-stem-equals-id invariant: a renamed file would
        otherwise be indexed under an id that does not match its own
        frontmatter, which silently breaks every reference to it.
        """
        memory = parse_memory_markdown(path.read_text(encoding="utf-8"), path=path)
        if path.stem != memory.id:
            raise ValidationError(
                [f"filename stem {path.stem!r} does not match id {memory.id!r} in {path}"]
            )
        return memory

    def require(self, mid: str) -> Memory:
        memory = self.read(mid)
        if memory is None:
            raise NotFound(f"no memory {mid}")
        return memory

    def exists(self, mid: str) -> bool:
        return self.locate(mid) is not None

    def iter_all(self, *, include_deleted: bool = False) -> Iterator[Memory]:
        """Every memory on disk, skipping unparseable files.

        Deliberately tolerant: one corrupt file must not stop a rebuild. The
        caller surfaces problems via ``verify``.
        """
        for path in self.paths.memory_glob():
            try:
                memory = self.read_path(path)
            except (ValidationError, OSError, UnicodeDecodeError):
                continue
            if not include_deleted and memory.status is MemoryStatus.DELETED:
                continue
            yield memory

    def iter_problems(self) -> Iterator[tuple[Path, str]]:
        """Files that exist but cannot be read as memories."""
        for path in self.paths.memory_glob():
            try:
                self.read_path(path)
            except ValidationError as exc:
                yield path, "; ".join(exc.problems)
            except (OSError, UnicodeDecodeError) as exc:
                yield path, str(exc)

    # ------------------------------------------------------------------ tombstones

    def read_tombstone(self, mid: str) -> Tombstone | None:
        path = self.paths.tombstone_file(mid)
        if not path.exists():
            return None
        try:
            return Tombstone.from_json(json.loads(path.read_text(encoding="utf-8")), path=path)
        except (json.JSONDecodeError, ValidationError):
            return None

    def iter_tombstones(self) -> Iterator[Tombstone]:
        for path in self.paths.tombstone_glob():
            try:
                yield Tombstone.from_json(json.loads(path.read_text(encoding="utf-8")), path=path)
            except (json.JSONDecodeError, ValidationError, OSError):
                continue

    def check_resurrection(self, m: Memory) -> None:
        """Refuse to recreate something permanently deleted.

        Three guards, in order of how often they fire in practice:

        1. content hash -- the same memory written again verbatim;
        2. source + locator overlap -- the case that actually matters, because
           a re-extraction from a retained conversation usually rewords
           slightly but cites the same span;
        3. title hash -- a renamed body with the same claim.

        Soft-deleted memories do not block creation; only permanent ones do.
        """
        new_keys = m.source_keys()
        for tomb in self.iter_tombstones():
            if not tomb.permanent:
                continue
            if tomb.content_hash and tomb.content_hash == m.content_hash:
                raise Tombstoned(tomb, "content_hash")
            overlap = new_keys & tomb.source_keys()
            if overlap:
                raise Tombstoned(tomb, f"source {sorted(overlap)[0][1]}")
            if tomb.title_hash and tomb.title_hash == m.title_hash:
                raise Tombstoned(tomb, "title_hash")

    # ------------------------------------------------------------------ writes

    def write_new(self, m: Memory, *, force_recreate: bool = False) -> WriteResult:
        """Create a memory that does not yet exist."""
        ensure_valid(m)
        if self.exists(m.id):
            raise Conflict(f"memory {m.id} already exists")
        if not force_recreate:
            self.check_resurrection(m)

        target = self.paths.memory_file(m.type.value, m.id)
        payload = render_memory_markdown(m).encode("utf-8")
        plan = [
            Step(
                i=0,
                act="write",
                path=self.paths.rel(target),
                new_hash=_hash_bytes(payload),
                new_bytes=len(payload),
            )
        ]
        op_id = self.journal.begin(OpKind.CREATE_MEMORY, [m.id], plan, db={"reindex": [m.id]})
        try:
            atomic_write(target, payload, tmp_suffix=f".tmp-{op_id}")
        except BaseException:
            self.journal.abort(op_id, "write failed")
            raise
        self._path_cache[m.id] = target
        return WriteResult(target, op_id, _hash_bytes(payload))

    def overwrite(self, m: Memory, *, expect_hash: str | None = None) -> WriteResult | None:
        """Replace a memory's content in place, keeping its id and type.

        Returns without writing when the rendered bytes are unchanged, so an
        idempotent retry costs nothing and leaves no journal record.
        """
        ensure_valid(m)
        current_path = self.locate(m.id)
        if current_path is None:
            raise NotFound(f"no memory {m.id}")
        if current_path.name != f"{m.id}.md":
            raise Conflict(f"path {current_path} does not belong to {m.id}")

        existing_hash = hash_file(current_path)
        if expect_hash is not None and existing_hash != expect_hash:
            raise Conflict(
                f"memory {m.id} changed on disk (expected {expect_hash}, found {existing_hash})"
            )

        payload = render_memory_markdown(m).encode("utf-8")
        new_hash = _hash_bytes(payload)
        if new_hash == existing_hash:
            return None

        target = self.paths.memory_file(m.type.value, m.id)
        moving = target != current_path

        plan: list[Step] = []

        # The pre-image must exist before intent is recorded, and the undo
        # area is keyed by op id, so the id is reserved up front.
        pre_op = self.journal.reserve_op_id()
        backup = self.journal.backup(pre_op, 0, current_path)
        # On a move the new path has no pre-image -- nothing was there before,
        # so rolling back means deleting it, not restoring a backup. The
        # pre-image belongs solely to the old path, whose unlink step carries
        # it. Attaching it to both steps would resurrect the file at the new
        # location and leave two files sharing one id.
        plan.append(
            Step(
                i=0,
                act="write",
                path=self.paths.rel(target),
                new_hash=new_hash,
                new_bytes=len(payload),
                prev_hash=None if moving else existing_hash,
                prev_backup=None if moving else backup,
                from_path=self.paths.rel(current_path) if moving else None,
            )
        )
        if moving:
            plan.append(
                Step(
                    i=1,
                    act="unlink",
                    path=self.paths.rel(current_path),
                    prev_hash=existing_hash,
                    prev_backup=backup,
                )
            )

        op_id = self.journal.begin(
            OpKind.UPDATE_MEMORY,
            [m.id],
            plan,
            db={"reindex": [m.id]},
            op_id=pre_op,
        )
        try:
            atomic_write(target, payload, tmp_suffix=f".tmp-{op_id}")
        except BaseException:
            # Nothing landed, so the pre-image is still the truth: abort and
            # let the old file stand.
            self.journal.abort(op_id, "overwrite failed")
            raise

        if moving:
            # The new file is durable now. Removing the old one must not be
            # skippable -- an abort here would leave two files sharing one id,
            # so this runs before the caller commits and any failure leaves
            # the op open for recovery to finish.
            current_path.unlink(missing_ok=True)
        self._path_cache[m.id] = target
        return WriteResult(target, op_id, new_hash)

    def soft_delete(self, mid: str, *, reason: str | None = None) -> tuple[Tombstone, str]:
        """Mark a memory deleted, keeping its file and body on disk."""
        memory = self.require(mid)
        tomb = tombstone_for(memory, permanent=False, reason=reason)
        deleted = memory.with_updates(status=MemoryStatus.DELETED, updated_at=now_utc())

        path = self.paths.memory_file(memory.type.value, mid)
        payload = render_memory_markdown(deleted).encode("utf-8")
        tomb_path = self.paths.tombstone_file(mid)
        tomb_payload = _json_bytes(tomb.to_json())

        pre_op = self.journal.reserve_op_id()
        backup = self.journal.backup(pre_op, 1, path)
        plan = [
            Step(
                i=0,
                act="write",
                path=self.paths.rel(tomb_path),
                new_hash=_hash_bytes(tomb_payload),
                new_bytes=len(tomb_payload),
            ),
            Step(
                i=1,
                act="write",
                path=self.paths.rel(path),
                new_hash=_hash_bytes(payload),
                new_bytes=len(payload),
                prev_hash=hash_file(path),
                prev_backup=backup,
            ),
        ]
        op_id = self.journal.begin(
            OpKind.DELETE_MEMORY, [mid], plan, db={"reindex": [mid]}, op_id=pre_op
        )
        try:
            atomic_write(tomb_path, tomb_payload, tmp_suffix=f".tmp-{op_id}")
            atomic_write(path, payload, tmp_suffix=f".tmp-{op_id}")
        except BaseException:
            self.journal.abort(op_id, "soft delete failed")
            raise
        return tomb, op_id

    def purge(self, mid: str, *, reason: str | None = None) -> tuple[Tombstone, str]:
        """Permanently remove a memory's file, leaving a tombstone.

        The tombstone is written **first** and the unlink second, so a crash
        between them leaves a recoverable state that rolls *forward*: the
        intent to destroy is already durable.
        """
        memory = self.require(mid)
        tomb = tombstone_for(memory, permanent=True, reason=reason)
        path = self.paths.memory_file(memory.type.value, mid)
        tomb_path = self.paths.tombstone_file(mid)
        tomb_payload = _json_bytes(tomb.to_json())

        pre_op = self.journal.reserve_op_id()
        backup = self.journal.backup(pre_op, 1, path)
        plan = [
            Step(
                i=0,
                act="write",
                path=self.paths.rel(tomb_path),
                new_hash=_hash_bytes(tomb_payload),
                new_bytes=len(tomb_payload),
            ),
            Step(
                i=1,
                act="unlink",
                path=self.paths.rel(path),
                prev_hash=hash_file(path),
                prev_backup=backup,
            ),
        ]
        op_id = self.journal.begin(
            OpKind.PURGE_MEMORY, [mid], plan, db={"unindex": [mid]}, op_id=pre_op
        )
        # Deliberately no abort-on-failure here. Once the tombstone is
        # durable the intent to destroy is recorded, and aborting would leave
        # a tombstone next to a live memory file -- two files disagreeing
        # about whether the memory exists. Leaving the op open hands it to
        # recovery, which rolls destructive operations forward.
        atomic_write(tomb_path, tomb_payload, tmp_suffix=f".tmp-{op_id}")
        path.unlink(missing_ok=True)
        self._path_cache.pop(mid, None)
        return tomb, op_id

    def write_tombstone(self, tomb: Tombstone) -> Path:
        path = self.paths.tombstone_file(tomb.memory_id)
        atomic_write(path, _json_bytes(tomb.to_json()))
        return path

    # ------------------------------------------------------------------ blobs

    def put_blob(self, src: Path | bytes) -> BlobResult:
        """Store bytes content-addressed, deduplicating by hash.

        Writing the same hash twice is a no-op, which is why blob steps need
        no undo pre-image: the operation is naturally idempotent.
        """
        if isinstance(src, bytes):
            digest = _raw_sha256(src)
            size = len(src)
            target = self.paths.asset_blob(digest)
            if target.exists() and target.stat().st_size == size:
                return BlobResult(digest, size, True, target)
            atomic_write(target, src)
            return BlobResult(digest, size, False, target)

        if not src.is_file():
            raise NotFound(f"no such file: {src}")
        digest, size = sha256_file(src)
        target = self.paths.asset_blob(digest)
        if target.exists() and target.stat().st_size == size:
            return BlobResult(digest, size, True, target)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + f".tmp-{new_id()}")
        try:
            shutil.copyfile(src, tmp)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return BlobResult(digest, size, False, target)

    def blob_exists(self, sha256: str) -> bool:
        return self.paths.asset_blob(sha256).exists()

    def read_blob(self, sha256: str) -> bytes:
        path = self.paths.asset_blob(sha256)
        if not path.exists():
            raise NotFound(f"no blob {sha256}")
        return path.read_bytes()

    def unlink_blob(self, sha256: str) -> bool:
        path = self.paths.asset_blob(sha256)
        if not path.exists():
            return False
        path.unlink()
        return True

    # ------------------------------------------------------------------ assets

    def write_asset_meta(self, asset: Asset) -> Path:
        path = self.paths.asset_meta(asset.id)
        atomic_write(path, _json_bytes(asset.to_json()))
        return path

    def read_asset_meta(self, asset_id: str) -> Asset | None:
        path = self.paths.asset_meta(asset_id)
        if not path.exists():
            return None
        try:
            return Asset.from_json(json.loads(path.read_text(encoding="utf-8")), path=path)
        except (json.JSONDecodeError, ValidationError):
            return None

    def require_asset(self, asset_id: str) -> Asset:
        asset = self.read_asset_meta(asset_id)
        if asset is None:
            raise NotFound(f"no asset {asset_id}")
        return asset

    def iter_assets(self) -> Iterator[Asset]:
        for path in self.paths.asset_meta_glob():
            try:
                yield Asset.from_json(json.loads(path.read_text(encoding="utf-8")), path=path)
            except (json.JSONDecodeError, ValidationError, OSError):
                continue

    def delete_asset_meta(self, asset_id: str) -> bool:
        path = self.paths.asset_meta(asset_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def blob_refcount(self, sha256: str) -> int:
        """How many assets still point at this blob.

        Counts prior versions too (``prev_sha256``), so garbage collection is
        conservative and never breaks a version chain.
        """
        count = 0
        for asset in self.iter_assets():
            if asset.status != "active":
                continue
            if asset.sha256 == sha256 or asset.prev_sha256 == sha256:
                count += 1
        return count

    @staticmethod
    def guess_media_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    # ------------------------------------------------------------------ misc

    def stat_mtime_ns(self, path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def invalidate_cache(self, mid: str | None = None) -> None:
        if mid is None:
            self._path_cache.clear()
        else:
            self._path_cache.pop(mid, None)


# ---------------------------------------------------------------- helpers


def _hash_bytes(data: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(data).hexdigest()


def _raw_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: object) -> bytes:
    """Canonical JSON: sorted keys and a trailing newline.

    Stable output keeps file diffs meaningful and content hashes comparable
    across writes.
    """
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def utc_from_stat(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


__all__ = [
    "BlobResult",
    "Conflict",
    "NotFound",
    "Store",
    "StoreError",
    "Tombstoned",
    "WriteResult",
    "content_hash",
    "title_hash",
]
