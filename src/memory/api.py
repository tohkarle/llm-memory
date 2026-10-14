"""The public API. CLI and MCP both call this, so there is one implementation.

Every mutating method follows the same bracket::

    lock -> journal.begin -> filesystem writes -> one SQLite tx -> commit

A database failure *after* the filesystem commit is deliberately not fatal:
the files already won, so the affected ids are marked dirty and reindexed on
the next open. The filesystem is the source of truth; the index catches up.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from . import db
from .journal import Journal, RecoveryReport
from .models import (
    Memory,
    MemoryStatus,
    MemoryType,
    Relation,
    Sensitivity,
    SourceRef,
    Tombstone,
    clean_entities,
    clean_tags,
    ensure_valid,
    now_utc,
)
from .paths import Paths, RootNotFound
from .store import Conflict, NotFound, Store, Tombstoned

#: Fields a caller may patch through ``update``. ``type`` is excluded on
#: purpose: changing it changes what the memory *is*, which is a supersession,
#: not an edit. See :meth:`MemoryAPI.update`.
PATCHABLE = frozenset(
    {
        "title",
        "body",
        "status",
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
        "sources",
        "relations",
        "extra",
    }
)


@dataclass(slots=True)
class DeleteResult:
    memory_id: str
    permanent: bool
    tombstone: dict[str, Any]
    removed_files: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "permanent": self.permanent,
            "tombstone": self.tombstone,
            "removed_files": self.removed_files,
        }


class MemoryAPI:
    """Open a memory root for reading and writing.

    Use as a context manager so recovery runs on open and the lock is always
    released::

        with MemoryAPI() as api:
            api.create(type="fact", title="...", body="...")
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        read_only: bool = False,
        lock_timeout: float = 10.0,
        auto_recover: bool = True,
    ) -> None:
        self.paths = Paths(root) if root is not None else Paths.discover()
        self.read_only = read_only
        self.auto_recover = auto_recover
        self.journal = Journal(self.paths, lock_timeout=lock_timeout)
        self.store = Store(self.paths, self.journal)
        self._conn: sqlite3.Connection | None = None
        self.recovery: RecoveryReport | None = None

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def initialize(cls, root: Path | str, **kw: Any) -> MemoryAPI:
        """Create the directory tree and index for a new root."""
        paths = Paths(root)
        paths.ensure_tree()
        api = cls(root, **kw)
        api.open()
        return api

    def open(self) -> Self:
        if self._conn is not None:
            return self
        if not self.paths.data.exists():
            raise RootNotFound(
                f"{self.paths.data} does not exist; run `mem init --root {self.paths.root}`"
            )
        conn = db.connect(self.paths, read_only=self.read_only)
        try:
            if not self.read_only:
                db.migrate(conn)
            self._conn = conn
        except BaseException:
            conn.close()
            raise

        if not self.read_only and self.auto_recover:
            self._recover_and_reconcile()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self.journal.release()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryAPI is not open; use it as a context manager")
        return self._conn

    def _recover_and_reconcile(self) -> None:
        """Settle interrupted operations, then make the index match the files."""
        with self.journal:
            report = self.journal.recover()
            self.recovery = report

            pending = set(report.reindex) | set(db.take_dirty(self.conn))
            if db.requires_rebuild(self.conn) is not None:
                from .lifecycle import rebuild_index

                rebuild_index(self.paths, self.conn, journal=self.journal)
                self.conn.commit()
                return

            for mid in report.unindex:
                db.unindex_memory(self.conn, mid)
            for mid in pending:
                self._reindex_one(mid)

            # A crash can land a file durably and still skip the index write
            # (the journal commits before the index transaction, and the undo
            # path may never run). The filesystem is authoritative, so close
            # the gap here rather than trusting the journal to have flagged it.
            drifted = self._reconcile_drift()

            if report.unindex or pending or drifted:
                self.conn.commit()

    def _reconcile_drift(self) -> int:
        """Index files the database has not seen, and drop rows with no file.

        Compares content hashes rather than mtimes: a hash cannot be fooled by
        a clock skew or a same-second rewrite.
        """
        indexed = {
            row["id"]: row["content_hash"]
            for row in self.conn.execute("SELECT id, content_hash FROM memories")
        }
        seen: set[str] = set()
        changed = 0

        for memory in self.store.iter_all(include_deleted=True):
            seen.add(memory.id)
            if indexed.get(memory.id) != memory.content_hash:
                self._reindex_one(memory.id)
                changed += 1

        for mid in set(indexed) - seen:
            db.unindex_memory(self.conn, mid)
            changed += 1
        return changed

    def _reindex_one(self, mid: str) -> None:
        """Re-derive one memory's index rows from whatever the file now says."""
        self.store.invalidate_cache(mid)
        memory = self.store.read(mid)
        if memory is None:
            db.unindex_memory(self.conn, mid)
            return
        path = self.paths.memory_file(memory.type.value, memory.id)
        db.index_memory(
            self.conn,
            memory,
            path=self.paths.rel(path),
            mtime_ns=self.store.stat_mtime_ns(path),
        )

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("this MemoryAPI was opened read-only")

    # ------------------------------------------------------------------ create

    def create(
        self,
        *,
        type: str | MemoryType,
        title: str,
        body: str = "",
        force_recreate: bool = False,
        **fields: Any,
    ) -> Memory:
        """Add a new memory."""
        self._require_writable()
        memory = _build_memory(type=type, title=title, body=body, fields=fields)

        with self.journal:
            result = self.store.write_new(memory, force_recreate=force_recreate)
            try:
                db.index_memory(
                    self.conn,
                    memory,
                    path=self.paths.rel(result.path),
                    mtime_ns=self.store.stat_mtime_ns(result.path),
                )
                self.conn.commit()
            except sqlite3.DatabaseError:
                # The file is durable, so the write stands; the index catches
                # up on the next open rather than losing the memory.
                self.conn.rollback()
                db.add_dirty(self.conn, [memory.id])
                self.conn.commit()
            self.journal.commit(result.op_id)
        return memory

    # ------------------------------------------------------------------ read

    def get(self, mid: str, *, with_related: bool = False, follow: bool = False) -> Memory | None:
        memory = self.store.read(mid)
        if memory is None:
            return None
        if follow:
            memory = self._follow_tip(memory)
        return memory

    def _follow_tip(self, memory: Memory) -> Memory:
        """Walk ``superseded_by`` to the live version, with a cycle guard."""
        seen = {memory.id}
        current = memory
        while current.superseded_by and current.superseded_by not in seen:
            nxt = self.store.read(current.superseded_by)
            if nxt is None:
                break
            seen.add(nxt.id)
            current = nxt
        return current

    def require(self, mid: str) -> Memory:
        memory = self.get(mid)
        if memory is None:
            raise NotFound(f"no memory {mid}")
        return memory

    def tombstone(self, mid: str) -> Tombstone | None:
        return self.store.read_tombstone(mid)

    # ------------------------------------------------------------------ update

    def update(self, mid: str, *, expect_hash: str | None = None, **patch: Any) -> Memory:
        """Patch fields on an existing memory, keeping its id.

        Rejects a ``type`` change: that changes what the memory is, so it
        belongs to :meth:`replace`, which preserves the old version. Allowing
        it here would make the two verbs overlap and agents would pick
        arbitrarily.
        """
        self._require_writable()
        if "type" in patch:
            raise ValueError(
                "update cannot change `type`; use replace() so the old version is preserved"
            )
        unknown = set(patch) - PATCHABLE
        if unknown:
            raise ValueError(f"not updatable: {', '.join(sorted(unknown))}")

        current = self.store.require(mid)
        updated = _apply_patch(current, patch)
        if updated == current:
            return current
        updated = updated.with_updates(updated_at=now_utc())
        ensure_valid(updated)

        with self.journal:
            result = self.store.overwrite(updated, expect_hash=expect_hash)
            if result is None:
                return current
            try:
                db.index_memory(
                    self.conn,
                    updated,
                    path=self.paths.rel(result.path),
                    mtime_ns=self.store.stat_mtime_ns(result.path),
                )
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                db.add_dirty(self.conn, [updated.id])
                self.conn.commit()
            self.journal.commit(result.op_id)
        return updated

    # ------------------------------------------------------------------ delete

    def delete(
        self, mid: str, *, permanent: bool = False, reason: str | None = None
    ) -> DeleteResult:
        """Soft-delete (reversible) or permanently purge a memory.

        Permanent deletion removes the memory file and its index rows. It does
        **not** touch the conversations or documents it cites, nor asset
        blobs: those are usually shared with other memories.
        """
        self._require_writable()
        memory = self.store.require(mid)
        removed: list[str] = []

        with self.journal:
            if permanent:
                path = self.paths.memory_file(memory.type.value, mid)
                removed.append(self.paths.rel(path))
                tomb, op_id = self.store.purge(mid, reason=reason)
                try:
                    db.unindex_memory(self.conn, mid)
                    db.index_tombstone(
                        self.conn, tomb, path=self.paths.rel(self.paths.tombstone_file(mid))
                    )
                    self.conn.commit()
                except sqlite3.DatabaseError:
                    self.conn.rollback()
                    db.add_dirty(self.conn, [mid])
                    self.conn.commit()
            else:
                tomb, op_id = self.store.soft_delete(mid, reason=reason)
                try:
                    self._reindex_one(mid)
                    db.index_tombstone(
                        self.conn, tomb, path=self.paths.rel(self.paths.tombstone_file(mid))
                    )
                    self.conn.commit()
                except sqlite3.DatabaseError:
                    self.conn.rollback()
                    db.add_dirty(self.conn, [mid])
                    self.conn.commit()
            self.journal.commit(op_id)

        return DeleteResult(
            memory_id=mid,
            permanent=permanent,
            tombstone=tomb.to_json(),
            removed_files=removed,
        )

    # ------------------------------------------------------------------ maintenance

    def verify(self) -> dict[str, Any]:
        from .lifecycle import verify

        return verify(self.paths, self.conn, self.store)

    def rebuild_index(self, *, verify_after: bool = False) -> dict[str, Any]:
        self._require_writable()
        from .lifecycle import rebuild_index

        with self.journal:
            report = rebuild_index(
                self.paths, self.conn, journal=self.journal, verify_after=verify_after
            )
            self.conn.commit()
        return report

    def stats(self) -> dict[str, Any]:
        cur = self.conn
        by_type = {
            row["type"]: row["c"]
            for row in cur.execute(
                "SELECT type, count(*) c FROM memories WHERE status='active' GROUP BY type"
            )
        }
        by_status = {
            row["status"]: row["c"]
            for row in cur.execute("SELECT status, count(*) c FROM memories GROUP BY status")
        }
        totals = cur.execute(
            "SELECT (SELECT count(*) FROM memories) memories,"
            " (SELECT count(*) FROM assets) assets,"
            " (SELECT count(*) FROM sources) sources,"
            " (SELECT count(*) FROM relations WHERE implied=0) relations,"
            " (SELECT count(*) FROM tombstones) tombstones,"
            " (SELECT count(*) FROM segments) segments,"
            " (SELECT count(*) FROM derivations) derivations"
        ).fetchone()
        blob_bytes = sum(p.stat().st_size for p in self.paths.blob_glob() if p.is_file())
        return {
            "memories": totals["memories"],
            "by_type": by_type,
            "by_status": by_status,
            "assets": totals["assets"],
            "sources": totals["sources"],
            "relations": totals["relations"],
            "tombstones": totals["tombstones"],
            "segments": totals["segments"],
            "derivations": totals["derivations"],
            "blob_bytes": blob_bytes,
            "index_bytes": (
                self.paths.index_db.stat().st_size if self.paths.index_db.exists() else 0
            ),
        }

    def doctor(self) -> dict[str, Any]:
        """Environment and index health, for troubleshooting."""
        import shutil
        import sys

        with self.journal:
            open_ops = [
                {"op_id": r.op_id, "kind": r.kind.value, "seq": r.seq, "ts": r.ts}
                for r in self.journal.open_ops()
            ]
        return {
            "root": str(self.paths.root),
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "fts5": db.has_fts5(self.conn),
            "schema_version": db.schema_version(self.conn),
            "requires_rebuild": db.requires_rebuild(self.conn),
            "open_journal_ops": open_ops,
            "last_recovery": self.recovery.to_json() if self.recovery else None,
            "ffmpeg": shutil.which("ffmpeg"),
        }


# ---------------------------------------------------------------- helpers


def _build_memory(
    *, type: str | MemoryType, title: str, body: str, fields: dict[str, Any]
) -> Memory:
    mem_type = MemoryType(type) if not isinstance(type, MemoryType) else type
    kwargs: dict[str, Any] = {"type": mem_type, "title": title, "body": body}

    if "status" in fields:
        kwargs["status"] = MemoryStatus(fields.pop("status"))
    if "sensitivity" in fields:
        kwargs["sensitivity"] = Sensitivity(fields.pop("sensitivity"))
    if "tags" in fields:
        kwargs["tags"] = clean_tags(list(fields.pop("tags")))
    if "entities" in fields:
        kwargs["entities"] = clean_entities(list(fields.pop("entities")))

    for key in (
        "id",
        "body",
        "created_at",
        "updated_at",
        "occurred_at",
        "valid_from",
        "valid_until",
        "review_after",
        "confidence",
        "salience",
        "profile",
        "assets",
        "superseded_by",
        "supersedes",
        "sources",
        "relations",
        "extra",
    ):
        if key in fields:
            kwargs[key] = fields.pop(key)

    if fields:
        raise ValueError(f"unknown field(s): {', '.join(sorted(fields))}")
    return ensure_valid(Memory(**kwargs))


def _apply_patch(current: Memory, patch: dict[str, Any]) -> Memory:
    changes: dict[str, Any] = {}
    for key, value in patch.items():
        if key == "status":
            changes[key] = MemoryStatus(value)
        elif key == "sensitivity":
            changes[key] = Sensitivity(value)
        elif key == "tags":
            changes[key] = clean_tags(list(value))
        elif key == "entities":
            changes[key] = clean_entities(list(value))
        else:
            changes[key] = value
    return current.with_updates(**changes)


__all__ = [
    "Conflict",
    "DeleteResult",
    "MemoryAPI",
    "NotFound",
    "Relation",
    "SourceRef",
    "Tombstoned",
]
