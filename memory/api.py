"""The public API. CLI and MCP both call this, so there is one implementation.

Every mutating method follows the same bracket::

    lock -> journal.begin -> filesystem writes -> one SQLite tx -> commit

A database failure *after* the filesystem commit is deliberately not fatal:
the files already won, so the affected ids are marked dirty and reindexed on
the next open. The filesystem is the source of truth; the index catches up.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from . import db, media
from .context import ContextResult, get_context
from .journal import Journal, RecoveryReport
from .models import (
    SENSITIVITY_ORDER,
    Asset,
    Memory,
    MemoryStatus,
    MemoryType,
    Relation,
    Segment,
    Sensitivity,
    SourceRef,
    Tombstone,
    clean_entities,
    clean_tags,
    ensure_valid,
    fmt_ts,
    now_utc,
)
from .paths import Paths, RootNotFound
from .search import (
    Hit,
    RelatedHit,
    SearchError,
    SearchFilters,
    get_related,
    profile_memories,
    search,
)
from .store import Conflict, NotFound, Store, Tombstoned

if TYPE_CHECKING:  # pragma: no cover
    from .backup import BackupResult, ExportResult, ImportResult
    from .cascade import DeletionPlan, DeletionResult
    from .consolidate import ConsolidationReport
    from .extract import ExtractionReport
    from .providers import LLMProvider

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
class ReplaceResult:
    new: Memory
    superseded: Memory
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "new": self.new.to_public(),
            "superseded": {
                "id": self.superseded.id,
                "status": self.superseded.status.value,
                "superseded_by": self.superseded.superseded_by,
                "valid_until": fmt_ts(self.superseded.valid_until),
            },
            "reason": self.reason,
        }


@dataclass(slots=True)
class MergeResult:
    merged: Memory
    superseded: list[Memory]
    strategy: str = "union"

    def to_json(self) -> dict[str, Any]:
        return {
            "merged": self.merged.to_public(),
            "superseded": [p.id for p in self.superseded],
            "strategy": self.strategy,
        }


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
            if self.read_only:
                # A reader cannot migrate, but it must not pretend an index it
                # does not understand is empty: a newer build's schema may have
                # moved the rows this one queries, and every search would
                # return [] with no error at all. Raise the same actionable
                # message the writer path raises.
                db.check_not_newer(conn)
            else:
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

            # A worker killed mid-task leaves a claimed queue row. Returning
            # it here means the next command picks it up, rather than it
            # waiting for someone to run the queue explicitly.
            from . import workqueue

            reaped = workqueue.reap_expired(self.conn)

            if report.unindex or pending or drifted or reaped:
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

    # ------------------------------------------------------------------ supersede

    def replace(
        self,
        mid: str,
        *,
        title: str,
        body: str = "",
        type: str | MemoryType | None = None,
        reason: str | None = None,
        **fields: Any,
    ) -> ReplaceResult:
        """Supersede a memory with a new one, preserving the old version.

        Unlike :meth:`update`, this mints a new id and keeps the old memory on
        disk with ``status=superseded`` and its full body intact. That is the
        difference between "I misspelled this" and "this is no longer what I
        think": the latter is history worth keeping.
        """
        self._require_writable()
        old = self.store.require(mid)
        if old.status is MemoryStatus.DELETED:
            raise Conflict(f"memory {mid} is deleted; create a new memory instead")

        carried = _carry_forward(old, fields)
        new_memory = _build_memory(
            type=type or old.type,
            title=title,
            body=body,
            fields={**carried, "supersedes": [*old.supersedes, mid]},
        )
        stamp = now_utc()
        superseded = old.with_updates(
            status=MemoryStatus.SUPERSEDED,
            superseded_by=new_memory.id,
            valid_until=old.valid_until or stamp,
            updated_at=stamp,
        )

        with self.journal:
            created = self.store.write_new(new_memory)
            updated = self.store.overwrite(superseded)
            try:
                db.index_memory(
                    self.conn,
                    new_memory,
                    path=self.paths.rel(created.path),
                    mtime_ns=self.store.stat_mtime_ns(created.path),
                )
                self._reindex_one(mid)
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                db.add_dirty(self.conn, [new_memory.id, mid])
                self.conn.commit()
            self.journal.commit(created.op_id)
            if updated is not None:
                self.journal.commit(updated.op_id)

        return ReplaceResult(new=new_memory, superseded=superseded, reason=reason)

    def merge(
        self,
        ids: list[str],
        *,
        into: str | None = None,
        title: str | None = None,
        body: str | None = None,
        strategy: str = "union",
    ) -> MergeResult:
        """Consolidate duplicates into one memory.

        By default this mints a new id and leaves every parent intact and
        marked superseded, so the merge is auditable. ``into`` folds into an
        existing parent instead, when a stable id matters more than history.

        The union of sources is the part that must not be lost: a merged
        memory has to cite everything its parents cited, or provenance
        silently narrows.
        """
        self._require_writable()
        if len(ids) < 2:
            raise ValueError("merge needs at least two memories")
        if into is not None and into not in ids:
            raise ValueError(f"--into {into} must be one of the merged ids")
        if strategy not in ("union", "prefer-newest", "prefer-highest-confidence"):
            raise ValueError(f"unknown merge strategy: {strategy}")

        parents = [self.store.require(mid) for mid in dict.fromkeys(ids)]
        for parent in parents:
            if parent.status is MemoryStatus.DELETED:
                raise Conflict(f"memory {parent.id} is deleted and cannot be merged")

        merged_fields = _merge_fields(parents, strategy=strategy)
        stamp = now_utc()

        if into is not None:
            target = next(p for p in parents if p.id == into)
            others = [p for p in parents if p.id != into]
            merged = target.with_updates(
                title=title or target.title,
                body=body if body is not None else merged_fields["body"],
                updated_at=stamp,
                supersedes=sorted({*target.supersedes, *(o.id for o in others)}),
                **{
                    k: v
                    for k, v in merged_fields.items()
                    if k
                    in (
                        "tags",
                        "entities",
                        "assets",
                        "sources",
                        "relations",
                        "confidence",
                        "salience",
                        "occurred_at",
                        "created_at",
                    )
                },
            )
        else:
            merged = _build_memory(
                type=merged_fields["type"],
                title=title or parents[0].title,
                body=body if body is not None else merged_fields["body"],
                fields={
                    **{k: v for k, v in merged_fields.items() if k not in ("type", "body")},
                    "supersedes": sorted({p.id for p in parents}),
                },
            )
            others = parents

        ensure_valid(merged)
        superseded: list[Memory] = [
            p.with_updates(
                status=MemoryStatus.SUPERSEDED,
                superseded_by=merged.id,
                valid_until=p.valid_until or stamp,
                updated_at=stamp,
            )
            for p in others
        ]

        with self.journal:
            op_ids: list[str] = []
            if into is not None:
                written = self.store.overwrite(merged)
                if written is not None:
                    op_ids.append(written.op_id)
            else:
                created = self.store.write_new(merged)
                op_ids.append(created.op_id)

            for parent in superseded:
                result = self.store.overwrite(parent)
                if result is not None:
                    op_ids.append(result.op_id)

            try:
                self._reindex_one(merged.id)
                for parent in superseded:
                    self._reindex_one(parent.id)
                self.conn.commit()
            except sqlite3.DatabaseError:
                self.conn.rollback()
                db.add_dirty(self.conn, [merged.id, *(p.id for p in superseded)])
                self.conn.commit()

            for op_id in op_ids:
                self.journal.commit(op_id)

        return MergeResult(merged=merged, superseded=superseded, strategy=strategy)

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
                    # Drop sources nothing cites any more. The raw files under
                    # conversations/ and documents/ are untouched -- this is
                    # only the index row, which a rebuild would not recreate
                    # because no memory references it.
                    db.prune_orphan_sources(self.conn)
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

    # ------------------------------------------------------------------ retrieval

    def search(
        self,
        query: str | None = None,
        *,
        filters: SearchFilters | None = None,
        **kwargs: Any,
    ) -> list[Hit]:
        """Keyword and/or metadata search. See :class:`SearchFilters`."""
        if filters is not None and kwargs:
            raise ValueError("pass either filters= or keyword filters, not both")
        return search(self.conn, query, filters or SearchFilters(**kwargs))

    def get_related(self, mid: str, **kwargs: Any) -> list[RelatedHit]:
        return get_related(self.conn, mid, **kwargs)

    def get_context(
        self,
        query: str = "",
        *,
        token_budget: int = 2000,
        filters: SearchFilters | None = None,
        **kwargs: Any,
    ) -> ContextResult:
        """Assemble a token-budgeted context pack. See :mod:`memory.context`."""
        return get_context(
            self.conn,
            query,
            token_budget=token_budget,
            filters=filters,
            **kwargs,
        )

    def profile(self, *, limit: int = 50) -> list[Hit]:
        """Memories flagged as always-relevant."""
        return profile_memories(self.conn, limit=limit)

    def review_due(self, *, limit: int = 20) -> list[Hit]:
        return self.search(filters=SearchFilters(review_due=True, limit=limit, order="updated"))

    # ------------------------------------------------------------------ assets

    def store_asset(
        self,
        path_or_bytes: Path | str | bytes,
        *,
        media_type: str | None = None,
        filename: str | None = None,
        tags: list[str] | None = None,
        source: SourceRef | None = None,
        meta: dict[str, Any] | None = None,
        link_to: list[str] | None = None,
    ) -> Asset:
        """Import a file into managed storage.

        Identical bytes are stored once, but each import gets its own asset
        id: the same PDF can be two different things in the user's world, and
        collapsing them would conflate those.
        """
        self._require_writable()
        src = Path(path_or_bytes) if isinstance(path_or_bytes, str | Path) else path_or_bytes

        with self.journal:
            blob = self.store.put_blob(src)
            asset = Asset(
                sha256=blob.sha256,
                media_type=(
                    media_type
                    or (self.store.guess_media_type(src) if isinstance(src, Path) else None)
                    or "application/octet-stream"
                ),
                bytes=blob.bytes,
                original_filename=filename or (src.name if isinstance(src, Path) else None),
                tags=clean_tags(tags or []),
                meta=dict(meta or {}),
                source=source,
            )
            meta_path = self.store.write_asset_meta(asset)
            db.index_asset(
                self.conn,
                asset,
                blob_path=self.paths.rel(blob.path),
                meta_path=self.paths.rel(meta_path),
            )
            for mid in link_to or []:
                memory = self.store.require(mid)
                if asset.id not in memory.assets:
                    updated = memory.with_updates(
                        assets=[*memory.assets, asset.id], updated_at=now_utc()
                    )
                    result = self.store.overwrite(updated)
                    self._reindex_one(mid)
                    if result is not None:
                        self.journal.commit(result.op_id)
            self.conn.commit()
        return asset

    def get_asset(self, asset_id: str, *, with_derived: bool = False) -> dict[str, Any] | None:
        asset = self.store.read_asset_meta(asset_id)
        if asset is None:
            return None
        payload = asset.to_json()
        payload["blob_path"] = self.paths.rel(self.paths.asset_blob(asset.sha256))
        payload["exists"] = self.paths.asset_blob(asset.sha256).exists()
        if with_derived:
            payload["derivations"] = [
                dict(row)
                for row in self.conn.execute(
                    "SELECT id, kind, processor, processor_version, status, output_path "
                    "FROM derivations WHERE asset_id = ? ORDER BY kind, created_at",
                    (asset_id,),
                )
            ]
        return payload

    def replace_asset(
        self,
        asset_id: str,
        path_or_bytes: Path | str | bytes,
        *,
        discard_previous: bool = False,
        media_type: str | None = None,
    ) -> Asset:
        """Point an existing asset at new bytes, keeping its id.

        The id identifies the thing, not the bytes, so every memory that
        references it keeps working. The prior blob is retained by default --
        cheap version history is the point of content addressing.
        """
        self._require_writable()
        existing = self.store.require_asset(asset_id)
        src = Path(path_or_bytes) if isinstance(path_or_bytes, str | Path) else path_or_bytes

        with self.journal:
            blob = self.store.put_blob(src)
            updated = Asset(
                id=existing.id,
                sha256=blob.sha256,
                media_type=media_type or existing.media_type,
                bytes=blob.bytes,
                original_filename=(
                    src.name if isinstance(src, Path) else existing.original_filename
                ),
                created_at=existing.created_at,
                tags=existing.tags,
                meta=existing.meta,
                source=existing.source,
                prev_sha256=None if discard_previous else existing.sha256,
                version=existing.version + 1,
            )
            meta_path = self.store.write_asset_meta(updated)
            db.index_asset(
                self.conn,
                updated,
                blob_path=self.paths.rel(blob.path),
                meta_path=self.paths.rel(meta_path),
            )
            self.conn.commit()
            if (
                discard_previous
                and existing.sha256 != blob.sha256
                and self.store.blob_refcount(existing.sha256) == 0
            ):
                self.store.unlink_blob(existing.sha256)
        return updated

    def delete_asset(
        self, asset_id: str, *, permanent: bool = False, gc_blobs: bool = False
    ) -> dict[str, Any]:
        """Mark an asset deleted, or remove its metadata entirely.

        Blobs survive unless ``gc_blobs`` is set and nothing else references
        them, since the same bytes may back another asset or an older version.
        """
        self._require_writable()
        asset = self.store.require_asset(asset_id)
        removed_blobs: list[str] = []

        with self.journal:
            if permanent:
                self.store.delete_asset_meta(asset_id)
                db.unindex_asset(self.conn, asset_id)
            else:
                marked = Asset(
                    id=asset.id,
                    sha256=asset.sha256,
                    media_type=asset.media_type,
                    bytes=asset.bytes,
                    original_filename=asset.original_filename,
                    created_at=asset.created_at,
                    imported_at=asset.imported_at,
                    status="deleted",
                    tags=asset.tags,
                    meta=asset.meta,
                    source=asset.source,
                    prev_sha256=asset.prev_sha256,
                    version=asset.version,
                )
                meta_path = self.store.write_asset_meta(marked)
                db.index_asset(
                    self.conn,
                    marked,
                    blob_path=self.paths.rel(self.paths.asset_blob(marked.sha256)),
                    meta_path=self.paths.rel(meta_path),
                )
            self.conn.commit()

            if gc_blobs:
                for digest in filter(None, (asset.sha256, asset.prev_sha256)):
                    if self.store.blob_refcount(digest) == 0 and self.store.unlink_blob(digest):
                        removed_blobs.append(digest)

        return {
            "asset_id": asset_id,
            "permanent": permanent,
            "removed_blobs": removed_blobs,
        }

    def register_derivation(
        self,
        *,
        asset_id: str,
        kind: str,
        processor: str,
        processor_version: str,
        output_path: Path,
        params_hash: str = "default",
        segments: list[dict[str, Any]] | None = None,
        status: str = "complete",
        cost_ms: int | None = None,
    ) -> dict[str, Any]:
        """Index a derived output a processor has already written.

        The output file must embed its own cache key (a ``_derivation`` key in
        JSON, YAML frontmatter in Markdown), because that header is what lets
        ``rebuild_index`` reconstruct these rows from the filesystem alone.
        This method and the rebuild path therefore agree by construction.

        ``cost_ms`` is accepted for call-site convenience but is only indexed
        when the processor embedded it in that header, for the same reason:
        anything this method stores that the header lacks is state a rebuild
        would silently drop.
        """
        del cost_ms
        self._require_writable()
        from .lifecycle import read_derivation_header

        asset = self.store.require_asset(asset_id)
        derivation_id = output_path.stem

        # The sidecar header in the output file is authoritative: it is what a
        # rebuild will read, so taking the row from it keeps the live path and
        # the rebuild path in exact agreement.
        header = read_derivation_header(output_path) or {}
        row: dict[str, Any] = {
            "id": str(header.get("id") or derivation_id),
            "asset_id": asset.id,
            "source_id": header.get("source_id"),
            "kind": str(header.get("kind") or kind),
            "processor": str(header.get("processor") or processor),
            "processor_version": str(header.get("processor_version") or processor_version),
            "input_hash": str(header.get("input_hash") or asset.sha256),
            "params_hash": str(header.get("params_hash") or params_hash),
            "output_path": self.paths.rel(output_path),
            "output_hash": header.get("output_hash"),
            "status": str(header.get("status") or status),
            "error": header.get("error"),
            "created_at": str(header.get("created_at") or fmt_ts(now_utc())),
            # Absent in the header means absent, full stop. Inventing a value
            # here would be a field the rebuild cannot reproduce.
            "completed_at": header.get("completed_at"),
            # `cost_ms` obeys the same rule. The caller's measurement is only
            # used when the processor embedded it, because a timing that lives
            # in the index alone makes `rebuild` produce a different row --
            # which tests/test_rebuild_equivalence.py asserts cannot happen.
            "cost_ms": header.get("cost_ms"),
        }
        with self.journal:
            db.index_derivation(self.conn, row)
            for i, raw in enumerate(segments or []):
                db.index_segment(
                    self.conn,
                    Segment(
                        id=str(raw.get("id") or f"{row['id']}:{i}"),
                        text=str(raw["text"]),
                        ordinal=int(raw.get("ordinal", i)),
                        asset_id=asset.id,
                        derivation_id=str(row["id"]),
                        start_ms=raw.get("start_ms"),
                        end_ms=raw.get("end_ms"),
                        page=raw.get("page"),
                        bbox=raw.get("bbox"),
                        speaker=raw.get("speaker"),
                        meta=dict(raw.get("meta") or {}),
                    ),
                )
            self.conn.commit()
        return row

    # ------------------------------------------------------------------ ingestion

    def ingest(self, path: Path | str, **kwargs: Any) -> dict[str, Any]:
        """Import a file and derive searchable content from it.

        Cheap processors run inline; expensive ones are queued. See
        :mod:`memory.ingestion`.
        """
        self._require_writable()
        from .ingestion import ingest

        return ingest(self, Path(path), **kwargs).to_json()

    def ingest_directory(self, directory: Path | str, **kwargs: Any) -> list[dict[str, Any]]:
        self._require_writable()
        from .ingestion import ingest_directory

        return [r.to_json() for r in ingest_directory(self, Path(directory), **kwargs)]

    def process_asset(self, asset_id: str, **kwargs: Any) -> dict[str, Any]:
        """Derive content for an asset that is already stored."""
        self._require_writable()
        from .ingestion import process_asset

        return process_asset(self, asset=asset_id, **kwargs)

    def search_segments(
        self,
        query: str,
        *,
        asset_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search over segments, with provenance on every hit.

        Returns the locator a citation needs: the asset, the derivation, and
        whichever of timestamp/page/bbox the processor recorded.
        """
        from .search import fts_query

        sql = """
            SELECT s.id, s.asset_id, s.derivation_id, s.source_id, s.ordinal,
                   s.text, s.start_ms, s.end_ms, s.page, s.bbox, s.speaker,
                   s.meta, bm25(segments_fts) AS rank,
                   d.kind, d.processor, d.output_path,
                   a.media_type, a.original_filename
              FROM segments_fts
              JOIN segments    s ON s.id = segments_fts.segment_id
              LEFT JOIN derivations d ON d.id = s.derivation_id
              LEFT JOIN assets      a ON a.id = s.asset_id
             WHERE segments_fts MATCH ?
        """
        params: list[Any] = [fts_query(query)]
        if asset_id:
            sql += " AND s.asset_id = ?"
            params.append(asset_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        out: list[dict[str, Any]] = []
        for row in self.conn.execute(sql, params):
            out.append(
                {
                    "segment_id": row["id"],
                    "asset_id": row["asset_id"],
                    "derivation_id": row["derivation_id"],
                    "source_id": row["source_id"],
                    "ordinal": row["ordinal"],
                    "text": row["text"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "page": row["page"],
                    "bbox": row["bbox"],
                    "speaker": row["speaker"],
                    "meta": json.loads(row["meta"] or "{}"),
                    "kind": row["kind"],
                    "processor": row["processor"],
                    "output_path": row["output_path"],
                    "media_type": row["media_type"],
                    "original_filename": row["original_filename"],
                    "score": -float(row["rank"]) if row["rank"] is not None else 0.0,
                }
            )
        return out

    def get_segment(self, segment_id: str) -> dict[str, Any] | None:
        """One segment with its full provenance chain."""
        row = self.conn.execute(
            """
            SELECT s.*, d.kind, d.processor, d.processor_version, d.output_path,
                   a.media_type, a.original_filename, a.sha256
              FROM segments s
              LEFT JOIN derivations d ON d.id = s.derivation_id
              LEFT JOIN assets      a ON a.id = s.asset_id
             WHERE s.id = ?
            """,
            (segment_id,),
        ).fetchone()
        if row is None:
            return None
        payload = _row_to_dict(row)
        payload["meta"] = json.loads(payload.get("meta") or "{}")
        return payload

    def asset_segments(self, asset_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        """Every segment derived from one asset, in order."""
        rows = self.conn.execute(
            "SELECT id, ordinal, text, start_ms, end_ms, page, speaker, derivation_id "
            "FROM segments WHERE asset_id = ? ORDER BY ordinal LIMIT ?",
            (asset_id, limit),
        )
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------ queue

    def queue_list(self, **kwargs: Any) -> list[dict[str, Any]]:
        from . import workqueue

        return [t.to_json() for t in workqueue.listing(self.conn, **kwargs)]

    def queue_stats(self) -> dict[str, Any]:
        from . import workqueue

        return workqueue.stats(self.conn)

    def queue_run(self, *, limit: int | None = None, lease_s: int | None = None) -> dict[str, Any]:
        """Drain ready tasks, running each one's processor."""
        self._require_writable()
        from . import workqueue
        from .ingestion import run_queued_task

        def handler(task: workqueue.Task) -> dict[str, Any]:
            return run_queued_task(self, task)

        kwargs: dict[str, Any] = {"limit": limit}
        if lease_s is not None:
            kwargs["lease_s"] = lease_s
        return workqueue.drain(self.conn, handler, **kwargs)

    def queue_retry(self, **kwargs: Any) -> dict[str, Any]:
        self._require_writable()
        from . import workqueue

        revived = workqueue.retry(self.conn, **kwargs)
        return {"retried": revived, "count": len(revived), **workqueue.stats(self.conn)}

    def queue_cancel(self, task_id: str) -> dict[str, Any]:
        self._require_writable()
        from . import workqueue

        return {"task_id": task_id, "cancelled": workqueue.cancel(self.conn, task_id)}

    def queue_purge(self, *, keep_days: int = 7) -> dict[str, Any]:
        self._require_writable()
        from . import workqueue

        return {"removed": workqueue.purge(self.conn, keep_days=keep_days)}

    # ------------------------------------------------------------------ intelligence

    def provider(self) -> LLMProvider | None:
        """The configured LLM provider, or ``None`` when none is set up.

        ``None`` is the normal state, not a failure: every caller either
        degrades to a deterministic path or raises with instructions. See
        :mod:`memory.providers`.
        """
        from .providers import load_provider

        return load_provider(self.paths.data)

    def provider_config(self) -> dict[str, Any]:
        from .providers import read_config

        return read_config(self.paths.data).to_json()

    def consolidate(self, *, use_llm: bool = False, **kwargs: Any) -> ConsolidationReport:
        """Scan for duplicates, contradictions and stale memories.

        Read-only: it returns proposals naming a verb and its arguments.
        Nothing is written unless the caller passes them to
        :meth:`apply_consolidation`.
        """
        from . import consolidate as consolidation

        provider = self.provider() if use_llm else None
        return consolidation.detect(self, provider=provider, **kwargs)

    def apply_consolidation(
        self, report: ConsolidationReport, *, kinds: list[str] | None = None
    ) -> ConsolidationReport:
        from . import consolidate as consolidation

        self._require_writable()
        return consolidation.apply_all(self, report, kinds=kinds)

    def extract(
        self,
        text: str,
        *,
        source: SourceRef,
        use_llm: bool = False,
        check_duplicates: bool = True,
        **kwargs: Any,
    ) -> ExtractionReport:
        """Propose candidate memories from source material. Writes nothing."""
        from . import extract as extraction

        provider = self.provider() if use_llm else None
        report = extraction.from_text(text, source=source, provider=provider, **kwargs)
        if check_duplicates:
            extraction.annotate_duplicates(self, report)
        return report

    def extract_suggestions(
        self,
        suggestions: list[dict[str, Any]],
        *,
        source: SourceRef | None = None,
        check_duplicates: bool = True,
    ) -> ExtractionReport:
        """Turn ``ProcessResult.memory_suggestions`` into candidates."""
        from . import extract as extraction

        report = extraction.from_suggestions(suggestions, source=source)
        if check_duplicates:
            extraction.annotate_duplicates(self, report)
        return report

    def commit_extraction(self, report: ExtractionReport, **kwargs: Any) -> ExtractionReport:
        from . import extract as extraction

        self._require_writable()
        return extraction.commit(self, report, **kwargs)

    # ------------------------------------------------------------------ cascade

    def plan_source_deletion(self, source_ref: str, **kwargs: Any) -> DeletionPlan:
        """The full blast radius of deleting a source. Read-only."""
        from . import cascade

        return cascade.plan_deletion(self, source_ref, **kwargs)

    def delete_source(self, source_ref: str, **kwargs: Any) -> DeletionResult:
        from . import cascade

        self._require_writable()
        return cascade.delete_source(self, source_ref, **kwargs)

    def orphaned_memories(self) -> list[dict[str, Any]]:
        from . import cascade

        return cascade.orphaned_memories(self)

    # ------------------------------------------------------------------ backup

    def backup(self, **kwargs: Any) -> BackupResult:
        from . import backup as backups

        self._require_writable()
        return backups.backup(self, **kwargs)

    def list_backups(self, **kwargs: Any) -> list[dict[str, Any]]:
        from . import backup as backups

        return backups.list_backups(self.paths, **kwargs)

    def prune_backups(self, **kwargs: Any) -> dict[str, Any]:
        from . import backup as backups

        return backups.prune(self.paths, **kwargs)

    def export_memories(
        self, *, format: str = "jsonl", out: Path | None = None, **kw: Any
    ) -> ExportResult:
        from . import backup as backups

        if format == "jsonl":
            return backups.export_jsonl(self, out=out, **kw)
        if format == "markdown":
            if out is None:
                raise ValueError("the markdown format writes a directory; pass out=")
            return backups.export_markdown_dir(self, out, **kw)
        raise ValueError(f"unknown export format: {format}")

    def import_memories(
        self, source: Path | str, *, format: str = "jsonl", **kw: Any
    ) -> ImportResult:
        from . import backup as backups

        self._require_writable()
        if format == "jsonl":
            return backups.import_jsonl(self, source, **kw)
        if format == "markdown":
            return backups.import_markdown_dir(self, Path(source), **kw)
        raise ValueError(f"unknown import format: {format}")

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

    def tags(
        self,
        *,
        facet: str = "tag",
        prefix: str | None = None,
        min_count: int = 1,
        limit: int = 100,
        include_inactive: bool = False,
        include_secret: bool = False,
        near_misses: bool = False,
    ) -> dict[str, Any]:
        """Existing tag (or entity) vocabulary with usage counts.

        The point is reuse: an agent that cannot see the vocabulary coins a new
        tag every session, and ``--tag synctasks`` then silently returns a
        subset of the memories about SyncTasks. Counting is a GROUP BY over the
        normalized side table, which is already indexed by value.

        Counts cover ``active`` memories only unless ``include_inactive``: a
        tag carried solely by superseded or deleted memories is not something
        to scope a recall by, and including it would overstate what a filter
        would return.

        ``secret`` memories are excluded unless ``include_secret``, matching
        the default of ``search`` and ``get_context``. Two reasons, and the
        second is the stronger one. A count that included them would stop
        predicting what ``--tag <value>`` returns, which is the whole job of
        this listing. And the tag string is itself a disclosure: ``secret``
        exists for memories whose *subject* -- an unannounced project, a
        client name -- should not surface by default, so publishing the slug
        and a usage count would leak exactly what the sensitivity protects.
        """
        if facet not in ("tag", "entity"):
            raise ValueError(f"facet: expected 'tag' or 'entity', got {facet!r}")
        if limit < 1:
            raise ValueError(f"limit: must be >= 1, got {limit}")
        if min_count < 1:
            raise ValueError(f"min_count: must be >= 1, got {min_count}")

        table = "memory_tags" if facet == "tag" else "memory_entities"
        column = facet
        where = ["1=1"] if include_inactive else ["m.status = 'active'"]
        if not include_secret:
            where.append("m.sensitivity != 'secret'")
        params: list[Any] = []
        if prefix:
            # LIKE would treat % and _ in the prefix as wildcards, and `_` is
            # legal in a tag. Compare the slice instead.
            where.append(f"substr(t.{column}, 1, ?) = ?")
            params += [len(prefix), prefix]

        rows = self.conn.execute(
            # `table` and `column` come from the facet check above, never from input.
            f"SELECT t.{column} AS value, count(*) AS c"
            f" FROM {table} t JOIN memories m ON m.id = t.memory_id"
            f" WHERE {' AND '.join(where)}"
            f" GROUP BY t.{column} HAVING c >= ?"
            # Name is the tiebreaker, so equal counts never reorder between
            # runs -- otherwise SQLite's grouping order leaks into the output.
            f" ORDER BY c DESC, value ASC LIMIT ?",
            [*params, min_count, limit],
        ).fetchall()

        items = [{facet: row["value"], "count": row["c"]} for row in rows]
        total = self.conn.execute(
            f"SELECT count(DISTINCT t.{column}) AS n"
            f" FROM {table} t JOIN memories m ON m.id = t.memory_id"
            f" WHERE {' AND '.join(where)}",
            params,
        ).fetchone()["n"]

        out: dict[str, Any] = {
            "facet": facet,
            "total": total,
            "returned": len(items),
            "tags" if facet == "tag" else "entities": items,
        }
        if near_misses:
            out["near_misses"] = _near_misses([(i[facet], i["count"]) for i in items])
        return out

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
            "install": _install_health(),
            "sqlite": sqlite3.sqlite_version,
            "fts5": db.has_fts5(self.conn),
            "schema_version": db.schema_version(self.conn),
            "requires_rebuild": db.requires_rebuild(self.conn),
            "open_journal_ops": open_ops,
            "last_recovery": self.recovery.to_json() if self.recovery else None,
            "ffmpeg": shutil.which("ffmpeg"),
            "media": media.capability_report(),
            "queue": self.queue_stats(),
        }


# ---------------------------------------------------------------- helpers


def _skeleton(value: str) -> str:
    """Collapse the variations ``normalize_tag`` deliberately keeps distinct.

    ``normalize_tag`` lowercases and turns whitespace into hyphens, so
    ``sync-tasks``, ``sync_tasks`` and ``synctasks`` all survive as three
    separate tags. That is exactly the drift an agent produces across
    sessions, and dropping the separators is what makes the three collapse
    onto one key.
    """
    return value.casefold().replace("-", "").replace("_", "")


# Shortest skeleton the one-edit rule is allowed to judge. At four characters
# a single edit is still a quarter of the tag, and real vocabularies are dense
# with distinct four-letter tags one edit apart (`code`/`core`/`mode`/`node`,
# `data`/`date`, `mail`/`main`, `beta`/`meta`), so a floor of four re-creates
# the chaining it is meant to stop. Five admits none of those while keeping
# every genuine drift pair, which is longer: `backend`/`backen`,
# `synctask`/`synctasks`, `sync-tasks`/`synk-tasks`. The cost is `doc`/`docs`,
# accepted -- silence is better here than confidently advising a bad merge.
_MIN_EDIT_SKELETON = 5


def _within_one_edit(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by one insertion, deletion or swap.

    Exact rather than a similarity score, and cheap: a length gap above one
    already answers no, so the single scan never needs an edit matrix.
    """
    if a == b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    spent = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if spent:
            return False
        spent = True
        j += 1
        if len(a) == len(b):  # substitution: advance both
            i += 1
    return True


def _near_misses(items: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Group values that look like drifted spellings of one another.

    Deterministic and dependency-free, over the listed values only, so the
    common path pays nothing for it. The counts are the tell: ``synctasks 40``
    beside ``sync-tasks 12`` is plainly one vocabulary that split in two, not
    two separate topics.
    """
    counts = dict(items)
    values = list(counts)
    parent = {v: v for v in values}

    def find(v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_skeleton: dict[str, list[str]] = {}
    for value in values:
        by_skeleton.setdefault(_skeleton(value), []).append(value)
    for members in by_skeleton.values():
        for other in members[1:]:
            union(members[0], other)

    for index, value in enumerate(values):
        for other in values[index + 1 :]:
            a, b = _skeleton(value), _skeleton(other)
            if min(len(a), len(b)) < _MIN_EDIT_SKELETON:
                # On a short tag one edit is most of the tag, so distance 1 is
                # no evidence of drift: `ci`/`ui`, `code`/`core`, `data`/`date`
                # are distinct topics, and union-find would chain them into one
                # group (`ci`-`ui`-`ux`) and advise merging unrelated
                # vocabulary. Separator drift is unaffected -- the skeleton
                # rule above has already grouped it at any length.
                continue
            if _within_one_edit(a, b):
                union(value, other)

    groups: dict[str, list[str]] = {}
    for value in values:
        groups.setdefault(find(value), []).append(value)

    drifted = [sorted(members) for members in groups.values() if len(members) > 1]
    drifted.sort(key=lambda members: (-sum(counts[v] for v in members), members[0]))
    return [
        {
            "members": [{"value": v, "count": counts[v]} for v in members],
            "total": sum(counts[v] for v in members),
        }
        for members in drifted
    ]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """``sqlite3.Row`` is not a mapping -- ``in`` tests values, not keys."""
    return {key: row[key] for key in row.keys()}  # noqa: SIM118


def _install_health() -> dict[str, Any]:
    """Report whether this package is importable outside its own directory.

    An editable install normally works via a ``.pth`` path hook. Installers
    on macOS may write that file with the ``UF_HIDDEN`` flag, which CPython
    >= 3.12 skips, so the package imports only when the project directory
    happens to be on the path -- the installed console script then fails
    everywhere else. Surfacing it here turns a confusing ImportError into a
    diagnosis.
    """
    import sys

    import memory

    package_dir = Path(memory.__file__).resolve().parent
    site_dirs = [Path(p) for p in sys.path if p.endswith("site-packages")]
    installed = any(package_dir.is_relative_to(d) for d in site_dirs)

    skipped: list[str] = []
    for site_dir in site_dirs:
        for pth in site_dir.glob("*.pth"):
            try:
                flags = pth.stat().st_flags
            except (AttributeError, OSError):
                continue
            if flags & 0x8000:  # UF_HIDDEN
                skipped.append(pth.name)

    health: dict[str, Any] = {
        "mode": "regular" if installed else "editable",
        "package_dir": str(package_dir),
    }
    if not installed and skipped:
        health["warning"] = (
            "editable install path hook is hidden and will be skipped by this "
            f"Python ({', '.join(sorted(skipped))}); the `mem` console script "
            "will fail outside the project directory. Fix: uv pip install ."
        )
    return health


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


#: Fields a replacement inherits from the memory it supersedes unless the
#: caller overrides them. Deliberately excludes status/supersedes/id and the
#: timestamps, which the supersession itself sets.
CARRIED = (
    "tags",
    "entities",
    "assets",
    "sources",
    "relations",
    "confidence",
    "salience",
    "profile",
    "sensitivity",
    "occurred_at",
    "valid_from",
    "review_after",
    "extra",
)


def _carry_forward(old: Memory, overrides: dict[str, Any]) -> dict[str, Any]:
    """Inherit context from the superseded memory.

    A replacement that silently dropped the old memory's sources or tags
    would quietly destroy provenance, so inheritance is the default and the
    caller opts out per field.
    """
    carried: dict[str, Any] = {}
    for key in CARRIED:
        carried[key] = overrides.pop(key, getattr(old, key))
    carried.update(overrides)
    return carried


def _merge_fields(parents: list[Memory], *, strategy: str) -> dict[str, Any]:
    """Combine parents into one field set.

    Sources, tags, entities, assets and relations are always unioned -- the
    merged memory must cite everything its parents cited. ``strategy`` only
    decides which parent's *type* and body ordering lead.
    """
    # Timestamps are second-precision, so two memories written in the same
    # second would otherwise order arbitrarily. Falling back to the caller's
    # argument order keeps a merge reproducible.
    position = {m.id: i for i, m in enumerate(parents)}
    if strategy == "prefer-newest":
        ordered = sorted(parents, key=lambda m: (m.updated_at, -position[m.id]), reverse=True)
    elif strategy == "prefer-highest-confidence":
        ordered = sorted(
            parents,
            key=lambda m: (m.confidence, m.updated_at, -position[m.id]),
            reverse=True,
        )
    else:
        ordered = list(parents)
    lead = ordered[0]

    sources: list[SourceRef] = []
    seen_sources: set[tuple[str, str, str]] = set()
    for parent in ordered:
        for src in parent.sources:
            if src.key() not in seen_sources:
                seen_sources.add(src.key())
                sources.append(src)

    relations: list[Relation] = []
    seen_relations: set[tuple[str, str]] = set()
    merged_ids = {p.id for p in parents}
    for parent in ordered:
        for rel in parent.relations:
            # A relation pointing at another parent becomes self-referential
            # once they collapse into one memory.
            if rel.target in merged_ids:
                continue
            key = (rel.type.value, rel.target)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append(rel)

    occurred = [p.occurred_at for p in parents if p.occurred_at is not None]
    body = "\n\n".join(
        f"## from {p.title} ({p.id[:8]})\n\n{p.body}".rstrip() for p in ordered if p.body.strip()
    )

    return {
        "type": lead.type,
        "body": body,
        "tags": clean_tags([t for p in ordered for t in p.tags]),
        "entities": clean_entities([e for p in ordered for e in p.entities]),
        "assets": list(dict.fromkeys(a for p in ordered for a in p.assets)),
        "sources": sources,
        "relations": relations,
        "confidence": max(p.confidence for p in parents),
        "salience": max(p.salience for p in parents),
        "profile": any(p.profile for p in parents),
        "sensitivity": max(
            (p.sensitivity for p in parents), key=lambda s: SENSITIVITY_ORDER[s.value]
        ),
        "created_at": min(p.created_at for p in parents),
        "occurred_at": min(occurred) if occurred else None,
    }


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
    "ContextResult",
    "DeleteResult",
    "Hit",
    "MemoryAPI",
    "MergeResult",
    "NotFound",
    "Relation",
    "ReplaceResult",
    "SearchError",
    "SearchFilters",
    "SourceRef",
    "Tombstoned",
]
