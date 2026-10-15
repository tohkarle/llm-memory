"""Dependency-aware deletion: removing a source removes what was derived from it.

Deleting a source file is not one deletion. A conversation or document has
descendants -- transcripts, descriptions, keyframes, segment rows, the
derivation cache entries that memoize them, and the memories that cite it --
and leaving any of them behind means the user asked for something to be gone
and it is still on disk.

The walk is over the ``derivations`` table, which is the system's record of
"X was computed from Y". Three rules shape this module:

**Plan first, then execute.** :func:`plan_deletion` is read-only and returns
the full blast radius; :func:`execute` performs it. Permanent deletion cannot
be undone, so the user (or an agent) must be able to see the whole list
before any of it happens. ``mem source delete`` prints the plan unless
``--yes``.

**Memories are never silently destroyed.** A memory citing a deleted source
has lost its provenance, but it may still be true and the user may have
curated it by hand. The default is to *orphan* it -- drop the source
reference, flag it for review -- and only ``--delete-memories`` removes them,
through the ordinary delete verb so a tombstone is written.

**Tombstones outlive everything.** They are the only defense against
re-extraction from retained material, so nothing here removes one, and a
permanent cascade writes them for every memory it deletes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import db
from .models import SourceKind, SourceRef, fmt_ts, now_utc

if TYPE_CHECKING:  # pragma: no cover
    from .api import MemoryAPI

#: What happens to a memory whose source is being deleted.
#:
#: ``orphan`` keeps the memory and drops the dangling reference. ``delete``
#: soft-deletes it. ``purge`` removes it permanently. ``keep`` leaves the
#: reference in place, which is only right when the raw file is being removed
#: for space rather than because the user wants the content gone.
MemoryPolicy = tuple[str, ...]
MEMORY_POLICIES: MemoryPolicy = ("orphan", "delete", "purge", "keep")

#: Tag applied to an orphaned memory, so ``mem search --tag`` finds the set
#: whose provenance needs re-establishing.
ORPHAN_TAG = "provenance-lost"


@dataclass(slots=True)
class DeletionPlan:
    """Everything a source deletion would touch. Nothing has happened yet."""

    source_ref: str
    source_kind: str
    source_id: str | None
    raw_files: list[str] = field(default_factory=list)
    derivations: list[dict[str, Any]] = field(default_factory=list)
    derived_files: list[str] = field(default_factory=list)
    segments: int = 0
    assets: list[dict[str, Any]] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    memory_policy: str = "orphan"
    warnings: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (
            self.raw_files
            or self.derivations
            or self.derived_files
            or self.segments
            or self.memories
            or self.assets
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source": {
                "ref": self.source_ref,
                "kind": self.source_kind,
                "source_id": self.source_id,
            },
            "memory_policy": self.memory_policy,
            "counts": {
                "raw_files": len(self.raw_files),
                "derivations": len(self.derivations),
                "derived_files": len(self.derived_files),
                "segments": self.segments,
                "assets": len(self.assets),
                "memories": len(self.memories),
            },
            "raw_files": self.raw_files,
            "derivations": self.derivations,
            "derived_files": self.derived_files,
            "assets": self.assets,
            "memories": self.memories,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class DeletionResult:
    plan: DeletionPlan
    removed_files: list[str] = field(default_factory=list)
    removed_derivations: list[str] = field(default_factory=list)
    removed_assets: list[str] = field(default_factory=list)
    removed_segments: int = 0
    orphaned_memories: list[str] = field(default_factory=list)
    deleted_memories: list[str] = field(default_factory=list)
    kept_memories: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_json(),
            "removed_files": self.removed_files,
            "removed_derivations": self.removed_derivations,
            "removed_assets": self.removed_assets,
            "removed_segments": self.removed_segments,
            "orphaned_memories": self.orphaned_memories,
            "deleted_memories": self.deleted_memories,
            "kept_memories": self.kept_memories,
            "errors": self.errors,
        }


# ---------------------------------------------------------------- planning


def plan_deletion(
    api: MemoryAPI,
    source_ref: str,
    *,
    kind: str | None = None,
    memory_policy: str = "orphan",
) -> DeletionPlan:
    """Compute the full blast radius of deleting a source. Read-only.

    ``source_ref`` is the ref as memories cite it (``conversations/x.md``),
    which is deliberately the same string ``--source`` takes: a user should
    not need to know an internal id to delete something they named.
    """
    if memory_policy not in MEMORY_POLICIES:
        raise ValueError(
            f"unknown memory policy {memory_policy!r}; one of {', '.join(MEMORY_POLICIES)}"
        )

    row = _find_source(api, source_ref, kind=kind)
    source_id = str(row["id"]) if row is not None else None
    source_kind = str(row["kind"]) if row is not None else (kind or "unknown")

    plan = DeletionPlan(
        source_ref=source_ref,
        source_kind=source_kind,
        source_id=source_id,
        memory_policy=memory_policy,
    )

    if row is None:
        # Not in the index is not the same as not on disk: the file may exist
        # with nothing citing it yet. Removing it is still the right action,
        # so plan for the file and say why nothing else was found.
        plan.warnings.append(
            f"no indexed source matches {source_ref!r}; "
            "only raw files will be removed (nothing cites it)"
        )

    # ---- raw files
    for path in _raw_paths(api, source_ref):
        plan.raw_files.append(api.paths.rel(path))

    # ---- assets imported from this source
    plan.assets = _assets_from_source(api, source_ref, source_id=source_id)

    # ---- derivations computed from this source, directly or via its assets
    asset_ids = [str(a["id"]) for a in plan.assets]
    for deriv in _derivation_rows(api, source_id=source_id, asset_ids=asset_ids):
        plan.derivations.append(deriv)
        output = deriv.get("output_path")
        if output:
            plan.derived_files.append(str(output))

    # ---- segment rows those derivations produced
    plan.segments = _segment_count(
        api, source_id=source_id, derivation_ids=[str(d["id"]) for d in plan.derivations]
    )

    # ---- memories citing this source
    plan.memories = _citing_memories(api, source_ref)

    return plan


def _find_source(api: MemoryAPI, source_ref: str, *, kind: str | None) -> Any:
    sql = "SELECT id, kind, ref FROM sources WHERE ref = ?"
    params: list[Any] = [source_ref]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    row = api.conn.execute(sql, params).fetchone()
    if row is not None:
        return row
    # Fall back to a suffix match so `mem source delete 2026-10-15.md` works
    # the way a user would expect when they do not recall the directory.
    sql = "SELECT id, kind, ref FROM sources WHERE ref LIKE ?"
    params = [f"%{source_ref}"]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    rows = api.conn.execute(sql + " LIMIT 2", params).fetchall()
    return rows[0] if len(rows) == 1 else None


def _raw_paths(api: MemoryAPI, source_ref: str) -> list[Path]:
    """Locate the raw file(s) a source ref names, under ``data/``.

    Confined to ``data/``: a ref is a relative path by contract, and resolving
    one that escapes the tree would let a crafted memory point deletion at an
    arbitrary file.
    """
    candidate = api.paths.data / source_ref
    out: list[Path] = []
    try:
        resolved = candidate.resolve()
        if resolved.is_relative_to(api.paths.data.resolve()) and resolved.is_file():
            out.append(resolved)
    except (OSError, ValueError):
        return out
    return out


def _assets_from_source(
    api: MemoryAPI, source_ref: str, *, source_id: str | None
) -> list[dict[str, Any]]:
    """Assets imported from this source.

    Read from the asset sidecars, because that is where an asset's origin
    actually lives: ``Asset.source`` is written to ``assets/meta/`` and fed
    to ``upsert_source``, but the ``assets`` index table has no source
    column. Deriving this from the index would have silently found nothing --
    and then a rebuild, which *does* re-derive ``sources`` from asset
    sidecars, would resurrect the source row for a source the user deleted.
    """
    out: list[dict[str, Any]] = []
    for asset in api.store.iter_assets():
        if asset.source is None or asset.source.ref != source_ref:
            continue
        out.append(
            {
                "id": asset.id,
                "sha256": asset.sha256,
                "media_type": asset.media_type,
                "original_filename": asset.original_filename,
            }
        )
    # Assets reached only through a derivation of this source still count:
    # the derivation says they were computed from it.
    if source_id is not None:
        known = {a["id"] for a in out}
        for row in api.conn.execute(
            "SELECT DISTINCT asset_id FROM derivations "
            "WHERE source_id = ? AND asset_id IS NOT NULL",
            (source_id,),
        ):
            aid = str(row["asset_id"])
            if aid in known:
                continue
            derived_asset = api.store.read_asset_meta(aid)
            if derived_asset is not None:
                out.append(
                    {
                        "id": derived_asset.id,
                        "sha256": derived_asset.sha256,
                        "media_type": derived_asset.media_type,
                        "original_filename": derived_asset.original_filename,
                    }
                )
    return sorted(out, key=lambda a: str(a["id"]))


def _derivation_rows(
    api: MemoryAPI, *, source_id: str | None, asset_ids: Sequence[str]
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if asset_ids:
        clauses.append(f"asset_id IN ({', '.join('?' * len(asset_ids))})")
        params.extend(asset_ids)
    if not clauses:
        return []
    rows = api.conn.execute(
        "SELECT id, kind, processor, processor_version, status, output_path, asset_id, source_id "
        f"FROM derivations WHERE {' OR '.join(clauses)} ORDER BY kind, id",
        params,
    )
    return [dict(r) for r in rows]


def _segment_count(api: MemoryAPI, *, source_id: str | None, derivation_ids: Sequence[str]) -> int:
    clauses: list[str] = []
    params: list[Any] = []
    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if derivation_ids:
        clauses.append(f"derivation_id IN ({', '.join('?' * len(derivation_ids))})")
        params.extend(derivation_ids)
    if not clauses:
        return 0
    row = api.conn.execute(
        f"SELECT count(*) c FROM segments WHERE {' OR '.join(clauses)}", params
    ).fetchone()
    return int(row["c"]) if row else 0


def _citing_memories(api: MemoryAPI, source_ref: str) -> list[dict[str, Any]]:
    """Memories whose frontmatter cites this ref.

    Read from the filesystem rather than the index: the files are
    authoritative, and this list drives destructive action.
    """
    out: list[dict[str, Any]] = []
    for memory in api.store.iter_all(include_deleted=True):
        matching = [s for s in memory.sources if s.ref == source_ref]
        if not matching:
            continue
        out.append(
            {
                "memory_id": memory.id,
                "title": memory.title,
                "type": memory.type.value,
                "status": memory.status.value,
                "locators": [s.locator for s in matching],
                "other_sources": len(memory.sources) - len(matching),
            }
        )
    return sorted(out, key=lambda m: str(m["memory_id"]))


# ---------------------------------------------------------------- execution


def execute(
    api: MemoryAPI,
    plan: DeletionPlan,
    *,
    remove_raw: bool = True,
    reason: str | None = None,
) -> DeletionResult:
    """Carry out a plan. Destructive and, for permanent parts, irreversible.

    Order matters: memories first, then derived state, then the raw file.
    A memory read after its source file is gone can still be soft-deleted,
    but a crash between the two is less confusing if the record of *why*
    (the memory's tombstone, naming the reason) is already durable.
    """
    result = DeletionResult(plan=plan)
    note = reason or f"source {plan.source_ref} deleted"

    # ---- memories
    for entry in plan.memories:
        mid = str(entry["memory_id"])
        try:
            if plan.memory_policy == "keep":
                result.kept_memories.append(mid)
            elif plan.memory_policy == "orphan":
                _orphan_memory(api, mid, plan.source_ref, note)
                result.orphaned_memories.append(mid)
            elif plan.memory_policy == "delete":
                api.delete(mid, permanent=False, reason=note)
                result.deleted_memories.append(mid)
            elif plan.memory_policy == "purge":
                api.delete(mid, permanent=True, reason=note)
                result.deleted_memories.append(mid)
        except Exception as exc:
            result.errors.append({"memory_id": mid, "error": f"{type(exc).__name__}: {exc}"})

    # ---- derived outputs and their index rows
    for deriv in plan.derivations:
        did = str(deriv["id"])
        output = deriv.get("output_path")
        if output:
            path = api.paths.abs(str(output))
            try:
                if path.is_file():
                    path.unlink()
                    result.removed_files.append(str(output))
            except OSError as exc:
                result.errors.append({"path": str(output), "error": str(exc)})
        try:
            result.removed_segments += db.unindex_segments_for(api.conn, derivation_id=did)
            api.conn.execute("DELETE FROM derivations WHERE id = ?", (did,))
            result.removed_derivations.append(did)
        except Exception as exc:
            result.errors.append({"derivation_id": did, "error": str(exc)})

    # ---- assets imported from this source. These must go, and not only to
    # reclaim space: an asset sidecar records its own source, and
    # `rebuild_index` re-derives the `sources` table from those sidecars. An
    # asset left behind would resurrect the source row for a source the user
    # deleted, which is the filesystem quietly overruling the deletion.
    for asset_entry in plan.assets:
        aid = str(asset_entry["id"])
        if plan.memory_policy == "keep":
            break
        try:
            api.delete_asset(aid, permanent=True, gc_blobs=True)
            result.removed_assets.append(aid)
        except Exception as exc:
            result.errors.append({"asset_id": aid, "error": f"{type(exc).__name__}: {exc}"})

    # Segments attached to the source itself rather than to a derivation.
    if plan.source_id is not None:
        try:
            cur = api.conn.execute("SELECT id FROM segments WHERE source_id = ?", (plan.source_id,))
            stragglers = [str(r["id"]) for r in cur]
            for sid in stragglers:
                api.conn.execute("DELETE FROM segments_fts WHERE segment_id = ?", (sid,))
            api.conn.execute("DELETE FROM segments WHERE source_id = ?", (plan.source_id,))
            result.removed_segments += len(stragglers)
        except Exception as exc:  # pragma: no cover - defensive
            result.errors.append({"source_id": plan.source_id, "error": str(exc)})

    # ---- raw files
    if remove_raw:
        for relpath in plan.raw_files:
            path = api.paths.abs(relpath)
            try:
                if path.is_file():
                    path.unlink()
                    result.removed_files.append(relpath)
            except OSError as exc:
                result.errors.append({"path": relpath, "error": str(exc)})

    # ---- the source row itself, once nothing cites it. A rebuild derives
    # `sources` from what memories cite, so a row left here would reappear or
    # linger depending on which ran last.
    if plan.source_id is not None:
        api.conn.execute("DELETE FROM memory_sources WHERE source_id = ?", (plan.source_id,))
        db.prune_orphan_sources(api.conn)
    api.conn.commit()

    return result


def _orphan_memory(api: MemoryAPI, mid: str, source_ref: str, note: str) -> None:
    """Drop a dangling source reference, keeping the memory and flagging it.

    The memory may well still be true, and it may have been hand-curated, so
    destroying it because its citation went away would lose information the
    user never asked to lose. Tagging it is what makes the loss recoverable:
    ``mem search --tag provenance-lost`` is the review queue.
    """
    memory = api.require(mid)
    remaining = [s for s in memory.sources if s.ref != source_ref]
    patch: dict[str, Any] = {"sources": remaining}
    if ORPHAN_TAG not in memory.tags:
        patch["tags"] = [*memory.tags, ORPHAN_TAG]
    if not remaining:
        # Nothing left to check the claim against, so it needs a human.
        patch["review_after"] = now_utc()
        extra = dict(memory.extra)
        extra["provenance_lost"] = {
            "source_ref": source_ref,
            "at": fmt_ts(now_utc()),
            "reason": note,
        }
        patch["extra"] = extra
    api.update(mid, **patch)


def delete_source(
    api: MemoryAPI,
    source_ref: str,
    *,
    kind: str | None = None,
    memory_policy: str = "orphan",
    remove_raw: bool = True,
    reason: str | None = None,
) -> DeletionResult:
    """Plan and execute in one call. See :func:`plan_deletion` for the rules."""
    plan = plan_deletion(api, source_ref, kind=kind, memory_policy=memory_policy)
    return execute(api, plan, remove_raw=remove_raw, reason=reason)


def orphaned_memories(api: MemoryAPI) -> list[dict[str, Any]]:
    """Memories whose provenance was removed and not re-established."""
    rows = api.conn.execute(
        "SELECT m.id, m.title, m.type, m.review_after FROM memories m "
        "JOIN memory_tags t ON t.memory_id = m.id "
        "WHERE t.tag = ? AND m.status = 'active' ORDER BY m.updated_at DESC",
        (ORPHAN_TAG,),
    )
    return [dict(r) for r in rows]


def source_ref_for(kind: str, ref: str) -> SourceRef:
    return SourceRef(kind=SourceKind(kind), ref=ref)


__all__ = [
    "MEMORY_POLICIES",
    "ORPHAN_TAG",
    "DeletionPlan",
    "DeletionResult",
    "delete_source",
    "execute",
    "orphaned_memories",
    "plan_deletion",
    "source_ref_for",
]
