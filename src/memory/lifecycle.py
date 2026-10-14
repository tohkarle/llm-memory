"""Index rebuilding, verification, and (in step 6) supersession.

``rebuild_index`` is the load-bearing guarantee of the whole design: given
only ``data/``, it reconstructs every index row. If it drifts from what the
write path produces, the filesystem-is-truth claim is false, so
``tests/test_rebuild_equivalence.py`` asserts byte-equality of a full table
dump before and after.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import db
from .journal import Journal
from .models import (
    Segment,
    SourceRef,
    ValidationError,
    is_uuid,
)
from .paths import Paths
from .store import Store


def rebuild_index(
    paths: Paths,
    conn: sqlite3.Connection,
    *,
    journal: Journal | None = None,
    verify_after: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Reconstruct the entire index from the filesystem.

    Tolerant by design: an unparseable file is reported and skipped rather
    than aborting the rebuild, because partial corruption must not cost you
    the other 199 memories.
    """
    store = Store(paths, journal or Journal(paths))
    errors: list[dict[str, str]] = []
    counts = {
        "memories": 0,
        "assets": 0,
        "sources": 0,
        "tombstones": 0,
        "segments": 0,
        "derivations": 0,
    }

    if db.missing_tables(conn):
        db.init_schema(conn)
    db.clear_all(conn)

    # ---- memories (base rows first, FTS in one pass at the end)
    for path in paths.memory_glob():
        try:
            memory = store.read_path(path)
        except (ValidationError, OSError, UnicodeDecodeError) as exc:
            errors.append({"path": paths.rel(path), "error": str(exc)})
            continue
        try:
            db.index_memory(
                conn,
                memory,
                path=paths.rel(path),
                mtime_ns=store.stat_mtime_ns(path),
            )
        except sqlite3.DatabaseError as exc:
            errors.append({"path": paths.rel(path), "error": str(exc)})
            continue
        counts["memories"] += 1
        if progress is not None:
            progress(counts["memories"])

    # ---- tombstones
    for tomb in store.iter_tombstones():
        db.index_tombstone(conn, tomb, path=paths.rel(paths.tombstone_file(tomb.memory_id)))
        counts["tombstones"] += 1

    # ---- assets
    for asset in store.iter_assets():
        db.index_asset(
            conn,
            asset,
            blob_path=paths.rel(paths.asset_blob(asset.sha256)),
            meta_path=paths.rel(paths.asset_meta(asset.id)),
        )
        counts["assets"] += 1

    # ---- derivations, reconstructed from the sidecar header each derived
    # file carries. Without that header these rows would be index-only state
    # and `rm memory.sqlite` would silently lose the processor cache.
    for path in paths.derived_glob():
        header = read_derivation_header(path)
        if header is None:
            continue
        header.setdefault("id", path.stem)
        header["output_path"] = paths.rel(path)
        try:
            db.index_derivation(conn, header)
        except (sqlite3.DatabaseError, KeyError) as exc:
            errors.append({"path": paths.rel(path), "error": f"derivation: {exc}"})
            continue
        counts["derivations"] += 1
        for seg in _segments_from_derived(path, header):
            db.index_segment(conn, seg)
            counts["segments"] += 1

    counts["sources"] = conn.execute("SELECT count(*) c FROM sources").fetchone()["c"]

    db.set_meta(conn, "schema_version", str(db.SCHEMA_VERSION))
    db.set_meta(conn, "built_at", _now())
    db.set_meta(conn, "dirty_ids", "[]")

    report: dict[str, Any] = {**counts, "errors": errors}
    if verify_after:
        report["integrity"] = db.integrity_check(conn)
        report["verified"] = not report["integrity"]
    return report


def read_derivation_header(path: Path) -> dict[str, Any] | None:
    """Extract the embedded cache key from a derived output file.

    JSON outputs carry a ``_derivation`` key; Markdown outputs carry YAML
    frontmatter. Anything else is not a derivation product.
    """
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            header = payload.get("_derivation")
            return dict(header) if isinstance(header, dict) else None
        if path.suffix in (".md", ".markdown"):
            import yaml

            text = path.read_text(encoding="utf-8")
            if not text.startswith("---"):
                return None
            _, _, rest = text.partition("---")
            front, _, _ = rest.partition("\n---")
            loaded = yaml.safe_load(front)
            if isinstance(loaded, dict) and loaded.get("kind"):
                return dict(loaded)
            return None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return None
    return None


def _segments_from_derived(path: Path, header: dict[str, Any]) -> list[Segment]:
    if path.suffix != ".json":
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []

    out: list[Segment] = []
    for i, raw in enumerate(raw_segments):
        if not isinstance(raw, dict) or not raw.get("text"):
            continue
        seg_id = str(raw.get("id") or "")
        out.append(
            Segment(
                id=seg_id if is_uuid(seg_id) else f"{header['id']}:{i}",
                text=str(raw["text"]),
                ordinal=int(raw.get("ordinal", i)),
                asset_id=header.get("asset_id"),
                source_id=header.get("source_id"),
                derivation_id=str(header["id"]),
                start_ms=raw.get("start_ms"),
                end_ms=raw.get("end_ms"),
                page=raw.get("page"),
                bbox=raw.get("bbox"),
                speaker=raw.get("speaker"),
                meta=dict(raw.get("meta") or {}),
            )
        )
    return out


def verify(paths: Paths, conn: sqlite3.Connection, store: Store | None = None) -> dict[str, Any]:
    """Compare the filesystem against the index and report every divergence.

    Read-only. Everything it reports is actionable: ``mem rebuild`` fixes the
    index-side findings, and the filesystem-side ones need a human.
    """
    store = store or Store(paths, Journal(paths))

    fs_ids: dict[str, str] = {}
    hash_mismatch: list[dict[str, str]] = []
    unreadable: list[dict[str, str]] = []

    for path, problem in store.iter_problems():
        unreadable.append({"path": paths.rel(path), "error": problem})

    for memory in store.iter_all(include_deleted=True):
        path = paths.memory_file(memory.type.value, memory.id)
        fs_ids[memory.id] = memory.content_hash
        row = conn.execute(
            "SELECT content_hash, path FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()
        if row is not None and row["content_hash"] != memory.content_hash:
            hash_mismatch.append(
                {
                    "memory_id": memory.id,
                    "filesystem": memory.content_hash,
                    "index": row["content_hash"],
                }
            )

    db_ids = {row["id"] for row in conn.execute("SELECT id FROM memories")}
    fs_only = sorted(set(fs_ids) - db_ids)
    db_only = sorted(db_ids - set(fs_ids))

    dangling = [
        {"src": row["src_id"], "dst": row["dst_id"], "type": row["type"]}
        for row in conn.execute(
            "SELECT src_id, dst_id, type FROM relations WHERE implied = 0 "
            "AND dst_id NOT IN (SELECT id FROM memories)"
        )
    ]

    missing_tombstones = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM memories WHERE status = 'deleted' "
            "AND id NOT IN (SELECT memory_id FROM tombstones)"
        )
    ]

    orphan_blobs: list[str] = []
    referenced = {
        row["sha256"] for row in conn.execute("SELECT sha256 FROM assets WHERE status='active'")
    } | {
        row["prev_sha256"]
        for row in conn.execute("SELECT prev_sha256 FROM assets WHERE prev_sha256 IS NOT NULL")
    }
    for blob in paths.blob_glob():
        if blob.is_file() and blob.name not in referenced:
            orphan_blobs.append(paths.rel(blob))

    missing_blobs = [
        row["sha256"]
        for row in conn.execute("SELECT sha256 FROM assets WHERE status='active'")
        if not paths.asset_blob(row["sha256"]).exists()
    ]

    return {
        "fs_only": fs_only,
        "db_only": db_only,
        "hash_mismatch": hash_mismatch,
        "unreadable": unreadable,
        "dangling_relations": dangling,
        "missing_tombstones": missing_tombstones,
        "orphan_blobs": orphan_blobs,
        "missing_blobs": missing_blobs,
        "integrity": db.integrity_check(conn),
        "clean": not any(
            [
                fs_only,
                db_only,
                hash_mismatch,
                unreadable,
                missing_tombstones,
                missing_blobs,
                db.integrity_check(conn),
            ]
        ),
    }


def gc_blobs(paths: Paths, conn: sqlite3.Connection, *, dry_run: bool = True) -> dict[str, Any]:
    """Remove blobs no active asset or prior version references."""
    referenced = {
        row["sha256"] for row in conn.execute("SELECT sha256 FROM assets WHERE status='active'")
    } | {
        row["prev_sha256"]
        for row in conn.execute("SELECT prev_sha256 FROM assets WHERE prev_sha256 IS NOT NULL")
    }
    removed: list[str] = []
    freed = 0
    for blob in paths.blob_glob():
        if not blob.is_file() or blob.name in referenced:
            continue
        freed += blob.stat().st_size
        removed.append(paths.rel(blob))
        if not dry_run:
            blob.unlink(missing_ok=True)
    return {"removed": removed, "bytes_freed": freed, "dry_run": dry_run}


def _now() -> str:
    from .models import fmt_ts, now_utc

    return fmt_ts(now_utc()) or ""


def source_rows(memory_sources: list[SourceRef]) -> list[dict[str, Any]]:
    return [s.to_yaml() for s in memory_sources]
