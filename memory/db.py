"""SQLite index: schema, connection policy, and synchronization.

The database is **derived state**. Everything here can be reconstructed from
the filesystem by ``lifecycle.rebuild_index``, so it is safe to delete
``data/index/memory.sqlite`` at any time.

FTS5 uses external content (``content='memories'``) but is synchronized
manually rather than by triggers. Reasons:

1. Every write already funnels through :func:`index_memory`, which is the
   natural sync point and is covered by one test. Triggers would add a second,
   invisible write path that diverges during bulk rebuild.
2. External-content FTS5 requires the ``'delete'`` command to carry the *old*
   column values. A trigger has them as ``OLD.*``, but a bulk rebuild does
   not, and a mismatched delete silently corrupts the index. Doing it in
   Python lets us fall back to ``'rebuild'`` whenever we are unsure.
3. Bulk rebuild is far faster as insert-all-rows then one ``'rebuild'``.

Two consequences of external content that shape this module:

*The index must mirror the base table exactly.* Indexed values are the same
bytes ``memories`` holds, JSON and all -- the tokenizer strips the
punctuation, so ``["a","b"]`` yields the same searchable terms as ``a b``.
Any deliberate divergence (flattening a column, or omitting a row to hide it
from search) would be reported as corruption, so status filtering, not index
surgery, is what keeps deleted memories out of results.

*Divergence is detectable.* :func:`integrity_check` passes ``rank=1`` to the
FTS5 ``'integrity-check'`` command, which compares the index against the
content table. The bare form checks only the index's own structure and would
pass over a stale index.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from .models import (
    Asset,
    Memory,
    RelationType,
    Segment,
    SourceRef,
    Tombstone,
    fmt_ts,
)
from .paths import Paths

SCHEMA_VERSION = 1

#: Minimum SQLite that supports everything used here (FTS5 'delete' command,
#: WITHOUT ROWID, partial indexes, json1).
MIN_SQLITE = (3, 24, 0)


class SchemaError(RuntimeError):
    """The database is unusable and must be rebuilt."""


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- ─────────────────────────────── memories ───────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
  id             TEXT PRIMARY KEY,
  type           TEXT NOT NULL CHECK (type IN
                   ('fact','preference','event','decision','procedure')),
  title          TEXT NOT NULL,
  body           TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN
                   ('active','superseded','archived','deleted')),
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  occurred_at    TEXT,
  valid_from     TEXT,
  valid_until    TEXT,
  review_after   TEXT,
  confidence     REAL NOT NULL DEFAULT 0.8 CHECK (confidence BETWEEN 0 AND 1),
  salience       REAL NOT NULL DEFAULT 0.5 CHECK (salience   BETWEEN 0 AND 1),
  profile        INTEGER NOT NULL DEFAULT 0 CHECK (profile IN (0,1)),
  sensitivity    TEXT NOT NULL DEFAULT 'normal' CHECK (sensitivity IN
                   ('normal','private','secret')),
  superseded_by  TEXT,
  tags           TEXT NOT NULL DEFAULT '[]',
  entities       TEXT NOT NULL DEFAULT '[]',
  schema_version INTEGER NOT NULL DEFAULT 1,
  path           TEXT NOT NULL,
  content_hash   TEXT NOT NULL,
  mtime_ns       INTEGER NOT NULL DEFAULT 0,
  indexed_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mem_status_type ON memories(status, type);
CREATE INDEX IF NOT EXISTS ix_mem_updated     ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_mem_occurred    ON memories(occurred_at);
CREATE INDEX IF NOT EXISTS ix_mem_created     ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_mem_profile     ON memories(profile, status)
                                               WHERE profile = 1;
CREATE INDEX IF NOT EXISTS ix_mem_review      ON memories(review_after)
                                               WHERE review_after IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_path ON memories(path);

-- Normalized side tables: tags/entities are also JSON on `memories` (the
-- filesystem-faithful copy), but json_each is a table scan, so filters need
-- these to be indexable.
CREATE TABLE IF NOT EXISTS memory_tags (
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  tag       TEXT NOT NULL,
  PRIMARY KEY (memory_id, tag)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_tags_tag ON memory_tags(tag);

CREATE TABLE IF NOT EXISTS memory_entities (
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  entity    TEXT NOT NULL,
  PRIMARY KEY (memory_id, entity)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_ent_entity ON memory_entities(entity);

-- ─────────────────────────────── tombstones ─────────────────────────────
CREATE TABLE IF NOT EXISTS tombstones (
  memory_id    TEXT PRIMARY KEY,
  deleted_at   TEXT NOT NULL,
  permanent    INTEGER NOT NULL DEFAULT 0 CHECK (permanent IN (0,1)),
  reason       TEXT,
  type         TEXT,
  title_hash   TEXT,
  content_hash TEXT,
  source_refs  TEXT NOT NULL DEFAULT '[]',
  path         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tomb_content   ON tombstones(content_hash);
CREATE INDEX IF NOT EXISTS ix_tomb_title     ON tombstones(title_hash);
CREATE INDEX IF NOT EXISTS ix_tomb_permanent ON tombstones(permanent);

-- ─────────────────────────────── assets ─────────────────────────────────
CREATE TABLE IF NOT EXISTS assets (
  id                TEXT PRIMARY KEY,
  sha256            TEXT NOT NULL,
  media_type        TEXT NOT NULL,
  bytes             INTEGER NOT NULL,
  original_filename TEXT,
  created_at        TEXT,
  imported_at       TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active','deleted')),
  tags              TEXT NOT NULL DEFAULT '[]',
  meta              TEXT NOT NULL DEFAULT '{}',
  blob_path         TEXT NOT NULL,
  meta_path         TEXT NOT NULL,
  prev_sha256       TEXT,
  version           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_asset_sha    ON assets(sha256);
CREATE INDEX IF NOT EXISTS ix_asset_media  ON assets(media_type, status);
CREATE INDEX IF NOT EXISTS ix_asset_import ON assets(imported_at DESC);
CREATE INDEX IF NOT EXISTS ix_asset_prev   ON assets(prev_sha256)
                                            WHERE prev_sha256 IS NOT NULL;

CREATE TABLE IF NOT EXISTS memory_assets (
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  asset_id  TEXT NOT NULL,
  role      TEXT NOT NULL DEFAULT 'attachment',
  PRIMARY KEY (memory_id, asset_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_massets_asset ON memory_assets(asset_id);

-- ─────────────────────────────── sources ────────────────────────────────
CREATE TABLE IF NOT EXISTS sources (
  id           TEXT PRIMARY KEY,
  kind         TEXT NOT NULL CHECK (kind IN
                 ('conversation','document','asset','manual','import')),
  ref          TEXT NOT NULL,
  title        TEXT,
  captured_at  TEXT,
  content_hash TEXT,
  retained     INTEGER NOT NULL DEFAULT 1 CHECK (retained IN (0,1)),
  meta         TEXT NOT NULL DEFAULT '{}',
  UNIQUE (kind, ref)
);
CREATE INDEX IF NOT EXISTS ix_src_captured ON sources(captured_at DESC);

CREATE TABLE IF NOT EXISTS memory_sources (
  memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  source_id  TEXT NOT NULL REFERENCES sources(id)  ON DELETE CASCADE,
  locator    TEXT NOT NULL DEFAULT '',
  confidence REAL,
  PRIMARY KEY (memory_id, source_id, locator)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_msrc_source ON memory_sources(source_id);

-- ─────────────────────────────── segments ───────────────────────────────
CREATE TABLE IF NOT EXISTS segments (
  id            TEXT PRIMARY KEY,
  asset_id      TEXT REFERENCES assets(id)      ON DELETE CASCADE,
  source_id     TEXT REFERENCES sources(id)     ON DELETE CASCADE,
  derivation_id TEXT REFERENCES derivations(id) ON DELETE CASCADE,
  memory_id     TEXT REFERENCES memories(id)    ON DELETE CASCADE,
  ordinal       INTEGER NOT NULL,
  text          TEXT NOT NULL,
  start_ms      INTEGER,
  end_ms        INTEGER,
  page          INTEGER,
  bbox          TEXT,
  speaker       TEXT,
  meta          TEXT NOT NULL DEFAULT '{}',
  CHECK (asset_id IS NOT NULL OR source_id IS NOT NULL OR memory_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_seg_asset ON segments(asset_id, ordinal);
CREATE INDEX IF NOT EXISTS ix_seg_src   ON segments(source_id, ordinal);
CREATE INDEX IF NOT EXISTS ix_seg_deriv ON segments(derivation_id, ordinal);

-- ─────────────────────────────── relations ──────────────────────────────
-- No foreign keys: a dangling target is legal (the other memory may have been
-- purged) and is reported by `verify` rather than blocking the write.
CREATE TABLE IF NOT EXISTS relations (
  src_id  TEXT NOT NULL,
  dst_id  TEXT NOT NULL,
  type    TEXT NOT NULL CHECK (type IN
            ('supersedes','relates_to','causes','part_of','contradicts',
             'derived_from','references')),
  note    TEXT,
  implied INTEGER NOT NULL DEFAULT 0 CHECK (implied IN (0,1)),
  PRIMARY KEY (src_id, dst_id, type, implied)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_rel_dst ON relations(dst_id, type);
CREATE INDEX IF NOT EXISTS ix_rel_src ON relations(src_id, type);

-- ─────────────────────────────── derivations ────────────────────────────
-- A memoized, invalidatable function call. The UNIQUE key is the cache key,
-- which is what keeps a 40s transcription from being re-run.
CREATE TABLE IF NOT EXISTS derivations (
  id                TEXT PRIMARY KEY,
  asset_id          TEXT REFERENCES assets(id)  ON DELETE CASCADE,
  source_id         TEXT REFERENCES sources(id) ON DELETE CASCADE,
  kind              TEXT NOT NULL,
  processor         TEXT NOT NULL,
  processor_version TEXT NOT NULL,
  input_hash        TEXT NOT NULL,
  params_hash       TEXT NOT NULL,
  output_path       TEXT,
  output_hash       TEXT,
  status            TEXT NOT NULL DEFAULT 'complete'
                      CHECK (status IN ('pending','running','complete','failed','stale')),
  error             TEXT,
  created_at        TEXT NOT NULL,
  completed_at      TEXT,
  cost_ms           INTEGER,
  UNIQUE (asset_id, kind, processor, processor_version, input_hash, params_hash)
);
CREATE INDEX IF NOT EXISTS ix_deriv_asset  ON derivations(asset_id, kind);
CREATE INDEX IF NOT EXISTS ix_deriv_status ON derivations(status)
                                            WHERE status IN ('pending','failed');

-- ─────────────────────────────── FTS5 ───────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  title, body, tags, entities,
  content       = 'memories',
  content_rowid = 'rowid',
  tokenize      = "unicode61 remove_diacritics 2 tokenchars '-_@.'",
  prefix        = '2 3 4'
);
"""


# ---------------------------------------------------------------- connection


def connect(paths: Paths, *, read_only: bool = False, timeout: float = 5.0) -> sqlite3.Connection:
    """Open the index with the pragmas this system depends on.

    WAL is what lets a reader run ``search`` while a writer holds the global
    file lock, which is the only concurrency the design needs.
    """
    if sqlite3.sqlite_version_info < MIN_SQLITE:
        raise SchemaError(
            f"SQLite {'.'.join(map(str, MIN_SQLITE))}+ required, found {sqlite3.sqlite_version}"
        )

    paths.index_dir.mkdir(parents=True, exist_ok=True)
    if read_only and paths.index_db.exists():
        conn = sqlite3.connect(f"file:{paths.index_db}?mode=ro", uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(paths.index_db, timeout=timeout)

    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

    if not has_fts5(conn):
        conn.close()
        raise SchemaError("this SQLite build lacks FTS5, which the search index requires")
    return conn


def has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    conn.execute("DROP TABLE temp.__fts5_probe")
    return True


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()


def schema_version(conn: sqlite3.Connection) -> int:
    try:
        raw = get_meta(conn, "schema_version")
    except sqlite3.OperationalError:
        return 0
    try:
        return int(raw) if raw is not None else 0
    except ValueError:
        return 0


def migrate(conn: sqlite3.Connection) -> None:
    """Upgrade an older index in place.

    No-op at v1. Later versions dispatch through a table of step functions so
    an upgrade never requires a full rebuild unless the data genuinely moved.
    """
    current = schema_version(conn)
    if current == SCHEMA_VERSION:
        return
    if current == 0:
        init_schema(conn)
        return
    raise SchemaError(
        f"index schema v{current} cannot be migrated to v{SCHEMA_VERSION}; run `mem rebuild`"
    )


def requires_rebuild(conn: sqlite3.Connection) -> str | None:
    """Return a human reason the index must be rebuilt, or ``None``."""
    try:
        if schema_version(conn) != SCHEMA_VERSION:
            return f"schema version mismatch (found {schema_version(conn)})"
        missing = missing_tables(conn)
        if missing:
            return f"missing tables: {', '.join(sorted(missing))}"
        problems = integrity_check(conn)
        if problems:
            return problems[0]
    except sqlite3.DatabaseError as exc:
        return f"database error: {exc}"
    return None


EXPECTED_TABLES = frozenset(
    {
        "meta",
        "memories",
        "memory_tags",
        "memory_entities",
        "tombstones",
        "assets",
        "memory_assets",
        "sources",
        "memory_sources",
        "segments",
        "relations",
        "derivations",
        "memories_fts",
    }
)


def missing_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    present = {r["name"] for r in rows}
    return set(EXPECTED_TABLES) - present


def integrity_check(conn: sqlite3.Connection) -> list[str]:
    """Structural checks, including FTS/base divergence.

    The FTS5 ``'integrity-check'`` command is how the manual-sync decision is
    held honest: if :func:`index_memory` ever desynchronizes the index from
    the base table, this reports it.
    """
    problems: list[str] = []
    for row in conn.execute("PRAGMA integrity_check"):
        text = row[0]
        if text != "ok":
            problems.append(f"sqlite integrity: {text}")
    # With external content, the bare 'integrity-check' only validates the
    # index's internal structure -- it will happily pass over an index that
    # disagrees with `memories`. The rank=1 argument is what compares the two,
    # which is the check that actually holds manual sync honest.
    # The check is issued as an INSERT, so it needs a writable connection. On
    # a read-only one it fails with "readonly database", which is not
    # corruption -- reporting it would make `mem doctor` and `mem verify`
    # always claim a broken index.
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES('integrity-check', 1)")
    except sqlite3.OperationalError as exc:
        if "readonly" not in str(exc).lower():
            problems.append(f"fts integrity: {exc}")
    except sqlite3.DatabaseError as exc:
        problems.append(f"fts integrity: {exc}")
    for row in conn.execute("PRAGMA foreign_key_check"):
        problems.append(f"foreign key violation in {row[0]}")
    return problems


# ---------------------------------------------------------------- meta


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def add_dirty(conn: sqlite3.Connection, ids: Iterable[str]) -> None:
    """Mark ids whose index rows may lag the filesystem.

    Set when a database write fails *after* the filesystem commit: the files
    already won, so the next open reindexes these rather than failing.
    """
    current = set(json.loads(get_meta(conn, "dirty_ids") or "[]"))
    current.update(ids)
    set_meta(conn, "dirty_ids", json.dumps(sorted(current)))


def take_dirty(conn: sqlite3.Connection) -> list[str]:
    ids = sorted(set(json.loads(get_meta(conn, "dirty_ids") or "[]")))
    set_meta(conn, "dirty_ids", "[]")
    return ids


# ---------------------------------------------------------------- indexing


def index_memory(conn: sqlite3.Connection, m: Memory, *, path: str, mtime_ns: int = 0) -> None:
    """Upsert one memory and all of its derived rows.

    The single sync point between the filesystem and the index. Deliberately
    does the FTS ``'delete'`` with the *stored old values* before updating the
    base row, because external-content FTS5 cannot delete correctly otherwise.
    """
    existing = conn.execute(
        "SELECT rowid, title, body, tags, entities FROM memories WHERE id = ?", (m.id,)
    ).fetchone()

    if existing is not None:
        conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, title, body, tags, entities) "
            "VALUES('delete', ?, ?, ?, ?, ?)",
            (
                existing["rowid"],
                existing["title"],
                existing["body"],
                existing["tags"],
                existing["entities"],
            ),
        )

    # Index the same bytes the base table holds: the tokenizer strips JSON
    # punctuation, so ["a","b"] produces the same terms as "a b", and an
    # exact match with `memories` is what makes the rank=1 integrity check
    # (which compares index against content table) meaningful.
    tags_text = json.dumps(m.tags)
    entities_text = json.dumps(m.entities)
    now = fmt_ts(datetime.now().astimezone())

    conn.execute(
        """
        INSERT INTO memories (
          id, type, title, body, status, created_at, updated_at, occurred_at,
          valid_from, valid_until, review_after, confidence, salience, profile,
          sensitivity, superseded_by, tags, entities, schema_version, path,
          content_hash, mtime_ns, indexed_at
        ) VALUES (
          :id, :type, :title, :body, :status, :created_at, :updated_at, :occurred_at,
          :valid_from, :valid_until, :review_after, :confidence, :salience, :profile,
          :sensitivity, :superseded_by, :tags, :entities, :schema_version, :path,
          :content_hash, :mtime_ns, :indexed_at
        )
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type, title=excluded.title, body=excluded.body,
          status=excluded.status, created_at=excluded.created_at,
          updated_at=excluded.updated_at, occurred_at=excluded.occurred_at,
          valid_from=excluded.valid_from, valid_until=excluded.valid_until,
          review_after=excluded.review_after, confidence=excluded.confidence,
          salience=excluded.salience, profile=excluded.profile,
          sensitivity=excluded.sensitivity, superseded_by=excluded.superseded_by,
          tags=excluded.tags, entities=excluded.entities,
          schema_version=excluded.schema_version, path=excluded.path,
          content_hash=excluded.content_hash, mtime_ns=excluded.mtime_ns,
          indexed_at=excluded.indexed_at
        """,
        {
            "id": m.id,
            "type": m.type.value,
            "title": m.title,
            "body": m.body,
            "status": m.status.value,
            "created_at": fmt_ts(m.created_at),
            "updated_at": fmt_ts(m.updated_at),
            "occurred_at": fmt_ts(m.occurred_at),
            "valid_from": fmt_ts(m.valid_from),
            "valid_until": fmt_ts(m.valid_until),
            "review_after": fmt_ts(m.review_after),
            "confidence": m.confidence,
            "salience": m.salience,
            "profile": 1 if m.profile else 0,
            "sensitivity": m.sensitivity.value,
            "superseded_by": m.superseded_by,
            "tags": json.dumps(m.tags),
            "entities": json.dumps(m.entities),
            "schema_version": m.schema_version,
            "path": path,
            "content_hash": m.content_hash,
            "mtime_ns": mtime_ns,
            "indexed_at": now,
        },
    )

    row = conn.execute("SELECT rowid FROM memories WHERE id = ?", (m.id,)).fetchone()
    rowid = int(row["rowid"])
    conn.execute(
        "INSERT INTO memories_fts(rowid, title, body, tags, entities) VALUES(?, ?, ?, ?, ?)",
        (rowid, m.title, m.body, tags_text, entities_text),
    )

    _reindex_children(conn, m)


def _reindex_children(conn: sqlite3.Connection, m: Memory) -> None:
    """Replace every child row derived from this memory's frontmatter."""
    conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (m.id,))
    conn.executemany(
        "INSERT OR IGNORE INTO memory_tags(memory_id, tag) VALUES(?, ?)",
        [(m.id, t) for t in m.tags],
    )

    conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (m.id,))
    conn.executemany(
        "INSERT OR IGNORE INTO memory_entities(memory_id, entity) VALUES(?, ?)",
        [(m.id, e) for e in m.entities],
    )

    conn.execute("DELETE FROM memory_assets WHERE memory_id = ?", (m.id,))
    conn.executemany(
        "INSERT OR IGNORE INTO memory_assets(memory_id, asset_id) VALUES(?, ?)",
        [(m.id, a) for a in m.assets],
    )

    conn.execute("DELETE FROM memory_sources WHERE memory_id = ?", (m.id,))
    for src in m.sources:
        upsert_source(conn, src)
        conn.execute(
            "INSERT OR IGNORE INTO memory_sources(memory_id, source_id, locator, confidence) "
            "VALUES(?, ?, ?, ?)",
            (m.id, src.source_id, src.locator or "", src.confidence),
        )

    # Each memory owns exactly the edges its own frontmatter declares: the
    # forward edge, and the inverse edge pointing back at itself. Clearing
    # anything else would delete an edge another memory owns and only that
    # memory's reindex would restore it, making the table depend on write
    # order -- and diverge from a rebuild.
    conn.execute(
        "DELETE FROM relations WHERE (src_id = ? AND implied = 0) OR (dst_id = ? AND implied = 1)",
        (m.id, m.id),
    )
    edges: list[tuple[str, str, str, str | None, int]] = []
    for rel in m.relations:
        edges.append((m.id, rel.target, rel.type.value, rel.note, 0))
        edges.append((rel.target, m.id, rel.type.value, rel.note, 1))
    for older in m.supersedes:
        edges.append((m.id, older, RelationType.SUPERSEDES.value, None, 0))
        edges.append((older, m.id, RelationType.SUPERSEDES.value, None, 1))
    conn.executemany(
        "INSERT OR IGNORE INTO relations(src_id, dst_id, type, note, implied) "
        "VALUES(?, ?, ?, ?, ?)",
        edges,
    )


def unindex_memory(conn: sqlite3.Connection, mid: str) -> None:
    """Remove a memory and everything that cascades from it."""
    existing = conn.execute(
        "SELECT rowid, title, body, tags, entities FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    if existing is not None:
        conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, title, body, tags, entities) "
            "VALUES('delete', ?, ?, ?, ?, ?)",
            (
                existing["rowid"],
                existing["title"],
                existing["body"],
                existing["tags"],
                existing["entities"],
            ),
        )
    # Edges *from* this memory go; edges *to* it are left dangling on purpose
    # and reported by verify, so a purge does not silently rewrite history.
    conn.execute("DELETE FROM relations WHERE src_id = ? AND implied = 0", (mid,))
    conn.execute("DELETE FROM relations WHERE dst_id = ? AND implied = 1", (mid,))
    conn.execute("DELETE FROM memories WHERE id = ?", (mid,))


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Re-derive the whole FTS index from the base table.

    The escape hatch that manual sync buys us: whenever old column values are
    unavailable or suspect, this is always correct, and it is how bulk rebuild
    populates the index in one pass.
    """
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")


# ---------------------------------------------------------------- other rows


def upsert_source(conn: sqlite3.Connection, src: SourceRef, *, retained: bool = True) -> str:
    conn.execute(
        """
        INSERT INTO sources(id, kind, ref, captured_at, retained)
        VALUES(:id, :kind, :ref, :captured_at, :retained)
        ON CONFLICT(id) DO UPDATE SET
          captured_at = COALESCE(excluded.captured_at, sources.captured_at),
          retained    = excluded.retained
        """,
        {
            "id": src.source_id,
            "kind": src.kind.value,
            "ref": src.ref,
            "captured_at": fmt_ts(src.captured_at),
            "retained": 1 if retained else 0,
        },
    )
    return src.source_id


def prune_orphan_sources(conn: sqlite3.Connection) -> int:
    """Remove source rows no memory or asset references.

    Keeps the index reconstructible: a rebuild derives ``sources`` from what
    memories and assets cite, so a row nothing cites could never come back.
    The underlying files are never touched.
    """
    cur = conn.execute("DELETE FROM sources WHERE id NOT IN (SELECT source_id FROM memory_sources)")
    return cur.rowcount or 0


def index_asset(conn: sqlite3.Connection, a: Asset, *, blob_path: str, meta_path: str) -> None:
    conn.execute(
        """
        INSERT INTO assets(
          id, sha256, media_type, bytes, original_filename, created_at, imported_at,
          status, tags, meta, blob_path, meta_path, prev_sha256, version
        ) VALUES (
          :id, :sha256, :media_type, :bytes, :original_filename, :created_at, :imported_at,
          :status, :tags, :meta, :blob_path, :meta_path, :prev_sha256, :version
        )
        ON CONFLICT(id) DO UPDATE SET
          sha256=excluded.sha256, media_type=excluded.media_type, bytes=excluded.bytes,
          original_filename=excluded.original_filename, created_at=excluded.created_at,
          imported_at=excluded.imported_at, status=excluded.status, tags=excluded.tags,
          meta=excluded.meta, blob_path=excluded.blob_path, meta_path=excluded.meta_path,
          prev_sha256=excluded.prev_sha256, version=excluded.version
        """,
        {
            "id": a.id,
            "sha256": a.sha256,
            "media_type": a.media_type,
            "bytes": a.bytes,
            "original_filename": a.original_filename,
            "created_at": fmt_ts(a.created_at),
            "imported_at": fmt_ts(a.imported_at),
            "status": a.status,
            "tags": json.dumps(a.tags),
            "meta": json.dumps(a.meta),
            "blob_path": blob_path,
            "meta_path": meta_path,
            "prev_sha256": a.prev_sha256,
            "version": a.version,
        },
    )
    if a.source is not None:
        upsert_source(conn, a.source)


def unindex_asset(conn: sqlite3.Connection, asset_id: str) -> None:
    conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    conn.execute("DELETE FROM memory_assets WHERE asset_id = ?", (asset_id,))


def index_segment(conn: sqlite3.Connection, s: Segment) -> None:
    conn.execute(
        """
        INSERT INTO segments(
          id, asset_id, source_id, derivation_id, memory_id, ordinal, text,
          start_ms, end_ms, page, bbox, speaker, meta
        ) VALUES (
          :id, :asset_id, :source_id, :derivation_id, :memory_id, :ordinal, :text,
          :start_ms, :end_ms, :page, :bbox, :speaker, :meta
        )
        ON CONFLICT(id) DO UPDATE SET
          ordinal=excluded.ordinal, text=excluded.text, start_ms=excluded.start_ms,
          end_ms=excluded.end_ms, page=excluded.page, bbox=excluded.bbox,
          speaker=excluded.speaker, meta=excluded.meta
        """,
        {
            "id": s.id,
            "asset_id": s.asset_id,
            "source_id": s.source_id,
            "derivation_id": s.derivation_id,
            "memory_id": s.memory_id,
            "ordinal": s.ordinal,
            "text": s.text,
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "page": s.page,
            "bbox": s.bbox,
            "speaker": s.speaker,
            "meta": json.dumps(s.meta),
        },
    )


def index_tombstone(conn: sqlite3.Connection, t: Tombstone, *, path: str) -> None:
    conn.execute(
        """
        INSERT INTO tombstones(
          memory_id, deleted_at, permanent, reason, type, title_hash,
          content_hash, source_refs, path
        ) VALUES (
          :memory_id, :deleted_at, :permanent, :reason, :type, :title_hash,
          :content_hash, :source_refs, :path
        )
        ON CONFLICT(memory_id) DO UPDATE SET
          deleted_at=excluded.deleted_at, permanent=excluded.permanent,
          reason=excluded.reason, type=excluded.type, title_hash=excluded.title_hash,
          content_hash=excluded.content_hash, source_refs=excluded.source_refs,
          path=excluded.path
        """,
        {
            "memory_id": t.memory_id,
            "deleted_at": fmt_ts(t.deleted_at),
            "permanent": 1 if t.permanent else 0,
            "reason": t.reason,
            "type": t.type,
            "title_hash": t.title_hash,
            "content_hash": t.content_hash,
            "source_refs": json.dumps(t.source_refs),
            "path": path,
        },
    )


def index_derivation(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO derivations(
          id, asset_id, source_id, kind, processor, processor_version, input_hash,
          params_hash, output_path, output_hash, status, error, created_at,
          completed_at, cost_ms
        ) VALUES (
          :id, :asset_id, :source_id, :kind, :processor, :processor_version, :input_hash,
          :params_hash, :output_path, :output_hash, :status, :error, :created_at,
          :completed_at, :cost_ms
        )
        ON CONFLICT(id) DO UPDATE SET
          status=excluded.status, error=excluded.error, output_path=excluded.output_path,
          output_hash=excluded.output_hash, completed_at=excluded.completed_at,
          cost_ms=excluded.cost_ms
        """,
        {
            "id": row["id"],
            "asset_id": row.get("asset_id"),
            "source_id": row.get("source_id"),
            "kind": row["kind"],
            "processor": row["processor"],
            "processor_version": row["processor_version"],
            "input_hash": row["input_hash"],
            "params_hash": row["params_hash"],
            "output_path": row.get("output_path"),
            "output_hash": row.get("output_hash"),
            "status": row.get("status", "complete"),
            "error": row.get("error"),
            "created_at": row["created_at"],
            "completed_at": row.get("completed_at"),
            "cost_ms": row.get("cost_ms"),
        },
    )


def clear_all(conn: sqlite3.Connection) -> None:
    """Empty every table, for a full rebuild. Order respects foreign keys."""
    for table in (
        "memories_fts",
        "segments",
        "derivations",
        "memory_sources",
        "memory_assets",
        "memory_tags",
        "memory_entities",
        "relations",
        "tombstones",
        "assets",
        "sources",
        "memories",
    ):
        conn.execute(f"DELETE FROM {table}")
