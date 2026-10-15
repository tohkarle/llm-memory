"""Durable, retryable background work.

The ``Cost`` hint on a processor decides the lane: ``cheap`` runs inline
during ingestion, ``expensive`` is enqueued here and run later by
``mem queue run``. A 40-minute recording must not block the command that
imported it.

Durability comes from the rows themselves. Nothing is held in memory, so a
worker killed with SIGKILL loses no task:

* A claim writes ``status='running'`` plus a *lease* (``lease_expires_at``),
  not a lock. A process that dies mid-task leaves a row whose lease expires,
  and the next worker reclaims it. A lock would strand it until a human
  intervened.
* ``attempts`` increments at claim time, not at failure time. A task that
  crashes the interpreter hard enough to skip the failure handler still
  counts its attempt, so a poison input cannot be retried forever.
* Retries back off exponentially via ``not_before``, because the common
  transient failure (a model file still downloading) needs seconds, not
  microseconds.

The queue is index-resident but is *not* derived state: a pending task is not
reconstructible from the filesystem, which is why ``rebuild_index`` leaves
``work_queue`` alone while clearing every other table.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .models import fmt_ts, now_utc, parse_ts

#: How long a claim is valid. Comfortably longer than the slowest expected
#: task (transcribing a long recording) so a live worker is never treated as
#: dead, but short enough that a crashed one is reclaimed the same session.
DEFAULT_LEASE_S = 3600

#: Backoff schedule, in seconds, indexed by attempt number.
BACKOFF_S = (5, 30, 300, 1800)

DEFAULT_MAX_ATTEMPTS = 3

TERMINAL = ("done", "failed", "cancelled")


class QueueError(RuntimeError):
    """A queue operation could not be completed."""


@dataclass(slots=True)
class Task:
    """One unit of deferred work."""

    kind: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str | None = None
    source_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    priority: int = 100
    attempts: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    last_error: str | None = None
    enqueued_at: str = ""
    claimed_at: str | None = None
    lease_expires_at: str | None = None
    finished_at: str | None = None
    not_before: str | None = None
    dedupe_key: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Task:
        return cls(
            id=row["id"],
            kind=row["kind"],
            asset_id=row["asset_id"],
            source_id=row["source_id"],
            payload=json.loads(row["payload"] or "{}"),
            status=row["status"],
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            last_error=row["last_error"],
            enqueued_at=row["enqueued_at"],
            claimed_at=row["claimed_at"],
            lease_expires_at=row["lease_expires_at"],
            finished_at=row["finished_at"],
            not_before=row["not_before"],
            dedupe_key=row["dedupe_key"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "asset_id": self.asset_id,
            "source_id": self.source_id,
            "payload": self.payload,
            "status": self.status,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "enqueued_at": self.enqueued_at,
            "claimed_at": self.claimed_at,
            "lease_expires_at": self.lease_expires_at,
            "finished_at": self.finished_at,
            "not_before": self.not_before,
            "dedupe_key": self.dedupe_key,
        }


def _now() -> str:
    return fmt_ts(now_utc()) or ""


def _plus(seconds: float) -> str:
    return fmt_ts(now_utc() + timedelta(seconds=seconds)) or ""


def enqueue(
    conn: sqlite3.Connection,
    *,
    kind: str,
    asset_id: str | None = None,
    source_id: str | None = None,
    payload: dict[str, Any] | None = None,
    priority: int = 100,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    dedupe_key: str | None = None,
    not_before: str | None = None,
) -> Task:
    """Add a task, or return the live one that already covers this work.

    The dedupe key is enforced by a partial unique index over the non-terminal
    statuses, so re-ingesting an asset cannot queue a second transcription of
    it while the first is still pending.
    """
    if dedupe_key is not None:
        existing = conn.execute(
            "SELECT * FROM work_queue WHERE dedupe_key = ? AND status IN ('pending','running')",
            (dedupe_key,),
        ).fetchone()
        if existing is not None:
            return Task.from_row(existing)

    task = Task(
        kind=kind,
        asset_id=asset_id,
        source_id=source_id,
        payload=dict(payload or {}),
        priority=priority,
        max_attempts=max_attempts,
        enqueued_at=_now(),
        not_before=not_before,
        dedupe_key=dedupe_key,
    )
    try:
        conn.execute(
            """
            INSERT INTO work_queue(
              id, kind, asset_id, source_id, payload, status, priority,
              attempts, max_attempts, enqueued_at, not_before, dedupe_key
            ) VALUES (
              :id, :kind, :asset_id, :source_id, :payload, 'pending', :priority,
              0, :max_attempts, :enqueued_at, :not_before, :dedupe_key
            )
            """,
            {
                "id": task.id,
                "kind": task.kind,
                "asset_id": task.asset_id,
                "source_id": task.source_id,
                "payload": json.dumps(task.payload),
                "priority": task.priority,
                "max_attempts": task.max_attempts,
                "enqueued_at": task.enqueued_at,
                "not_before": task.not_before,
                "dedupe_key": task.dedupe_key,
            },
        )
    except sqlite3.IntegrityError:
        # Lost a race on the dedupe index: the other writer's task is as good
        # as ours, so adopt it rather than failing the caller's ingestion.
        if dedupe_key is None:
            raise
        row = conn.execute(
            "SELECT * FROM work_queue WHERE dedupe_key = ? AND status IN ('pending','running')",
            (dedupe_key,),
        ).fetchone()
        if row is None:
            raise
        return Task.from_row(row)
    return task


def claim(
    conn: sqlite3.Connection,
    *,
    kinds: list[str] | None = None,
    lease_s: int = DEFAULT_LEASE_S,
) -> Task | None:
    """Take the next ready task, or ``None``.

    Ready means pending and past ``not_before``, or running with an expired
    lease -- the latter is how a task orphaned by a killed worker returns to
    circulation.

    The UPDATE is guarded by the status and lease the SELECT saw, so two
    workers racing for the same row cannot both win: the loser's UPDATE
    matches zero rows and it moves to the next candidate.
    """
    now = _now()
    clauses = [
        "(status = 'pending' AND (not_before IS NULL OR not_before <= :now))",
        "(status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= :now)",
    ]
    params: dict[str, Any] = {"now": now}
    sql = f"SELECT * FROM work_queue WHERE ({' OR '.join(clauses)})"
    if kinds:
        placeholders = ",".join(f":k{i}" for i in range(len(kinds)))
        sql += f" AND kind IN ({placeholders})"
        params.update({f"k{i}": k for i, k in enumerate(kinds)})
    sql += " ORDER BY priority ASC, enqueued_at ASC LIMIT 20"

    for row in conn.execute(sql, params).fetchall():
        task = Task.from_row(row)
        if task.attempts >= task.max_attempts:
            # Exhausted but never marked: the worker died before it could
            # record the final failure.
            _finish(conn, task.id, "failed", error="attempts exhausted")
            continue
        cur = conn.execute(
            """
            UPDATE work_queue
               SET status = 'running',
                   attempts = attempts + 1,
                   claimed_at = :now,
                   lease_expires_at = :lease
             WHERE id = :id
               AND status = :expected_status
               AND COALESCE(lease_expires_at, '') = COALESCE(:expected_lease, '')
            """,
            {
                "now": now,
                "lease": _plus(lease_s),
                "id": task.id,
                "expected_status": task.status,
                "expected_lease": task.lease_expires_at,
            },
        )
        if cur.rowcount == 1:
            conn.commit()
            fresh = conn.execute("SELECT * FROM work_queue WHERE id = ?", (task.id,)).fetchone()
            return Task.from_row(fresh)
    conn.commit()
    return None


def complete(
    conn: sqlite3.Connection, task_id: str, *, result: dict[str, Any] | None = None
) -> None:
    _finish(conn, task_id, "done", result=result)


def fail(conn: sqlite3.Connection, task_id: str, error: str, *, retry: bool = True) -> Task | None:
    """Record a failure, scheduling a backed-off retry if attempts remain."""
    row = conn.execute("SELECT * FROM work_queue WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    task = Task.from_row(row)
    message = error[:2000]

    if retry and task.attempts < task.max_attempts:
        index = min(max(task.attempts - 1, 0), len(BACKOFF_S) - 1)
        delay = BACKOFF_S[index]
        conn.execute(
            """
            UPDATE work_queue
               SET status = 'pending', last_error = :error, claimed_at = NULL,
                   lease_expires_at = NULL, not_before = :not_before
             WHERE id = :id
            """,
            {"error": message, "not_before": _plus(delay), "id": task_id},
        )
    else:
        conn.execute(
            """
            UPDATE work_queue
               SET status = 'failed', last_error = :error, finished_at = :now,
                   lease_expires_at = NULL
             WHERE id = :id
            """,
            {"error": message, "now": _now(), "id": task_id},
        )
    conn.commit()
    return get(conn, task_id)


def _finish(
    conn: sqlite3.Connection,
    task_id: str,
    status: str,
    *,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    payload_sql = ""
    params: dict[str, Any] = {
        "status": status,
        "now": _now(),
        "id": task_id,
        "error": error,
    }
    if result is not None:
        # Merge the outcome into the payload so `mem queue list` can show what
        # a finished task produced without a second table.
        row = conn.execute("SELECT payload FROM work_queue WHERE id = ?", (task_id,)).fetchone()
        current = json.loads(row["payload"] or "{}") if row else {}
        current["result"] = result
        params["payload"] = json.dumps(current)
        payload_sql = ", payload = :payload"
    conn.execute(
        f"""
        UPDATE work_queue
           SET status = :status, finished_at = :now, last_error = :error,
               lease_expires_at = NULL{payload_sql}
         WHERE id = :id
        """,
        params,
    )
    conn.commit()


def get(conn: sqlite3.Connection, task_id: str) -> Task | None:
    row = conn.execute("SELECT * FROM work_queue WHERE id = ?", (task_id,)).fetchone()
    return Task.from_row(row) if row else None


def listing(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    kind: str | None = None,
    asset_id: str | None = None,
    limit: int = 50,
) -> list[Task]:
    sql = "SELECT * FROM work_queue"
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if asset_id:
        where.append("asset_id = ?")
        params.append(asset_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    # Pending work first and oldest-first within a status, which is the order
    # a human asking "what is outstanding" wants to read.
    sql += """
        ORDER BY CASE status
                   WHEN 'running' THEN 0 WHEN 'pending' THEN 1
                   WHEN 'failed' THEN 2 ELSE 3 END,
                 priority ASC, enqueued_at ASC
        LIMIT ?
    """
    params.append(limit)
    return [Task.from_row(row) for row in conn.execute(sql, params)]


def retry(
    conn: sqlite3.Connection,
    *,
    task_id: str | None = None,
    all_failed: bool = False,
    reset_attempts: bool = True,
) -> list[str]:
    """Return failed tasks to pending.

    ``reset_attempts`` because the usual reason to retry by hand is that the
    cause was fixed -- the missing extra got installed -- and the old attempt
    count would otherwise exhaust the task immediately.
    """
    if task_id:
        targets = [task_id]
    elif all_failed:
        targets = [r["id"] for r in conn.execute("SELECT id FROM work_queue WHERE status='failed'")]
    else:
        raise QueueError("pass task_id or all_failed")

    revived: list[str] = []
    for tid in targets:
        cur = conn.execute(
            f"""
            UPDATE work_queue
               SET status = 'pending', not_before = NULL, claimed_at = NULL,
                   lease_expires_at = NULL, finished_at = NULL
                   {", attempts = 0" if reset_attempts else ""}
             WHERE id = ? AND status IN ('failed','cancelled')
            """,
            (tid,),
        )
        if cur.rowcount == 1:
            revived.append(tid)
    conn.commit()
    return revived


def cancel(conn: sqlite3.Connection, task_id: str) -> bool:
    cur = conn.execute(
        "UPDATE work_queue SET status='cancelled', finished_at=?, lease_expires_at=NULL "
        "WHERE id = ? AND status IN ('pending','running')",
        (_now(), task_id),
    )
    conn.commit()
    return cur.rowcount == 1


def purge(conn: sqlite3.Connection, *, keep_days: int = 7) -> int:
    """Drop old terminal rows. Pending and running work is never purged."""
    cutoff = fmt_ts(now_utc() - timedelta(days=keep_days))
    cur = conn.execute(
        f"DELETE FROM work_queue WHERE status IN {TERMINAL} "  # the tuple is a module constant
        "AND finished_at IS NOT NULL AND finished_at < ?",
        (cutoff,),
    )
    conn.commit()
    return int(cur.rowcount)


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        row["status"]: row["c"]
        for row in conn.execute("SELECT status, count(*) c FROM work_queue GROUP BY status")
    }
    overdue = conn.execute(
        "SELECT count(*) c FROM work_queue WHERE status='running' "
        "AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?",
        (_now(),),
    ).fetchone()["c"]
    return {
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "done": counts.get("done", 0),
        "failed": counts.get("failed", 0),
        "cancelled": counts.get("cancelled", 0),
        "expired_leases": overdue,
    }


def drain(
    conn: sqlite3.Connection,
    handler: Callable[[Task], dict[str, Any] | None],
    *,
    limit: int | None = None,
    kinds: list[str] | None = None,
    lease_s: int = DEFAULT_LEASE_S,
) -> dict[str, Any]:
    """Run ready tasks until none remain (or ``limit`` is reached).

    The handler's exceptions are caught and recorded as task failures: one
    poison input must not stop the drain, since the whole point is to work
    through a backlog unattended.
    """
    done: list[str] = []
    failed: list[dict[str, str]] = []
    processed = 0
    while limit is None or processed < limit:
        task = claim(conn, kinds=kinds, lease_s=lease_s)
        if task is None:
            break
        processed += 1
        try:
            result = handler(task)
        except Exception as exc:
            fail(conn, task.id, f"{type(exc).__name__}: {exc}")
            failed.append({"id": task.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        complete(conn, task.id, result=result)
        done.append(task.id)
    return {"processed": processed, "done": done, "failed": failed, **stats(conn)}


def iter_ready(conn: sqlite3.Connection) -> Iterator[Task]:
    """Ready tasks, without claiming them. For inspection only."""
    now = _now()
    rows = conn.execute(
        "SELECT * FROM work_queue WHERE status='pending' "
        "AND (not_before IS NULL OR not_before <= ?) "
        "ORDER BY priority ASC, enqueued_at ASC",
        (now,),
    )
    for row in rows:
        yield Task.from_row(row)


def reap_expired(conn: sqlite3.Connection) -> int:
    """Return leases that outlived their worker to pending.

    Called at open time, so a crashed worker's task is picked up by the next
    command rather than waiting for someone to run the queue explicitly.
    """
    now = _now()
    cur = conn.execute(
        """
        UPDATE work_queue
           SET status = 'pending', claimed_at = NULL, lease_expires_at = NULL,
               last_error = COALESCE(last_error, 'lease expired; worker presumed dead')
         WHERE status = 'running'
           AND lease_expires_at IS NOT NULL
           AND lease_expires_at <= ?
           AND attempts < max_attempts
        """,
        (now,),
    )
    exhausted = conn.execute(
        """
        UPDATE work_queue
           SET status = 'failed', finished_at = ?, lease_expires_at = NULL,
               last_error = COALESCE(last_error, 'lease expired; attempts exhausted')
         WHERE status = 'running'
           AND lease_expires_at IS NOT NULL
           AND lease_expires_at <= ?
           AND attempts >= max_attempts
        """,
        (now, now),
    )
    conn.commit()
    return int(cur.rowcount) + int(exhausted.rowcount)


def ensure_parsed(stamp: str | None) -> Any:
    """Parse a queue timestamp, tolerating a hand-edited row."""
    if not stamp:
        return None
    try:
        return parse_ts(stamp)
    except Exception:
        return None


__all__ = [
    "BACKOFF_S",
    "DEFAULT_LEASE_S",
    "DEFAULT_MAX_ATTEMPTS",
    "QueueError",
    "Task",
    "cancel",
    "claim",
    "complete",
    "drain",
    "enqueue",
    "fail",
    "get",
    "iter_ready",
    "listing",
    "purge",
    "reap_expired",
    "retry",
    "stats",
]
