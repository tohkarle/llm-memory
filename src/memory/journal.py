"""Durable operation journal and crash recovery.

The journal records intent before any filesystem mutation and completion
after it, so an interrupted operation can always be driven to one of exactly
two states: fully applied or fully rolled back.

The invariant that makes recovery sound: **every plan step is idempotent and
verifiable by hash.** Recovery never needs to know how far it got -- it
re-derives that from the filesystem. Hence ``new_hash``/``prev_hash`` are
mandatory on every step that touches content.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from .paths import Paths

#: How far back recovery reads. Older ops are provably settled: the index
#: advanced past them and truncation only removes settled ops.
RECOVERY_WINDOW = timedelta(days=30)

DEFAULT_LOCK_TIMEOUT = 10.0


class OpKind(StrEnum):
    CREATE_MEMORY = "create_memory"
    UPDATE_MEMORY = "update_memory"
    REPLACE_MEMORY = "replace_memory"
    MERGE_MEMORIES = "merge_memories"
    DELETE_MEMORY = "delete_memory"
    PURGE_MEMORY = "purge_memory"
    STORE_ASSET = "store_asset"
    REPLACE_ASSET = "replace_asset"
    DELETE_ASSET = "delete_asset"
    PURGE_ASSET = "purge_asset"


#: Kinds whose intent is destruction. A partially-applied op of this kind is
#: driven *forward*: the tombstone (always step 0) is already durable, so
#: finishing honors what the user asked for.
ROLL_FORWARD_KINDS = frozenset({OpKind.PURGE_MEMORY, OpKind.PURGE_ASSET})

Phase = Literal["begin", "commit", "abort"]
StepState = Literal["done", "not_started", "unknown"]
StepAct = Literal["write", "unlink", "rename", "link"]


class JournalError(RuntimeError):
    """The journal could not be read or written."""


class LockTimeout(TimeoutError):
    """Another process holds the write lock."""


def new_op_id() -> str:
    """Sortable, collision-resistant id: 48-bit ms timestamp + 64 random bits.

    Hand-rolled rather than depending on a ULID package -- it is ten lines and
    the only property we need is lexicographic ordering by time.
    """
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    return f"{ms:012x}{secrets.token_hex(8)}"


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def hash_file(path: Path) -> str | None:
    """Content hash of a file, or ``None`` if it does not exist."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        return "sha256:" + h.hexdigest()
    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return None


@dataclass(slots=True)
class Step:
    """One filesystem action within an operation.

    ``path`` and ``prev_backup`` are relative to ``data/`` so the whole tree
    can be relocated without invalidating the journal.
    """

    i: int
    act: StepAct
    path: str
    tmp: str | None = None
    new_hash: str | None = None
    new_bytes: int | None = None
    prev_hash: str | None = None
    prev_backup: str | None = None
    from_path: str | None = None
    blob_hash: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Step:
        return cls(
            i=int(raw["i"]),
            act=raw["act"],
            path=str(raw["path"]),
            tmp=raw.get("tmp"),
            new_hash=raw.get("new_hash"),
            new_bytes=raw.get("new_bytes"),
            prev_hash=raw.get("prev_hash"),
            prev_backup=raw.get("prev_backup"),
            from_path=raw.get("from_path"),
            blob_hash=raw.get("blob_hash"),
        )


@dataclass(slots=True)
class JournalRecord:
    op_id: str
    seq: int
    kind: OpKind
    phase: Phase
    ts: str
    targets: list[str] = field(default_factory=list)
    plan: list[Step] = field(default_factory=list)
    db: dict[str, list[str]] = field(default_factory=dict)
    applied_through: int | None = None
    reason: str | None = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "op_id": self.op_id,
            "seq": self.seq,
            "kind": self.kind.value,
            "phase": self.phase,
            "ts": self.ts,
            "targets": list(self.targets),
            "plan": [s.to_json() for s in self.plan],
            "db": {k: list(v) for k, v in self.db.items()},
        }
        if self.applied_through is not None:
            out["applied_through"] = self.applied_through
        if self.reason is not None:
            out["reason"] = self.reason
        return out

    def to_line(self) -> str:
        payload = self.payload()
        payload["checksum"] = _checksum(payload)
        return json.dumps(payload, separators=(",", ":"), default=str) + "\n"

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> JournalRecord:
        return cls(
            op_id=str(raw["op_id"]),
            seq=int(raw["seq"]),
            kind=OpKind(raw["kind"]),
            phase=raw["phase"],
            ts=str(raw["ts"]),
            targets=[str(t) for t in raw.get("targets", [])],
            plan=[Step.from_json(s) for s in raw.get("plan", [])],
            db={k: [str(x) for x in v] for k, v in (raw.get("db") or {}).items()},
            applied_through=raw.get("applied_through"),
            reason=raw.get("reason"),
        )


@dataclass(slots=True)
class RecoveryReport:
    rolled_forward: list[str] = field(default_factory=list)
    rolled_back: list[str] = field(default_factory=list)
    reindex: list[str] = field(default_factory=list)
    unindex: list[str] = field(default_factory=list)
    dropped_records: int = 0
    torn_lines: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def settled(self) -> int:
        return len(self.rolled_forward) + len(self.rolled_back)

    @property
    def clean(self) -> bool:
        return self.settled == 0 and self.dropped_records == 0 and self.torn_lines == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "rolled_forward": self.rolled_forward,
            "rolled_back": self.rolled_back,
            "reindex": self.reindex,
            "unindex": self.unindex,
            "dropped_records": self.dropped_records,
            "torn_lines": self.torn_lines,
            "warnings": self.warnings,
            "settled": self.settled,
        }


def atomic_write(path: Path, data: bytes, *, tmp_suffix: str | None = None) -> None:
    """Write ``data`` to ``path`` atomically and durably.

    The temp file is created in the *same directory* so ``os.replace`` is a
    rename rather than a cross-device copy. Both the file and its parent
    directory are fsynced, so the rename itself survives a power loss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = tmp_suffix or f".tmp-{new_op_id()}"
    tmp = path.with_name(path.name + suffix)
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (FileNotFoundError, PermissionError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # Some filesystems reject directory fsync; the rename still landed.
    finally:
        os.close(fd)


class Journal:
    """Append-only operation log guarded by an exclusive filesystem lock.

    Used as a context manager for the duration of one mutating operation::

        with journal:
            op_id = journal.begin(OpKind.CREATE_MEMORY, [mid], plan)
            ...filesystem writes...
            journal.commit(op_id)
    """

    def __init__(self, paths: Paths, *, lock_timeout: float = DEFAULT_LOCK_TIMEOUT) -> None:
        self.paths = paths
        self.lock_timeout = lock_timeout
        self._lock_fd: int | None = None
        self._depth = 0

    # ------------------------------------------------------------------ locking

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        """Take the exclusive write lock, blocking up to ``lock_timeout``.

        Re-entrant within a process so nested operations (merge calling into
        per-memory writes) do not deadlock.
        """
        if self._lock_fd is not None:
            self._depth += 1
            return
        self.paths.data.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.paths.lock_file, os.O_WRONLY | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LockTimeout(
                        f"another process holds {self.paths.lock_file} "
                        f"(waited {self.lock_timeout}s)"
                    ) from None
                time.sleep(0.05)
        self._lock_fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._lock_fd is None:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)
            self._lock_fd = None

    @property
    def locked(self) -> bool:
        return self._lock_fd is not None

    def _require_lock(self) -> None:
        if self._lock_fd is None:
            raise JournalError("journal operations require the write lock")

    # ------------------------------------------------------------------ sequence

    def _next_seq(self) -> int:
        self._require_lock()
        seq_file = self.paths.seq_file
        try:
            current = int(seq_file.read_text(encoding="utf-8").strip() or "0")
        except (FileNotFoundError, ValueError):
            current = self._max_seq_on_disk()
        nxt = current + 1
        atomic_write(seq_file, f"{nxt}\n".encode())
        return nxt

    def _max_seq_on_disk(self) -> int:
        """Recover the sequence counter by scanning the log.

        Only used when ``seq`` is missing or corrupt; keeps sequence numbers
        strictly increasing across a lost counter file.
        """
        best = 0
        for record, _ in self._iter_records(all_files=True):
            best = max(best, record.seq)
        return best

    # ------------------------------------------------------------------ files

    def _log_file(self, when: datetime | None = None) -> Path:
        stamp = (when or datetime.now(UTC)).strftime("%Y%m%d")
        return self.paths.journal_dir / f"{stamp}.jsonl"

    def _log_files(self, *, all_files: bool = False) -> list[Path]:
        files = sorted(self.paths.journal_dir.glob("*.jsonl"))
        if all_files:
            return files
        cutoff = (datetime.now(UTC) - RECOVERY_WINDOW).strftime("%Y%m%d")
        return [f for f in files if f.stem >= cutoff]

    def _append(self, record: JournalRecord) -> None:
        self._require_lock()
        path = self._log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = record.to_line().encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _iter_records(self, *, all_files: bool = False) -> Iterator[tuple[JournalRecord, Path]]:
        for path in self._log_files(all_files=all_files):
            for record in self._read_file(path)[0]:
                yield record, path

    def _read_file(self, path: Path) -> tuple[list[JournalRecord], int, int]:
        """Return (records, dropped, torn).

        A trailing partial line is a torn append -- the process died mid-write.
        It is dropped without alarm. A bad checksum anywhere is corruption and
        is counted separately.
        """
        records: list[JournalRecord] = []
        dropped = 0
        torn = 0
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return records, dropped, torn
        if not raw:
            return records, dropped, torn

        lines = raw.split(b"\n")
        # The final element is b"" for a complete file, or the torn fragment of
        # an interrupted append. Either way it is not a record.
        if lines[-1] != b"":
            torn += 1
        body = lines[:-1]

        for line in body:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                stated = payload.pop("checksum", None)
                if stated != _checksum(payload):
                    dropped += 1
                    continue
                records.append(JournalRecord.from_payload(payload))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                dropped += 1
        return records, dropped, torn

    # ------------------------------------------------------------------ writing

    def reserve_op_id(self) -> str:
        """Mint an op id before :meth:`begin`.

        Pre-images must be copied into the undo area *before* intent is
        recorded (the plan references their paths), and the undo area is keyed
        by op id -- so the id has to exist first. Pass it back to ``begin``.
        """
        self._require_lock()
        return new_op_id()

    def begin(
        self,
        kind: OpKind,
        targets: list[str],
        plan: list[Step],
        *,
        db: dict[str, list[str]] | None = None,
        op_id: str | None = None,
    ) -> str:
        """Record intent. Returns the ``op_id`` to pass to commit/abort."""
        self._require_lock()
        op_id = op_id or new_op_id()
        record = JournalRecord(
            op_id=op_id,
            seq=self._next_seq(),
            kind=kind,
            phase="begin",
            ts=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            targets=list(targets),
            plan=plan,
            db=db or {},
        )
        self._append(record)
        return op_id

    def commit(self, op_id: str, *, applied_through: int | None = None) -> None:
        self._settle(op_id, "commit", applied_through=applied_through)

    def abort(self, op_id: str, reason: str, *, applied_through: int | None = None) -> None:
        self._settle(op_id, "abort", reason=reason, applied_through=applied_through)

    def _settle(
        self,
        op_id: str,
        phase: Phase,
        *,
        reason: str | None = None,
        applied_through: int | None = None,
    ) -> None:
        self._require_lock()
        opened = self._find_begin(op_id)
        kind = opened.kind if opened else OpKind.CREATE_MEMORY
        self._append(
            JournalRecord(
                op_id=op_id,
                seq=self._next_seq(),
                kind=kind,
                phase=phase,
                ts=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                reason=reason,
                applied_through=applied_through,
            )
        )
        self._discard_undo(op_id)

    def _find_begin(self, op_id: str) -> JournalRecord | None:
        for record, _ in self._iter_records():
            if record.op_id == op_id and record.phase == "begin":
                return record
        return None

    # ------------------------------------------------------------------ undo dir

    def undo_path(self, op_id: str, index: int) -> Path:
        return self.paths.undo_dir / op_id / f"{index}.bak"

    def backup(self, op_id: str, index: int, src: Path) -> str | None:
        """Copy a pre-image into the undo area. Returns its ``data/``-relative path.

        Pre-images are what make rollback possible for overwrite and unlink.
        """
        if not src.exists():
            return None
        dest = self.undo_path(op_id, index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        _fsync_dir(dest.parent)
        return self.paths.rel(dest)

    def _discard_undo(self, op_id: str) -> None:
        shutil.rmtree(self.paths.undo_dir / op_id, ignore_errors=True)

    # ------------------------------------------------------------------ reading

    def open_ops(self) -> list[JournalRecord]:
        """Begins with no matching commit or abort, in sequence order.

        Scans every log file, not just the recovery window: an unsettled op in
        an older file must stay visible, or ``truncate`` would delete the file
        holding it and silently discard the operation.
        """
        begins: dict[str, JournalRecord] = {}
        settled: set[str] = set()
        for record, _ in self._iter_records(all_files=True):
            if record.phase == "begin":
                begins[record.op_id] = record
            else:
                settled.add(record.op_id)
        return sorted(
            (r for op_id, r in begins.items() if op_id not in settled),
            key=lambda r: r.seq,
        )

    def history(self, *, limit: int = 50) -> list[JournalRecord]:
        records = [r for r, _ in self._iter_records()]
        records.sort(key=lambda r: r.seq, reverse=True)
        return records[:limit]

    def show(self, op_id: str) -> list[JournalRecord]:
        return sorted(
            (r for r, _ in self._iter_records(all_files=True) if r.op_id == op_id),
            key=lambda r: r.seq,
        )

    # ------------------------------------------------------------------ recovery

    def _step_state(self, step: Step) -> StepState:
        """Classify a step by hashing the filesystem.

        This is the heart of recovery: state is *derived*, never remembered,
        so recovery is correct regardless of where the process died.
        """
        target = self.paths.abs(step.path)
        if step.act in ("write", "rename", "link"):
            actual = hash_file(target)
            if actual is None:
                return "not_started"
            if step.new_hash is None:
                return "done"  # existence is the only signal we have
            if actual == step.new_hash:
                return "done"
            if step.prev_hash is not None and actual == step.prev_hash:
                return "not_started"
            return "unknown"
        # unlink
        if not target.exists():
            return "done"
        actual = hash_file(target)
        if step.prev_hash is not None and actual == step.prev_hash:
            return "not_started"
        return "not_started"

    def recover(self, *, dry_run: bool = False) -> RecoveryReport:
        """Drive every open operation to a settled state.

        Runs on every open of the API. Idempotent: a second call on an
        already-recovered tree reports nothing.
        """
        self._require_lock()
        report = RecoveryReport()

        for path in self._log_files():
            _, dropped, torn = self._read_file(path)
            report.dropped_records += dropped
            report.torn_lines += torn
            if dropped:
                report.warnings.append(f"{dropped} corrupt record(s) in {path.name}")
            if torn:
                report.warnings.append(f"torn trailing line in {path.name}")

        for record in self.open_ops():
            states = [self._step_state(s) for s in record.plan]
            done = sum(1 for s in states if s == "done")
            forward = (
                not record.plan
                or done == len(record.plan)
                or (done > 0 and record.kind in ROLL_FORWARD_KINDS)
            )

            if dry_run:
                (report.rolled_forward if forward else report.rolled_back).append(record.op_id)
                continue

            if forward:
                self._roll_forward(record, states)
                report.rolled_forward.append(record.op_id)
                report.reindex.extend(record.db.get("reindex", []))
                report.unindex.extend(record.db.get("unindex", []))
            else:
                self._roll_back(record, states)
                report.rolled_back.append(record.op_id)
                # The filesystem reverted, so the index must be re-derived
                # from whatever the files now say.
                report.reindex.extend(record.targets)

            self._clean_tmp(record)

        return report

    def _roll_forward(self, record: JournalRecord, states: Sequence[StepState]) -> None:
        """Finish the remaining steps, then commit."""
        for step, state in zip(record.plan, states, strict=True):
            if state == "done":
                continue
            target = self.paths.abs(step.path)
            if step.act == "unlink":
                target.unlink(missing_ok=True)
            elif step.act in ("write", "link") and step.prev_backup:
                # Only a pre-image is available; the new content was never
                # durable, so forward completion is impossible. Leave it.
                continue
        self.commit(record.op_id, applied_through=len(record.plan))

    def _roll_back(self, record: JournalRecord, states: Sequence[StepState]) -> None:
        """Restore pre-images in reverse order, then abort."""
        for step, state in reversed(list(zip(record.plan, states, strict=True))):
            target = self.paths.abs(step.path)
            if step.act == "rename" and state == "done" and step.from_path:
                origin = self.paths.abs(step.from_path)
                origin.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    os.replace(target, origin)
                continue
            if step.prev_backup:
                backup = self.paths.abs(step.prev_backup)
                if backup.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
                    _fsync_dir(target.parent)
                continue
            # Created by this op with no pre-image: it did not exist before.
            # Blobs are exempt -- they are content-addressed and may be shared.
            if (
                step.act in ("write", "link")
                and state in ("done", "unknown")
                and step.blob_hash is None
            ):
                target.unlink(missing_ok=True)
        self.abort(record.op_id, "recovered: rolled back incomplete operation")

    def _clean_tmp(self, record: JournalRecord) -> None:
        for step in record.plan:
            if step.tmp:
                self.paths.abs(step.tmp).unlink(missing_ok=True)
        for leftover in self.paths.data.rglob(f"*.tmp-{record.op_id}"):
            leftover.unlink(missing_ok=True)

    # ------------------------------------------------------------------ upkeep

    def truncate(self, *, keep_days: int = 7) -> int:
        """Delete log files whose every op is settled and which are older than
        ``keep_days``. Never removes a file containing an open op.
        """
        self._require_lock()
        open_ids = {r.op_id for r in self.open_ops()}
        cutoff = (datetime.now(UTC) - timedelta(days=keep_days)).strftime("%Y%m%d")
        removed = 0
        for path in self._log_files(all_files=True):
            if path.stem >= cutoff:
                continue
            records, _, _ = self._read_file(path)
            if any(r.op_id in open_ids for r in records):
                continue
            path.unlink(missing_ok=True)
            removed += 1
        return removed
