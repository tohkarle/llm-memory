"""Crash recovery: the tree must land in exactly the pre-op or post-op state.

The central assertion of the whole design. For each operation and each point
a crash can occur, reopening the API must settle the interrupted operation
and leave a full-tree hash manifest equal to **either** the manifest before
the operation **or** the one after it. No third state is permitted.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.journal import hash_file
from memory.models import MemoryType, new_id
from memory.paths import Paths


class SimulatedCrash(BaseException):
    """Raised at a chosen syscall to model a process death mid-operation.

    Inherits BaseException so ordinary ``except Exception`` handlers in the
    code under test cannot swallow it -- a crash is not a recoverable error.
    """


@pytest.fixture
def root(tmp_path: Path) -> Path:
    api = MemoryAPI.initialize(tmp_path)
    api.close()
    return tmp_path


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def manifest(root: Path) -> dict[str, str]:
    """Hash every authoritative file, ignoring derived and in-flight state.

    Excludes the index (rebuildable), the journal (its growth is expected),
    and undo pre-images and temp files (which recovery is required to clean).
    """
    paths = Paths(root)
    out: dict[str, str] = {}
    for path in sorted(paths.data.rglob("*")):
        if not path.is_file():
            continue
        rel = paths.rel(path)
        if rel.startswith((".journal/", "index/")) or ".tmp-" in rel:
            continue
        out[rel] = hash_file(path) or ""
    return out


def shape(root: Path) -> list[str]:
    """The tree's shape, with per-run identifiers normalized away.

    Two runs of the same operation mint different UUIDs and therefore
    different filenames and content hashes, so comparing raw manifests across
    runs is meaningless. What must match is *which* files exist -- whether the
    memory landed in facts/ or decisions/, whether a tombstone appeared. The
    named tests below assert specific file *contents*, where a single root
    makes raw hashes comparable.
    """
    entries: list[str] = []
    for rel in manifest(root):
        if rel in (".lock", ".memory-root") or rel.endswith(".gitkeep"):
            continue
        entries.append(_UUID_RE.sub("<id>", rel))
    return sorted(entries)


def residue(root: Path) -> list[str]:
    paths = Paths(root)
    leftovers = [paths.rel(p) for p in paths.data.rglob("*.tmp-*")]
    undo = paths.undo_dir
    if undo.exists():
        leftovers += [paths.rel(p) for p in undo.rglob("*") if p.is_file()]
    return sorted(leftovers)


class CrashPoint:
    """Monkeypatch a syscall to raise on the Nth call."""

    def __init__(self, target: str, nth: int) -> None:
        self.target = target
        self.nth = nth
        self.calls = 0
        self._real: Any = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._real = getattr(os, self.target)
        real = self._real
        state = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state.calls += 1
            if state.calls == state.nth:
                raise SimulatedCrash(f"{state.target} call #{state.nth}")
            return real(*args, **kwargs)

        monkeypatch.setattr(os, self.target, wrapper)

    def count_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Count calls without raising, to discover how many a run makes."""
        self._real = getattr(os, self.target)
        real = self._real
        state = self

        def counter(*args: Any, **kwargs: Any) -> Any:
            state.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(os, self.target, counter)


def seed(api: MemoryAPI, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "type": MemoryType.FACT,
        "title": "Seeded memory",
        "body": "Original body text.",
    }
    defaults.update(kw)
    return api.create(**defaults)


# ---------------------------------------------------------------- operations
# Each entry returns a callable that performs one operation on an open API.


def op_create(api: MemoryAPI, state: dict[str, Any]) -> None:
    api.create(type=MemoryType.PREFERENCE, title="Created under crash", body="New body.")


def op_update(api: MemoryAPI, state: dict[str, Any]) -> None:
    api.update(state["mid"], title="Updated under crash", body="Replacement body.")


def op_update_tags(api: MemoryAPI, state: dict[str, Any]) -> None:
    api.update(state["mid"], tags=["alpha", "beta"], salience=0.9)


def op_type_change(api: MemoryAPI, state: dict[str, Any]) -> None:
    # A type change is a file move, so the plan spans two filesystem actions.
    memory = api.require(state["mid"])
    with api.journal:
        result = api.store.overwrite(memory.with_updates(type=MemoryType.DECISION))
        if result is not None:
            api._reindex_one(memory.id)
            api.conn.commit()
            api.journal.commit(result.op_id)


def op_soft_delete(api: MemoryAPI, state: dict[str, Any]) -> None:
    api.delete(state["mid"], reason="crash test")


def op_permanent_delete(api: MemoryAPI, state: dict[str, Any]) -> None:
    api.delete(state["mid"], permanent=True, reason="crash test")


OPERATIONS: dict[str, Callable[[MemoryAPI, dict[str, Any]], None]] = {
    "create": op_create,
    "update": op_update,
    "update_tags": op_update_tags,
    "type_change": op_type_change,
    "soft_delete": op_soft_delete,
    "permanent_delete": op_permanent_delete,
}

#: Operations whose intent is destruction. A partially-applied one of these
#: rolls *forward*, so the post-op state is the only acceptable landing.
DESTRUCTIVE = {"permanent_delete"}

SYSCALLS = ("replace", "fsync", "unlink")


def run_clean(root: Path, op_name: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Run the operation to completion, returning the post-op manifest."""
    state: dict[str, Any] = {}
    with MemoryAPI(root) as api:
        if op_name != "create":
            state["mid"] = seed(api).id
    before = shape(root)
    with MemoryAPI(root) as api:
        OPERATIONS[op_name](api, state)
    return before, {"after": shape(root), "state": state}


def count_syscalls(root: Path, op_name: str, syscall: str) -> int:
    """How many times the operation invokes a syscall, for parametrizing."""
    state: dict[str, Any] = {}
    with MemoryAPI(root) as api:
        if op_name != "create":
            state["mid"] = seed(api).id
    probe = CrashPoint(syscall, -1)
    with pytest.MonkeyPatch.context() as mp:
        probe.count_only(mp)
        with MemoryAPI(root) as api:
            OPERATIONS[op_name](api, state)
    return probe.calls


@pytest.mark.parametrize("op_name", sorted(OPERATIONS))
@pytest.mark.parametrize("syscall", SYSCALLS)
def test_crash_leaves_exactly_one_of_two_states(tmp_path: Path, op_name: str, syscall: str) -> None:
    """The load-bearing assertion: no third state is reachable.

    For every point in the operation where the named syscall is used, crash
    there, reopen, and require the tree to equal the pre-op or post-op
    manifest exactly.
    """
    # Discover the shape of a clean run in a throwaway root.
    probe_root = tmp_path / "probe"
    MemoryAPI.initialize(probe_root).close()
    total = count_syscalls(probe_root, op_name, syscall)
    if total == 0:
        pytest.skip(f"{op_name} does not call os.{syscall}")

    reference_root = tmp_path / "reference"
    MemoryAPI.initialize(reference_root).close()
    before, outcome = run_clean(reference_root, op_name)
    after = outcome["after"]

    for nth in range(1, total + 1):
        root = tmp_path / f"case-{syscall}-{nth}"
        MemoryAPI.initialize(root).close()

        state: dict[str, Any] = {}
        with MemoryAPI(root) as api:
            if op_name != "create":
                state["mid"] = seed(api).id

        pre_crash = shape(root)
        assert pre_crash == before, "seeding diverged from the reference run"

        crash = CrashPoint(syscall, nth)
        with pytest.MonkeyPatch.context() as mp:
            crash.install(mp)
            try:
                with MemoryAPI(root) as api:
                    OPERATIONS[op_name](api, state)
            except SimulatedCrash:
                pass
            except Exception:
                # A real error path (not the simulated crash) still has to
                # leave a recoverable tree, so keep going.
                pass

        # Reopen: recovery runs on open and must settle the operation.
        with MemoryAPI(root) as api:
            report = api.recovery
            final = shape(root)

        assert final in (before, after), (
            f"{op_name} crashed at os.{syscall} #{nth} left a third state\n"
            f"unexpected: {sorted(set(final) ^ set(before))}"
        )
        if op_name in DESTRUCTIVE and final != before:
            assert final == after, "a destructive op must roll forward, not partially"
        assert residue(root) == [], f"recovery left residue after {syscall} #{nth}"
        assert report is not None

        # Recovery must be idempotent.
        with MemoryAPI(root) as api:
            again = api.recovery
            assert again is not None and again.settled == 0
            assert shape(root) == final

        # And the index must agree with the files it landed on.
        with MemoryAPI(root) as api:
            result = api.verify()
            assert result["clean"], f"index diverged after {syscall} #{nth}: {result}"


class TestSpecificGuarantees:
    """Named cases the parametrized sweep covers but that deserve to be explicit."""

    def test_update_crashed_mid_write_restores_original_bytes(self, root: Path) -> None:
        with MemoryAPI(root) as api:
            memory = seed(api, title="Original title", body="Original body.")
            original = hash_file(Paths(root).memory_file("fact", memory.id))

        crash = CrashPoint("replace", 1)
        with pytest.MonkeyPatch.context() as mp:
            crash.install(mp)
            with pytest.raises(SimulatedCrash), MemoryAPI(root) as api:
                api.update(memory.id, title="Never lands", body="Never lands either.")

        with MemoryAPI(root) as api:
            back = api.get(memory.id)
            assert back is not None
            assert back.title == "Original title"
            assert back.body == "Original body."
            assert hash_file(Paths(root).memory_file("fact", memory.id)) == original

    def test_permanent_delete_after_tombstone_rolls_forward(self, root: Path) -> None:
        """The tombstone is durable, so destruction is finished, not undone."""
        with MemoryAPI(root) as api:
            memory = seed(api)

        # Crash on the unlink, after the tombstone write has landed.
        crash = CrashPoint("unlink", 1)
        with pytest.MonkeyPatch.context() as mp:
            crash.install(mp)
            try:
                with MemoryAPI(root) as api:
                    api.delete(memory.id, permanent=True)
            except SimulatedCrash:
                pass

        paths = Paths(root)
        assert paths.tombstone_file(memory.id).exists(), "tombstone landed before the crash"

        with MemoryAPI(root) as api:
            assert api.get(memory.id) is None, "recovery completed the deletion"
            assert not paths.memory_file("fact", memory.id).exists()
            tomb = api.tombstone(memory.id)
            assert tomb is not None and tomb.permanent is True

    def test_create_crashed_before_write_leaves_nothing(self, root: Path) -> None:
        before = manifest(root)  # same root, so raw paths are comparable here
        crash = CrashPoint("replace", 1)
        with pytest.MonkeyPatch.context() as mp:
            crash.install(mp)
            try:
                with MemoryAPI(root) as api:
                    api.create(type=MemoryType.FACT, title="Doomed", body="x")
            except SimulatedCrash:
                pass

        with MemoryAPI(root) as api:
            assert manifest(root) == before
            assert api.verify()["clean"]

    def test_index_catches_up_when_db_write_fails(self, root: Path, monkeypatch) -> None:
        """A database failure after the file lands must not lose the memory."""
        import sqlite3

        from memory import db as db_module

        with MemoryAPI(root) as api:
            real_index = db_module.index_memory
            calls = {"n": 0}

            def flaky(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise sqlite3.OperationalError("simulated index failure")
                return real_index(*args, **kwargs)

            monkeypatch.setattr(db_module, "index_memory", flaky)
            memory = api.create(type=MemoryType.FACT, title="Survives", body="body")
            monkeypatch.setattr(db_module, "index_memory", real_index)

            # The file is authoritative, so the memory exists despite the
            # index write failing.
            assert api.get(memory.id) is not None

        # Reopening reconciles the index from the dirty marker.
        with MemoryAPI(root) as api:
            row = api.conn.execute(
                "SELECT title FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
            assert row is not None and row["title"] == "Survives"
            assert api.verify()["clean"]

    def test_torn_journal_line_does_not_block_recovery(self, root: Path) -> None:
        with MemoryAPI(root) as api:
            seed(api)
        paths = Paths(root)
        log = next(paths.journal_dir.glob("*.jsonl"))
        with open(log, "a", encoding="utf-8") as fh:
            fh.write('{"op_id":"partial","seq":999,"kind":"create_mem')

        with MemoryAPI(root) as api:
            assert api.recovery is not None
            assert api.recovery.torn_lines == 1
            assert api.verify()["clean"]

    def test_multiple_interrupted_ops_all_settle(self, root: Path) -> None:
        paths = Paths(root)
        with MemoryAPI(root) as api:
            kept = seed(api, title="Kept")

        # Fabricate three unsettled ops by beginning without committing.
        from memory.journal import Journal, OpKind, Step

        journal = Journal(paths)
        with journal:
            for i in range(3):
                target = paths.type_dir("fact") / f"{new_id()}.md"
                journal.begin(
                    OpKind.CREATE_MEMORY,
                    [new_id()],
                    [
                        Step(
                            i=0,
                            act="write",
                            path=paths.rel(target),
                            new_hash="sha256:" + f"{i}" * 64,
                        )
                    ],
                )

        with MemoryAPI(root) as api:
            assert api.recovery is not None
            assert api.recovery.settled == 3
            assert api.get(kept.id) is not None
            assert api.verify()["clean"]


class TestRecoveryIsIdempotent:
    def test_repeated_opens_are_quiet(self, root: Path) -> None:
        with MemoryAPI(root) as api:
            seed(api)
        for _ in range(3):
            with MemoryAPI(root) as api:
                assert api.recovery is not None
                assert api.recovery.clean

    def test_no_residue_after_normal_operation(self, root: Path) -> None:
        with MemoryAPI(root) as api:
            memory = seed(api)
            api.update(memory.id, title="Changed")
            api.delete(memory.id)
        assert residue(root) == []
