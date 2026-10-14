"""Root discovery and all path math.

Every filesystem location in the system is computed here, so the rest of the
codebase never joins paths by hand and a relocation of ``data/`` cannot leave
stale absolute paths behind.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

#: Singular enum values; the on-disk directories use the plural forms below.
MEMORY_TYPES: tuple[str, ...] = ("fact", "preference", "event", "decision", "procedure")

#: Explicit map rather than inflection logic -- five entries, no surprises.
TYPE_DIRS: dict[str, str] = {
    "fact": "facts",
    "preference": "preferences",
    "event": "events",
    "decision": "decisions",
    "procedure": "procedures",
}

DERIVED_KINDS: tuple[str, ...] = ("transcripts", "descriptions", "keyframes")

ROOT_MARKER = ".memory-root"
ENV_VAR = "MEMORY_ROOT"


class RootNotFound(RuntimeError):
    """No memory root could be discovered."""


def plural(mem_type: str) -> str:
    try:
        return TYPE_DIRS[mem_type]
    except KeyError:
        raise ValueError(f"unknown memory type: {mem_type!r}") from None


class Paths:
    """Resolved locations for one memory root.

    ``root`` is the project directory that contains ``data/``. Everything
    durable lives under ``self.data``.
    """

    __slots__ = ("root",)

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    # ------------------------------------------------------------------ discovery

    @classmethod
    def discover(cls, start: Path | str | None = None) -> Paths:
        """Find the memory root.

        Order: ``$MEMORY_ROOT`` > nearest ancestor containing
        ``data/.memory-root`` > ``<start>/data`` if it exists.
        """
        env = os.environ.get(ENV_VAR)
        if env:
            return cls(env)

        here = Path(start).expanduser().resolve() if start else Path.cwd()
        for candidate in (here, *here.parents):
            if (candidate / "data" / ROOT_MARKER).exists():
                return cls(candidate)

        if (here / "data").is_dir():
            return cls(here)

        raise RootNotFound(
            f"no memory root found from {here} (set ${ENV_VAR}, pass --root, or run `mem init`)"
        )

    # ------------------------------------------------------------------ top level

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def memories(self) -> Path:
        return self.data / "memories"

    @property
    def conversations(self) -> Path:
        return self.data / "conversations"

    @property
    def documents(self) -> Path:
        return self.data / "documents"

    @property
    def assets(self) -> Path:
        return self.data / "assets"

    @property
    def blobs(self) -> Path:
        return self.assets / "blobs"

    @property
    def asset_meta_dir(self) -> Path:
        return self.assets / "meta"

    @property
    def derived(self) -> Path:
        return self.data / "derived"

    @property
    def index_dir(self) -> Path:
        return self.data / "index"

    @property
    def index_db(self) -> Path:
        return self.index_dir / "memory.sqlite"

    @property
    def journal_dir(self) -> Path:
        return self.data / ".journal"

    @property
    def undo_dir(self) -> Path:
        return self.journal_dir / "undo"

    @property
    def seq_file(self) -> Path:
        return self.journal_dir / "seq"

    @property
    def tombstones(self) -> Path:
        return self.data / ".tombstones"

    @property
    def asset_tombstones(self) -> Path:
        return self.tombstones / "assets"

    @property
    def lock_file(self) -> Path:
        return self.data / ".lock"

    @property
    def marker(self) -> Path:
        return self.data / ROOT_MARKER

    # ------------------------------------------------------------------ memories

    def type_dir(self, mem_type: str) -> Path:
        return self.memories / plural(mem_type)

    def memory_file(self, mem_type: str, mid: str) -> Path:
        return self.type_dir(mem_type) / f"{mid}.md"

    def memory_glob(self) -> Iterator[Path]:
        """Every memory file on disk, in a deterministic order."""
        for mem_type in MEMORY_TYPES:
            yield from sorted(self.type_dir(mem_type).glob("*.md"))

    def tombstone_file(self, mid: str) -> Path:
        return self.tombstones / f"{mid}.json"

    def tombstone_glob(self) -> Iterator[Path]:
        yield from sorted(self.tombstones.glob("*.json"))

    # ------------------------------------------------------------------ assets

    def asset_blob(self, sha256: str) -> Path:
        """Content-addressed blob path, sharded two levels deep.

        ``ab/cd/abcdef...`` gives 65536 buckets, keeping any one directory
        small enough for APFS and for file-sync tools up to ~65M blobs.
        """
        if len(sha256) < 4:
            raise ValueError(f"not a sha256 hex digest: {sha256!r}")
        return self.blobs / sha256[:2] / sha256[2:4] / sha256

    def asset_meta(self, asset_id: str) -> Path:
        if len(asset_id) < 2:
            raise ValueError(f"not an asset id: {asset_id!r}")
        return self.asset_meta_dir / asset_id[:2] / f"{asset_id}.json"

    def asset_meta_glob(self) -> Iterator[Path]:
        yield from sorted(self.asset_meta_dir.glob("*/*.json"))

    def blob_glob(self) -> Iterator[Path]:
        yield from sorted(self.blobs.glob("*/*/*"))

    def asset_tombstone_file(self, asset_id: str) -> Path:
        return self.asset_tombstones / f"{asset_id}.json"

    # ------------------------------------------------------------------ derived

    def derived_dir(self, kind: str, asset_id: str) -> Path:
        if kind not in DERIVED_KINDS:
            raise ValueError(f"unknown derived kind: {kind!r}")
        if len(asset_id) < 2:
            raise ValueError(f"not an asset id: {asset_id!r}")
        return self.derived / kind / asset_id[:2] / asset_id

    def derived_glob(self) -> Iterator[Path]:
        for kind in DERIVED_KINDS:
            base = self.derived / kind
            if base.is_dir():
                yield from sorted(p for p in base.rglob("*") if p.is_file())

    # ------------------------------------------------------------------ relative

    def rel(self, path: Path | str) -> str:
        """Path relative to ``data/`` -- the only form ever persisted.

        Storing relative paths is what lets the whole tree move without
        invalidating the journal or the index.
        """
        p = Path(path)
        if not p.is_absolute():
            return p.as_posix()
        return p.resolve().relative_to(self.data).as_posix()

    def abs(self, relpath: str) -> Path:
        return self.data / relpath

    # ------------------------------------------------------------------ setup

    def all_dirs(self) -> list[Path]:
        dirs = [
            self.memories,
            *(self.type_dir(t) for t in MEMORY_TYPES),
            self.conversations,
            self.documents,
            self.assets,
            self.blobs,
            self.asset_meta_dir,
            self.derived,
            *(self.derived / k for k in DERIVED_KINDS),
            self.index_dir,
            self.journal_dir,
            self.undo_dir,
            self.tombstones,
            self.asset_tombstones,
        ]
        return dirs

    def ensure_tree(self) -> None:
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        if not self.marker.exists():
            self.marker.write_text(
                "This file marks a personal-ai-memory data root.\n", encoding="utf-8"
            )

    def __repr__(self) -> str:
        return f"Paths(root={str(self.root)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Paths) and other.root == self.root

    def __hash__(self) -> int:
        return hash(self.root)
