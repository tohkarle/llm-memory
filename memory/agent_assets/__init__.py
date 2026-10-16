"""Files the package ships *for an agent* rather than for Python to import.

The skills under ``skills/`` are the agent-facing half of this system: the
CLI is useless to a model that does not know when to reach for it. They are
carried inside the wheel so that installing the distribution and installing
the agent's instructions are one act -- see ``mem install``.

The repository's own ``.claude/skills`` is a symlink to this directory, so
the copy being dogfooded and the copy being shipped cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

#: Skill directory names, each holding a ``SKILL.md``.
SKILL_NAMES: tuple[str, ...] = ("memory-recall", "memory-write")

#: Frontmatter key naming the package version a skill copy shipped with.
VERSION_KEY = "x-memory-version"


def skills_dir() -> Path:
    """Directory holding the bundled skills.

    A plain path join rather than ``importlib.resources``: these are read as
    a directory tree and copied wholesale, and the wheel installs them as
    real files, so there is no zip case to abstract over.
    """
    return Path(__file__).resolve().parent / "skills"


def skill_source(name: str) -> Path:
    if name not in SKILL_NAMES:
        raise ValueError(f"unknown bundled skill: {name!r}")
    return skills_dir() / name


def skill_version(skill_md: Path) -> str | None:
    """The ``x-memory-version`` stamp in a ``SKILL.md``, if it carries one.

    Deliberately a line scan rather than a YAML parse: this runs against
    files a user may have hand-edited, and a malformed document should yield
    "unknown version", not an exception out of ``mem doctor``.
    """
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        key, sep, value = line.partition(":")
        if sep and key.strip() == VERSION_KEY:
            return value.strip() or None
    return None
