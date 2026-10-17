"""Completing the install: data root, agent skills, MCP wiring, shell env.

Installing the wheel gives you a ``mem`` binary and nothing to point it at.
Four things still have to happen before an agent in a fresh sandbox can use
this system, and all four are things a human currently does by hand:

1. a data root has to exist and be initialized;
2. the bundled skills have to land where the agent will read them;
3. an MCP client has to be told the server exists;
4. ``$MEMORY_ROOT`` has to be set for every later shell.

``mem install`` does those. Every step is **idempotent** -- sandboxes get
re-provisioned, and a setup script that cannot be re-run is a setup script
that breaks on the second build -- and every step reports what it did rather
than whether it felt successful, so a calling agent can verify the outcome
from the JSON instead of trusting an exit code.

Nothing here writes outside the paths it reports, and the JSON settings merge
preserves keys it does not own: this runs against a user's real
``~/.claude/settings.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .agent_assets import (
    SKILL_NAMES,
    persona_text,
    skill_source,
    skill_version,
)
from .paths import ENV_VAR, Paths

#: Where the data root goes when the caller names none: the working
#: directory, so that a default install writes nothing outside the tree it
#: was run in. An agent in a sandbox frequently *cannot* write to ``$HOME``,
#: and one that can should not have a corpus appear there as a side effect
#: of a command about something else.
#:
#: The cost is real and belongs in the docs rather than in a different
#: default: a corpus under a scratch checkout dies with the checkout. Where
#: the data must outlive the directory, name the root -- ``--root`` or
#: ``$MEMORY_ROOT`` -- and keep it somewhere durable.
DEFAULT_ROOT = Path(".")

#: Agent config directory. Claude Code reads skills from ``<dir>/skills``
#: and MCP servers from ``<dir>/settings.json``.
#:
#: Local for the same reason as the root: installing a memory system is not
#: a licence to edit the user's global agent configuration. ``--agent-dir
#: ~/.claude`` still installs globally for someone who wants that, and says
#: so explicitly.
DEFAULT_AGENT_DIR = Path("./.claude")

#: Overrides the agent config directory, the way ``$MEMORY_ROOT`` overrides
#: the data root. An agent confined to its working directory installs skills
#: there with ``--agent-dir``, and without this every later command would go
#: back to looking under ``$HOME`` and report them missing.
AGENT_DIR_ENV_VAR = "MEMORY_AGENT_DIR"

#: Name the MCP server is registered under in ``settings.json``.
MCP_SERVER_NAME = "memory"

#: Shell profiles to append ``MEMORY_ROOT`` to, by ``$SHELL`` basename.
#: ``.profile`` is the fallback because every POSIX login shell reads it.
SHELL_PROFILES: dict[str, str] = {
    "zsh": ".zshrc",
    "bash": ".bashrc",
    "fish": ".config/fish/config.fish",
}
FALLBACK_PROFILE = ".profile"

#: Marker opening the block we manage in a shell profile, so a re-run can
#: recognize its own previous work instead of appending a second copy.
ENV_MARKER = "# >>> personal-ai-memory >>>"
ENV_MARKER_END = "# <<< personal-ai-memory <<<"

#: File the persona is merged into. Claude Code loads it at session start
#: without being asked, which is the one property a skill cannot offer:
#: skills are selected by the model, so nothing in them is guaranteed to
#: reach a fresh agent's first turn.
PERSONA_FILE = "CLAUDE.md"

#: The cross-runtime counterpart, written beside it. Cursor, Codex, Copilot
#: and others read ``AGENTS.md``; Claude Code reads ``CLAUDE.md``. Same
#: text, two filenames, because the convention is split and a system that
#: claims to work with any agent cannot pick one and call it portable.
PERSONA_FILE_PORTABLE = "AGENTS.md"

#: Cross-runtime skills location, the counterpart to ``.claude/skills``.
#: Written relative to the directory holding the agent config rather than
#: inside it: ``.claude/`` is one runtime's directory, and a second
#: runtime's skills do not belong under it.
PORTABLE_SKILLS_DIR = Path(".agents") / "skills"

#: Markers bounding the block we manage inside that file. Same device as the
#: shell profile's: the file belongs to the user and carries their own
#: instructions, so a re-run has to recognize its previous work rather than
#: append a second copy or overwrite the lot.
PERSONA_MARKER = "<!-- >>> personal-ai-memory >>> -->"
PERSONA_MARKER_END = "<!-- <<< personal-ai-memory <<< -->"

#: Dependency sets a caller can ask for by name. These are reported as the
#: command to run, not installed: `mem install` runs *inside* the environment
#: it would be mutating, and a tool that pip-installs into its own
#: interpreter mid-run is a tool that can break itself.
PROFILES: dict[str, tuple[str, ...]] = {
    "minimal": (),
    "text": ("media",),
    "full": ("all-media", "mcp", "tokenizer"),
}


class InstallError(RuntimeError):
    """A step could not be completed."""


def resolve_agent_dir(agent_dir: Path | None = None) -> Path:
    """Where the agent reads its skills and settings from.

    Order mirrors root discovery: an explicit argument wins, then
    ``$MEMORY_AGENT_DIR``, then ``~/.claude``.
    """
    if agent_dir is not None:
        return agent_dir.expanduser().resolve()
    env = os.environ.get(AGENT_DIR_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    # Resolved like the other two branches: the default is relative now, and
    # an unresolved path would report as "./.claude" while comparing unequal
    # to the same directory named absolutely -- which reads to the caller as
    # a non-default dir and triggers a warning about an env var they do not
    # need.
    return DEFAULT_AGENT_DIR.expanduser().resolve()


@dataclass
class Step:
    """One unit of install work and what it actually did.

    ``changed`` is the field a caller should branch on: it distinguishes "I
    made this true" from "this was already true", which is what makes a
    re-run's output readable.
    """

    name: str
    changed: bool
    detail: str
    path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "step": self.name,
            "changed": self.changed,
            "detail": self.detail,
        }
        if self.path is not None:
            out["path"] = self.path
        if self.warnings:
            out["warnings"] = self.warnings
        return out


# ---------------------------------------------------------------- root


def install_root(root: Path, *, dry_run: bool) -> Step:
    """Create and initialize the data tree.

    Delegates the tree itself to :meth:`Paths.ensure_tree` so there is one
    definition of what a root contains, and opens the API afterwards only to
    build the index -- which is what makes the root usable rather than just
    present.
    """
    paths = Paths(root)
    existed = paths.marker.exists()
    if dry_run:
        return Step(
            "root",
            changed=not existed,
            detail="exists" if existed else "would create",
            path=str(paths.root),
        )

    from .api import MemoryAPI

    if existed:
        api = MemoryAPI(paths.root)
        api.open()
        api.close()
        return Step("root", changed=False, detail="already initialized", path=str(paths.root))

    api = MemoryAPI.initialize(paths.root)
    api.close()
    return Step("root", changed=True, detail="created and indexed", path=str(paths.root))


# ---------------------------------------------------------------- skills


def install_skills(
    agent_dir: Path,
    *,
    dry_run: bool,
    force: bool = False,
    portable: bool = False,
) -> Step:
    """Copy the bundled skills into the agent's skills directory.

    With ``portable``, they also go to ``.agents/skills/`` beside it -- the
    cross-runtime location, for a harness that is not Claude Code. The two
    are copies of one source rather than a link, because a link is not what
    survives a ``git clone`` on every platform, and a drifted hand-made copy
    is the failure this is replacing.

    A skill already present is overwritten only when its version stamp
    differs from ours or ``force`` is set. The stamp is what makes an upgrade
    detectable: without it, a copy edited into place by a previous version
    looks identical to a current one and silently keeps stale instructions in
    front of the model.
    """
    dest_root = agent_dir.expanduser() / "skills"
    # Beside the agent dir, not inside it: `.claude/` belongs to one
    # runtime, and another runtime's skills do not live under it.
    portable_root = agent_dir.expanduser().parent / PORTABLE_SKILLS_DIR
    dest_roots = [dest_root, portable_root] if portable else [dest_root]
    installed: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    for name in SKILL_NAMES:
        src = skill_source(name)
        if not (src / "SKILL.md").is_file():
            # The wheel was built without its artifacts. Worth a warning
            # rather than an exception: the data root is still worth having.
            warnings.append(f"{name}: bundled SKILL.md missing from the installed package")
            continue

        # Each destination is judged on its own: a current copy under
        # `.claude` must not mask a missing or stale one under `.agents`,
        # which is exactly how a second location silently rots.
        states: list[str] = []
        for root in dest_roots:
            dest_md = root / name / "SKILL.md"
            if dest_md.is_file() and not force:
                have = skill_version(dest_md)
                # Content, not just the stamp. A stamp alone cannot see an
                # edit made within one release -- the bundled skill changes
                # and the version does not -- which leaves a stale copy
                # reporting itself current for the rest of the release.
                if have == __version__ and _same_file(src.resolve() / "SKILL.md", dest_md):
                    states.append("current")
                    continue
                states.append(
                    f"{have or 'unstamped'} -> {__version__}"
                    if have != __version__
                    else "content changed"
                )
            elif dest_md.is_file():
                states.append("forced")
            else:
                states.append("new")

            if not dry_run:
                # The repository ships `.claude/skills` as a symlink to the
                # package copy. Resolving the source keeps that case from
                # copying a link into the destination.
                dest_md.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src.resolve() / "SKILL.md", dest_md)

        if all(s == "current" for s in states):
            skipped.append(name)
        elif all(s == "new" for s in states):
            installed.append(name)
        else:
            moved = next((s for s in states if s not in ("current", "new")), "refreshed")
            updated.append(f"{name} ({moved})")

    # Tense follows dry_run: these strings are read by a human scanning a
    # provisioning log, where "would install" and "installed" must not blur.
    add_verb, up_verb = ("would install", "would update") if dry_run else ("installed", "updated")
    parts = []
    if installed:
        parts.append(f"{add_verb} {', '.join(installed)}")
    if updated:
        parts.append(f"{up_verb} {', '.join(updated)}")
    if skipped:
        parts.append(f"already current: {', '.join(skipped)}")
    detail = "; ".join(parts) if parts else "nothing to do"

    return Step(
        "skills",
        changed=bool(installed or updated),
        detail=detail,
        path=" + ".join(str(r) for r in dest_roots),
        warnings=warnings,
    )


# ---------------------------------------------------------------- mcp


def mcp_entry(root: Path) -> dict[str, Any]:
    """The ``mcpServers`` entry for this install.

    ``$MEMORY_ROOT`` goes in ``env`` rather than as a ``--root`` argument
    because the server resolves its root through :meth:`Paths.discover`, and
    an MCP client launches it with an unpredictable working directory, so
    discovery-by-ancestor cannot be relied on.
    """
    command = shutil.which("mem-mcp")
    if command:
        return {"command": command, "env": {ENV_VAR: str(root)}}
    # No console script on PATH -- an editable install that was never
    # `pip install`ed, or a PATH the installer did not extend. Naming this
    # interpreter explicitly still produces a config that works.
    return {
        "command": sys.executable,
        "args": ["-m", "memory.mcp_server"],
        "env": {ENV_VAR: str(root)},
    }


def install_mcp(agent_dir: Path, root: Path, *, dry_run: bool) -> Step:
    """Register the MCP server in the agent's ``settings.json``.

    Merges: the file belongs to the user and carries unrelated keys. A
    malformed existing file is an error rather than something to overwrite --
    clobbering a user's settings to install an optional feature is never the
    right trade.
    """
    settings = agent_dir.expanduser() / "settings.json"
    desired = mcp_entry(root)

    existing: dict[str, Any] = {}
    if settings.is_file():
        try:
            raw = json.loads(settings.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise InstallError(
                f"{settings} is not valid JSON ({exc}); fix or move it, then re-run"
            ) from exc
        if not isinstance(raw, dict):
            raise InstallError(f"{settings} does not contain a JSON object")
        existing = raw

    servers = existing.get("mcpServers")
    if servers is None:
        servers = {}
    elif not isinstance(servers, dict):
        raise InstallError(f"{settings}: 'mcpServers' is not a JSON object")

    current = servers.get(MCP_SERVER_NAME)
    if current == desired or _mcp_equivalent(current, desired, root):
        return Step("mcp", changed=False, detail="already registered", path=str(settings))

    replacing = MCP_SERVER_NAME in servers
    if dry_run:
        verb = "would update" if replacing else "would register"
        return Step("mcp", changed=True, detail=f"{verb} '{MCP_SERVER_NAME}'", path=str(settings))

    servers[MCP_SERVER_NAME] = desired
    existing["mcpServers"] = servers
    settings.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(settings, json.dumps(existing, indent=2) + "\n")

    return Step(
        "mcp",
        changed=True,
        detail=f"{'updated' if replacing else 'registered'} '{MCP_SERVER_NAME}'",
        path=str(settings),
    )


def _mcp_equivalent(current: Any, desired: dict[str, Any], root: Path) -> bool:
    """Whether an existing entry already launches this server at this root.

    Not the same as equality. ``mem-mcp`` may or may not be on PATH at
    install time, so the same root legitimately yields either the console
    script or an ``<interpreter> -m memory.mcp_server`` fallback. Treating
    those as different would make every run report a change -- and rewrite a
    working config -- whenever PATH differed between runs, which in a sandbox
    is most of them. A config that launches the right server at the right
    root is already correct, so leave it alone.
    """
    if not isinstance(current, dict):
        return False
    if (current.get("env") or {}).get(ENV_VAR) != str(root):
        return False
    command = str(current.get("command") or "")
    args = current.get("args") or []
    if Path(command).name in ("mem-mcp", "mem-mcp.exe") and not args:
        return True
    return isinstance(args, list) and args[:2] == ["-m", "memory.mcp_server"]


# ---------------------------------------------------------------- persona


def persona_block(*, inline_skills: bool, agent_dir: Path) -> str:
    """The managed block, stamped and bounded by its markers.

    ``inline_skills`` carries the skill bodies into the block itself. That
    is for the confined install, where no harness registers the skills
    directory: a pointer there costs a tool call *and* the agent's
    willingness to make it, and the whole reason this file exists is that
    willingness cannot be assumed. Where a harness does load skills, the
    pointer is right -- duplicating two long documents into every session's
    context to say what the harness already said is pure waste.
    """
    body = persona_text()
    if inline_skills:
        parts = [body, "", "---", ""]
        for name in SKILL_NAMES:
            md = skill_source(name) / "SKILL.md"
            if not md.is_file():
                continue
            parts.append(_strip_frontmatter(md.read_text(encoding="utf-8")).strip())
            parts.append("")
        body = "\n".join(parts).rstrip()
    stamp = f"<!-- personal-ai-memory {__version__} -- edits here are overwritten -->"
    return f"{PERSONA_MARKER}\n{stamp}\n{body}\n{PERSONA_MARKER_END}\n"


def _same_file(src: Path, dest: Path) -> bool:
    """Whether two files hold identical bytes.

    Read rather than compared by mtime or size: these are small text files
    a user may have hand-edited, and a same-size edit is exactly the case
    that must not read as unchanged.
    """
    try:
        return src.read_bytes() == dest.read_bytes()
    except OSError:
        return False


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block.

    A skill's frontmatter is addressed to the harness that registers it --
    inlined into prose that no harness is reading, ``name:`` and
    ``description:`` are noise competing with the instructions beneath them.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def installed_persona_version(target: Path) -> str | None:
    """The version stamped inside the managed block, if there is one.

    A line scan for the same reason ``skill_version`` is one: this runs
    against a file a user hand-edits, and a surprise there should read as
    "unknown version", not raise out of ``mem doctor``.
    """
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return None
    if PERSONA_MARKER not in text:
        return None
    start = text.index(PERSONA_MARKER)
    end = text.find(PERSONA_MARKER_END, start)
    segment = text[start : end if end != -1 else len(text)]
    for line in segment.split("\n"):
        line = line.strip()
        if line.startswith("<!-- personal-ai-memory "):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return None


def install_persona(
    agent_dir: Path,
    *,
    dry_run: bool,
    force: bool = False,
    inline_skills: bool = False,
    target: Path | None = None,
    portable: bool = False,
) -> Step:
    """Merge the persona into the agent's ``CLAUDE.md``.

    The file is the user's; only the marked block is ours. An unmarked file
    is appended to, never replaced -- the instructions already in it are the
    reason it exists.

    With ``portable``, the same block also goes to ``AGENTS.md`` beside it,
    for the runtimes that read that name instead. Both files are written in
    full rather than one pointing at the other: a pointer is only followed
    by an agent that chooses to, and being read without being chosen is the
    entire reason this file exists.
    """
    dest = target if target is not None else agent_dir.expanduser() / PERSONA_FILE
    block = persona_block(inline_skills=inline_skills, agent_dir=agent_dir)

    if portable:
        # AGENTS.md goes beside `.agents/`, not inside the Claude config
        # directory: a runtime that reads AGENTS.md has no reason to look
        # in `.claude/`, so a copy there would be read by nobody. An
        # explicit --persona-target names one file and is honored as given.
        portable_dest = (
            dest.parent / PERSONA_FILE_PORTABLE
            if target is not None
            else agent_dir.expanduser().parent / PERSONA_FILE_PORTABLE
        )
        primary = _merge_persona(dest, block, dry_run=dry_run, force=force)
        second = _merge_persona(portable_dest, block, dry_run=dry_run, force=force)
        both = f"{primary.name_detail}; {second.name_detail}"
        return Step(
            "persona",
            changed=primary.changed or second.changed,
            detail=both,
            path=f"{dest} + {portable_dest}",
        )
    return _merge_persona(dest, block, dry_run=dry_run, force=force).as_step()


@dataclass
class _MergeResult:
    """What merging the block into one file did."""

    path: Path
    changed: bool
    detail: str

    @property
    def name_detail(self) -> str:
        return f"{self.path.name}: {self.detail}"

    def as_step(self) -> Step:
        return Step("persona", changed=self.changed, detail=self.detail, path=str(self.path))


def _merge_persona(dest: Path, block: str, *, dry_run: bool, force: bool) -> _MergeResult:
    """Write the managed block into one file, preserving everything else."""
    text = dest.read_text(encoding="utf-8") if dest.is_file() else ""

    if PERSONA_MARKER in text:
        start = text.index(PERSONA_MARKER)
        end_marker = text.find(PERSONA_MARKER_END, start)
        if end_marker == -1:
            raise InstallError(
                f"{dest}: found {PERSONA_MARKER!r} with no closing "
                f"{PERSONA_MARKER_END!r}; remove the partial block, then re-run"
            )
        end = end_marker + len(PERSONA_MARKER_END)
        have = installed_persona_version(dest)
        if text[start:end] == block.rstrip("\n") and not force:
            return _MergeResult(dest, False, "already current")
        moved = f"({have or 'unstamped'} -> {__version__})"
        if dry_run:
            return _MergeResult(dest, True, f"would update block {moved}")
        _write_atomic(dest, text[:start] + block.rstrip("\n") + text[end:])
        return _MergeResult(dest, True, f"updated block {moved}")

    if dry_run:
        verb = "would append block to" if text else "would create"
        return _MergeResult(dest, True, f"{verb} {dest.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if text:
        sep = "" if text.endswith("\n") else "\n"
        _write_atomic(dest, f"{text}{sep}\n{block}")
        return _MergeResult(dest, True, f"appended block to existing {dest.name}")
    _write_atomic(dest, block)
    return _MergeResult(dest, True, f"created {dest.name}")


# ---------------------------------------------------------------- shell env


def profile_path(home: Path, shell: str | None) -> Path:
    """The profile to append to, chosen from ``$SHELL``."""
    name = Path(shell).name if shell else ""
    return home / SHELL_PROFILES.get(name, FALLBACK_PROFILE)


def install_shell_env(root: Path, *, home: Path, shell: str | None, dry_run: bool) -> Step:
    """Append a marked ``MEMORY_ROOT`` export to the shell profile.

    Written between markers so a re-run replaces its own block instead of
    stacking exports. A profile that already exports the right root is left
    untouched.
    """
    target = profile_path(home, shell)
    is_fish = target.name == "config.fish"
    export = f"set -gx {ENV_VAR} {root}" if is_fish else f'export {ENV_VAR}="{root}"'
    block = f"{ENV_MARKER}\n{export}\n{ENV_MARKER_END}\n"

    text = target.read_text(encoding="utf-8") if target.is_file() else ""
    if ENV_MARKER in text:
        start = text.index(ENV_MARKER)
        end_marker = text.find(ENV_MARKER_END, start)
        if end_marker == -1:
            raise InstallError(
                f"{target}: found {ENV_MARKER!r} with no closing {ENV_MARKER_END!r}; "
                "remove the partial block, then re-run"
            )
        end = end_marker + len(ENV_MARKER_END)
        current = text[start:end]
        if export in current:
            return Step("shell_env", changed=False, detail="already set", path=str(target))
        if dry_run:
            return Step("shell_env", changed=True, detail="would update block", path=str(target))
        # Keep whatever followed the block (a trailing newline, later lines).
        new_text = text[:start] + block.rstrip("\n") + text[end:]
        _write_atomic(target, new_text)
        return Step("shell_env", changed=True, detail="updated block", path=str(target))

    if dry_run:
        return Step("shell_env", changed=True, detail="would append block", path=str(target))

    sep = "" if not text or text.endswith("\n") else "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(f"{sep}\n{block}")
    return Step(
        "shell_env",
        changed=True,
        detail="appended block",
        path=str(target),
        warnings=[f"open a new shell, or run: export {ENV_VAR}={root}"],
    )


# ---------------------------------------------------------------- driver


def install(
    *,
    root: Path | None = None,
    agent_dir: Path | None = None,
    skills: bool = False,
    persona: bool = False,
    inline_skills: bool = False,
    persona_target: Path | None = None,
    portable: bool = False,
    mcp: bool = False,
    shell_env: bool = False,
    profile: str = "minimal",
    force: bool = False,
    dry_run: bool = False,
    home: Path | None = None,
    shell: str | None = None,
) -> dict[str, Any]:
    """Run the requested install steps and report what each one did.

    The root step always runs: it is the one thing without which nothing else
    is meaningful. The other three are opt-in because each writes outside the
    data root, into files the user owns.
    """
    if profile not in PROFILES:
        raise InstallError(f"unknown profile {profile!r}; choose from {', '.join(PROFILES)}")

    resolved_root = (root or DEFAULT_ROOT).expanduser().resolve()
    resolved_agent = resolve_agent_dir(agent_dir)
    resolved_home = home or Path.home()

    steps: list[Step] = [install_root(resolved_root, dry_run=dry_run)]
    if skills:
        steps.append(
            install_skills(resolved_agent, dry_run=dry_run, force=force, portable=portable)
        )
    if persona:
        steps.append(
            install_persona(
                resolved_agent,
                dry_run=dry_run,
                force=force,
                inline_skills=inline_skills,
                target=persona_target,
                portable=portable,
            )
        )
    if mcp:
        steps.append(install_mcp(resolved_agent, resolved_root, dry_run=dry_run))
    if shell_env:
        steps.append(
            install_shell_env(
                resolved_root,
                home=resolved_home,
                shell=shell if shell is not None else os.environ.get("SHELL"),
                dry_run=dry_run,
            )
        )

    extras = PROFILES[profile]
    warnings = [w for s in steps for w in s.warnings]

    # Warn only when the environment genuinely will not find this root.
    # Discovery walks up from the working directory to the root marker, so a
    # root at or above the CWD needs no variable at all -- and saying
    # otherwise teaches the reader that an unset $MEMORY_ROOT is a missing
    # step, which invites them to "fix" it with a path that points at an
    # empty tree.
    if not dry_run and not shell_env and not _discoverable(resolved_root):
        warnings.append(
            f"{resolved_root} is not discoverable from the working directory; "
            f"pass --root or export {ENV_VAR}"
        )

    # Same test for the agent dir, which has no ancestor walk: a later
    # command looks at the default unless the variable names this one.
    if (
        not dry_run
        and (skills or persona)
        and resolved_agent != DEFAULT_AGENT_DIR.expanduser().resolve()
        and os.environ.get(AGENT_DIR_ENV_VAR) != str(resolved_agent)
    ):
        warnings.append(
            f"{AGENT_DIR_ENV_VAR} is not set to {resolved_agent}; "
            f"export it so later commands find these skills"
        )

    return {
        "version": __version__,
        "root": str(resolved_root),
        "agent_dir": str(resolved_agent),
        "dry_run": dry_run,
        "profile": profile,
        "extras_hint": (
            f"uv pip install 'personal-ai-memory[{','.join(extras)}]'" if extras else None
        ),
        "changed": any(s.changed for s in steps),
        "steps": [s.to_json() for s in steps],
        "warnings": warnings,
    }


def skills_health(agent_dir: Path | None = None) -> dict[str, Any]:
    """Whether the installed skills are present and current, for ``doctor``.

    Reported rather than repaired: ``doctor`` is read-only, and a stale skill
    is a thing the user should be told about, with the one command that fixes
    it.
    """
    dest_root = resolve_agent_dir(agent_dir) / "skills"
    entries: dict[str, Any] = {}
    stale: list[str] = []
    for name in SKILL_NAMES:
        md = dest_root / name / "SKILL.md"
        if not md.is_file():
            entries[name] = {"installed": False}
            stale.append(name)
            continue
        found = skill_version(md)
        current = found == __version__
        entries[name] = {"installed": True, "version": found, "current": current}
        if not current:
            stale.append(name)
    out: dict[str, Any] = {
        "dir": str(dest_root),
        "expected_version": __version__,
        "skills": entries,
    }
    if stale:
        out["hint"] = "run `mem install --skills` to install or refresh"
    return out


def _discoverable(root: Path) -> bool:
    """Whether ``Paths.discover`` would find this root with no variable set.

    True when the root is the working directory or an ancestor of it: the
    walk starts at the CWD and climbs. Anything else -- a sibling directory,
    a path under ``$HOME`` while the shell sits elsewhere -- needs
    ``$MEMORY_ROOT`` or an explicit ``--root`` on every later call.
    """
    if os.environ.get(ENV_VAR) == str(root):
        return True
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return False
    return root == cwd or root in cwd.parents


def persona_health(agent_dir: Path | None = None) -> dict[str, Any]:
    """Whether the always-loaded instructions are in place, for ``doctor``.

    Reported, not repaired, like the skills: this writes into a file the
    user owns and may have reorganized, and ``doctor`` is read-only.
    """
    dest = resolve_agent_dir(agent_dir) / PERSONA_FILE
    if not dest.is_file():
        return {
            "path": str(dest),
            "installed": False,
            "expected_version": __version__,
            "hint": "run `mem install --persona` to install",
        }
    found = installed_persona_version(dest)
    text = dest.read_text(encoding="utf-8")
    present = PERSONA_MARKER in text
    out: dict[str, Any] = {
        "path": str(dest),
        "installed": present,
        "version": found,
        "current": present and found == __version__,
        "expected_version": __version__,
    }
    if not out["current"]:
        out["hint"] = "run `mem install --persona` to install or refresh"
    return out


def _write_atomic(path: Path, text: str) -> None:
    """Replace a file's contents without a window where it is truncated.

    These are a user's config files; a crash mid-write that leaves
    ``settings.json`` empty is worse than the install failing.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
