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
from .agent_assets import SKILL_NAMES, skill_source, skill_version
from .paths import ENV_VAR, Paths

#: Where the data root goes when the caller names none. Under ``$HOME`` and
#: not the CWD: a sandbox's working directory is frequently a scratch checkout
#: that gets deleted, and losing the memory corpus with it is the one failure
#: this system exists to prevent.
DEFAULT_ROOT = Path("~/.memory")

#: Agent config directory. Claude Code reads skills from ``<dir>/skills`` and
#: MCP servers from ``<dir>/settings.json``.
DEFAULT_AGENT_DIR = Path("~/.claude")

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


def install_skills(agent_dir: Path, *, dry_run: bool, force: bool = False) -> Step:
    """Copy the bundled skills into the agent's skills directory.

    A skill already present is overwritten only when its version stamp
    differs from ours or ``force`` is set. The stamp is what makes an upgrade
    detectable: without it, a copy edited into place by a previous version
    looks identical to a current one and silently keeps stale instructions in
    front of the model.
    """
    dest_root = agent_dir.expanduser() / "skills"
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

        dest = dest_root / name
        dest_md = dest / "SKILL.md"
        if dest_md.is_file() and not force:
            have = skill_version(dest_md)
            if have == __version__:
                skipped.append(name)
                continue
            updated.append(f"{name} ({have or 'unstamped'} -> {__version__})")
        elif dest_md.is_file():
            updated.append(f"{name} (forced)")
        else:
            installed.append(name)

        if not dry_run:
            # The repository ships `.claude/skills` as a symlink to the
            # package copy. Resolving the source keeps that case from
            # copying a link into the destination.
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src.resolve() / "SKILL.md", dest_md)

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
        path=str(dest_root),
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
    resolved_agent = (agent_dir or DEFAULT_AGENT_DIR).expanduser()
    resolved_home = home or Path.home()

    steps: list[Step] = [install_root(resolved_root, dry_run=dry_run)]
    if skills:
        steps.append(install_skills(resolved_agent, dry_run=dry_run, force=force))
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

    # Warn when the environment will not find the root on its own. A caller
    # who skipped --shell-env needs to know their next `mem` call will not
    # see this root unless they pass --root.
    if not dry_run and not shell_env and os.environ.get(ENV_VAR) != str(resolved_root):
        warnings.append(f"{ENV_VAR} is not set to {resolved_root}; pass --root or export it")

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
    dest_root = (agent_dir or DEFAULT_AGENT_DIR).expanduser() / "skills"
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


def _write_atomic(path: Path, text: str) -> None:
    """Replace a file's contents without a window where it is truncated.

    These are a user's config files; a crash mid-write that leaves
    ``settings.json`` empty is worse than the install failing.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
