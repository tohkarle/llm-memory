"""`mem install`: the bootstrap must be idempotent and must not clobber.

The two properties worth testing here are the ones a sandbox depends on:
running it twice changes nothing the second time, and it never damages a
config file it does not own.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from memory import __version__
from memory import install as install_mod
from memory.agent_assets import (
    SKILL_NAMES,
    persona_source,
    persona_text,
    skill_source,
    skill_version,
)
from memory.cli import EXIT_VALIDATION, main
from memory.install import (
    ENV_MARKER,
    MCP_SERVER_NAME,
    PERSONA_FILE,
    PERSONA_MARKER,
    InstallError,
    install,
    persona_health,
    skills_health,
)
from memory.paths import ENV_VAR, Paths


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite off the developer's real root and shell profile."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("SHELL", raising=False)


# ---------------------------------------------------------------- bundled assets


def test_every_declared_skill_ships_a_file() -> None:
    """A wheel built without its artifacts is a broken wheel."""
    for name in SKILL_NAMES:
        md = skill_source(name) / "SKILL.md"
        assert md.is_file(), f"{name} is declared but has no SKILL.md"
        assert skill_version(md) == __version__, f"{name} version stamp is stale"


def test_persona_ships_a_file() -> None:
    """Same contract as the skills: the wheel must carry it."""
    assert persona_source().is_file(), "PERSONA.md is not in the package"
    assert persona_text().strip(), "PERSONA.md is empty"


def test_persona_states_both_standing_rules() -> None:
    """The two rules are the whole reason this file is always loaded.

    Asserted on substance rather than wording: if an edit drops the
    delegation half, every fresh agent silently goes back to doing the work
    itself, and nothing else in the system would notice.
    """
    text = persona_text().lower()
    assert "mem context" in text, "persona does not tell the agent how to recall"
    assert "never through a dispatched agent" in text, "persona does not reserve memory"
    assert "dispatch a new agent" in text, "persona does not delegate execution"
    assert "mem ingest" in text, "persona does not route long documents to ingest"


def test_persona_warns_against_setting_a_root() -> None:
    """The one failure the persona exists to prevent, after the first.

    An unset `MEMORY_ROOT` looks like a missing step to an agent checking
    its environment, and the obvious-looking repair is to export a root
    read off some document -- which silently points writes at an empty tree
    while discovery was already working.
    """
    text = persona_text()
    assert "MEMORY_ROOT" in text, "persona does not mention the root at all"
    assert "--root" in text, "persona does not warn off the flag"


def test_skills_do_not_assume_a_project_checkout() -> None:
    """An installed skill must not tell the agent to run `uv run` in a repo."""
    for name in SKILL_NAMES:
        text = (skill_source(name) / "SKILL.md").read_text(encoding="utf-8")
        assert "uv run" not in text, f"{name} assumes a project checkout"


def test_skill_version_tolerates_garbage(tmp_path: Path) -> None:
    missing = tmp_path / "nope.md"
    assert skill_version(missing) is None
    no_fm = tmp_path / "plain.md"
    no_fm.write_text("# just a heading\n", encoding="utf-8")
    assert skill_version(no_fm) is None
    unstamped = tmp_path / "unstamped.md"
    unstamped.write_text("---\nname: x\n---\nbody\n", encoding="utf-8")
    assert skill_version(unstamped) is None


# ---------------------------------------------------------------- idempotency


def test_install_is_idempotent(tmp_path: Path) -> None:
    """The whole point: a provisioning script can run twice."""
    root = tmp_path / "root"
    agent = tmp_path / "agent"
    home = tmp_path / "home"
    home.mkdir()

    kw: dict[str, Any] = {
        "root": root,
        "agent_dir": agent,
        "skills": True,
        "mcp": True,
        "shell_env": True,
        "home": home,
        "shell": "/bin/zsh",
    }

    first = install(**kw)
    assert first["changed"] is True
    assert {s["step"] for s in first["steps"]} == {"root", "skills", "mcp", "shell_env"}
    assert all(s["changed"] for s in first["steps"])

    second = install(**kw)
    assert second["changed"] is False, second["steps"]
    for step in second["steps"]:
        assert step["changed"] is False, step

    # And a third run still leaves exactly one env block.
    install(**kw)
    profile = (home / ".zshrc").read_text(encoding="utf-8")
    assert profile.count(ENV_MARKER) == 1


def test_root_step_runs_without_opt_ins(tmp_path: Path) -> None:
    """The data root is the one thing that is not optional."""
    result = install(root=tmp_path / "r", home=tmp_path)
    assert [s["step"] for s in result["steps"]] == ["root"]
    assert Paths(tmp_path / "r").marker.exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    agent = tmp_path / "agent"
    home = tmp_path / "home"
    home.mkdir()

    result = install(
        root=root,
        agent_dir=agent,
        skills=True,
        mcp=True,
        shell_env=True,
        home=home,
        shell="/bin/bash",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["changed"] is True
    assert all("would" in s["detail"] for s in result["steps"]), result["steps"]

    assert not root.exists()
    assert not agent.exists()
    assert not (home / ".bashrc").exists()


# ---------------------------------------------------------------- skills


def test_stale_skill_is_refreshed(tmp_path: Path) -> None:
    """A copy from an older version must not stay in front of the model."""
    agent = tmp_path / "agent"
    dest = agent / "skills" / "memory-recall"
    dest.mkdir(parents=True)
    dest.joinpath("SKILL.md").write_text(
        "---\nname: memory-recall\nx-memory-version: 0.0.1\n---\nSTALE-BODY-MARKER\n",
        encoding="utf-8",
    )

    result = install(root=tmp_path / "root", agent_dir=agent, skills=True, home=tmp_path)
    step = next(s for s in result["steps"] if s["step"] == "skills")
    assert step["changed"] is True
    assert "0.0.1 -> " in step["detail"]
    assert skill_version(dest / "SKILL.md") == __version__
    assert "STALE-BODY-MARKER" not in (dest / "SKILL.md").read_text(encoding="utf-8")


def test_unstamped_skill_is_refreshed(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    dest = agent / "skills" / "memory-write"
    dest.mkdir(parents=True)
    dest.joinpath("SKILL.md").write_text("---\nname: memory-write\n---\nhand-edited\n", "utf-8")

    result = install(root=tmp_path / "root", agent_dir=agent, skills=True, home=tmp_path)
    step = next(s for s in result["steps"] if s["step"] == "skills")
    assert "unstamped ->" in step["detail"]


def test_force_rewrites_a_current_skill(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    kw: dict[str, Any] = {
        "root": tmp_path / "root",
        "agent_dir": agent,
        "skills": True,
        "home": tmp_path,
    }
    install(**kw)
    md = agent / "skills" / "memory-recall" / "SKILL.md"
    md.write_text(f"---\nx-memory-version: {__version__}\n---\nmangled\n", encoding="utf-8")

    result = install(**kw, force=True)
    assert next(s for s in result["steps"] if s["step"] == "skills")["changed"] is True
    assert "mangled" not in md.read_text(encoding="utf-8")


def test_installed_skill_is_a_real_file_not_a_link(tmp_path: Path) -> None:
    """The repo ships .claude/skills as a symlink; copies must be files."""
    agent = tmp_path / "agent"
    install(root=tmp_path / "root", agent_dir=agent, skills=True, home=tmp_path)
    for name in SKILL_NAMES:
        md = agent / "skills" / name / "SKILL.md"
        assert md.is_file() and not md.is_symlink()


# ---------------------------------------------------------------- mcp


def test_mcp_merge_preserves_unrelated_settings(tmp_path: Path) -> None:
    """settings.json belongs to the user; we add one key to it."""
    agent = tmp_path / "agent"
    agent.mkdir()
    settings = agent / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "theme": "dark",
                "mcpServers": {"other": {"command": "keep-me"}},
            }
        ),
        encoding="utf-8",
    )

    root = tmp_path / "root"
    install(root=root, agent_dir=agent, mcp=True, home=tmp_path)

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert data["mcpServers"]["other"] == {"command": "keep-me"}
    entry = data["mcpServers"][MCP_SERVER_NAME]
    assert entry["env"][ENV_VAR] == str(root.resolve())


def test_mcp_entry_carries_the_root_in_env(tmp_path: Path) -> None:
    """The server discovers its root from the environment, not the CWD."""
    entry = install_mod.mcp_entry(tmp_path / "r")
    assert entry["env"][ENV_VAR] == str(tmp_path / "r")
    assert entry["command"]


def test_mcp_accepts_an_equivalent_existing_entry(tmp_path: Path) -> None:
    """PATH differing between runs must not make the install never settle.

    `mem-mcp` on PATH yields the console script; absent, the interpreter
    fallback. Both launch the same server at the same root, so neither should
    be rewritten into the other.
    """
    agent = tmp_path / "agent"
    agent.mkdir()
    root = (tmp_path / "root").resolve()
    for entry in (
        {"command": "/somewhere/bin/mem-mcp", "env": {ENV_VAR: str(root)}},
        {
            "command": "/other/python",
            "args": ["-m", "memory.mcp_server"],
            "env": {ENV_VAR: str(root)},
        },
    ):
        (agent / "settings.json").write_text(
            json.dumps({"mcpServers": {MCP_SERVER_NAME: entry}}), encoding="utf-8"
        )
        result = install(root=root, agent_dir=agent, mcp=True, home=tmp_path)
        step = next(s for s in result["steps"] if s["step"] == "mcp")
        assert step["changed"] is False, entry
        kept = json.loads((agent / "settings.json").read_text(encoding="utf-8"))
        assert kept["mcpServers"][MCP_SERVER_NAME] == entry


def test_mcp_rewrites_an_entry_pointing_at_another_root(tmp_path: Path) -> None:
    """Equivalence is per-root: a stale root must still be corrected."""
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    MCP_SERVER_NAME: {
                        "command": "/somewhere/bin/mem-mcp",
                        "env": {ENV_VAR: "/an/old/root"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    root = (tmp_path / "root").resolve()
    result = install(root=root, agent_dir=agent, mcp=True, home=tmp_path)
    assert next(s for s in result["steps"] if s["step"] == "mcp")["changed"] is True
    data = json.loads((agent / "settings.json").read_text(encoding="utf-8"))
    assert data["mcpServers"][MCP_SERVER_NAME]["env"][ENV_VAR] == str(root)


def test_malformed_settings_is_an_error_not_an_overwrite(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    settings = agent / "settings.json"
    settings.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(InstallError, match="not valid JSON"):
        install(root=tmp_path / "root", agent_dir=agent, mcp=True, home=tmp_path)
    assert settings.read_text(encoding="utf-8") == "{ this is not json"


def test_non_object_mcp_servers_is_an_error(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text(json.dumps({"mcpServers": []}), encoding="utf-8")
    with pytest.raises(InstallError, match="not a JSON object"):
        install(root=tmp_path / "root", agent_dir=agent, mcp=True, home=tmp_path)


def test_empty_settings_file_is_fine(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text("", encoding="utf-8")
    install(root=tmp_path / "root", agent_dir=agent, mcp=True, home=tmp_path)
    data = json.loads((agent / "settings.json").read_text(encoding="utf-8"))
    assert MCP_SERVER_NAME in data["mcpServers"]


# ---------------------------------------------------------------- shell env


@pytest.mark.parametrize(
    ("shell", "profile", "expected"),
    [
        ("/bin/zsh", ".zshrc", 'export MEMORY_ROOT="'),
        ("/bin/bash", ".bashrc", 'export MEMORY_ROOT="'),
        ("/usr/bin/fish", ".config/fish/config.fish", "set -gx MEMORY_ROOT "),
        ("/bin/ksh", ".profile", 'export MEMORY_ROOT="'),
        (None, ".profile", 'export MEMORY_ROOT="'),
    ],
)
def test_shell_env_targets_the_right_profile(
    tmp_path: Path, shell: str | None, profile: str, expected: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    install(root=tmp_path / "root", shell_env=True, home=home, shell=shell)
    text = (home / profile).read_text(encoding="utf-8")
    assert expected in text
    assert ENV_MARKER in text


def test_shell_env_preserves_existing_profile_content(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    rc = home / ".zshrc"
    rc.write_text("alias ll='ls -la'\n", encoding="utf-8")

    install(root=tmp_path / "root", shell_env=True, home=home, shell="/bin/zsh")
    text = rc.read_text(encoding="utf-8")
    assert "alias ll='ls -la'" in text
    assert ENV_MARKER in text


def test_shell_env_updates_a_block_for_a_moved_root(tmp_path: Path) -> None:
    """Re-pointing the root rewrites the block instead of stacking one."""
    home = tmp_path / "home"
    home.mkdir()
    rc = home / ".zshrc"
    rc.write_text("# before\n", encoding="utf-8")

    install(root=tmp_path / "one", shell_env=True, home=home, shell="/bin/zsh")
    install(root=tmp_path / "two", shell_env=True, home=home, shell="/bin/zsh")

    text = rc.read_text(encoding="utf-8")
    assert text.count(ENV_MARKER) == 1
    assert str(tmp_path / "two") in text
    assert str(tmp_path / "one") not in text
    assert "# before" in text


def test_unterminated_env_block_is_an_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").write_text(f"{ENV_MARKER}\nhalf a block\n", encoding="utf-8")
    with pytest.raises(InstallError, match="no closing"):
        install(root=tmp_path / "root", shell_env=True, home=home, shell="/bin/zsh")


# ---------------------------------------------------------------- profiles, doctor


def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="unknown profile"):
        install(root=tmp_path / "root", profile="enormous", home=tmp_path)


def test_profile_reports_the_install_command(tmp_path: Path) -> None:
    minimal = install(root=tmp_path / "a", profile="minimal", home=tmp_path)
    assert minimal["extras_hint"] is None
    full = install(root=tmp_path / "b", profile="full", home=tmp_path)
    assert "all-media" in (full["extras_hint"] or "")


def test_default_install_writes_nothing_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The containment guarantee: a bare install stays in the working tree.

    An agent in a sandbox frequently cannot write to ``$HOME`` at all, and
    one that can should not find a corpus there as a side effect of a
    command about something else. Asserted against a fake home that is
    watched rather than trusted: a default that drifts back under ``$HOME``
    is the one regression this test exists to catch.
    """
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(work)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(install_mod.AGENT_DIR_ENV_VAR, raising=False)

    report = install(skills=True, persona=True)

    assert list(home.iterdir()) == [], "a default install wrote under $HOME"
    assert (work / "data").is_dir(), "no corpus in the working directory"
    assert (work / ".claude" / "skills").is_dir()
    assert (work / ".claude" / PERSONA_FILE).is_file()
    assert Path(report["root"]) == work.resolve()


def test_default_install_needs_no_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local root is discoverable, so nothing should ask for a variable.

    The warning this guards used to fire whenever ``$MEMORY_ROOT`` was
    unset, which taught the reader that the normal state was a missing step
    -- and invited them to set it to a documented default pointing at an
    empty tree.
    """
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.chdir(work)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(install_mod.AGENT_DIR_ENV_VAR, raising=False)

    report = install(skills=True, persona=True)

    assert report["warnings"] == [], report["warnings"]


def test_out_of_tree_root_still_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root the walk cannot reach genuinely does need a variable."""
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.chdir(work)
    monkeypatch.delenv(ENV_VAR, raising=False)

    report = install(root=elsewhere)

    assert any("not discoverable" in w for w in report["warnings"])


def test_portable_install_writes_both_conventions(tmp_path: Path) -> None:
    """Claude Code reads CLAUDE.md and .claude/skills; others read AGENTS.md
    and .agents/skills. Both get the same text, in full.

    AGENTS.md sits beside `.agents/`, not inside the Claude config
    directory: a runtime that reads AGENTS.md has no reason to look in
    `.claude/`, so a copy there would be read by nobody.
    """
    work = tmp_path / "work"
    agent = work / ".claude"
    agent.mkdir(parents=True)

    install_mod.install_skills(agent, dry_run=False, portable=True)
    install_mod.install_persona(agent, dry_run=False, portable=True)

    claude_md = agent / PERSONA_FILE
    agents_md = work / "AGENTS.md"
    assert claude_md.is_file() and agents_md.is_file()
    assert claude_md.read_text(encoding="utf-8") == agents_md.read_text(encoding="utf-8")
    assert (agent / "skills" / "memory-recall" / "SKILL.md").is_file()
    assert (work / ".agents" / "skills" / "memory-recall" / "SKILL.md").is_file()
    assert not (agent / "AGENTS.md").exists(), "AGENTS.md hidden inside .claude/"


def test_portable_repairs_one_destination_without_the_other(tmp_path: Path) -> None:
    """Each copy is judged on its own, or the second one silently rots."""
    work = tmp_path / "work"
    agent = work / ".claude"
    agent.mkdir(parents=True)
    install_mod.install_skills(agent, dry_run=False, portable=True)
    shutil.rmtree(work / ".agents")

    step = install_mod.install_skills(agent, dry_run=False, portable=True)

    assert step.changed, "a missing portable copy reported as current"
    assert (work / ".agents" / "skills" / "memory-write" / "SKILL.md").is_file()


def test_skill_content_change_within_a_release_is_detected(tmp_path: Path) -> None:
    """A stamp alone cannot see an edit that does not bump the version.

    This is not hypothetical: editing a bundled skill mid-release leaves
    every installed copy reporting itself current until the next bump.
    """
    agent = tmp_path / "agent"
    agent.mkdir()
    install_mod.install_skills(agent, dry_run=False)
    dest = agent / "skills" / "memory-recall" / "SKILL.md"
    dest.write_text(
        dest.read_text(encoding="utf-8").replace("# Recalling memory", "# Edited"),
        encoding="utf-8",
    )

    step = install_mod.install_skills(agent, dry_run=False)

    assert step.changed, "a hand-edited copy reported as current"
    assert "# Recalling memory" in dest.read_text(encoding="utf-8")


def test_persona_preserves_a_users_own_file(tmp_path: Path) -> None:
    """CLAUDE.md belongs to the user; only the marked block is ours."""
    agent = tmp_path / "agent"
    agent.mkdir()
    target = agent / PERSONA_FILE
    target.write_text("# My notes\n\nAlways run pytest -x.\n", encoding="utf-8")

    step = install_mod.install_persona(agent, dry_run=False)

    assert step.changed
    text = target.read_text(encoding="utf-8")
    assert "Always run pytest -x." in text, "the user's own instructions were lost"
    assert PERSONA_MARKER in text


def test_persona_is_idempotent(tmp_path: Path) -> None:
    """A re-run replaces its own block rather than stacking a second copy."""
    agent = tmp_path / "agent"
    agent.mkdir()

    first = install_mod.install_persona(agent, dry_run=False)
    second = install_mod.install_persona(agent, dry_run=False)

    assert first.changed and not second.changed
    text = (agent / PERSONA_FILE).read_text(encoding="utf-8")
    assert text.count(PERSONA_MARKER) == 1


def test_persona_refreshes_a_stale_block(tmp_path: Path) -> None:
    """An older stamp is what makes an upgrade detectable."""
    agent = tmp_path / "agent"
    agent.mkdir()
    target = agent / PERSONA_FILE
    install_mod.install_persona(agent, dry_run=False)
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            f"personal-ai-memory {__version__}", "personal-ai-memory 0.0.1"
        ),
        encoding="utf-8",
    )

    step = install_mod.install_persona(agent, dry_run=False)

    assert step.changed
    assert "0.0.1" in step.detail and __version__ in step.detail
    assert (agent / PERSONA_FILE).read_text(encoding="utf-8").count(PERSONA_MARKER) == 1


def test_persona_unclosed_marker_is_an_error(tmp_path: Path) -> None:
    """Half a block means a hand-edit went wrong; refuse rather than guess."""
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / PERSONA_FILE).write_text(f"{PERSONA_MARKER}\nstuff\n", encoding="utf-8")

    with pytest.raises(InstallError, match="no closing"):
        install_mod.install_persona(agent, dry_run=False)


def test_persona_inlines_skills_for_a_confined_install(tmp_path: Path) -> None:
    """Where no harness registers the skills, a pointer to them is not enough.

    The confined install has nothing that auto-loads the skills directory,
    so the block has to carry the instructions themselves -- minus the
    frontmatter, which is addressed to a registrar that is not there.
    """
    agent = tmp_path / "agent"
    agent.mkdir()

    install_mod.install_persona(agent, dry_run=False, inline_skills=True)

    text = (agent / PERSONA_FILE).read_text(encoding="utf-8")
    assert "# Recalling memory" in text
    assert "# Writing memory" in text
    assert "name: memory-recall" not in text, "skill frontmatter leaked into the block"


def test_persona_target_scopes_the_rules_to_one_project(tmp_path: Path) -> None:
    """A project-local CLAUDE.md is the point of --persona-target.

    The default lands in the agent dir, which is user-global; a repo root
    wants the file itself, and the rules then travel with the checkout
    rather than following the user into every other project.
    """
    agent = tmp_path / "agent"
    agent.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "CLAUDE.md"

    step = install_mod.install_persona(agent, dry_run=False, target=target)

    assert step.changed
    assert PERSONA_MARKER in target.read_text(encoding="utf-8")
    assert not (agent / PERSONA_FILE).exists(), "wrote to the agent dir as well"


def test_persona_dry_run_writes_nothing(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()

    step = install_mod.install_persona(agent, dry_run=True)

    assert step.changed
    assert not (agent / PERSONA_FILE).exists()


def test_persona_health_reports_missing_then_current(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()

    missing = persona_health(agent)
    assert missing["installed"] is False
    assert "hint" in missing

    install_mod.install_persona(agent, dry_run=False)
    healthy = persona_health(agent)
    assert healthy["installed"] is True
    assert healthy["current"] is True
    assert "hint" not in healthy


def test_skills_health_reports_missing_then_current(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    before = skills_health(agent)
    assert all(not s["installed"] for s in before["skills"].values())
    assert "hint" in before

    install(root=tmp_path / "root", agent_dir=agent, skills=True, home=tmp_path)
    after = skills_health(agent)
    assert all(s["current"] for s in after["skills"].values())
    assert "hint" not in after


def test_warns_when_memory_root_is_not_set(tmp_path: Path) -> None:
    """A caller who skipped --shell-env needs to know `mem` won't find this."""
    result = install(root=tmp_path / "root", home=tmp_path)
    assert any(ENV_VAR in w for w in result["warnings"])


# ---------------------------------------------------------------- cli surface


def test_cli_install_emits_one_json_document(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            "--root",
            str(tmp_path / "root"),
            "install",
            "--agent-dir",
            str(tmp_path / "agent"),
            "--skills",
            "--mcp",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["command"] == "install"
    assert payload["data"]["changed"] is True


def test_cli_all_flag_runs_every_step(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    result = runner.invoke(
        main,
        [
            "--root",
            str(tmp_path / "root"),
            "install",
            "--agent-dir",
            str(tmp_path / "agent"),
            "--all",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    steps = {s["step"] for s in json.loads(result.stdout)["data"]["steps"]}
    assert steps == {"root", "skills", "persona", "mcp", "shell_env"}


def test_cli_install_failure_is_a_classified_envelope(runner: CliRunner, tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "settings.json").write_text("not json", encoding="utf-8")

    result = runner.invoke(
        main,
        ["--root", str(tmp_path / "root"), "install", "--agent-dir", str(agent), "--mcp"],
        catch_exceptions=False,
    )
    assert result.exit_code == EXIT_VALIDATION
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "install_failed"


def test_cli_doctor_reports_agent_skills(runner: CliRunner, tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    result = runner.invoke(main, ["--root", str(root), "doctor"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "agent_skills" in json.loads(result.stdout)["data"]


# ---------------------------------------------------------------- mem-mcp shim


def test_mem_mcp_reports_the_missing_extra_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent reads the error; a traceback buries the line that fixes it.

    ``memory.mcp_server`` raises a hand-built ``ModuleNotFoundError`` whose
    ``name`` is None, so a shim dispatching on ``exc.name`` misses it and
    lets the traceback through. Pin the catch.
    """
    import builtins

    import mem_launcher

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kw: Any) -> Any:
        if name == "memory.mcp_server":
            raise ModuleNotFoundError(
                "the MCP server needs the optional 'mcp' extra: uv sync --extra mcp"
            )
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as caught:
        mem_launcher.run_mcp()

    message = str(caught.value)
    assert "mcp" in message
    assert "uv pip install" in message


# ---------------------------------------------------------------- confined installs


def test_agent_dir_env_var_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sandboxed install's skill location must survive into later commands.

    Without this, `mem doctor` looks under $HOME, finds nothing, and reports
    skills missing that are installed -- which sends an agent into a loop
    re-installing them.
    """
    agent = tmp_path / "workdir" / ".claude"
    monkeypatch.setenv(install_mod.AGENT_DIR_ENV_VAR, str(agent))

    result = install(root=tmp_path / "root", skills=True, home=tmp_path)
    assert result["agent_dir"] == str(agent)

    health = skills_health()
    assert health["dir"] == str(agent / "skills")
    assert all(s["current"] for s in health["skills"].values())
    assert "hint" not in health


def test_explicit_agent_dir_beats_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(install_mod.AGENT_DIR_ENV_VAR, str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    result = install(root=tmp_path / "root", agent_dir=explicit, skills=True, home=tmp_path)
    assert result["agent_dir"] == str(explicit)


def test_non_default_agent_dir_warns_to_export_it(tmp_path: Path) -> None:
    """The warning is what tells an agent how to make the location stick."""
    result = install(
        root=tmp_path / "root",
        agent_dir=tmp_path / "workdir" / ".claude",
        skills=True,
        home=tmp_path,
    )
    assert any(install_mod.AGENT_DIR_ENV_VAR in w for w in result["warnings"])


def test_workdir_only_install_writes_nothing_under_home(tmp_path: Path) -> None:
    """The sandboxed case: an agent confined to its working directory.

    Everything the install touches must land in the workdir -- no data root,
    no skills, no settings and no profile edit under $HOME.
    """
    home = tmp_path / "home"
    home.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    install(
        root=workdir / ".memory",
        agent_dir=workdir / ".claude",
        skills=True,
        home=home,
    )

    assert list(home.iterdir()) == [], "install wrote under $HOME"
    assert Paths(workdir / ".memory").marker.exists()
    for name in SKILL_NAMES:
        assert (workdir / ".claude" / "skills" / name / "SKILL.md").is_file()


def test_cli_doctor_accepts_an_agent_dir(runner: CliRunner, tmp_path: Path) -> None:
    root = tmp_path / "root"
    agent = tmp_path / "wd" / ".claude"
    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert (
        runner.invoke(
            main, ["--root", str(root), "install", "--skills", "--agent-dir", str(agent)]
        ).exit_code
        == 0
    )

    result = runner.invoke(
        main, ["--root", str(root), "doctor", "--agent-dir", str(agent)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    health = json.loads(result.stdout)["data"]["agent_skills"]
    assert health["dir"] == str(agent / "skills")
    assert all(s["current"] for s in health["skills"].values())
