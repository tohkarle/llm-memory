"""The Phase 4 commands, held to the same CLI contract as everything else:
one JSON envelope on stdout, classified exit codes, and no write without
explicit confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from memory.cli import EXIT_VALIDATION, main
from memory.providers import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_TIMEOUT,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (ENV_PROVIDER, ENV_MODEL, ENV_BASE_URL, ENV_API_KEY, ENV_TIMEOUT):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def root(tmp_path: Path, runner: CliRunner) -> Path:
    result = runner.invoke(main, ["--root", str(tmp_path), "init"])
    assert result.exit_code == 0, result.output
    return tmp_path


def run(runner: CliRunner, root: Path, *args: str) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(main, ["--root", str(root), *args], catch_exceptions=False)
    # The contract: exactly one JSON document on stdout, always.
    return result.exit_code, json.loads(result.stdout)


def create(runner: CliRunner, root: Path, title: str, body: str, *extra: str) -> str:
    code, payload = run(
        runner,
        root,
        "create",
        "--type",
        "fact",
        "--title",
        title,
        "--body",
        body,
        *extra,
    )
    assert code == 0, payload
    return str(payload["data"]["memory"]["id"])


def conversation(root: Path, name: str = "chat.md") -> str:
    path = root / "data" / "conversations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("I always run tests before pushing. I prefer uv over pip.\n", encoding="utf-8")
    return f"conversations/{name}"


# ---------------------------------------------------------------- provider


def test_provider_show_reports_no_provider(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "provider", "show")
    assert code == 0
    assert payload["data"]["configured"] is False
    assert "note" in payload["data"], "the absence must be explained, not just reported"


def test_provider_set_writes_config_without_calling_anything(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "provider", "set", "--provider", "echo")
    assert code == 0
    assert payload["data"]["config"]["provider"] == "echo"
    assert (root / "data" / "config" / "llm.json").is_file()

    code, payload = run(runner, root, "provider", "show")
    assert payload["data"]["configured"] is True


def test_provider_set_rejects_an_incomplete_http_config(runner: CliRunner, root: Path) -> None:
    """Fail here, with a clear message, rather than on first use elsewhere."""
    code, payload = run(runner, root, "provider", "set", "--provider", "openai-compatible")
    assert code != 0
    assert payload["ok"] is False
    assert "base_url" in payload["error"]["message"]


def test_provider_set_never_echoes_the_key(runner: CliRunner, root: Path) -> None:
    code, payload = run(
        runner,
        root,
        "provider",
        "set",
        "--provider",
        "echo",
        "--api-key",
        "sk-secret-value",
    )
    assert code == 0
    assert "sk-secret-value" not in json.dumps(payload)


def test_provider_set_rejects_an_unknown_provider(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "provider", "set", "--provider", "some-vendor")
    assert code != 0
    assert payload["ok"] is False


# ---------------------------------------------------------------- consolidate


def test_consolidate_reports_duplicates(runner: CliRunner, root: Path) -> None:
    create(runner, root, "Prefers uv over pip", "Uses uv for every Python project.")
    create(
        runner,
        root,
        "Prefers uv over pip",
        "Uses uv for every Python project.",
        "--force-recreate",
    )
    code, payload = run(runner, root, "consolidate")
    assert code == 0
    assert payload["data"]["by_kind"].get("duplicate") == 1
    proposal = payload["data"]["proposals"][0]
    assert proposal["verb"] == "merge"
    assert proposal["arguments"]["ids"], "a proposal must carry runnable arguments"


def test_consolidate_writes_nothing_by_default(runner: CliRunner, root: Path) -> None:
    first = create(runner, root, "Prefers uv over pip", "Uses uv everywhere.")
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.", "--force-recreate")
    run(runner, root, "consolidate")
    code, payload = run(runner, root, "get", first)
    assert code == 0
    assert payload["data"]["memory"]["status"] == "active"


def test_consolidate_apply_requires_confirmation(runner: CliRunner, root: Path) -> None:
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.")
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.", "--force-recreate")
    code, payload = run(runner, root, "consolidate", "--apply")
    assert code == EXIT_VALIDATION
    assert payload["error"]["code"] == "confirmation_required"


def test_consolidate_apply_merges(runner: CliRunner, root: Path) -> None:
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.")
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.", "--force-recreate")
    code, payload = run(runner, root, "consolidate", "--apply", "--yes")
    assert code == 0
    assert payload["data"]["applied"]
    assert payload["data"]["applied"][0]["verb"] == "merge"


def test_consolidate_on_an_empty_root_is_not_an_error(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "consolidate")
    assert code == 0
    assert payload["data"]["total"] == 0


def test_consolidate_llm_without_a_provider_still_works(runner: CliRunner, root: Path) -> None:
    """--llm degrades to the lexical pass rather than failing the command."""
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.")
    create(runner, root, "Prefers uv over pip", "Uses uv everywhere.", "--force-recreate")
    code, payload = run(runner, root, "consolidate", "--llm")
    assert code == 0
    assert payload["data"]["provider"] is None
    assert payload["data"]["by_kind"].get("duplicate") == 1


def test_consolidate_filters_by_type(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "consolidate", "--type", "preference")
    assert code == 0
    assert payload["data"]["total"] == 0


# ---------------------------------------------------------------- extract


def test_extract_requires_a_source(runner: CliRunner, root: Path) -> None:
    result = runner.invoke(
        main, ["--root", str(root), "extract", "--text", "I prefer uv over pip."]
    )
    assert result.exit_code != 0
    assert "--source" in result.output


def test_extract_requires_exactly_one_input(runner: CliRunner, root: Path) -> None:
    code, payload = run(runner, root, "extract", "--source", "conversation:conversations/a.md")
    assert code == EXIT_VALIDATION
    assert "exactly one" in payload["error"]["message"]


def test_extract_from_text_reports_candidates(runner: CliRunner, root: Path) -> None:
    code, payload = run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing to main.",
        "--source",
        "conversation:conversations/a.md",
    )
    assert code == 0
    assert payload["data"]["extracted_by"] == "heuristic"
    assert payload["data"]["candidates"]
    assert payload["data"]["candidates"][0]["sources"], "every candidate cites its origin"


def test_extract_from_a_file(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    code, payload = run(
        runner,
        root,
        "extract",
        "--file",
        str(root / "data" / ref),
        "--source",
        f"conversation:{ref}",
    )
    assert code == 0
    assert payload["data"]["total"] >= 1


def test_extract_writes_nothing_without_commit(runner: CliRunner, root: Path) -> None:
    run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing.",
        "--source",
        "conversation:conversations/a.md",
    )
    code, payload = run(runner, root, "stats")
    assert code == 0
    assert payload["data"]["memories"] == 0


def test_extract_commit_creates_memories(runner: CliRunner, root: Path) -> None:
    code, payload = run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing to main.",
        "--source",
        "conversation:conversations/a.md",
        "--commit",
    )
    assert code == 0
    assert payload["data"]["created"]
    code, stats = run(runner, root, "stats")
    assert stats["data"]["memories"] == 1


def test_extract_commit_respects_min_confidence(runner: CliRunner, root: Path) -> None:
    """The heuristic's 0.5 is below this floor, so nothing should be created."""
    code, payload = run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing.",
        "--source",
        "conversation:conversations/a.md",
        "--commit",
        "--min-confidence",
        "0.9",
    )
    assert code == 0
    assert payload["data"]["created"] == []
    assert payload["data"]["skipped"]


def test_extract_commit_skips_an_existing_duplicate(runner: CliRunner, root: Path) -> None:
    run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing to main.",
        "--source",
        "conversation:conversations/a.md",
        "--commit",
    )
    code, payload = run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing to main.",
        "--source",
        "conversation:conversations/a.md",
        "--commit",
    )
    assert code == 0
    assert payload["data"]["created"] == []


def test_extract_with_echo_provider_end_to_end(runner: CliRunner, root: Path) -> None:
    """Verifies the --llm wiring without a live model, as a user would."""
    run(runner, root, "provider", "set", "--provider", "echo")
    code, payload = run(
        runner,
        root,
        "extract",
        "--text",
        "I always run tests before pushing to main.",
        "--source",
        "conversation:conversations/a.md",
        "--llm",
    )
    assert code == 0
    assert payload["data"]["provider"] == "echo"


# ---------------------------------------------------------------- source


def test_source_list_shows_citation_counts(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, payload = run(runner, root, "source", "list")
    assert code == 0
    assert payload["data"]["sources"][0]["memories"] == 1


def test_source_plan_delete_is_read_only(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, payload = run(runner, root, "source", "plan-delete", ref)
    assert code == 0
    assert payload["data"]["counts"]["memories"] == 1
    assert (root / "data" / ref).is_file()


def test_source_delete_without_yes_is_a_dry_run(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, payload = run(runner, root, "source", "delete", ref)
    assert code == 0
    assert payload["data"]["dry_run"] is True
    assert (root / "data" / ref).is_file(), "a dry run must not delete"


def test_source_delete_with_yes_removes_and_orphans(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    mid = create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, payload = run(runner, root, "source", "delete", ref, "--yes")
    assert code == 0
    assert payload["data"]["orphaned_memories"] == [mid]
    assert not (root / "data" / ref).exists()

    code, got = run(runner, root, "get", mid)
    assert code == 0
    assert got["data"]["memory"]["sources"] == []


def test_source_orphans_lists_the_review_queue(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    mid = create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    run(runner, root, "source", "delete", ref, "--yes")
    code, payload = run(runner, root, "source", "orphans")
    assert code == 0
    assert [m["id"] for m in payload["data"]["memories"]] == [mid]


def test_source_delete_can_purge_memories(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    mid = create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, payload = run(runner, root, "source", "delete", ref, "--memories", "purge", "--yes")
    assert code == 0
    assert payload["data"]["deleted_memories"] == [mid]
    code, got = run(runner, root, "get", mid)
    assert code != 0
    assert got["error"]["code"] == "not_found"


def test_source_delete_keep_raw_retains_the_original(runner: CliRunner, root: Path) -> None:
    ref = conversation(root)
    create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    code, _ = run(runner, root, "source", "delete", ref, "--keep-raw", "--yes")
    assert code == 0
    assert (root / "data" / ref).is_file()


# ---------------------------------------------------------------- backup


def test_backup_create_and_list(runner: CliRunner, root: Path) -> None:
    create(runner, root, "A claim", "Body.")
    code, payload = run(runner, root, "backup", "create", "--label", "demo")
    assert code == 0
    assert payload["data"]["memories"] == 1
    archive = payload["data"]["path"]

    code, listing = run(runner, root, "backup", "list")
    assert [b["path"] for b in listing["data"]["backups"]] == [archive]


def test_backup_show_reads_the_manifest(runner: CliRunner, root: Path) -> None:
    create(runner, root, "A claim", "Body.")
    _, created = run(runner, root, "backup", "create")
    code, payload = run(runner, root, "backup", "show", created["data"]["path"])
    assert code == 0
    assert payload["data"]["manifest"]["memories"] == 1


def test_backup_prune_is_a_dry_run_without_yes(runner: CliRunner, root: Path) -> None:
    create(runner, root, "A claim", "Body.")
    run(runner, root, "backup", "create")
    code, payload = run(runner, root, "backup", "prune", "--keep-count", "1")
    assert code == 0
    assert payload["data"]["dry_run"] is True


def test_backup_restore_rebuilds_the_index(runner: CliRunner, root: Path, tmp_path: Path) -> None:
    create(runner, root, "Prefers uv over pip", "Uses uv.")
    _, created = run(runner, root, "backup", "create")
    target = tmp_path / "restored"

    code, payload = run(
        runner, root, "backup", "restore", created["data"]["path"], "--to", str(target)
    )
    assert code == 0
    assert payload["data"]["rebuild"]["memories"] == 1
    assert payload["data"]["rebuild"]["verified"] is True

    code, found = run(runner, target, "search", "uv")
    assert code == 0
    assert found["data"]["total"] == 1


def test_backup_restore_refuses_a_populated_root(runner: CliRunner, root: Path) -> None:
    create(runner, root, "A claim", "Body.")
    _, created = run(runner, root, "backup", "create")
    code, payload = run(
        runner, root, "backup", "restore", created["data"]["path"], "--to", str(root)
    )
    assert code != 0
    assert payload["ok"] is False


# ---------------------------------------------------------------- export/import


def test_export_still_works_unchanged(runner: CliRunner, root: Path) -> None:
    """The pre-existing export contract must not have shifted underneath callers."""
    create(runner, root, "A claim", "Body.")
    code, payload = run(runner, root, "export")
    assert code == 0
    assert payload["data"]["count"] == 1
    assert payload["data"]["format"] == "jsonl"


def test_import_round_trips_into_a_fresh_root(
    runner: CliRunner, root: Path, tmp_path: Path
) -> None:
    create(runner, root, "Prefers uv over pip", "Uses uv for every project.")
    out = tmp_path / "export.jsonl"
    code, _ = run(runner, root, "export", "--out", str(out))
    assert code == 0

    target = tmp_path / "target"
    assert runner.invoke(main, ["--root", str(target), "init"]).exit_code == 0
    code, payload = run(runner, target, "import", str(out))
    assert code == 0
    assert payload["data"]["created_count"] == 1

    code, found = run(runner, target, "search", "uv")
    assert found["data"]["total"] == 1


def test_import_is_idempotent(runner: CliRunner, root: Path, tmp_path: Path) -> None:
    create(runner, root, "A claim", "Body.")
    out = tmp_path / "export.jsonl"
    run(runner, root, "export", "--out", str(out))
    code, payload = run(runner, root, "import", str(out))
    assert code == 0
    assert payload["data"]["created_count"] == 0
    assert payload["data"]["skipped_count"] == 1


def test_import_will_not_resurrect_a_purged_memory(
    runner: CliRunner, root: Path, tmp_path: Path
) -> None:
    ref = conversation(root)
    mid = create(runner, root, "A claim", "Body.", "--source", f"conversation:{ref}")
    out = tmp_path / "export.jsonl"
    run(runner, root, "export", "--out", str(out))
    run(runner, root, "delete", mid, "--permanent", "--yes")

    code, payload = run(runner, root, "import", str(out))
    assert code == 0
    assert payload["data"]["created_count"] == 0
    reasons = [entry["reason"] for entry in payload["data"]["skipped"]]
    assert "permanently deleted in this root" in reasons


def test_every_new_command_appears_in_help(runner: CliRunner) -> None:
    """A command nobody can discover is a command nobody will use."""
    output = runner.invoke(main, ["--help"]).output
    for name in ("consolidate", "extract", "provider", "source", "backup", "import"):
        assert name in output, name


def test_extract_file_accepts_a_root_relative_path(runner: CliRunner, root: Path) -> None:
    """`--file conversations/x.md` pairs with `--source conversation:conversations/x.md`.

    Click's exists= only resolves against the process cwd, so that natural
    pairing failed with a bare "does not exist" from the wrong directory.
    """
    ref = conversation(root)
    code, payload = run(runner, root, "extract", "--file", ref, "--source", f"conversation:{ref}")
    assert code == 0, payload
    assert payload["data"]["total"] >= 1


def test_extract_missing_file_is_a_classified_error(runner: CliRunner, root: Path) -> None:
    code, payload = run(
        runner, root, "extract", "--file", "nope.md", "--source", "conversation:x.md"
    )
    assert code == 1
    assert payload["error"]["code"] == "not_found"
    # The message must say where it looked, or the user cannot act on it.
    assert "looked in" in payload["error"]["message"]
