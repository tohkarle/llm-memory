"""The CLI contract: one JSON document on stdout, classified exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from memory.cli import (
    EXIT_CONFLICT,
    EXIT_NOT_FOUND,
    EXIT_VALIDATION,
    main,
    parse_relation,
    parse_source,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def root(tmp_path: Path, runner: CliRunner) -> Path:
    result = runner.invoke(main, ["--root", str(tmp_path), "init"])
    assert result.exit_code == 0, result.output
    return tmp_path


def run(runner: CliRunner, root: Path, *args: str) -> tuple[int, dict[str, Any], str]:
    """Invoke a command and parse its envelope.

    Asserts the contract on every call: stdout is exactly one JSON document.
    """
    result = runner.invoke(main, ["--root", str(root), *args], catch_exceptions=False)
    payload = json.loads(result.stdout)
    return result.exit_code, payload, result.stdout


def create(runner: CliRunner, root: Path, **kw: str) -> str:
    args = [
        "create",
        "--type",
        kw.pop("type", "fact"),
        "--title",
        kw.pop("title", "A memory"),
        "--body",
        kw.pop("body", "Body text."),
    ]
    for key, value in kw.items():
        args += [f"--{key.replace('_', '-')}", value]
    code, payload, _ = run(runner, root, *args)
    assert code == 0, payload
    return str(payload["data"]["memory"]["id"])


class TestEnvelope:
    def test_success_shape(self, runner: CliRunner, root: Path) -> None:
        code, payload, _ = run(runner, root, "stats")
        assert code == 0
        assert payload["ok"] is True
        assert payload["command"] == "stats"
        assert "data" in payload
        assert payload["warnings"] == []
        assert isinstance(payload["elapsed_ms"], int)

    def test_error_shape(self, runner: CliRunner, root: Path) -> None:
        code, payload, _ = run(runner, root, "get", "00000000-0000-4000-8000-000000000000")
        assert code == EXIT_NOT_FOUND
        assert payload["ok"] is False
        assert payload["error"]["code"] == "not_found"
        assert payload["error"]["message"]

    def test_stdout_is_exactly_one_json_document(self, runner: CliRunner, root: Path) -> None:
        """An agent must never have to find JSON inside other output."""
        create(runner, root, title="Findable memory")
        for args in (
            ["stats"],
            ["search", "findable"],
            ["doctor"],
            ["verify"],
            ["review"],
            ["export"],
            ["journal", "list"],
            ["asset", "list"],
            ["context", "findable"],
        ):
            result = runner.invoke(main, ["--root", str(root), *args])
            decoder = json.JSONDecoder()
            parsed, end = decoder.raw_decode(result.stdout.strip())
            assert parsed["ok"] is True, args
            assert result.stdout.strip()[end:].strip() == "", f"trailing output from {args}"

    def test_every_failure_still_emits_json(self, runner: CliRunner, root: Path) -> None:
        """Regression: --permanent without --yes exited 2 printing nothing."""
        mid = create(runner, root, title="Doomed")
        code, payload, out = run(runner, root, "delete", mid, "--permanent")
        assert code == EXIT_VALIDATION
        assert payload["ok"] is False
        assert payload["error"]["code"] == "confirmation_required"
        assert out.strip()


class TestExitCodes:
    def test_not_found(self, runner: CliRunner, root: Path) -> None:
        code, _, _ = run(runner, root, "get", "00000000-0000-4000-8000-000000000000")
        assert code == EXIT_NOT_FOUND

    def test_validation_on_empty_title(self, runner: CliRunner, root: Path) -> None:
        code, payload, _ = run(runner, root, "create", "--type", "fact", "--title", "   ")
        assert code == EXIT_VALIDATION
        assert payload["error"]["code"] == "validation"

    def test_validation_on_nothing_to_update(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root)
        code, payload, _ = run(runner, root, "update", mid)
        assert code == EXIT_VALIDATION
        assert "nothing to update" in payload["error"]["message"]

    def test_conflict_on_tombstoned_content(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Unique claim", body="unique body")
        run(runner, root, "delete", mid, "--permanent", "--yes")
        code, payload, _ = run(
            runner,
            root,
            "create",
            "--type",
            "fact",
            "--title",
            "Unique claim",
            "--body",
            "unique body",
        )
        assert code == EXIT_CONFLICT
        assert payload["error"]["code"] == "tombstoned"
        assert "force-recreate" in payload["error"]["details"]["hint"]

    def test_force_recreate_succeeds(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Unique claim", body="unique body")
        run(runner, root, "delete", mid, "--permanent", "--yes")
        code, _, _ = run(
            runner,
            root,
            "create",
            "--type",
            "fact",
            "--title",
            "Unique claim",
            "--body",
            "unique body",
            "--force-recreate",
        )
        assert code == 0

    def test_invalid_query_is_not_an_error(self, runner: CliRunner, root: Path) -> None:
        """Punctuation-only text matches nothing; it is not a failure."""
        code, payload, _ = run(runner, root, "search", "((( ***")
        assert code == 0
        assert payload["data"]["total"] == 0

    def test_usage_error_is_clicks_to_report(self, runner: CliRunner, root: Path) -> None:
        result = runner.invoke(main, ["--root", str(root), "create", "--type", "nonsense"])
        assert result.exit_code == 2


class TestLifecycle:
    def test_create_get_round_trip(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Prefers uv", body="Uses uv.", type="preference")
        code, payload, _ = run(runner, root, "get", mid)
        assert code == 0
        assert payload["data"]["memory"]["title"] == "Prefers uv"
        assert payload["data"]["memory"]["type"] == "preference"

    def test_get_raw_includes_file_text(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Raw test")
        _, payload, _ = run(runner, root, "get", mid, "--raw")
        assert payload["data"]["raw"].startswith("---")

    def test_update_reports_changed_fields(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Before")
        _, payload, _ = run(runner, root, "update", mid, "--title", "After")
        assert payload["data"]["memory"]["title"] == "After"
        assert "title" in payload["data"]["changed"]

    def test_update_tag_arithmetic(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, tag="keep")
        run(runner, root, "update", mid, "--add-tag", "added")
        _, payload, _ = run(runner, root, "get", mid)
        assert set(payload["data"]["memory"]["tags"]) == {"keep", "added"}

        run(runner, root, "update", mid, "--remove-tag", "keep")
        _, payload, _ = run(runner, root, "get", mid)
        assert payload["data"]["memory"]["tags"] == ["added"]

        run(runner, root, "update", mid, "--clear-tags")
        _, payload, _ = run(runner, root, "get", mid)
        assert payload["data"]["memory"]["tags"] == []

    def test_if_unchanged_guard(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Guarded")
        _, payload, _ = run(runner, root, "get", mid)
        stale = payload["data"]["memory"]["content_hash"]
        run(runner, root, "update", mid, "--title", "Changed by someone else")
        code, payload, _ = run(
            runner, root, "update", mid, "--title", "Mine", "--if-unchanged", stale
        )
        assert code == EXIT_CONFLICT

    def test_replace_supersedes(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Original")
        _, payload, _ = run(runner, root, "replace", mid, "--title", "Revised", "--body", "New.")
        assert payload["data"]["superseded"]["status"] == "superseded"
        assert payload["data"]["new"]["id"] != mid

    def test_merge_consolidates(self, runner: CliRunner, root: Path) -> None:
        a = create(runner, root, title="Dup A", body="body a")
        b = create(runner, root, title="Dup B", body="body b")
        _, payload, _ = run(runner, root, "merge", a, b, "--title", "Consolidated")
        assert len(payload["data"]["superseded"]) == 2
        assert payload["data"]["merged"]["title"] == "Consolidated"

    def test_merge_into_keeps_id(self, runner: CliRunner, root: Path) -> None:
        a = create(runner, root, title="Keeper", body="body a")
        b = create(runner, root, title="Folded", body="body b")
        _, payload, _ = run(runner, root, "merge", a, b, "--into", a)
        assert payload["data"]["merged"]["id"] == a

    def test_soft_delete_then_get(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Soft")
        _, payload, _ = run(runner, root, "delete", mid, "--reason", "testing")
        assert payload["data"]["permanent"] is False
        code, payload, _ = run(runner, root, "get", mid)
        assert code == 0
        assert payload["data"]["memory"]["status"] == "deleted"

    def test_permanent_delete_reports_tombstone(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Hard")
        _, payload, _ = run(runner, root, "delete", mid, "--permanent", "--yes")
        assert payload["data"]["permanent"] is True
        assert payload["data"]["tombstone"]["permanent"] is True

        code, payload, _ = run(runner, root, "get", mid)
        assert code == EXIT_NOT_FOUND
        assert payload["error"]["details"]["tombstone"]["permanent"] is True

    def test_related_traversal(self, runner: CliRunner, root: Path) -> None:
        target = create(runner, root, title="Target")
        anchor_code, anchor_payload, _ = run(
            runner,
            root,
            "create",
            "--type",
            "fact",
            "--title",
            "Anchor",
            "--body",
            "b",
            "--relation",
            f"causes:{target}",
        )
        assert anchor_code == 0
        anchor = anchor_payload["data"]["memory"]["id"]
        _, payload, _ = run(runner, root, "related", anchor)
        assert payload["data"]["total"] == 1
        assert payload["data"]["related"][0]["memory_id"] == target


class TestSearchCommand:
    def test_returns_hits(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Prefers uv over pip")
        _, payload, _ = run(runner, root, "search", "uv")
        assert payload["data"]["total"] == 1
        assert payload["data"]["hits"][0]["title"] == "Prefers uv over pip"

    def test_metadata_only_search(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Tagged", tag="findme")
        _, payload, _ = run(runner, root, "search", "--tag", "findme")
        assert payload["data"]["total"] == 1

    def test_secret_requires_flag(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Classified thing", sensitivity="secret")
        _, payload, _ = run(runner, root, "search", "classified")
        assert payload["data"]["total"] == 0
        _, payload, _ = run(runner, root, "search", "classified", "--include-secret")
        assert payload["data"]["total"] == 1

    def test_limit(self, runner: CliRunner, root: Path) -> None:
        for i in range(5):
            create(runner, root, title=f"Paged item {i}", body="shared body")
        _, payload, _ = run(runner, root, "search", "paged", "--limit", "2")
        assert payload["data"]["total"] == 2


class TestContextCommand:
    def test_json_format_is_enveloped(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Context memory", body="useful detail")
        _, payload, _ = run(runner, root, "context", "context", "--budget", "800")
        assert payload["ok"] is True
        assert payload["data"]["token_budget"] == 800
        assert payload["data"]["blocks"]

    def test_markdown_format_prints_the_pack(self, runner: CliRunner, root: Path) -> None:
        """The one command whose stdout is prose, because it is pasted into a prompt."""
        mid = create(runner, root, title="Pastable memory", body="useful detail")
        result = runner.invoke(
            main, ["--root", str(root), "context", "pastable", "--format", "markdown"]
        )
        assert result.exit_code == 0
        assert result.stdout.startswith("##")
        assert f"<!-- id:{mid}" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)

    def test_budget_is_respected(self, runner: CliRunner, root: Path) -> None:
        for i in range(20):
            create(runner, root, title=f"Budget item {i}", body="a reasonably long body " * 10)
        _, payload, _ = run(runner, root, "context", "budget", "--budget", "300")
        assert payload["data"]["tokens_used"] <= 300


class TestAssets:
    def test_add_get_list(self, runner: CliRunner, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("asset content", encoding="utf-8")

        _, payload, _ = run(runner, root, "asset", "add", str(source), "--tag", "docs")
        asset_id = payload["data"]["asset"]["id"]
        assert payload["data"]["asset"]["media_type"] == "text/plain"

        _, payload, _ = run(runner, root, "asset", "get", asset_id)
        assert payload["data"]["asset"]["exists"] is True

        _, payload, _ = run(runner, root, "asset", "list")
        assert payload["data"]["total"] == 1

    def test_link_to_memory(self, runner: CliRunner, root: Path, tmp_path: Path) -> None:
        mid = create(runner, root, title="Has an attachment")
        source = tmp_path / "doc.txt"
        source.write_text("content", encoding="utf-8")
        _, payload, _ = run(runner, root, "asset", "add", str(source), "--link-to", mid)
        asset_id = payload["data"]["asset"]["id"]

        _, payload, _ = run(runner, root, "get", mid)
        assert asset_id in payload["data"]["memory"]["assets"]

    def test_replace_keeps_id(self, runner: CliRunner, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "v1.txt"
        source.write_text("version one", encoding="utf-8")
        _, payload, _ = run(runner, root, "asset", "add", str(source))
        asset_id = payload["data"]["asset"]["id"]

        newer = tmp_path / "v2.txt"
        newer.write_text("version two", encoding="utf-8")
        _, payload, _ = run(runner, root, "asset", "replace", asset_id, str(newer))
        assert payload["data"]["asset"]["id"] == asset_id
        assert payload["data"]["asset"]["version"] == 2

    def test_permanent_delete_needs_confirmation(
        self, runner: CliRunner, root: Path, tmp_path: Path
    ) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("content", encoding="utf-8")
        _, payload, _ = run(runner, root, "asset", "add", str(source))
        asset_id = payload["data"]["asset"]["id"]
        code, payload, _ = run(runner, root, "asset", "delete", asset_id, "--permanent")
        assert code == EXIT_VALIDATION
        assert payload["error"]["code"] == "confirmation_required"


class TestMaintenance:
    def test_doctor_reports_a_healthy_index(self, runner: CliRunner, root: Path) -> None:
        """Regression: the FTS check needs a writable connection, and doctor
        opens read-only -- it used to always report a broken index."""
        create(runner, root, title="Something")
        _, payload, _ = run(runner, root, "doctor")
        assert payload["data"]["fts5"] is True
        assert payload["data"]["requires_rebuild"] is None
        assert payload["data"]["open_journal_ops"] == []

    def test_verify_is_clean(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Something")
        _, payload, _ = run(runner, root, "verify")
        assert payload["data"]["clean"] is True

    def test_rebuild_after_deleting_the_index(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Survivor memory")
        for suffix in ("", "-wal", "-shm"):
            Path(str(root / "data" / "index" / "memory.sqlite") + suffix).unlink(missing_ok=True)

        _, payload, _ = run(runner, root, "rebuild", "--verify")
        assert payload["data"]["memories"] == 1
        assert payload["data"]["verified"] is True

        _, payload, _ = run(runner, root, "search", "survivor")
        assert payload["data"]["total"] == 1

    def test_stats_counts(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, type="fact", title="One")
        create(runner, root, type="preference", title="Two")
        _, payload, _ = run(runner, root, "stats")
        assert payload["data"]["memories"] == 2
        assert payload["data"]["by_type"]["fact"] == 1

    def test_export_round_trip(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Exported")
        _, payload, _ = run(runner, root, "export")
        assert payload["data"]["count"] == 1
        assert json.loads(payload["data"]["lines"][0])["title"] == "Exported"

    def test_export_to_file(self, runner: CliRunner, root: Path, tmp_path: Path) -> None:
        create(runner, root, title="Exported")
        out = tmp_path / "dump.jsonl"
        _, payload, _ = run(runner, root, "export", "--out", str(out))
        assert out.exists()
        assert payload["data"]["path"] == str(out)

    def test_review_lists_due(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Due now", review_after="2020-01-01")
        create(runner, root, title="Later", review_after="2099-01-01")
        _, payload, _ = run(runner, root, "review")
        assert payload["data"]["total"] == 1
        assert payload["data"]["hits"][0]["title"] == "Due now"

    def test_journal_records_operations(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Journaled")
        _, payload, _ = run(runner, root, "journal", "list")
        assert payload["data"]["total"] > 0
        op_id = payload["data"]["records"][0]["op_id"]

        _, payload, _ = run(runner, root, "journal", "show", op_id)
        assert payload["data"]["records"]

    def test_journal_recover_is_clean(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Settled")
        _, payload, _ = run(runner, root, "journal", "recover")
        assert payload["data"]["settled"] == 0

    def test_init_is_idempotent(self, runner: CliRunner, root: Path) -> None:
        code, payload, _ = run(runner, root, "init")
        assert code == 0
        assert payload["data"]["created"] is False


class TestDryRun:
    def test_create_dry_run_writes_nothing(self, runner: CliRunner, root: Path) -> None:
        _, payload, _ = run(
            runner, root, "create", "--type", "fact", "--title", "Phantom", "--dry-run"
        )
        assert payload["data"]["dry_run"] is True
        _, payload, _ = run(runner, root, "search", "phantom")
        assert payload["data"]["total"] == 0

    def test_delete_dry_run_keeps_the_memory(self, runner: CliRunner, root: Path) -> None:
        mid = create(runner, root, title="Spared")
        _, payload, _ = run(runner, root, "delete", mid, "--permanent", "--dry-run")
        assert payload["data"]["dry_run"] is True
        assert payload["data"]["would_remove"]
        code, _, _ = run(runner, root, "get", mid)
        assert code == 0


class TestParsers:
    def test_source_spec(self) -> None:
        src = parse_source("conversation:conversations/a.md:L1-L20")
        assert src.kind.value == "conversation"
        assert src.ref == "conversations/a.md"
        assert src.locator == "L1-L20"

    def test_source_without_locator(self) -> None:
        assert parse_source("manual:note").locator is None

    def test_source_rejects_unknown_kind(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="unknown source kind"):
            parse_source("telepathy:x")

    def test_source_rejects_missing_ref(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="KIND:REF"):
            parse_source("manual")

    def test_relation_spec(self) -> None:
        rel = parse_relation("contradicts:11111111-2222-3333-4444-555555555555:disputed")
        assert rel.type.value == "contradicts"
        assert rel.note == "disputed"

    def test_relation_rejects_unknown_type(self) -> None:
        import click

        with pytest.raises(click.BadParameter, match="unknown relation type"):
            parse_relation("telepathy:11111111-2222-3333-4444-555555555555")


class TestBodyInput:
    def test_body_file(self, runner: CliRunner, root: Path, tmp_path: Path) -> None:
        body = tmp_path / "body.md"
        body.write_text("Body from a file.", encoding="utf-8")
        code, payload, _ = run(
            runner,
            root,
            "create",
            "--type",
            "fact",
            "--title",
            "From file",
            "--body-file",
            str(body),
        )
        assert code == 0
        assert payload["data"]["memory"]["body"] == "Body from a file."

    def test_stdin(self, runner: CliRunner, root: Path) -> None:
        result = runner.invoke(
            main,
            ["--root", str(root), "create", "--type", "fact", "--title", "From stdin", "--stdin"],
            input="Body from stdin.",
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["data"]["memory"]["body"] == "Body from stdin."

    def test_mutually_exclusive_body_sources(
        self, runner: CliRunner, root: Path, tmp_path: Path
    ) -> None:
        body = tmp_path / "body.md"
        body.write_text("x", encoding="utf-8")
        result = runner.invoke(
            main,
            [
                "--root",
                str(root),
                "create",
                "--type",
                "fact",
                "--title",
                "T",
                "--body",
                "inline",
                "--body-file",
                str(body),
            ],
        )
        assert result.exit_code != 0
