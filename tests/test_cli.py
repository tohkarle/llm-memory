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


class TestQueueCancelExitCode:
    """A typo'd task id must not read as a successful cancel.

    `reprocess` and `segment get` both answer an unknown id with not_found and
    exit 1; `queue cancel` answered ok:true with "cancelled": false, which is
    indistinguishable from success without parsing the payload.
    """

    def test_unknown_task_is_not_found(self, runner: CliRunner, root: Path) -> None:
        code, payload, _ = run(runner, root, "queue", "cancel", "nosuchtask")
        assert code == EXIT_NOT_FOUND
        assert payload["ok"] is False
        assert payload["error"]["code"] == "not_found"

    def test_cancelling_a_real_task_still_succeeds(
        self, runner: CliRunner, root: Path, tmp_path: Path
    ) -> None:
        from tests.test_ingestion import wav

        code, payload, _ = run(runner, root, "ingest", str(wav(tmp_path)))
        assert code == 0
        task_id = str(payload["data"]["queued_task"])

        code, payload, _ = run(runner, root, "queue", "cancel", task_id)
        assert code == 0
        assert payload["data"]["cancelled"] is True

        # A second cancel is no longer cancellable, and says so.
        code, payload, _ = run(runner, root, "queue", "cancel", task_id)
        assert code == EXIT_NOT_FOUND
        assert "cancelled" in payload["error"]["message"]


class TestTagsCommand:
    """`mem tags` -- the tag vocabulary, so drift is visible and reuse possible.

    Every fixture here is deliberately asymmetric: counts differ from each
    other, and count order disagrees with alphabetical order, so an
    implementation that sorted by name (or returned rows in grouping order)
    would fail rather than coincidentally agree.
    """

    @staticmethod
    def seed(runner: CliRunner, root: Path) -> None:
        # `zeta` is alphabetically last but the most used, and `alpha` is
        # alphabetically first but the least used -- so name order is the
        # reverse of count order and the two cannot be confused.
        for i in range(3):
            create(runner, root, title=f"Zeta {i}", tag="zeta")
        for i in range(2):
            create(runner, root, title=f"Mid {i}", tag="mid")
        create(runner, root, title="Alpha only", tag="alpha")

    def test_counts_are_per_memory(self, runner: CliRunner, root: Path) -> None:
        self.seed(runner, root)
        _, payload, _ = run(runner, root, "tags")
        counts = {t["tag"]: t["count"] for t in payload["data"]["tags"]}
        assert counts == {"zeta": 3, "mid": 2, "alpha": 1}
        assert payload["data"]["total"] == 3

    def test_ordered_by_count_not_name(self, runner: CliRunner, root: Path) -> None:
        self.seed(runner, root)
        _, payload, _ = run(runner, root, "tags")
        names = [t["tag"] for t in payload["data"]["tags"]]
        assert names == ["zeta", "mid", "alpha"]
        # The assertion above is only meaningful because the two orders differ.
        assert names != sorted(names)

    def test_equal_counts_break_ties_by_name(self, runner: CliRunner, root: Path) -> None:
        # Insert in reverse-alphabetical order: if the tiebreaker were absent,
        # insertion/grouping order would surface `beta` first.
        for tag in ("delta", "charlie", "beta"):
            create(runner, root, title=f"Tie {tag}", tag=tag)
        _, payload, _ = run(runner, root, "tags")
        names = [t["tag"] for t in payload["data"]["tags"]]
        assert names == ["beta", "charlie", "delta"]

    def test_drifted_spellings_are_distinct_rows(self, runner: CliRunner, root: Path) -> None:
        """The silent-subset bug, made visible: one topic, two tags."""
        for i in range(3):
            create(runner, root, title=f"Sync {i}", tag="synctasks")
        create(runner, root, title="Drifted", tag="sync-tasks")

        _, payload, _ = run(runner, root, "tags")
        rows = {t["tag"]: t["count"] for t in payload["data"]["tags"]}
        assert rows == {"synctasks": 3, "sync-tasks": 1}

        # And the consequence: filtering by either tag misses the other's memory.
        _, payload, _ = run(runner, root, "search", "--tag", "synctasks")
        assert payload["data"]["total"] == 3
        _, payload, _ = run(runner, root, "search", "--tag", "sync-tasks")
        assert payload["data"]["total"] == 1

    def test_near_misses_group_separator_drift(self, runner: CliRunner, root: Path) -> None:
        for i in range(3):
            create(runner, root, title=f"Sync {i}", tag="synctasks")
        create(runner, root, title="Hyphen", tag="sync-tasks")
        create(runner, root, title="Underscore", tag="sync_tasks")
        create(runner, root, title="Unrelated", tag="billing")

        _, payload, _ = run(runner, root, "tags", "--near-misses")
        groups = payload["data"]["near_misses"]
        assert len(groups) == 1
        assert [m["value"] for m in groups[0]["members"]] == [
            "sync-tasks",
            "sync_tasks",
            "synctasks",
        ]
        assert groups[0]["total"] == 5
        # `billing` shares no skeleton with them, so it must not be dragged in.
        assert "billing" not in str(groups)

    def test_near_misses_catch_multi_separator_drift(self, runner: CliRunner, root: Path) -> None:
        """Two separators apart -- beyond one edit, so only the skeleton rule finds it.

        `end-to-end` vs `endtoend` differ by two characters, which the
        edit-distance pass rejects. This is what proves separator-stripping is
        doing the work rather than riding on the distance check.
        """
        for i in range(2):
            create(runner, root, title=f"E2E {i}", tag="end-to-end")
        create(runner, root, title="Collapsed", tag="endtoend")

        _, payload, _ = run(runner, root, "tags", "--near-misses")
        groups = payload["data"]["near_misses"]
        assert len(groups) == 1
        assert [m["value"] for m in groups[0]["members"]] == ["end-to-end", "endtoend"]
        assert groups[0]["total"] == 3

    def test_near_misses_catch_a_one_character_typo(self, runner: CliRunner, root: Path) -> None:
        """Different skeletons, one edit apart -- only the distance check finds it."""
        for i in range(2):
            create(runner, root, title=f"Backend {i}", tag="backend")
        create(runner, root, title="Typo", tag="backen")

        _, payload, _ = run(runner, root, "tags", "--near-misses")
        groups = payload["data"]["near_misses"]
        assert len(groups) == 1
        assert [m["value"] for m in groups[0]["members"]] == ["backen", "backend"]

    def test_near_misses_absent_without_the_flag(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="A", tag="synctasks")
        create(runner, root, title="B", tag="sync-tasks")
        _, payload, _ = run(runner, root, "tags")
        assert "near_misses" not in payload["data"]

    def test_near_misses_empty_on_a_clean_vocabulary(self, runner: CliRunner, root: Path) -> None:
        for tag in ("billing", "frontend", "research"):
            create(runner, root, title=f"Clean {tag}", tag=tag)
        _, payload, _ = run(runner, root, "tags", "--near-misses")
        assert payload["data"]["near_misses"] == []

    def test_prefix_scopes_the_listing(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="One", tag="project-synctasks")
        create(runner, root, title="Two", tag="project-notion-todos")
        create(runner, root, title="Three", tag="backend")

        _, payload, _ = run(runner, root, "tags", "--prefix", "project-")
        names = sorted(t["tag"] for t in payload["data"]["tags"])
        assert names == ["project-notion-todos", "project-synctasks"]
        # `total` must honour the prefix too, not report the whole vocabulary.
        assert payload["data"]["total"] == 2

    def test_prefix_underscore_is_not_a_wildcard(self, runner: CliRunner, root: Path) -> None:
        """`_` is legal in a tag, so a LIKE-based prefix would over-match."""
        create(runner, root, title="Literal", tag="a_b")
        create(runner, root, title="Wildcard bait", tag="axb")

        _, payload, _ = run(runner, root, "tags", "--prefix", "a_")
        assert [t["tag"] for t in payload["data"]["tags"]] == ["a_b"]

    def test_min_count_drops_rare_tags(self, runner: CliRunner, root: Path) -> None:
        self.seed(runner, root)
        _, payload, _ = run(runner, root, "tags", "--min-count", "2")
        assert [t["tag"] for t in payload["data"]["tags"]] == ["zeta", "mid"]

    def test_limit_truncates_but_total_does_not(self, runner: CliRunner, root: Path) -> None:
        self.seed(runner, root)
        _, payload, _ = run(runner, root, "tags", "--limit", "2")
        assert payload["data"]["returned"] == 2
        assert payload["data"]["total"] == 3
        # Truncation keeps the most-used, not an arbitrary two.
        assert [t["tag"] for t in payload["data"]["tags"]] == ["zeta", "mid"]

    def test_deleted_memories_are_excluded_by_default(self, runner: CliRunner, root: Path) -> None:
        kept = create(runner, root, title="Kept", tag="kept-tag")
        gone = create(runner, root, title="Gone", tag="gone-tag")
        code, _, _ = run(runner, root, "delete", gone, "--reason", "test")
        assert code == 0

        _, payload, _ = run(runner, root, "tags")
        assert [t["tag"] for t in payload["data"]["tags"]] == ["kept-tag"]

        _, payload, _ = run(runner, root, "tags", "--include-inactive")
        assert sorted(t["tag"] for t in payload["data"]["tags"]) == ["gone-tag", "kept-tag"]
        assert kept  # both memories existed; only status separates them

    def test_entities_facet_lists_entities(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="One", tag="backend", entity="SQLite")
        create(runner, root, title="Two", tag="backend", entity="SQLite")
        create(runner, root, title="Three", tag="backend", entity="Click")

        _, payload, _ = run(runner, root, "tags", "--entities")
        assert payload["data"]["facet"] == "entity"
        assert [(e["entity"], e["count"]) for e in payload["data"]["entities"]] == [
            ("SQLite", 2),
            ("Click", 1),
        ]
        # The tag facet is unaffected -- the two listings are separate.
        _, payload, _ = run(runner, root, "tags")
        assert [t["tag"] for t in payload["data"]["tags"]] == ["backend"]

    def test_entities_keep_their_case(self, runner: CliRunner, root: Path) -> None:
        """Unlike tags, entities are names; a lowercasing listing would fail."""
        create(runner, root, title="Cased", entity="SQLite")
        _, payload, _ = run(runner, root, "tags", "--entities")
        assert [e["entity"] for e in payload["data"]["entities"]] == ["SQLite"]

    def test_empty_vocabulary_is_not_an_error(self, runner: CliRunner, root: Path) -> None:
        create(runner, root, title="Untagged")
        code, payload, _ = run(runner, root, "tags")
        assert code == 0
        assert payload["data"] == {
            "facet": "tag",
            "total": 0,
            "returned": 0,
            "tags": [],
        }

    @pytest.mark.parametrize(("flag", "value"), [("--limit", "0"), ("--min-count", "0")])
    def test_rejects_nonsense_bounds(
        self, runner: CliRunner, root: Path, flag: str, value: str
    ) -> None:
        code, payload, _ = run(runner, root, "tags", flag, value)
        assert code == EXIT_VALIDATION
        assert payload["error"]["code"] == "validation"

    # ---------------------------------------------------------- sensitivity

    @staticmethod
    def search_total(runner: CliRunner, root: Path, tag: str) -> int:
        """What a default `mem search --tag <tag>` actually returns."""
        code, payload, _ = run(runner, root, "search", "--tag", tag)
        assert code == 0, payload
        return int(payload["data"]["total"])

    @staticmethod
    def tags_count(runner: CliRunner, root: Path, tag: str, *args: str) -> int | None:
        """The count `mem tags` reports for one tag, or None if unlisted."""
        code, payload, _ = run(runner, root, "tags", *args)
        assert code == 0, payload
        for row in payload["data"]["tags"]:
            if row["tag"] == tag:
                return int(row["count"])
        return None

    def test_count_predicts_what_search_returns(self, runner: CliRunner, root: Path) -> None:
        """The listed count must equal the default search's hit count.

        The tag is carried at three different sensitivities, so the two sides
        can genuinely disagree: the count comes from `tags` and the
        expectation from a separate `search` invocation, not from one call.
        """
        create(runner, root, title="Normal", tag="projectsec", sensitivity="normal")
        create(runner, root, title="Private", tag="projectsec", sensitivity="private")
        create(runner, root, title="Secret", tag="projectsec", sensitivity="secret")

        assert self.search_total(runner, root, "projectsec") == 2
        assert self.tags_count(runner, root, "projectsec") == 2

        # And with the flag, both sides move together to include the secret.
        code, payload, _ = run(runner, root, "search", "--tag", "projectsec", "--include-secret")
        assert code == 0
        assert payload["data"]["total"] == 3
        assert self.tags_count(runner, root, "projectsec", "--include-secret") == 3

    def test_a_secret_only_tag_is_not_listed(self, runner: CliRunner, root: Path) -> None:
        """The slug is itself the disclosure, so it must not surface by default."""
        create(runner, root, title="Only secret", tag="onlysecret", sensitivity="secret")
        create(runner, root, title="Visible", tag="shown", sensitivity="normal")

        code, payload, _ = run(runner, root, "tags")
        assert code == 0
        assert [t["tag"] for t in payload["data"]["tags"]] == ["shown"]
        assert payload["data"]["total"] == 1
        # Nothing to find by it either -- the listing would have been a dead end.
        assert self.search_total(runner, root, "onlysecret") == 0

        # Opt in and it appears, which proves the tag existed all along.
        assert self.tags_count(runner, root, "onlysecret", "--include-secret") == 1

    def test_secret_entities_are_hidden_too(self, runner: CliRunner, root: Path) -> None:
        """The entity facet shares the listing's SQL, so it shares the hole."""
        create(runner, root, title="Secret", entity="AcmeCorp", sensitivity="secret")
        create(runner, root, title="Normal", entity="SQLite", sensitivity="normal")

        _, payload, _ = run(runner, root, "tags", "--entities")
        assert [e["entity"] for e in payload["data"]["entities"]] == ["SQLite"]

        _, payload, _ = run(runner, root, "tags", "--entities", "--include-secret")
        assert sorted(e["entity"] for e in payload["data"]["entities"]) == [
            "AcmeCorp",
            "SQLite",
        ]

    def test_include_inactive_still_hides_secret(self, runner: CliRunner, root: Path) -> None:
        """The two flags are independent; neither implies the other."""
        create(runner, root, title="Secret", tag="secrettag", sensitivity="secret")
        create(runner, root, title="Normal", tag="plaintag", sensitivity="normal")

        _, payload, _ = run(runner, root, "tags", "--include-inactive")
        assert [t["tag"] for t in payload["data"]["tags"]] == ["plaintag"]
