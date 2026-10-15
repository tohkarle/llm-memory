"""MCP tool surface and dispatch, exercised through the API.

A live client is not needed: the tool definitions and handlers live in
``memory.mcp_tools``, so the contract an MCP client depends on is testable
without the optional ``mcp`` dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memory.api import MemoryAPI
from memory.mcp_tools import (
    HANDLERS,
    READ_ONLY_TOOLS,
    TOOLS,
    handle,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    api = MemoryAPI.initialize(tmp_path)
    api.close()
    return tmp_path


def call(root: Path, name: str, **args: Any) -> dict[str, Any]:
    return handle(name, args, root=root)


def seed(root: Path, **kw: Any) -> str:
    payload = call(
        root,
        "memory_create",
        type=kw.pop("type", "fact"),
        title=kw.pop("title", "A memory"),
        body=kw.pop("body", "Body text."),
        **kw,
    )
    return str(payload["memory"]["id"])


class TestToolSurface:
    def test_every_tool_has_a_handler(self) -> None:
        assert {spec["name"] for spec in TOOLS} == set(HANDLERS)

    def test_the_planned_tools_are_present(self) -> None:
        expected = {
            "memory_search",
            "memory_get_context",
            "memory_get",
            "memory_get_related",
            "memory_create",
            "memory_update",
            "memory_replace",
            "memory_merge",
            "memory_delete",
            "memory_store_asset",
            "memory_get_asset",
            "memory_replace_asset",
            "memory_delete_asset",
            "memory_rebuild_index",
            "memory_stats",
        }
        assert expected <= set(HANDLERS)

    def test_schemas_are_valid_json_schema_objects(self) -> None:
        for spec in TOOLS:
            schema = spec["schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            json.dumps(schema)  # must be serializable for the wire

    def test_required_fields_exist_in_properties(self) -> None:
        for spec in TOOLS:
            schema = spec["schema"]
            for field in schema.get("required", []):
                assert field in schema["properties"], f"{spec['name']}.{field}"

    def test_descriptions_guide_a_model(self) -> None:
        """Descriptions are the only instructions an MCP client receives."""
        for spec in TOOLS:
            assert len(spec["description"]) > 40, spec["name"]

    def test_confusable_tools_point_at_each_other(self) -> None:
        by_name = {spec["name"]: spec["description"] for spec in TOOLS}
        assert "memory_search" in by_name["memory_get_context"]
        assert "memory_get_context" in by_name["memory_search"]
        assert "memory_replace" in by_name["memory_update"]

    def test_search_description_documents_the_paraphrase_workaround(self) -> None:
        """Keyword-only recall is a real limit; the tool must say how to cope."""
        description = next(s["description"] for s in TOOLS if s["name"] == "memory_search")
        assert "fts:" in description

    def test_destructive_tools_are_marked(self) -> None:
        destructive = {s["name"] for s in TOOLS if s.get("destructive")}
        assert "memory_delete" in destructive
        assert "memory_delete_asset" in destructive
        assert "memory_replace_asset" in destructive

    def test_read_only_tools_do_not_mutate(self) -> None:
        for name in READ_ONLY_TOOLS:
            assert name in HANDLERS
        assert "memory_create" not in READ_ONLY_TOOLS
        assert "memory_delete" not in READ_ONLY_TOOLS

    def test_unknown_tool_raises(self, root: Path) -> None:
        with pytest.raises(ValueError, match="unknown tool"):
            handle("memory_nonsense", {}, root=root)


class TestReadTools:
    def test_stats(self, root: Path) -> None:
        seed(root)
        assert call(root, "memory_stats")["memories"] == 1

    def test_search(self, root: Path) -> None:
        seed(root, title="Prefers uv over pip")
        payload = call(root, "memory_search", query="uv")
        assert payload["total"] == 1
        assert payload["hits"][0]["title"] == "Prefers uv over pip"

    def test_search_hides_secret_by_default(self, root: Path) -> None:
        seed(root, title="Classified item", sensitivity="secret")
        assert call(root, "memory_search", query="classified")["total"] == 0
        assert call(root, "memory_search", query="classified", include_secret=True)["total"] == 1

    def test_get(self, root: Path) -> None:
        mid = seed(root, title="Readable")
        payload = call(root, "memory_get", memory_id=mid)
        assert payload["found"] is True
        assert payload["memory"]["title"] == "Readable"

    def test_get_missing_reports_not_found_without_raising(self, root: Path) -> None:
        """A missing memory is data, not an error, so the model can react."""
        payload = call(root, "memory_get", memory_id="00000000-0000-4000-8000-000000000000")
        assert payload["found"] is False
        assert payload["tombstone"] is None

    def test_get_deleted_includes_its_tombstone(self, root: Path) -> None:
        mid = seed(root, title="Gone")
        call(root, "memory_delete", memory_id=mid, permanent=True, confirm=True)
        payload = call(root, "memory_get", memory_id=mid)
        assert payload["found"] is False
        assert payload["tombstone"]["permanent"] is True

    def test_get_follow_resolves_to_the_tip(self, root: Path) -> None:
        mid = seed(root, title="Version one")
        call(root, "memory_replace", memory_id=mid, title="Version two", body="b")
        payload = call(root, "memory_get", memory_id=mid, follow=True)
        assert payload["memory"]["title"] == "Version two"

    def test_get_related(self, root: Path) -> None:
        target = seed(root, title="Target")
        anchor = seed(root, title="Anchor", relations=[f"causes:{target}"])
        payload = call(root, "memory_get_related", memory_id=anchor)
        assert payload["total"] == 1
        assert payload["related"][0]["memory_id"] == target

    def test_get_context_returns_markdown_and_blocks(self, root: Path) -> None:
        mid = seed(root, title="Context memory", body="useful detail", profile=True)
        payload = call(root, "memory_get_context", query="context", token_budget=800)
        assert payload["token_budget"] == 800
        assert payload["blocks"]
        assert f"<!-- id:{mid}" in payload["markdown"]

    def test_get_context_respects_budget(self, root: Path) -> None:
        for i in range(15):
            seed(root, title=f"Item {i}", body="a body with several words in it " * 5)
        payload = call(root, "memory_get_context", query="body", token_budget=300)
        assert payload["tokens_used"] <= 300


class TestWriteTools:
    def test_create_with_metadata(self, root: Path) -> None:
        payload = call(
            root,
            "memory_create",
            type="preference",
            title="Prefers uv",
            body="Uses uv.",
            tags=["tooling"],
            entities=["uv"],
            confidence=0.9,
            profile=True,
            sources=["conversation:conversations/a.md:L1-L10"],
        )
        memory = payload["memory"]
        assert memory["type"] == "preference"
        assert memory["tags"] == ["tooling"]
        assert memory["profile"] is True
        assert memory["sources"][0]["ref"] == "conversations/a.md"

    def test_create_parses_occurred_at(self, root: Path) -> None:
        payload = call(
            root, "memory_create", type="event", title="Happened", occurred_at="2026-10-15"
        )
        assert payload["memory"]["occurred_at"].startswith("2026-10-15")

    def test_create_ignores_malformed_source_specs(self, root: Path) -> None:
        payload = call(
            root, "memory_create", type="fact", title="Robust", sources=["nonsense", "manual:ok"]
        )
        refs = [s["ref"] for s in payload["memory"]["sources"]]
        assert refs == ["ok"]

    def test_update(self, root: Path) -> None:
        mid = seed(root, title="Before")
        payload = call(root, "memory_update", memory_id=mid, title="After")
        assert payload["memory"]["title"] == "After"

    def test_update_with_no_fields_raises(self, root: Path) -> None:
        mid = seed(root)
        with pytest.raises(ValueError, match="nothing to update"):
            call(root, "memory_update", memory_id=mid)

    def test_update_cannot_change_type(self, root: Path) -> None:
        """Changing what a memory *is* belongs to replace, which keeps history."""
        mid = seed(root)
        with pytest.raises(ValueError, match="replace"):
            handle("memory_update", {"memory_id": mid, "type": "event"}, root=root)

    def test_replace_supersedes(self, root: Path) -> None:
        mid = seed(root, title="Original")
        payload = call(root, "memory_replace", memory_id=mid, title="Revised", body="New.")
        assert payload["superseded"]["status"] == "superseded"
        assert payload["new"]["id"] != mid

    def test_merge_unions_sources(self, root: Path) -> None:
        a = seed(root, title="Dup A", body="body a", sources=["manual:source-a"])
        b = seed(root, title="Dup B", body="body b", sources=["manual:source-b"])
        payload = call(root, "memory_merge", memory_ids=[a, b], title="Consolidated")
        refs = {s["ref"] for s in payload["merged"]["sources"]}
        assert refs == {"source-a", "source-b"}
        assert len(payload["superseded"]) == 2

    def test_merge_into_keeps_an_id(self, root: Path) -> None:
        a = seed(root, title="Keeper", body="body a")
        b = seed(root, title="Folded", body="body b")
        payload = call(root, "memory_merge", memory_ids=[a, b], into=a)
        assert payload["merged"]["id"] == a

    def test_soft_delete_is_reversible_shaped(self, root: Path) -> None:
        mid = seed(root, title="Soft")
        payload = call(root, "memory_delete", memory_id=mid)
        assert payload["permanent"] is False
        assert call(root, "memory_get", memory_id=mid)["memory"]["status"] == "deleted"

    def test_permanent_delete_requires_confirmation(self, root: Path) -> None:
        """An irreversible action must not be one accidental argument away."""
        mid = seed(root, title="Doomed")
        with pytest.raises(ValueError, match="confirm=true"):
            call(root, "memory_delete", memory_id=mid, permanent=True)
        assert call(root, "memory_get", memory_id=mid)["found"] is True

        payload = call(root, "memory_delete", memory_id=mid, permanent=True, confirm=True)
        assert payload["permanent"] is True

    def test_tombstone_blocks_recreation(self, root: Path) -> None:
        mid = seed(root, title="Unique claim", body="unique body")
        call(root, "memory_delete", memory_id=mid, permanent=True, confirm=True)
        from memory.store import Tombstoned

        with pytest.raises(Tombstoned):
            seed(root, title="Unique claim", body="unique body")

    def test_rebuild_index(self, root: Path) -> None:
        seed(root, title="Survivor")
        for suffix in ("", "-wal", "-shm"):
            Path(str(root / "data" / "index" / "memory.sqlite") + suffix).unlink(missing_ok=True)
        payload = call(root, "memory_rebuild_index")
        assert payload["memories"] == 1
        assert payload["verified"] is True


class TestAssetTools:
    def test_store_and_get(self, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("asset content", encoding="utf-8")
        payload = call(root, "memory_store_asset", path=str(source), tags=["docs"])
        asset_id = payload["asset"]["id"]

        fetched = call(root, "memory_get_asset", asset_id=asset_id)
        assert fetched["found"] is True
        assert fetched["asset"]["exists"] is True
        assert fetched["asset"]["media_type"] == "text/plain"

    def test_get_missing_asset_is_data(self, root: Path) -> None:
        payload = call(root, "memory_get_asset", asset_id="00000000-0000-4000-8000-000000000000")
        assert payload["found"] is False

    def test_link_to_memory(self, root: Path, tmp_path: Path) -> None:
        mid = seed(root, title="Has attachment")
        source = tmp_path / "doc.txt"
        source.write_text("content", encoding="utf-8")
        payload = call(root, "memory_store_asset", path=str(source), link_to=[mid])
        asset_id = payload["asset"]["id"]
        assert asset_id in call(root, "memory_get", memory_id=mid)["memory"]["assets"]

    def test_replace_keeps_the_id(self, root: Path, tmp_path: Path) -> None:
        first = tmp_path / "v1.txt"
        first.write_text("one", encoding="utf-8")
        asset_id = call(root, "memory_store_asset", path=str(first))["asset"]["id"]

        second = tmp_path / "v2.txt"
        second.write_text("two", encoding="utf-8")
        payload = call(root, "memory_replace_asset", asset_id=asset_id, path=str(second))
        assert payload["asset"]["id"] == asset_id
        assert payload["asset"]["version"] == 2

    def test_delete_asset_requires_confirmation(self, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("content", encoding="utf-8")
        asset_id = call(root, "memory_store_asset", path=str(source))["asset"]["id"]
        with pytest.raises(ValueError, match="confirm=true"):
            call(root, "memory_delete_asset", asset_id=asset_id, permanent=True)


class TestIngestionTools:
    """The multimodal surface: extract, search inside media, drain the queue."""

    def test_ingest_extracts_and_indexes(self, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "notes.md"
        source.write_text(
            "Pricing tiers are free and pro.\n\nRollback reverts the release tag.\n",
            encoding="utf-8",
        )
        payload = call(root, "memory_ingest", path=str(source), tags=["doc"])
        assert payload["status"] == "complete"
        assert payload["segments"] >= 1
        assert payload["processor"] == "text"

    def test_search_segments_carries_provenance(self, root: Path, tmp_path: Path) -> None:
        source = tmp_path / "notes.md"
        source.write_text("Rollback reverts the release tag.", encoding="utf-8")
        ingested = call(root, "memory_ingest", path=str(source))
        found = call(root, "memory_search_segments", query="rollback")
        assert found["total"] == 1
        hit = found["hits"][0]
        assert hit["asset_id"] == ingested["asset_id"]
        assert hit["processor"] == "text"

    def test_segment_search_is_read_only(self) -> None:
        assert "memory_search_segments" in READ_ONLY_TOOLS

    def test_audio_is_queued_not_run_inline(self, root: Path, tmp_path: Path) -> None:
        """An MCP client must not be blocked by an unbounded transcription."""
        import struct

        data = b"\x00\x00" * 800
        source = tmp_path / "clip.wav"
        source.write_bytes(
            b"RIFF"
            + struct.pack("<I", 36 + len(data))
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
            + b"data"
            + struct.pack("<I", len(data))
            + data
        )
        payload = call(root, "memory_ingest", path=str(source))
        assert payload["status"] == "queued"
        assert payload["queued_task"]

        status = call(root, "memory_queue_run", status_only=True)
        assert status["ran"] is False
        assert status["stats"]["pending"] == 1

        ran = call(root, "memory_queue_run")
        assert ran["ran"] is True
        assert ran["processed"] == 1

    def test_inline_overrides_the_queue(self, root: Path, tmp_path: Path) -> None:
        import struct

        data = b"\x00\x00" * 400
        source = tmp_path / "c.wav"
        source.write_bytes(
            b"RIFF"
            + struct.pack("<I", 36 + len(data))
            + b"WAVE"
            + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
            + b"data"
            + struct.pack("<I", len(data))
            + data
        )
        payload = call(root, "memory_ingest", path=str(source), inline=True)
        assert payload["status"] != "queued"

    def test_ingest_a_directory(self, root: Path, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        for i in range(2):
            (folder / f"n{i}.md").write_text(f"Note {i} about staging.", encoding="utf-8")
        payload = call(root, "memory_ingest", path=str(folder), recursive=True)
        assert payload["count"] == 2

    def test_every_new_tool_has_a_handler_and_schema(self) -> None:
        for name in ("memory_ingest", "memory_search_segments", "memory_queue_run"):
            assert name in HANDLERS
            spec = next(t for t in TOOLS if t["name"] == name)
            assert spec["description"] and spec["schema"]["type"] == "object"


class TestServerShell:
    def test_builds_when_mcp_is_installed(self) -> None:
        pytest.importorskip("mcp")
        from memory.mcp_server import build_server

        assert build_server() is not None

    def test_advertises_every_tool_with_annotations(self) -> None:
        pytest.importorskip("mcp")
        from memory.mcp_server import tool_definitions

        tools = tool_definitions()
        assert {t.name for t in tools} == set(HANDLERS)
        for tool in tools:
            assert tool.description
            assert tool.input_schema["type"] == "object"
            assert tool.annotations is not None

    def test_destructive_tools_are_annotated(self) -> None:
        """A client surfaces these hints before letting a model act."""
        pytest.importorskip("mcp")
        from memory.mcp_server import tool_definitions

        by_name = {t.name: t for t in tool_definitions()}
        assert by_name["memory_delete"].annotations.destructive_hint is True
        assert by_name["memory_search"].annotations.read_only_hint is True
        assert by_name["memory_create"].annotations.read_only_hint is False

    def test_call_returns_an_envelope(self, root: Path, monkeypatch: Any) -> None:
        pytest.importorskip("mcp")
        monkeypatch.setenv("MEMORY_ROOT", str(root))
        from memory.mcp_server import call

        result = call("memory_create", {"type": "fact", "title": "Via MCP", "body": "b"})
        assert result.is_error is False
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is True
        assert payload["data"]["memory"]["title"] == "Via MCP"

    def test_failures_come_back_as_results_not_exceptions(
        self, root: Path, monkeypatch: Any
    ) -> None:
        """A model must be able to read the error and adapt, not just fail."""
        pytest.importorskip("mcp")
        monkeypatch.setenv("MEMORY_ROOT", str(root))
        from memory.mcp_server import call

        result = call("memory_get", {"memory_id": "not-a-uuid"})
        assert result.is_error is True
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False
        assert payload["error"]["message"]

    def test_unknown_tool_is_an_error_result(self, root: Path, monkeypatch: Any) -> None:
        pytest.importorskip("mcp")
        monkeypatch.setenv("MEMORY_ROOT", str(root))
        from memory.mcp_server import call

        result = call("memory_nonsense", {})
        assert result.is_error is True

    def test_import_error_is_actionable(self) -> None:
        """Without the extra, the message must name the fix."""
        try:
            import mcp  # noqa: F401
        except ModuleNotFoundError:
            with pytest.raises(ModuleNotFoundError, match="--extra mcp"):
                import memory.mcp_server  # noqa: F401
        else:
            pytest.skip("mcp is installed")

    def test_tools_import_without_mcp(self) -> None:
        """The dispatch layer must not depend on the optional package."""
        import importlib

        module = importlib.import_module("memory.mcp_tools")
        assert "mcp" not in getattr(module, "__dict__", {})
        assert module.TOOLS
