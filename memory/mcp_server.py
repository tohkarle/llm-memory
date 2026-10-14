"""Optional MCP server exposing the memory system over stdio.

A thin shell: the tool surface and handlers live in :mod:`memory.mcp_tools`,
which calls :class:`~memory.api.MemoryAPI`, so there is exactly one
implementation of each operation and the logic stays testable without this
module's optional dependency.

Targets the mcp 2.x API, where tool handlers are constructor callbacks
(``on_list_tools`` / ``on_call_tool``) rather than decorators.

The ``mcp`` package is an optional extra (``uv sync --extra mcp``).
Run with ``python -m memory.mcp_server``.
"""

from __future__ import annotations

import json
from typing import Any

from .mcp_tools import SERVER_NAME, TOOLS, handle

try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by hand
    raise ModuleNotFoundError(
        "the MCP server needs the optional 'mcp' extra: uv sync --extra mcp"
    ) from exc


def tool_definitions() -> list[types.Tool]:
    """Advertise the tool surface, carrying destructive-operation hints."""
    out: list[types.Tool] = []
    for spec in TOOLS:
        destructive = bool(spec.get("destructive"))
        out.append(
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                input_schema=spec["schema"],
                annotations=types.ToolAnnotations(
                    read_only_hint=spec["name"] in READ_ONLY,
                    destructive_hint=destructive,
                    idempotent_hint=not destructive,
                ),
            )
        )
    return out


READ_ONLY = frozenset(
    {
        "memory_get_context",
        "memory_search",
        "memory_get",
        "memory_get_related",
        "memory_get_asset",
        "memory_stats",
    }
)


def call(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
    """Run one tool, returning the same envelope shape the CLI emits.

    Failures come back as a result with ``is_error`` set rather than as a
    protocol error, so the calling model can read what went wrong and adapt.
    """
    try:
        payload: dict[str, Any] = {"ok": True, "data": handle(name, arguments or {})}
        is_error = False
    except Exception as exc:
        payload = {
            "ok": False,
            "error": {"code": type(exc).__name__, "message": str(exc)},
        }
        is_error = True
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2, default=str))],
        is_error=is_error,
    )


def build_server() -> Server[None]:
    async def on_list_tools(_ctx: Any, _params: Any = None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_definitions())

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return call(params.name, dict(params.arguments or {}))

    return Server(
        SERVER_NAME,
        version="0.1.0",
        instructions=(
            "Durable memory for this user: preferences, facts, events, decisions "
            "and procedures. Call memory_get_context before answering anything "
            "that depends on prior knowledge, and record durable statements with "
            "memory_create."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve() -> None:  # pragma: no cover - requires a live client
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:  # pragma: no cover - requires a live client
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":  # pragma: no cover
    main()
