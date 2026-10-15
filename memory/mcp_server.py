"""Optional MCP server exposing the memory system over stdio.

A thin shell: the tool surface and handlers live in :mod:`memory.mcp_tools`,
which calls :class:`~memory.api.MemoryAPI`, so there is exactly one
implementation of each operation and the logic stays testable without this
module's optional dependency.

Targets the mcp 1.x API that ``pyproject.toml`` pins: tool handlers are
registered with the ``@server.list_tools()`` / ``@server.call_tool()``
decorators, and ``types.Tool`` takes ``inputSchema``.

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
except ImportError as exc:  # pragma: no cover - exercised by hand
    # ImportError, not ModuleNotFoundError: an *installed* package missing an
    # expected name raises the bare parent class, which would otherwise
    # escape as a traceback instead of this actionable message. The two cases
    # need different fixes, so they get different messages.
    if isinstance(exc, ModuleNotFoundError) and (exc.name or "").split(".")[0] == "mcp":
        raise ModuleNotFoundError(
            "the MCP server needs the optional 'mcp' extra: uv sync --extra mcp"
        ) from exc
    raise ImportError(
        f"the installed 'mcp' package is not compatible with this server ({exc}). "
        "This module targets the mcp 1.x API pinned in pyproject.toml; "
        "run `uv sync --extra dev --extra mcp` to install it. If the package "
        "looks installed but will not import, the virtualenv may be corrupt: "
        "`rm -rf .venv && uv sync --extra dev --extra mcp`."
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
                inputSchema=spec["schema"],
                annotations=types.ToolAnnotations(
                    readOnlyHint=spec["name"] in READ_ONLY,
                    destructiveHint=destructive,
                    idempotentHint=not destructive,
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
    # isError, not is_error: pydantic ignores an unknown kwarg here, so the
    # snake_case spelling silently reported every failure as a success.
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2, default=str))],
        isError=is_error,
    )


def build_server() -> Server[None, None]:
    server: Server[None, None] = Server(
        SERVER_NAME,
        version="0.1.0",
        instructions=(
            "Durable memory for this user: preferences, facts, events, decisions "
            "and procedures. Call memory_get_context before answering anything "
            "that depends on prior knowledge, and record durable statements with "
            "memory_create."
        ),
    )

    # mcp 1.x ships these decorators unannotated, so --strict flags the call
    # and the functions they wrap. Suppressed at the two call sites rather
    # than module-wide, so real typing errors here are still reported.
    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def on_list_tools() -> list[types.Tool]:
        return tool_definitions()

    # validate_input=False: `handle` already validates, and its ValueError
    # becomes the readable is_error envelope below. Letting the transport
    # reject first would surface a schema error the model cannot act on.
    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def on_call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        return call(name, arguments)

    return server


async def serve() -> None:  # pragma: no cover - requires a live client
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:  # pragma: no cover - requires a live client
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":  # pragma: no cover
    main()
