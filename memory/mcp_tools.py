"""Tool definitions and dispatch for the MCP server.

Kept separate from :mod:`memory.mcp_server` so the tool surface and its
handlers can be tested without the optional ``mcp`` dependency installed --
the server shell is the only part that needs it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .api import MemoryAPI
from .context import ContextResult
from .models import Relation, RelationType, SourceKind, SourceRef
from .search import SearchFilters

SERVER_NAME = "memory"

#: Tool descriptions are written for a model, not a human: each says when to
#: reach for it and, where confusable, what to use instead.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_get_context",
        "description": (
            "Assemble a token-budgeted pack of relevant memories as Markdown, "
            "ready to paste into a prompt. Use this when preparing to answer "
            "anything that depends on what you already know about the user. "
            "For a targeted lookup of specific memories, use memory_search instead."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you need context about."},
                "token_budget": {"type": "integer", "default": 2000},
                "expand_depth": {"type": "integer", "default": 1},
                "include_profile": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Find memories by keyword and/or metadata, ranked. Keyword matching "
            "is lexical, so if a query returns little, retry with the user's own "
            "vocabulary or prefix the query with 'fts:' to OR synonyms together "
            "(e.g. 'fts:uv OR pip OR packaging'). Superseded and deleted "
            "memories are hidden unless requested via statuses. To assemble "
            "prompt context under a token limit, use memory_get_context instead."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "types": {"type": "array", "items": {"type": "string"}},
                "tags_any": {"type": "array", "items": {"type": "string"}},
                "statuses": {"type": "array", "items": {"type": "string"}},
                "profile": {"type": "boolean"},
                "limit": {"type": "integer", "default": 20},
                "include_secret": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "memory_get",
        "description": (
            "Read one memory in full by id, optionally with its neighbours. Use "
            "after a search or context pack to see untruncated text."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "with_related": {"type": "boolean", "default": False},
                "follow": {
                    "type": "boolean",
                    "default": False,
                    "description": "Resolve a superseded memory to its current version.",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_get_related",
        "description": (
            "Walk the relation graph from one memory to read its history, "
            "causes, or contradictions."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "depth": {"type": "integer", "default": 1},
                "types": {"type": "array", "items": {"type": "string"}},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "default": "both"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_create",
        "description": (
            "Record a durable new memory. Pick the type deliberately: "
            "preference (a standing choice), fact (a stable truth), event "
            "(something that happened at a time), decision (a choice and its "
            "reasoning), procedure (how to do something). Set profile=true only "
            "for identity-level facts that should inform every conversation. "
            "Always cite a source when one exists. Search first to avoid "
            "creating a near-duplicate."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["fact", "preference", "event", "decision", "procedure"],
                },
                "title": {
                    "type": "string",
                    "description": "A standalone assertion, not a topic label.",
                },
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "salience": {"type": "number", "minimum": 0, "maximum": 1},
                "profile": {"type": "boolean", "default": False},
                "sensitivity": {
                    "type": "string",
                    "enum": ["normal", "private", "secret"],
                    "default": "normal",
                },
                "occurred_at": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Each as 'kind:ref[:locator]'.",
                },
                "force_recreate": {"type": "boolean", "default": False},
            },
            "required": ["type", "title"],
        },
    },
    {
        "name": "memory_update",
        "description": (
            "Patch a memory in place, keeping its id. For corrections that do "
            "not change meaning: a typo, a tag, a confidence adjustment. If the "
            "meaning changed, use memory_replace instead so the old version is "
            "preserved. Cannot change type."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "salience": {"type": "number"},
                "profile": {"type": "boolean"},
                "status": {"type": "string", "enum": ["active", "archived"]},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_replace",
        "description": (
            "Supersede a memory with a new one when the user's position has "
            "changed. The old version is kept as history and hidden from "
            "ordinary search."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["memory_id", "title"],
        },
    },
    {
        "name": "memory_merge",
        "description": (
            "Consolidate duplicate memories into one, unioning their sources so "
            "no provenance is lost. Pass 'into' to keep one id."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "into": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["memory_ids"],
        },
    },
    {
        "name": "memory_delete",
        "description": (
            "Soft-delete a memory, hiding it from search while keeping it "
            "recoverable. Permanent deletion is irreversible, blocks the same "
            "content from being recorded again, and requires confirm=true -- "
            "only do it when the user explicitly asks for removal."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "permanent": {"type": "boolean", "default": False},
                "reason": {"type": "string"},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["memory_id"],
        },
        "destructive": True,
    },
    {
        "name": "memory_store_asset",
        "description": (
            "Import a local file (document, image, audio) into managed storage "
            "and optionally attach it to memories. Takes a path, not bytes."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "media_type": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "link_to": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path"],
        },
    },
    {
        "name": "memory_get_asset",
        "description": "Read an asset's metadata and storage path. Does not return bytes.",
        "schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "with_derived": {"type": "boolean", "default": False},
            },
            "required": ["asset_id"],
        },
    },
    {
        "name": "memory_replace_asset",
        "description": "Point an asset at new bytes, keeping its id so references survive.",
        "schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "path": {"type": "string"},
                "discard_previous": {"type": "boolean", "default": False},
            },
            "required": ["asset_id", "path"],
        },
        "destructive": True,
    },
    {
        "name": "memory_delete_asset",
        "description": "Mark an asset deleted, or permanently remove its metadata.",
        "schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "permanent": {"type": "boolean", "default": False},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["asset_id"],
        },
        "destructive": True,
    },
    {
        "name": "memory_rebuild_index",
        "description": (
            "Rebuild the search index from the authoritative files. Safe at any "
            "time; use after restoring a backup or if search looks stale."
        ),
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_stats",
        "description": "Counts by type and status, plus storage sizes.",
        "schema": {"type": "object", "properties": {}},
    },
]


def _root() -> Path | None:
    value = os.environ.get("MEMORY_ROOT")
    return Path(value) if value else None


def _parse_sources(raw: list[str] | None) -> list[SourceRef]:
    out: list[SourceRef] = []
    for spec in raw or []:
        parts = spec.split(":", 2)
        if len(parts) < 2 or parts[0] not in set(SourceKind):
            continue
        out.append(
            SourceRef(
                kind=SourceKind(parts[0]),
                ref=parts[1],
                locator=parts[2] if len(parts) > 2 else None,
            )
        )
    return out


def _parse_relations(raw: list[str] | None) -> list[Relation]:
    out: list[Relation] = []
    for spec in raw or []:
        parts = spec.split(":", 2)
        if len(parts) < 2 or parts[0] not in set(RelationType):
            continue
        out.append(
            Relation(
                type=RelationType(parts[0]),
                target=parts[1],
                note=parts[2] if len(parts) > 2 else None,
            )
        )
    return out


# ---------------------------------------------------------------- handlers


def handle(name: str, args: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Dispatch one tool call. Shared by the server and its tests."""
    handler = HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"unknown tool: {name}")
    read_only = name in READ_ONLY_TOOLS
    with MemoryAPI(root if root is not None else _root(), read_only=read_only) as api:
        return handler(api, args)


def _ctx(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    result: ContextResult = api.get_context(
        str(args.get("query", "")),
        token_budget=int(args.get("token_budget", 2000)),
        expand_depth=int(args.get("expand_depth", 1)),
        include_profile=bool(args.get("include_profile", True)),
    )
    return result.to_json()


def _search(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    filters = SearchFilters(
        types=args.get("types"),
        statuses=args.get("statuses"),
        tags_any=args.get("tags_any"),
        profile=args.get("profile"),
        limit=int(args.get("limit", 20)),
        sensitivity_max="secret" if args.get("include_secret") else "private",
    )
    hits = api.search(args.get("query"), filters=filters)
    return {"total": len(hits), "hits": [h.to_json() for h in hits]}


def _get(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    from .models import is_uuid

    if not is_uuid(args["memory_id"]):
        raise ValueError(f"not a memory id: {args['memory_id']!r}")
    memory = api.get(args["memory_id"], follow=bool(args.get("follow")))
    if memory is None:
        tomb = api.tombstone(args["memory_id"])
        return {
            "found": False,
            "memory_id": args["memory_id"],
            "tombstone": tomb.to_json() if tomb else None,
        }
    payload: dict[str, Any] = {"found": True, "memory": memory.to_public()}
    if args.get("with_related"):
        payload["related"] = [r.to_json() for r in api.get_related(memory.id)]
    return payload


def _related(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    hits = api.get_related(
        args["memory_id"],
        depth=int(args.get("depth", 1)),
        types=args.get("types"),
        direction=args.get("direction", "both"),
    )
    return {"total": len(hits), "related": [h.to_json() for h in hits]}


def _create(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        key: args[key]
        for key in ("tags", "entities", "confidence", "salience", "profile", "sensitivity")
        if key in args
    }
    if args.get("occurred_at"):
        from .models import parse_ts

        fields["occurred_at"] = parse_ts(args["occurred_at"])
    if args.get("sources"):
        fields["sources"] = _parse_sources(args["sources"])
    if args.get("relations"):
        fields["relations"] = _parse_relations(args["relations"])

    memory = api.create(
        type=args["type"],
        title=args["title"],
        body=args.get("body", ""),
        force_recreate=bool(args.get("force_recreate")),
        **fields,
    )
    return {"memory": memory.to_public()}


UPDATABLE = (
    "title",
    "body",
    "tags",
    "entities",
    "confidence",
    "salience",
    "profile",
    "sensitivity",
    "status",
)


def _update(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    if "type" in args:
        raise ValueError(
            "update cannot change `type`; use memory_replace so the old version is preserved"
        )
    unknown = set(args) - {"memory_id", *UPDATABLE}
    if unknown:
        raise ValueError(f"not updatable: {', '.join(sorted(unknown))}")
    patch = {key: args[key] for key in UPDATABLE if key in args}
    if not patch:
        raise ValueError("nothing to update")
    memory = api.update(args["memory_id"], **patch)
    return {"memory": memory.to_public()}


def _replace(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    return api.replace(
        args["memory_id"],
        title=args["title"],
        body=args.get("body", ""),
        type=args.get("type"),
        reason=args.get("reason"),
    ).to_json()


def _merge(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    return api.merge(
        list(args["memory_ids"]),
        into=args.get("into"),
        title=args.get("title"),
        body=args.get("body"),
    ).to_json()


def _delete(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    permanent = bool(args.get("permanent"))
    if permanent and not args.get("confirm"):
        raise ValueError("permanent deletion is irreversible; pass confirm=true")
    return api.delete(args["memory_id"], permanent=permanent, reason=args.get("reason")).to_json()


def _store_asset(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    asset = api.store_asset(
        Path(args["path"]),
        media_type=args.get("media_type"),
        tags=args.get("tags"),
        link_to=args.get("link_to"),
    )
    return {"asset": asset.to_json()}


def _get_asset(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    payload = api.get_asset(args["asset_id"], with_derived=bool(args.get("with_derived")))
    if payload is None:
        return {"found": False, "asset_id": args["asset_id"]}
    return {"found": True, "asset": payload}


def _replace_asset(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    asset = api.replace_asset(
        args["asset_id"],
        Path(args["path"]),
        discard_previous=bool(args.get("discard_previous")),
    )
    return {"asset": asset.to_json()}


def _delete_asset(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    permanent = bool(args.get("permanent"))
    if permanent and not args.get("confirm"):
        raise ValueError("permanent deletion is irreversible; pass confirm=true")
    return api.delete_asset(args["asset_id"], permanent=permanent)


def _rebuild(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    return api.rebuild_index(verify_after=True)


def _stats(api: MemoryAPI, args: dict[str, Any]) -> dict[str, Any]:
    return api.stats()


HANDLERS: dict[str, Callable[[MemoryAPI, dict[str, Any]], dict[str, Any]]] = {
    "memory_get_context": _ctx,
    "memory_search": _search,
    "memory_get": _get,
    "memory_get_related": _related,
    "memory_create": _create,
    "memory_update": _update,
    "memory_replace": _replace,
    "memory_merge": _merge,
    "memory_delete": _delete,
    "memory_store_asset": _store_asset,
    "memory_get_asset": _get_asset,
    "memory_replace_asset": _replace_asset,
    "memory_delete_asset": _delete_asset,
    "memory_rebuild_index": _rebuild,
    "memory_stats": _stats,
}

READ_ONLY_TOOLS = frozenset(
    {
        "memory_get_context",
        "memory_search",
        "memory_get",
        "memory_get_related",
        "memory_get_asset",
        "memory_stats",
    }
)
