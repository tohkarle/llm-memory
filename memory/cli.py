"""The ``mem`` command line. Primary interface for agents.

Contract, so a calling agent never has to parse prose:

* stdout carries exactly one JSON document per invocation;
* logs, warnings and progress go to stderr;
* the envelope is ``{"ok": true, "command": ..., "data": {...}}`` or
  ``{"ok": false, "command": ..., "error": {"code", "message", "details"}}``;
* the exit code classifies the failure.

The one exception is ``mem context --format markdown``, which prints the
pack itself because it is meant to be pasted into a prompt.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from . import __version__
from .api import MemoryAPI
from .journal import LockTimeout
from .models import (
    MemoryType,
    Relation,
    RelationType,
    SourceKind,
    SourceRef,
    ValidationError,
    parse_ts,
)
from .paths import Paths, RootNotFound
from .search import SearchError, SearchFilters
from .store import Conflict, NotFound, Tombstoned

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_VALIDATION = 2
EXIT_CONFLICT = 3
EXIT_LOCKED = 4
EXIT_CORRUPT = 5
EXIT_INTERNAL = 70


class CliError(click.ClickException):
    """An error already classified into an exit code and machine-readable code."""

    def __init__(self, code: str, message: str, exit_code: int, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details
        # ClickException declares exit_code as a class attribute; assigning
        # through type() keeps mypy --strict happy without shadowing it.
        self.__dict__["exit_code"] = exit_code

    @property
    def status(self) -> int:
        return int(self.__dict__["exit_code"])


def emit(command: str, data: Any, *, warnings: list[str] | None = None, started: float) -> None:
    payload = {
        "ok": True,
        "command": command,
        "data": data,
        "warnings": warnings or [],
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
    click.echo(json.dumps(payload, indent=2, default=str))


def fail(command: str, err: CliError) -> None:
    payload = {
        "ok": False,
        "command": command,
        "error": {"code": err.code, "message": err.message, "details": err.details},
    }
    click.echo(json.dumps(payload, indent=2, default=str))
    sys.exit(err.status)


def classify(exc: Exception) -> CliError:
    """Map an engine exception onto a stable code and exit status."""
    if isinstance(exc, Tombstoned):
        return CliError(
            "tombstoned",
            str(exc),
            EXIT_CONFLICT,
            memory_id=exc.tombstone.memory_id,
            matched_on=exc.matched_on,
            hint="pass --force-recreate to override",
        )
    if isinstance(exc, Conflict):
        return CliError("conflict", str(exc), EXIT_CONFLICT)
    if isinstance(exc, NotFound):
        return CliError("not_found", str(exc), EXIT_NOT_FOUND)
    if isinstance(exc, ValidationError):
        return CliError("validation", str(exc), EXIT_VALIDATION, problems=exc.problems)
    if isinstance(exc, SearchError):
        return CliError("invalid_query", str(exc), EXIT_VALIDATION)
    if isinstance(exc, LockTimeout):
        return CliError(
            "locked",
            str(exc),
            EXIT_LOCKED,
            hint="another process is writing; retry shortly",
        )
    if isinstance(exc, RootNotFound):
        return CliError("no_root", str(exc), EXIT_NOT_FOUND, hint="run `mem init`")
    if isinstance(exc, ValueError):
        return CliError("validation", str(exc), EXIT_VALIDATION)
    return CliError("internal", f"{type(exc).__name__}: {exc}", EXIT_INTERNAL)


def run(command: str, fn: Callable[[], Any]) -> None:
    """Execute one command, emitting exactly one JSON document."""
    started = time.monotonic()
    try:
        data = fn()
    except CliError as err:
        # Already classified. Envelope it rather than letting click print a
        # bare message: stdout must carry exactly one JSON document.
        fail(command, err)
        return
    except click.ClickException:
        # Usage errors (bad flags, missing arguments) are click's to report,
        # and happen before a command runs.
        raise
    except Exception as exc:
        fail(command, classify(exc))
        return
    emit(command, data, started=started)


def open_api(ctx: click.Context, *, read_only: bool = False) -> MemoryAPI:
    root = ctx.obj.get("root")
    api = MemoryAPI(root, read_only=read_only, lock_timeout=ctx.obj["lock_timeout"])
    api.open()
    recovery = api.recovery
    if recovery and not recovery.clean:
        click.echo(
            f"recovered {recovery.settled} interrupted operation(s)",
            err=True,
        )
    return api


# ---------------------------------------------------------------- parsing


def parse_source(raw: str) -> SourceRef:
    """``KIND:REF[:LOCATOR]`` -- e.g. ``conversation:conversations/a.md:L1-L20``."""
    parts = raw.split(":", 2)
    if len(parts) < 2:
        raise click.BadParameter(f"expected KIND:REF[:LOCATOR], got {raw!r}")
    kind, ref = parts[0], parts[1]
    if kind not in set(SourceKind):
        raise click.BadParameter(
            f"unknown source kind {kind!r}; one of {', '.join(sorted(set(SourceKind)))}"
        )
    return SourceRef(
        kind=SourceKind(kind),
        ref=ref,
        locator=parts[2] if len(parts) > 2 else None,
    )


def parse_relation(raw: str) -> Relation:
    """``TYPE:ID[:NOTE]`` -- e.g. ``contradicts:<uuid>:disputed by the user``."""
    parts = raw.split(":", 2)
    if len(parts) < 2:
        raise click.BadParameter(f"expected TYPE:ID[:NOTE], got {raw!r}")
    rtype, target = parts[0], parts[1]
    if rtype not in set(RelationType):
        raise click.BadParameter(
            f"unknown relation type {rtype!r}; one of {', '.join(sorted(set(RelationType)))}"
        )
    return Relation(
        type=RelationType(rtype),
        target=target,
        note=parts[2] if len(parts) > 2 else None,
    )


def parse_params(raw: Sequence[str]) -> dict[str, Any]:
    """``KEY=VALUE`` pairs into a processor params dict.

    Values are JSON-decoded when they parse, so ``--param ocr=false`` and
    ``--param target_chars=800`` arrive as a bool and an int rather than
    strings a processor would have to re-coerce.
    """
    out: dict[str, Any] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep:
            raise CliError("validation", f"expected KEY=VALUE, got {item!r}", EXIT_VALIDATION)
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def parse_when(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return parse_ts(raw)


def read_body(body: str | None, body_file: Path | None, use_stdin: bool) -> str:
    sources = [body is not None, body_file is not None, use_stdin]
    if sum(sources) > 1:
        raise click.BadParameter("pass only one of --body, --body-file, --stdin")
    if body is not None:
        return body
    if body_file is not None:
        return body_file.read_text(encoding="utf-8")
    if use_stdin:
        return sys.stdin.read()
    return ""


# ---------------------------------------------------------------- shared options

TYPE_CHOICE = click.Choice([t.value for t in MemoryType])
SENSITIVITY_CHOICE = click.Choice(["normal", "private", "secret"])


def content_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Metadata flags shared by ``create`` and ``replace``."""
    decorators = [
        click.option("--tag", "tags", multiple=True, help="Repeatable."),
        click.option("--entity", "entities", multiple=True, help="Repeatable."),
        click.option("--confidence", type=click.FloatRange(0, 1)),
        click.option("--salience", type=click.FloatRange(0, 1)),
        click.option("--profile/--no-profile", default=None, help="Always consider for context."),
        click.option("--sensitivity", type=SENSITIVITY_CHOICE),
        click.option("--occurred-at"),
        click.option("--valid-from"),
        click.option("--valid-until"),
        click.option("--review-after"),
        click.option("--source", "sources", multiple=True, metavar="KIND:REF[:LOCATOR]"),
        click.option("--relation", "relations", multiple=True, metavar="TYPE:ID[:NOTE]"),
        click.option("--asset", "assets", multiple=True, help="Asset id. Repeatable."),
    ]
    for decorator in reversed(decorators):
        fn = decorator(fn)
    return fn


def collect_fields(
    *,
    tags: Sequence[str],
    entities: Sequence[str],
    confidence: float | None,
    salience: float | None,
    profile: bool | None,
    sensitivity: str | None,
    occurred_at: str | None,
    valid_from: str | None,
    valid_until: str | None,
    review_after: str | None,
    sources: Sequence[str],
    relations: Sequence[str],
    assets: Sequence[str],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if tags:
        fields["tags"] = list(tags)
    if entities:
        fields["entities"] = list(entities)
    if confidence is not None:
        fields["confidence"] = confidence
    if salience is not None:
        fields["salience"] = salience
    if profile is not None:
        fields["profile"] = profile
    if sensitivity is not None:
        fields["sensitivity"] = sensitivity
    if occurred_at:
        fields["occurred_at"] = parse_when(occurred_at)
    if valid_from:
        fields["valid_from"] = parse_when(valid_from)
    if valid_until:
        fields["valid_until"] = parse_when(valid_until)
    if review_after:
        fields["review_after"] = parse_when(review_after)
    if sources:
        fields["sources"] = [parse_source(s) for s in sources]
    if relations:
        fields["relations"] = [parse_relation(r) for r in relations]
    if assets:
        fields["assets"] = list(assets)
    return fields


# ---------------------------------------------------------------- group


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", type=click.Path(path_type=Path), envvar="MEMORY_ROOT")
@click.option("--lock-timeout", type=float, default=10.0, show_default=True)
@click.version_option(__version__, prog_name="mem")
@click.pass_context
def main(ctx: click.Context, root: Path | None, lock_timeout: float) -> None:
    """Local-first memory for LLM agents. Emits JSON on stdout."""
    ctx.ensure_object(dict)
    ctx.obj["root"] = root
    ctx.obj["lock_timeout"] = lock_timeout


@main.command()
@click.option("--force", is_flag=True, help="Reinitialize an existing root.")
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """Create the data tree and index."""

    def action() -> dict[str, Any]:
        root = ctx.obj.get("root") or Path.cwd()
        paths = Paths(root)
        existed = paths.marker.exists()
        if existed and not force:
            api = MemoryAPI(root)
            api.open()
            try:
                return {"root": str(paths.root), "created": False, "stats": api.stats()}
            finally:
                api.close()
        api = MemoryAPI.initialize(root)
        try:
            return {"root": str(paths.root), "created": True, "stats": api.stats()}
        finally:
            api.close()

    run("init", action)


@main.command()
@click.option("--type", "mem_type", type=TYPE_CHOICE, required=True)
@click.option("--title", required=True)
@click.option("--body")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--stdin", "use_stdin", is_flag=True, help="Read the body from stdin.")
@click.option("--force-recreate", is_flag=True, help="Override a tombstone.")
@click.option("--dry-run", is_flag=True)
@content_options
@click.pass_context
def create(
    ctx: click.Context,
    /,
    mem_type: str,
    title: str,
    body: str | None,
    body_file: Path | None,
    use_stdin: bool,
    force_recreate: bool,
    dry_run: bool,
    **meta: Any,
) -> None:
    """Record a new memory."""

    def action() -> dict[str, Any]:
        text = read_body(body, body_file, use_stdin)
        fields = collect_fields(**meta)
        if dry_run:
            return {"dry_run": True, "type": mem_type, "title": title, "fields": list(fields)}
        api = open_api(ctx)
        try:
            memory = api.create(
                type=mem_type,
                title=title,
                body=text,
                force_recreate=force_recreate,
                **fields,
            )
            return {
                "memory": memory.to_public(),
                "path": api.paths.rel(api.paths.memory_file(memory.type.value, memory.id)),
            }
        finally:
            api.close()

    run("memory.create", action)


@main.command()
@click.argument("memory_id")
@click.option("--related", is_flag=True, help="Include neighbouring memories.")
@click.option("--follow", is_flag=True, help="Resolve to the live version.")
@click.option("--raw", is_flag=True, help="Include the file text verbatim.")
@click.pass_context
def get(ctx: click.Context, memory_id: str, related: bool, follow: bool, raw: bool) -> None:
    """Read one memory by id."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            memory = api.get(memory_id, follow=follow)
            if memory is None:
                tomb = api.tombstone(memory_id)
                raise CliError(
                    "not_found",
                    f"no memory {memory_id}"
                    + (" (permanently deleted)" if tomb and tomb.permanent else ""),
                    EXIT_NOT_FOUND,
                    tombstone=tomb.to_json() if tomb else None,
                )
            payload: dict[str, Any] = {"memory": memory.to_public()}
            if related:
                payload["related"] = [r.to_json() for r in api.get_related(memory.id)]
            if raw:
                path = api.paths.memory_file(memory.type.value, memory.id)
                payload["raw"] = path.read_text(encoding="utf-8")
            return payload
        finally:
            api.close()

    run("memory.get", action)


@main.command()
@click.argument("memory_id")
@click.option("--title")
@click.option("--body")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--stdin", "use_stdin", is_flag=True)
@click.option("--status", type=click.Choice(["active", "archived"]))
@click.option("--add-tag", multiple=True)
@click.option("--remove-tag", multiple=True)
@click.option("--clear-tags", is_flag=True)
@click.option("--if-unchanged", help="Content hash the memory must still have.")
@content_options
@click.pass_context
def update(
    ctx: click.Context,
    /,
    memory_id: str,
    title: str | None,
    body: str | None,
    body_file: Path | None,
    use_stdin: bool,
    status: str | None,
    add_tag: tuple[str, ...],
    remove_tag: tuple[str, ...],
    clear_tags: bool,
    if_unchanged: str | None,
    **meta: Any,
) -> None:
    """Patch an existing memory, keeping its id.

    Cannot change `type`: that changes what the memory is, which is `replace`.
    """

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            current = api.require(memory_id)
            patch = collect_fields(**meta)
            if title is not None:
                patch["title"] = title
            text = read_body(body, body_file, use_stdin)
            if body is not None or body_file is not None or use_stdin:
                patch["body"] = text
            if status is not None:
                patch["status"] = status

            if clear_tags:
                patch["tags"] = []
            elif add_tag or remove_tag:
                tags = set(patch.get("tags", current.tags))
                tags |= set(add_tag)
                tags -= set(remove_tag)
                patch["tags"] = sorted(tags)

            if not patch:
                raise CliError("validation", "nothing to update", EXIT_VALIDATION)

            memory = api.update(memory_id, expect_hash=if_unchanged, **patch)
            changed = [
                key for key in patch if getattr(memory, key, None) != getattr(current, key, None)
            ]
            return {"memory": memory.to_public(), "changed": changed}
        finally:
            api.close()

    run("memory.update", action)


@main.command()
@click.argument("memory_id")
@click.option("--title", required=True)
@click.option("--body")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--stdin", "use_stdin", is_flag=True)
@click.option("--type", "mem_type", type=TYPE_CHOICE)
@click.option("--reason")
@content_options
@click.pass_context
def replace(
    ctx: click.Context,
    /,
    memory_id: str,
    title: str,
    body: str | None,
    body_file: Path | None,
    use_stdin: bool,
    mem_type: str | None,
    reason: str | None,
    **meta: Any,
) -> None:
    """Supersede a memory, keeping the old version as history."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            result = api.replace(
                memory_id,
                title=title,
                body=read_body(body, body_file, use_stdin),
                type=mem_type,
                reason=reason,
                **collect_fields(**meta),
            )
            return result.to_json()
        finally:
            api.close()

    run("memory.replace", action)


@main.command()
@click.argument("memory_ids", nargs=-1, required=True)
@click.option("--into", help="Fold into this id instead of minting a new one.")
@click.option("--title")
@click.option("--body-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["union", "prefer-newest", "prefer-highest-confidence"]),
    default="union",
    show_default=True,
)
@click.pass_context
def merge(
    ctx: click.Context,
    memory_ids: tuple[str, ...],
    into: str | None,
    title: str | None,
    body_file: Path | None,
    strategy: str,
) -> None:
    """Consolidate duplicate memories."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            result = api.merge(
                list(memory_ids),
                into=into,
                title=title,
                body=body_file.read_text(encoding="utf-8") if body_file else None,
                strategy=strategy,
            )
            return result.to_json()
        finally:
            api.close()

    run("memory.merge", action)


@main.command()
@click.argument("memory_id")
@click.option("--permanent", is_flag=True, help="Unlink the file; keep a tombstone.")
@click.option("--reason")
@click.option("--yes", is_flag=True, help="Required for --permanent.")
@click.option("--dry-run", is_flag=True)
@click.pass_context
def delete(
    ctx: click.Context,
    memory_id: str,
    permanent: bool,
    reason: str | None,
    yes: bool,
    dry_run: bool,
) -> None:
    """Soft-delete a memory, or permanently remove it."""

    def action() -> dict[str, Any]:
        if permanent and not yes and not dry_run:
            raise CliError(
                "confirmation_required",
                "permanent deletion is irreversible; pass --yes to confirm",
                EXIT_VALIDATION,
            )
        api = open_api(ctx, read_only=dry_run)
        try:
            memory = api.require(memory_id)
            if dry_run:
                return {
                    "dry_run": True,
                    "memory_id": memory.id,
                    "permanent": permanent,
                    "would_remove": (
                        [api.paths.rel(api.paths.memory_file(memory.type.value, memory.id))]
                        if permanent
                        else []
                    ),
                }
            return api.delete(memory_id, permanent=permanent, reason=reason).to_json()
        finally:
            api.close()

    run("memory.delete", action)


@main.command()
@click.argument("query", required=False)
@click.option("--type", "types", multiple=True, type=TYPE_CHOICE)
@click.option("--tag", "tags_any", multiple=True)
@click.option("--all-tags", "tags_all", multiple=True)
@click.option("--entity", "entities_any", multiple=True)
@click.option("--status", "statuses", multiple=True)
@click.option("--profile/--no-profile", "profile", default=None)
@click.option("--min-confidence", type=click.FloatRange(0, 1))
@click.option("--created-after")
@click.option("--created-before")
@click.option("--occurred-after")
@click.option("--occurred-before")
@click.option("--source", "source_ref")
@click.option("--related-to")
@click.option("--include-secret", is_flag=True)
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--offset", type=int, default=0)
@click.option(
    "--order",
    type=click.Choice(["score", "updated", "created", "occurred"]),
    default="score",
    show_default=True,
)
@click.pass_context
def search(ctx: click.Context, /, query: str | None, include_secret: bool, **opts: Any) -> None:
    """Find memories by keyword and/or metadata."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            filters = SearchFilters(
                types=list(opts["types"]) or None,
                statuses=list(opts["statuses"]) or None,
                tags_any=list(opts["tags_any"]) or None,
                tags_all=list(opts["tags_all"]) or None,
                entities_any=list(opts["entities_any"]) or None,
                profile=opts["profile"],
                min_confidence=opts["min_confidence"],
                created_after=parse_when(opts["created_after"]),
                created_before=parse_when(opts["created_before"]),
                occurred_after=parse_when(opts["occurred_after"]),
                occurred_before=parse_when(opts["occurred_before"]),
                source_ref=opts["source_ref"],
                related_to=opts["related_to"],
                sensitivity_max="secret" if include_secret else "private",
                limit=opts["limit"],
                offset=opts["offset"],
                order=opts["order"],
            )
            hits = api.search(query, filters=filters)
            return {
                "query": query,
                "total": len(hits),
                "hits": [h.to_json() for h in hits],
            }
        finally:
            api.close()

    run("memory.search", action)


@main.command()
@click.argument("memory_id")
@click.option("--depth", type=int, default=1, show_default=True)
@click.option("--type", "types", multiple=True)
@click.option(
    "--direction",
    type=click.Choice(["out", "in", "both"]),
    default="both",
    show_default=True,
)
@click.option("--limit", type=int, default=50, show_default=True)
@click.pass_context
def related(
    ctx: click.Context,
    memory_id: str,
    depth: int,
    types: tuple[str, ...],
    direction: str,
    limit: int,
) -> None:
    """Walk the relation graph from one memory."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            api.require(memory_id)
            hits = api.get_related(
                memory_id,
                depth=depth,
                types=list(types) or None,
                direction=direction,
                limit=limit,
            )
            return {
                "memory_id": memory_id,
                "total": len(hits),
                "related": [h.to_json() for h in hits],
            }
        finally:
            api.close()

    run("memory.related", action)


@main.command()
@click.argument("query", required=False, default="")
@click.option("--budget", type=int, default=2000, show_default=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
)
@click.option("--depth", type=int, default=1, show_default=True)
@click.option("--profile-share", type=click.FloatRange(0, 1), default=0.25, show_default=True)
@click.option("--no-profile", is_flag=True)
@click.option("--type", "types", multiple=True, type=TYPE_CHOICE)
@click.option("--include-secret", is_flag=True)
@click.pass_context
def context(
    ctx: click.Context,
    query: str,
    budget: int,
    output_format: str,
    depth: int,
    profile_share: float,
    no_profile: bool,
    types: tuple[str, ...],
    include_secret: bool,
) -> None:
    """Assemble a token-budgeted context pack for a prompt."""
    started = time.monotonic()
    api: MemoryAPI | None = None
    try:
        api = open_api(ctx, read_only=True)
        result = api.get_context(
            query,
            token_budget=budget,
            filters=SearchFilters(
                types=list(types) or None,
                sensitivity_max="secret" if include_secret else "private",
            ),
            expand_depth=depth,
            profile_share=profile_share,
            include_profile=not no_profile,
        )
    except CliError as err:
        fail("memory.context", err)
        return
    except click.ClickException:
        raise
    except Exception as exc:
        fail("memory.context", classify(exc))
        return
    finally:
        if api is not None:
            api.close()

    if output_format == "markdown":
        # The single exception to JSON-only stdout: this output is meant to
        # be pasted straight into a prompt.
        click.echo(result.markdown, nl=False)
        if result.dropped:
            click.echo(
                f"note: {result.dropped} memory(ies) did not fit; raise --budget "
                "or narrow the query",
                err=True,
            )
        return
    emit("memory.context", result.to_json(), started=started)


# ---------------------------------------------------------------- assets


@main.group()
def asset() -> None:
    """Manage stored media and documents."""


@asset.command("add")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--media-type")
@click.option("--tag", "tags", multiple=True)
@click.option("--source", "source_spec", metavar="KIND:REF[:LOCATOR]")
@click.option("--link-to", "link_to", multiple=True, help="Attach to these memory ids.")
@click.option("--meta", help="JSON object of extra metadata.")
@click.pass_context
def asset_add(
    ctx: click.Context,
    path: Path,
    media_type: str | None,
    tags: tuple[str, ...],
    source_spec: str | None,
    link_to: tuple[str, ...],
    meta: str | None,
) -> None:
    """Import a file into managed storage."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            stored = api.store_asset(
                path,
                media_type=media_type,
                tags=list(tags),
                source=parse_source(source_spec) if source_spec else None,
                meta=json.loads(meta) if meta else None,
                link_to=list(link_to),
            )
            return {
                "asset": stored.to_json(),
                "blob": api.paths.rel(api.paths.asset_blob(stored.sha256)),
            }
        finally:
            api.close()

    run("asset.add", action)


@asset.command("get")
@click.argument("asset_id")
@click.option("--derived", is_flag=True, help="Include derived outputs.")
@click.option("--path-only", is_flag=True)
@click.pass_context
def asset_get(ctx: click.Context, asset_id: str, derived: bool, path_only: bool) -> None:
    """Read asset metadata."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            payload = api.get_asset(asset_id, with_derived=derived)
            if payload is None:
                raise CliError("not_found", f"no asset {asset_id}", EXIT_NOT_FOUND)
            if path_only:
                return {"path": payload["blob_path"]}
            return {"asset": payload}
        finally:
            api.close()

    run("asset.get", action)


@asset.command("replace")
@click.argument("asset_id")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--discard-previous", is_flag=True, help="Drop the prior version's blob.")
@click.pass_context
def asset_replace(ctx: click.Context, asset_id: str, path: Path, discard_previous: bool) -> None:
    """Point an asset at new bytes, keeping its id."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            updated = api.replace_asset(asset_id, path, discard_previous=discard_previous)
            return {"asset": updated.to_json()}
        finally:
            api.close()

    run("asset.replace", action)


@asset.command("delete")
@click.argument("asset_id")
@click.option("--permanent", is_flag=True)
@click.option("--gc-blobs", is_flag=True, help="Remove blobs nothing else references.")
@click.option("--yes", is_flag=True)
@click.pass_context
def asset_delete(
    ctx: click.Context, asset_id: str, permanent: bool, gc_blobs: bool, yes: bool
) -> None:
    """Mark an asset deleted, or remove its metadata."""

    def action() -> dict[str, Any]:
        if permanent and not yes:
            raise CliError(
                "confirmation_required",
                "permanent deletion is irreversible; pass --yes to confirm",
                EXIT_VALIDATION,
            )
        api = open_api(ctx)
        try:
            return api.delete_asset(asset_id, permanent=permanent, gc_blobs=gc_blobs)
        finally:
            api.close()

    run("asset.delete", action)


@asset.command("list")
@click.option("--media-type")
@click.option("--limit", type=int, default=50, show_default=True)
@click.pass_context
def asset_list(ctx: click.Context, media_type: str | None, limit: int) -> None:
    """List stored assets."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            sql = "SELECT id, media_type, bytes, original_filename, status, version FROM assets"
            params: list[Any] = []
            if media_type:
                sql += " WHERE media_type = ?"
                params.append(media_type)
            sql += " ORDER BY imported_at DESC LIMIT ?"
            params.append(limit)
            rows = [dict(r) for r in api.conn.execute(sql, params)]
            return {"total": len(rows), "assets": rows}
        finally:
            api.close()

    run("asset.list", action)


# ---------------------------------------------------------------- maintenance


@main.command()
@click.option("--verify", "verify_after", is_flag=True, help="Check integrity afterwards.")
@click.pass_context
def rebuild(ctx: click.Context, verify_after: bool) -> None:
    """Rebuild the index from the filesystem."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.rebuild_index(verify_after=verify_after)
        finally:
            api.close()

    run("rebuild", action)


@main.command()
@click.pass_context
def verify(ctx: click.Context) -> None:
    """Report divergence between the filesystem and the index."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            return api.verify()
        finally:
            api.close()

    run("verify", action)


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Environment and index health."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            return api.doctor()
        finally:
            api.close()

    run("doctor", action)


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Counts by type and status, plus storage sizes."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            return api.stats()
        finally:
            api.close()

    run("stats", action)


@main.command()
@click.option("--entities", "facet", flag_value="entity", help="List entities instead of tags.")
@click.option("--prefix", help="Only values starting with this, e.g. --prefix project-.")
@click.option("--min-count", type=int, default=1, show_default=True)
@click.option("--limit", type=int, default=100, show_default=True)
@click.option("--include-inactive", is_flag=True, help="Count superseded and deleted too.")
@click.option("--include-secret", is_flag=True, help="Count secret memories too.")
@click.option("--near-misses", is_flag=True, help="Group likely drifted spellings.")
@click.pass_context
def tags(
    ctx: click.Context,
    facet: str | None,
    prefix: str | None,
    min_count: int,
    limit: int,
    include_inactive: bool,
    include_secret: bool,
    near_misses: bool,
) -> None:
    """The existing tag vocabulary, with usage counts. Reuse before coining."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            return api.tags(
                facet=facet or "tag",
                prefix=prefix,
                min_count=min_count,
                limit=limit,
                include_inactive=include_inactive,
                include_secret=include_secret,
                near_misses=near_misses,
            )
        finally:
            api.close()

    run("tags", action)


@main.command()
@click.option("--limit", type=int, default=20, show_default=True)
@click.pass_context
def review(ctx: click.Context, limit: int) -> None:
    """Memories past their review date."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            hits = api.review_due(limit=limit)
            return {"total": len(hits), "hits": [h.to_json() for h in hits]}
        finally:
            api.close()

    run("review", action)


@main.group()
def journal() -> None:
    """Inspect the operation journal."""


@journal.command("list")
@click.option("--limit", type=int, default=20, show_default=True)
@click.pass_context
def journal_list(ctx: click.Context, limit: int) -> None:
    """Recent journal records."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            with api.journal:
                records = api.journal.history(limit=limit)
                return {
                    "total": len(records),
                    "records": [
                        {
                            "op_id": r.op_id,
                            "seq": r.seq,
                            "kind": r.kind.value,
                            "phase": r.phase,
                            "ts": r.ts,
                            "targets": r.targets,
                        }
                        for r in records
                    ],
                }
        finally:
            api.close()

    run("journal.list", action)


@journal.command("show")
@click.argument("op_id")
@click.pass_context
def journal_show(ctx: click.Context, op_id: str) -> None:
    """Every record for one operation."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            with api.journal:
                records = api.journal.show(op_id)
                if not records:
                    raise CliError("not_found", f"no operation {op_id}", EXIT_NOT_FOUND)
                return {"op_id": op_id, "records": [r.payload() for r in records]}
        finally:
            api.close()

    run("journal.show", action)


@journal.command("recover")
@click.option("--dry-run", is_flag=True)
@click.pass_context
def journal_recover(ctx: click.Context, dry_run: bool) -> None:
    """Settle interrupted operations."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            with api.journal:
                return api.journal.recover(dry_run=dry_run).to_json()
        finally:
            api.close()

    run("journal.recover", action)


@journal.command("truncate")
@click.option("--keep-days", type=int, default=7, show_default=True)
@click.pass_context
def journal_truncate(ctx: click.Context, keep_days: int) -> None:
    """Delete settled journal files older than a cutoff."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            with api.journal:
                return {"removed_files": api.journal.truncate(keep_days=keep_days)}
        finally:
            api.close()

    run("journal.truncate", action)


@main.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["jsonl"]),
    default="jsonl",
    show_default=True,
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path))
@click.pass_context
def export(ctx: click.Context, output_format: str, out: Path | None) -> None:
    """Export every memory as JSON Lines."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            lines = [
                json.dumps(m.to_public(), default=str)
                for m in api.store.iter_all(include_deleted=True)
            ]
            if out is not None:
                out.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return {"format": output_format, "count": len(lines), "path": str(out)}
            return {"format": output_format, "count": len(lines), "lines": lines}
        finally:
            api.close()

    run("export", action)


# ---------------------------------------------------------------- ingestion


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--media-type", help="Override type detection.")
@click.option("--tag", "tags", multiple=True)
@click.option("--source", "source_spec", metavar="KIND:REF[:LOCATOR]")
@click.option("--link-to", "link_to", multiple=True, help="Attach to these memory ids.")
@click.option("--meta", help="JSON object of extra metadata.")
@click.option("--param", "params", multiple=True, metavar="KEY=VALUE", help="Processor parameter.")
@click.option("--no-process", is_flag=True, help="Store the file without deriving content.")
@click.option("--force", is_flag=True, help="Reprocess even if a cached derivation matches.")
@click.option(
    "--defer/--inline",
    "defer",
    default=None,
    help="Override the cost-based lane. Default follows the processor.",
)
@click.option("--recursive", is_flag=True, help="For a directory: descend into subdirectories.")
@click.pass_context
def ingest(
    ctx: click.Context,
    path: Path,
    media_type: str | None,
    tags: tuple[str, ...],
    source_spec: str | None,
    link_to: tuple[str, ...],
    meta: str | None,
    params: tuple[str, ...],
    no_process: bool,
    force: bool,
    defer: bool | None,
    recursive: bool,
) -> None:
    """Import a file (or directory) and derive searchable content from it."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            kwargs: dict[str, Any] = {
                "media_type": media_type,
                "tags": list(tags),
                "source": parse_source(source_spec) if source_spec else None,
                "meta": json.loads(meta) if meta else None,
                "link_to": list(link_to),
                "params": parse_params(params),
                "process": not no_process,
                "force": force,
                "defer": defer,
            }
            if path.is_dir():
                results = api.ingest_directory(path, recursive=recursive, **kwargs)
                return {
                    "directory": str(path),
                    "count": len(results),
                    "segments": sum(r["segments"] for r in results),
                    "queued": [r["queued_task"] for r in results if r["queued_task"]],
                    "errors": [r for r in results if r["status"] == "error"],
                    "results": results,
                }
            return api.ingest(path, **kwargs)
        finally:
            api.close()

    run("ingest", action)


@main.command("reprocess")
@click.argument("asset_id")
@click.option("--param", "params", multiple=True, metavar="KEY=VALUE")
@click.option("--force", is_flag=True, help="Ignore a matching cached derivation.")
@click.option("--defer/--inline", "defer", default=None)
@click.pass_context
def reprocess(
    ctx: click.Context,
    asset_id: str,
    params: tuple[str, ...],
    force: bool,
    defer: bool | None,
) -> None:
    """Derive content again for an asset that is already stored."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.process_asset(
                asset_id, params=parse_params(params), force=force, defer=defer
            )
        finally:
            api.close()

    run("reprocess", action)


@main.group()
def segment() -> None:
    """Search and inspect segments -- the retrievable spans of an asset."""


@segment.command("search")
@click.argument("query")
@click.option("--asset", "asset_id", help="Restrict to one asset.")
@click.option("--limit", type=int, default=20, show_default=True)
@click.pass_context
def segment_search(ctx: click.Context, query: str, asset_id: str | None, limit: int) -> None:
    """Full-text search over segments. Every hit carries its provenance."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            hits = api.search_segments(query, asset_id=asset_id, limit=limit)
            return {"query": query, "total": len(hits), "hits": hits}
        finally:
            api.close()

    run("segment.search", action)


@segment.command("get")
@click.argument("segment_id")
@click.pass_context
def segment_get(ctx: click.Context, segment_id: str) -> None:
    """One segment with its full provenance chain."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            found = api.get_segment(segment_id)
            if found is None:
                raise CliError("not_found", f"no segment {segment_id}", EXIT_NOT_FOUND)
            return found
        finally:
            api.close()

    run("segment.get", action)


@segment.command("list")
@click.argument("asset_id")
@click.option("--limit", type=int, default=500, show_default=True)
@click.pass_context
def segment_list(ctx: click.Context, asset_id: str, limit: int) -> None:
    """Every segment derived from one asset, in order."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            rows = api.asset_segments(asset_id, limit=limit)
            return {"asset_id": asset_id, "total": len(rows), "segments": rows}
        finally:
            api.close()

    run("segment.list", action)


# ---------------------------------------------------------------- queue


@main.group()
def queue() -> None:
    """Inspect and run deferred processing work."""


@queue.command("list")
@click.option("--status", type=click.Choice(["pending", "running", "done", "failed", "cancelled"]))
@click.option("--kind")
@click.option("--asset", "asset_id")
@click.option("--limit", type=int, default=50, show_default=True)
@click.pass_context
def queue_list(
    ctx: click.Context,
    status: str | None,
    kind: str | None,
    asset_id: str | None,
    limit: int,
) -> None:
    """List queued tasks, outstanding work first."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            tasks = api.queue_list(status=status, kind=kind, asset_id=asset_id, limit=limit)
            return {"total": len(tasks), "stats": api.queue_stats(), "tasks": tasks}
        finally:
            api.close()

    run("queue.list", action)


@queue.command("run")
@click.option("--limit", type=int, help="Stop after this many tasks.")
@click.option("--lease", "lease_s", type=int, help="Seconds a claim stays valid.")
@click.pass_context
def queue_run(ctx: click.Context, limit: int | None, lease_s: int | None) -> None:
    """Run ready tasks until none remain."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.queue_run(limit=limit, lease_s=lease_s)
        finally:
            api.close()

    run("queue.run", action)


@queue.command("retry")
@click.argument("task_id", required=False)
@click.option("--all-failed", is_flag=True, help="Retry every failed task.")
@click.option("--keep-attempts", is_flag=True, help="Do not reset the attempt count.")
@click.pass_context
def queue_retry(
    ctx: click.Context, task_id: str | None, all_failed: bool, keep_attempts: bool
) -> None:
    """Return failed tasks to pending."""
    if not task_id and not all_failed:
        raise click.UsageError("pass a TASK_ID or --all-failed")

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.queue_retry(
                task_id=task_id,
                all_failed=all_failed,
                reset_attempts=not keep_attempts,
            )
        finally:
            api.close()

    run("queue.retry", action)


@queue.command("cancel")
@click.argument("task_id")
@click.pass_context
def queue_cancel(ctx: click.Context, task_id: str) -> None:
    """Cancel a pending or running task."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            result = api.queue_cancel(task_id)
            if not result["cancelled"]:
                # A typo'd id must not read as a successful cancel. Only
                # pending and running tasks are cancellable, so say which it
                # was rather than claiming the task does not exist.
                from . import workqueue

                existing = workqueue.get(api.conn, task_id)
                detail = (
                    f"task {task_id} is {existing.status}, not cancellable"
                    if existing is not None
                    else f"no task {task_id}"
                )
                raise CliError("not_found", detail, EXIT_NOT_FOUND)
            return result
        finally:
            api.close()

    run("queue.cancel", action)


@queue.command("purge")
@click.option("--keep-days", type=int, default=7, show_default=True)
@click.pass_context
def queue_purge(ctx: click.Context, keep_days: int) -> None:
    """Drop old finished tasks. Pending and running work is never purged."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.queue_purge(keep_days=keep_days)
        finally:
            api.close()

    run("queue.purge", action)


@main.command()
@click.pass_context
def capabilities(ctx: click.Context) -> None:
    """Report which optional media dependencies are installed."""

    def action() -> dict[str, Any]:
        from . import media
        from .processors import base as processors

        report = media.capability_report()
        report["processors"] = [
            {
                "name": p.name,
                "cost": p.cost,
                "version": p.version,
                "media_types": list(p.media_types),
                "extensions": list(p.extensions),
            }
            for p in sorted(processors.iter_registered(), key=lambda p: p.name)
        ]
        return report

    run("capabilities", action)


# ---------------------------------------------------------------- intelligence


@main.group()
def provider() -> None:
    """Inspect and configure the optional LLM provider."""


@provider.command("show")
@click.pass_context
def provider_show(ctx: click.Context) -> None:
    """Report the configured provider, if any."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            payload: dict[str, Any] = {"config": api.provider_config()}
            instance = api.provider()
            payload["configured"] = instance is not None
            payload["provider"] = instance.describe() if instance is not None else None
            if instance is None:
                payload["note"] = (
                    "No provider is configured. Consolidation and extraction still "
                    "work on their deterministic paths; `--llm` would fail with "
                    "instructions."
                )
            return payload
        finally:
            api.close()

    run("provider.show", action)


@provider.command("set")
@click.option("--provider", "name", required=True, help="none | echo | openai-compatible")
@click.option("--model")
@click.option("--base-url", help="Required for openai-compatible, e.g. http://localhost:11434/v1")
@click.option("--api-key", help="Stored in data/config/llm.json. Omit for a local runtime.")
@click.option("--timeout", type=float, default=60.0, show_default=True)
@click.pass_context
def provider_set(
    ctx: click.Context,
    name: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> None:
    """Write the provider config. No network call is made."""

    def action() -> dict[str, Any]:
        from .providers import ProviderConfig, load_provider, write_config

        api = open_api(ctx)
        try:
            config = ProviderConfig(
                provider=name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            # Build it once so a bad combination fails here, with a clear
            # message, rather than on first use inside another command.
            instance = load_provider(api.paths.data, config=config)
            path = write_config(api.paths.data, config)
            return {
                "path": api.paths.rel(path),
                "config": config.to_json(),
                "provider": instance.describe() if instance is not None else None,
            }
        finally:
            api.close()

    run("provider.set", action)


@main.command()
@click.option("--type", "types", multiple=True, type=TYPE_CHOICE)
@click.option("--tag", "tags", multiple=True)
@click.option("--id", "memory_ids", multiple=True, help="Restrict to these memories.")
@click.option("--threshold", type=click.FloatRange(0, 1), help="Duplicate cutoff.")
@click.option("--near-threshold", type=click.FloatRange(0, 1))
@click.option("--no-stale", is_flag=True, help="Skip review_after/valid_until findings.")
@click.option("--llm", "use_llm", is_flag=True, help="Adjudicate borderline pairs with a model.")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--apply", "apply_proposals", is_flag=True, help="Execute duplicate merges.")
@click.option(
    "--kind",
    "kinds",
    multiple=True,
    type=click.Choice(["duplicate", "contradiction"]),
    help="With --apply: which kinds to execute. Default: duplicate only.",
)
@click.option("--yes", is_flag=True, help="Required for --apply.")
@click.pass_context
def consolidate(
    ctx: click.Context,
    types: tuple[str, ...],
    tags: tuple[str, ...],
    memory_ids: tuple[str, ...],
    threshold: float | None,
    near_threshold: float | None,
    no_stale: bool,
    use_llm: bool,
    limit: int,
    apply_proposals: bool,
    kinds: tuple[str, ...],
    yes: bool,
) -> None:
    """Find duplicates, contradictions and stale memories.

    Reports proposals by default and writes nothing. Each proposal names the
    verb to run and its arguments, so you can execute them selectively.
    """

    def action() -> dict[str, Any]:
        if apply_proposals and not yes:
            raise CliError(
                "confirmation_required",
                "--apply merges memories; pass --yes to confirm",
                EXIT_VALIDATION,
            )
        from .consolidate import DUPLICATE_THRESHOLD, NEAR_THRESHOLD

        api = open_api(ctx, read_only=not apply_proposals)
        try:
            report = api.consolidate(
                use_llm=use_llm,
                types=list(types) or None,
                tags=list(tags) or None,
                memory_ids=list(memory_ids) or None,
                threshold=threshold if threshold is not None else DUPLICATE_THRESHOLD,
                near_threshold=(near_threshold if near_threshold is not None else NEAR_THRESHOLD),
                include_stale=not no_stale,
                limit=limit,
            )
            if apply_proposals:
                report = api.apply_consolidation(report, kinds=list(kinds) or None)
            return report.to_json()
        finally:
            api.close()

    run("consolidate", action)


@main.command()
@click.option(
    "--file",
    "path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the material. Resolved against the cwd, then the memory root.",
)
@click.option("--text")
@click.option("--stdin", "use_stdin", is_flag=True)
@click.option(
    "--source",
    "source_spec",
    required=True,
    metavar="KIND:REF[:LOCATOR]",
    help="Provenance for every candidate. Required: a memory must cite its origin.",
)
@click.option("--llm", "use_llm", is_flag=True, help="Use the configured model.")
@click.option("--no-fallback", is_flag=True, help="With --llm: do not fall back to patterns.")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--commit", "do_commit", is_flag=True, help="Create the candidates.")
@click.option("--min-confidence", type=click.FloatRange(0, 1), default=0.0, show_default=True)
@click.option("--keep-duplicates", is_flag=True, help="With --commit: create near-duplicates too.")
@click.pass_context
def extract(
    ctx: click.Context,
    path: Path | None,
    text: str | None,
    use_stdin: bool,
    source_spec: str,
    use_llm: bool,
    no_fallback: bool,
    limit: int,
    do_commit: bool,
    min_confidence: float,
    keep_duplicates: bool,
) -> None:
    """Propose memories from source material.

    Prints candidates by default; `--commit` creates them. Without `--llm`
    this runs a conservative pattern matcher over explicit statements.
    """

    def action() -> dict[str, Any]:
        sources = [path is not None, text is not None, use_stdin]
        if sum(sources) != 1:
            raise CliError(
                "validation",
                "pass exactly one of --file, --text, --stdin",
                EXIT_VALIDATION,
            )
        api = open_api(ctx, read_only=not do_commit)
        try:
            material = (
                _read_material(path, api)
                if path is not None
                else (text if text is not None else sys.stdin.read())
            )
            report = api.extract(
                material or "",
                source=parse_source(source_spec),
                use_llm=use_llm,
                limit=limit,
                fallback=not no_fallback,
            )
            if do_commit:
                report = api.commit_extraction(
                    report,
                    min_confidence=min_confidence,
                    skip_duplicates=not keep_duplicates,
                )
            return report.to_json()
        finally:
            api.close()

    run("extract", action)


def _read_material(path: Path, api: MemoryAPI) -> str:
    """Read extraction input, accepting a cwd- or root-relative path.

    A user pairs `--file conversations/x.md` with `--source
    conversation:conversations/x.md` and means the same file both times. Click's
    `exists=True` only ever resolves against the process cwd, so that pairing
    failed with a bare "does not exist" from somewhere other than the root.
    """
    if path.is_file():
        return path.read_text(encoding="utf-8")
    for candidate in (api.paths.data / path, api.paths.root / path):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise CliError(
        "not_found",
        f"no such file: {path} (looked in the current directory, "
        f"in {api.paths.data} and in {api.paths.root})",
        EXIT_NOT_FOUND,
    )


# ---------------------------------------------------------------- sources


@main.group()
def source() -> None:
    """Manage source material and what depends on it."""


@source.command("plan-delete")
@click.argument("source_ref")
@click.option("--kind", type=click.Choice([k.value for k in SourceKind]))
@click.option(
    "--memories",
    "memory_policy",
    type=click.Choice(["orphan", "delete", "purge", "keep"]),
    default="orphan",
    show_default=True,
)
@click.pass_context
def source_plan_delete(
    ctx: click.Context, source_ref: str, kind: str | None, memory_policy: str
) -> None:
    """Show everything deleting a source would touch. Writes nothing."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            plan = api.plan_source_deletion(source_ref, kind=kind, memory_policy=memory_policy)
            return plan.to_json()
        finally:
            api.close()

    run("source.plan-delete", action)


@source.command("delete")
@click.argument("source_ref")
@click.option("--kind", type=click.Choice([k.value for k in SourceKind]))
@click.option(
    "--memories",
    "memory_policy",
    type=click.Choice(["orphan", "delete", "purge", "keep"]),
    default="orphan",
    show_default=True,
    help="orphan keeps memories and drops the dangling citation.",
)
@click.option("--keep-raw", is_flag=True, help="Remove derived state but keep the original file.")
@click.option("--reason")
@click.option("--yes", is_flag=True, help="Required: this removes derived files.")
@click.pass_context
def source_delete(
    ctx: click.Context,
    source_ref: str,
    kind: str | None,
    memory_policy: str,
    keep_raw: bool,
    reason: str | None,
    yes: bool,
) -> None:
    """Delete a source and everything derived from it."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=not yes)
        try:
            if not yes:
                plan = api.plan_source_deletion(source_ref, kind=kind, memory_policy=memory_policy)
                return {"dry_run": True, "plan": plan.to_json(), "hint": "pass --yes to execute"}
            result = api.delete_source(
                source_ref,
                kind=kind,
                memory_policy=memory_policy,
                remove_raw=not keep_raw,
                reason=reason,
            )
            return result.to_json()
        finally:
            api.close()

    run("source.delete", action)


@source.command("list")
@click.option("--limit", type=int, default=50, show_default=True)
@click.pass_context
def source_list(ctx: click.Context, limit: int) -> None:
    """Sources memories cite, with how many cite each."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            rows = [
                dict(r)
                for r in api.conn.execute(
                    "SELECT s.id, s.kind, s.ref, s.captured_at, "
                    "(SELECT count(*) FROM memory_sources ms WHERE ms.source_id = s.id) "
                    "AS memories FROM sources s ORDER BY memories DESC, s.ref LIMIT ?",
                    (limit,),
                )
            ]
            return {"total": len(rows), "sources": rows}
        finally:
            api.close()

    run("source.list", action)


@source.command("orphans")
@click.pass_context
def source_orphans(ctx: click.Context) -> None:
    """Memories whose provenance was removed and not re-established."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            rows = api.orphaned_memories()
            return {"total": len(rows), "memories": rows}
        finally:
            api.close()

    run("source.orphans", action)


# ---------------------------------------------------------------- backup


@main.group()
def backup() -> None:
    """Archive, restore, and apply retention."""


@backup.command("create")
@click.option("--out", type=click.Path(path_type=Path), help="File or directory.")
@click.option("--label", help="A short name recorded in the filename and manifest.")
@click.option("--no-compress", is_flag=True)
@click.pass_context
def backup_create(
    ctx: click.Context, out: Path | None, label: str | None, no_compress: bool
) -> None:
    """Archive data/ (excluding the rebuildable index)."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            return api.backup(dest=out, label=label, compress=not no_compress).to_json()
        finally:
            api.close()

    run("backup.create", action)


@backup.command("list")
@click.option("--dir", "directory", type=click.Path(path_type=Path))
@click.pass_context
def backup_list(ctx: click.Context, directory: Path | None) -> None:
    """Backups on disk, newest first."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            rows = api.list_backups(directory=directory)
            return {"total": len(rows), "backups": rows}
        finally:
            api.close()

    run("backup.list", action)


@backup.command("prune")
@click.option("--keep-days", type=int, default=90, show_default=True)
@click.option("--keep-count", type=int, default=10, show_default=True)
@click.option("--dir", "directory", type=click.Path(path_type=Path))
@click.option("--yes", is_flag=True, help="Without this, reports what it would remove.")
@click.pass_context
def backup_prune(
    ctx: click.Context, keep_days: int, keep_count: int, directory: Path | None, yes: bool
) -> None:
    """Apply the retention policy. The newest backup is always kept."""

    def action() -> dict[str, Any]:
        api = open_api(ctx, read_only=True)
        try:
            return api.prune_backups(
                keep_days=keep_days,
                keep_count=keep_count,
                directory=directory,
                dry_run=not yes,
            )
        finally:
            api.close()

    run("backup.prune", action)


@backup.command("show")
@click.argument("archive", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def backup_show(archive: Path) -> None:
    """Read an archive's manifest without extracting it."""

    def action() -> dict[str, Any]:
        from .backup import read_manifest

        return {"archive": str(archive), "manifest": read_manifest(archive)}

    run("backup.show", action)


@backup.command("restore")
@click.argument("archive", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--to", "target", type=click.Path(path_type=Path), required=True)
@click.option("--overwrite", is_flag=True, help="Replace a non-empty data/ directory.")
@click.option("--rebuild/--no-rebuild", default=True, show_default=True)
@click.pass_context
def backup_restore(
    ctx: click.Context, archive: Path, target: Path, overwrite: bool, rebuild: bool
) -> None:
    """Extract an archive into a root and rebuild its index."""

    def action() -> dict[str, Any]:
        from .backup import restore as restore_archive

        report = restore_archive(archive, target, overwrite=overwrite)
        if rebuild:
            api = MemoryAPI(target)
            api.open()
            try:
                report["rebuild"] = api.rebuild_index(verify_after=True)
            finally:
                api.close()
        return report

    run("backup.restore", action)


@main.command("import")
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "input_format",
    type=click.Choice(["jsonl", "markdown"]),
    default="jsonl",
    show_default=True,
)
@click.option(
    "--force-recreate",
    is_flag=True,
    help="Override tombstones. Only on an explicit request: it undoes a deletion.",
)
@click.pass_context
def import_memories(
    ctx: click.Context, source_path: Path, input_format: str, force_recreate: bool
) -> None:
    """Import memories from an export. Additive; will not resurrect deletions."""

    def action() -> dict[str, Any]:
        api = open_api(ctx)
        try:
            result = api.import_memories(
                source_path, format=input_format, force_recreate=force_recreate
            )
            return result.to_json()
        finally:
            api.close()

    run("import", action)


if __name__ == "__main__":  # pragma: no cover
    main()
