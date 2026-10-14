# personal-ai-memory

Local-first memory system for LLM agents. No cloud, no server, no vector database.

**The filesystem is the source of truth.** Memories are Markdown files with YAML
frontmatter under `data/memories/`. SQLite is a rebuildable index for search and
relationships — delete `data/index/memory.sqlite` at any time and run `mem rebuild`.

## Quick start

```bash
uv sync --extra dev
uv run mem init
uv run mem create --type preference --title "Prefers uv over pip" \
  --body "Uses uv for all Python projects." --profile --tag tooling
uv run mem search "uv"
uv run mem context "python tooling" --budget 800 --format markdown
```

## How agents use it

Two skills in `.claude/skills/` teach any agent the system:

- **memory-recall** — when and how to look things up.
- **memory-write** — what to record, and which mutation verb to use.

The CLI emits a JSON envelope on stdout (`{"ok": true, "data": {...}}`); logs go to
stderr. `mem context --format markdown` is the one command that prints prose, because
it is meant to be pasted into a prompt.

An optional MCP server (`uv sync --extra mcp`) exposes the same operations to MCP
clients: `python -m memory.mcp_server`.

## Design

| Property | Choice |
| --- | --- |
| Truth | Markdown + YAML frontmatter on disk |
| Index | SQLite + FTS5, fully rebuildable |
| Search | Keyword (FTS5) + SQL filters + relation traversal; no embeddings |
| Durability | Operation journal + atomic writes; recovers from interrupted writes |
| Identity | UUID per memory, SHA-256 content addressing for media |
| Deletion | Soft (superseded) and permanent are separate; tombstones prevent resurrection |

Memory types: `fact`, `preference`, `event`, `decision`, `procedure`.

## Layout

```
data/memories/<type>/<uuid>.md   authoritative memory files
data/assets/blobs/ab/cd/<sha>    content-addressed media
data/derived/                    transcripts, descriptions, keyframes
data/index/memory.sqlite         rebuildable index
data/.journal/                   operation journal
data/.tombstones/                deletion records
```

Set `$MEMORY_ROOT` or pass `--root` to use a data directory elsewhere.

## Development

```bash
uv run pytest                    # full suite
uv run mypy --strict src/memory
uv run ruff check . && uv run ruff format --check .
uv run mem doctor                # environment + index health
```
