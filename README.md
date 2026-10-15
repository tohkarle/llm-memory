# personal-ai-memory

Local-first memory system for LLM agents. No cloud, no server, no vector database.

**The filesystem is the source of truth.** Memories are Markdown files with YAML
frontmatter under `data/memories/`. SQLite is a rebuildable index for search and
relationships — delete `data/index/memory.sqlite` at any time and run `mem rebuild`.

## Quick start

```bash
uv sync --extra dev        # dependencies and test tooling
uv pip install .           # installs the `mem` command
```

```bash
mem init
mem create --type preference --title "Prefers uv over pip" \
  --body "Uses uv for all Python projects." --profile --tag tooling
mem search "uv"
mem context "python tooling" --budget 800 --format markdown
mem doctor
```

Set `MEMORY_ROOT`, or pass `--root PATH`, to choose the data directory.

### If `mem` reports "No module named 'memory'"

`uv sync` installs this project in editable mode, which relies on a `.pth`
path hook in site-packages. On macOS, installers may write that file with the
`UF_HIDDEN` filesystem flag, and CPython 3.12+ skips hidden `.pth` files
outright — so the package imports fine inside the project directory, where
the CWD is already on the path, but the installed `mem` script fails
anywhere else.

`uv pip install .` (above) avoids the hook by installing real files, and
`mem doctor` reports `install.warning` when it detects the condition. Within
the project you can always use `uv run python -m memory …` instead. Note that
a later `uv sync` reverts to editable mode and undoes this; re-run
`uv pip install .` if `mem` stops resolving.

## Memory maintenance

A corpus that only grows accumulates duplicates, contradictions and claims
that have quietly gone stale. These commands find them; you decide.

```bash
mem consolidate                       # duplicates, contradictions, stale — reports only
mem consolidate --apply --yes         # execute the duplicate merges
mem extract --file notes.md --source document:notes.md
mem source plan-delete conversations/2026-10-15.md
mem backup create --label "before cleanup"
```

`consolidate` and `extract` **write nothing by default**. Each proposal names
the verb to run and its arguments, so you can apply them selectively or hand
the JSON to an agent. Duplicate detection is lexical — FTS5 plus token
overlap, no embeddings and no model needed.

Deleting source material is dependency-aware: `mem source delete` removes the
transcripts, segments, derivations and assets computed from it. Memories that
cited it are kept by default, tagged `provenance-lost` and flagged for
review, because a memory that outlived its citation may still be true.
`--memories purge` removes them permanently instead, writing tombstones so
they cannot be re-extracted.

### Backup and retention

```bash
mem backup create                     # tar of data/, excluding the index
mem backup list
mem backup restore <archive> --to /path/to/new-root
mem backup prune --keep-days 90 --keep-count 10 --yes
mem export --out memories.jsonl  &&  mem import memories.jsonl
```

Archives exclude `data/index/` (it is rebuildable) and always include
tombstones. `restore` extracts and rebuilds in one step, and refuses a
non-empty root — merging into a live tree is `mem import`, which is additive
and cannot resurrect a permanently deleted memory. `prune` never removes the
newest backup, whatever the policy says.

### Optional LLM provider

Every feature above runs without a language model. Configuring one adds
judgement on the cases a lexical pass cannot settle — paraphrases that are
not word-for-word duplicates, and extraction from free-form text.

```bash
mem provider show                     # reports whether one is configured
mem provider set --provider openai-compatible \
  --base-url http://localhost:11434/v1 --model qwen2.5
mem consolidate --llm
```

**No vendor is hardcoded and nothing is called by default.** The HTTP
provider speaks the chat-completions format that local runtimes (Ollama,
llama.cpp, vLLM, LM Studio) expose, and it refuses to construct without an
explicit `--base-url`, so no request can go anywhere you did not name.
Settings live in `data/config/llm.json` or `$MEMORY_LLM_*`. Without a
provider, `--llm` degrades to the deterministic path rather than failing.

## How agents use it

Two skills in `.claude/skills/` teach any agent the system:

- **memory-recall** — when and how to look things up.
- **memory-write** — what to record, and which mutation verb to use.

The CLI emits a JSON envelope on stdout (`{"ok": true, "data": {...}}`); logs go to
stderr. `mem context --format markdown` is the one command that prints prose, because
it is meant to be pasted into a prompt.

An optional MCP server (`uv sync --extra mcp`) exposes the same 15 operations
to MCP clients over stdio: `python -m memory.mcp_server`. Set `MEMORY_ROOT` to
choose the data directory. Destructive tools are annotated as such, and
permanent deletion requires `confirm: true`.

## Design

| Property | Choice |
| --- | --- |
| Truth | Markdown + YAML frontmatter on disk |
| Index | SQLite + FTS5, fully rebuildable |
| Search | Keyword (FTS5) + SQL filters + relation traversal; no embeddings |
| Durability | Operation journal + atomic writes; recovers from interrupted writes |
| Identity | UUID per memory, SHA-256 content addressing for media |
| Deletion | Soft (superseded) and permanent are separate; tombstones prevent resurrection |
| Dependencies | Deleting a source removes what was derived from it |
| Consolidation | Lexical duplicate/conflict detection; proposes, never writes |
| Intelligence | Optional, pluggable LLM provider; every feature works without one |
| Backup | tar of `data/` minus the index; explicit retention, lossless round trip |

Memory types: `fact`, `preference`, `event`, `decision`, `procedure`.

## Layout

```
data/memories/<type>/<uuid>.md   authoritative memory files
data/assets/blobs/ab/cd/<sha>    content-addressed media
data/derived/                    transcripts, descriptions, keyframes
data/index/memory.sqlite         rebuildable index
data/.journal/                   operation journal
data/.tombstones/                deletion records
data/config/llm.json             optional LLM provider settings
backups/                         archives, beside data/ not inside it
```

Set `$MEMORY_ROOT` or pass `--root` to use a data directory elsewhere.

## Development

```bash
uv run pytest                        # full suite
uv run mypy                          # strict, configured in pyproject
uv run ruff check . && uv run ruff format --check .
uv run python -m memory doctor       # environment + index health
uv run python -m memory verify       # filesystem vs. index divergence
```
