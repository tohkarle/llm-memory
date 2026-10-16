# personal-ai-memory

Local-first memory system for LLM agents. No cloud, no server, no vector database.

**The filesystem is the source of truth.** Memories are Markdown files with YAML
frontmatter under `data/memories/`. SQLite is a rebuildable index for search and
relationships — delete `data/index/memory.sqlite` at any time and run `mem rebuild`.

## Install

Any machine or sandbox, no checkout required:

```bash
uv tool install git+https://github.com/tohkarle/llm-memory
mem install --all
```

Setting this up as an agent in a sandbox? [**INSTALL-AGENT.md**](INSTALL-AGENT.md)
is written for you: the same steps, plus how to verify the result from the JSON
and the failures you are most likely to hit.

`mem install` is the bootstrap, and it is **idempotent** — safe to re-run from
a provisioning script or a Dockerfile. `--all` does four things:

| Step | What it does |
| --- | --- |
| `root` | Creates and indexes the data tree (default `~/.memory`) |
| `skills` | Copies the bundled agent skills into `~/.claude/skills/` |
| `mcp` | Registers the MCP server in `~/.claude/settings.json` |
| `shell_env` | Exports `MEMORY_ROOT` from your shell profile |

Each step is opt-in individually (`--skills`, `--mcp`, `--shell-env`); only the
data root always runs. `--dry-run` reports what would change and writes
nothing. Output is JSON like every other command, so a calling agent can
verify the result rather than trusting the exit code:

```bash
mem --root /workspace/memory install --skills --mcp --dry-run
```

The MCP registration merges into `settings.json` rather than replacing it, and
a malformed existing file is reported as an error instead of being
overwritten. The shell export is written between markers, so a re-run replaces
its own block instead of stacking another copy.

### Agent skills

The two skills under `memory/agent_assets/skills/` are the agent-facing half of
this system — the CLI is useless to a model that does not know when to reach
for it. They ship **inside the wheel**, so installing the package and
installing the agent's instructions are one act. Each carries an
`x-memory-version` stamp, and `mem doctor` reports whether the installed copies
are current:

```bash
mem doctor | jq .data.agent_skills
mem install --skills          # install or refresh; --force to overwrite
```

This repository's `.claude/skills` is a symlink to that directory, so the copy
being dogfooded and the copy being shipped cannot drift apart.

### Optional dependencies

The core install needs none. `--profile` names a set and reports the command
rather than installing into the interpreter it is running from:

```bash
mem install --profile minimal   # core only — the sandbox default
mem install --profile text      # + pypdf, python-docx, Pillow
mem install --profile full      # + transcription, OCR, MCP, tokenizer
```

## Quick start

```bash
mem create --type preference --title "Prefers uv over pip" \
  --body "Uses uv for all Python projects." --profile --tag tooling
mem search "uv"
mem context "python tooling" --budget 800 --format markdown
mem ingest ./some-notes.md
mem segment search "packaging"
mem doctor
```

`MEMORY_ROOT` selects the data directory. Note that `--root` is a group-level
option, so it goes **before** the subcommand: `mem --root PATH install`.

## Multimodal ingestion

`mem ingest` imports a file, derives searchable text from it, and indexes
that text as **segments** — spans that carry the locator a citation needs.

```bash
mem ingest ./paper.pdf --tag research       # page-numbered segments
mem ingest ./talk.mp4                       # transcript + keyframes (deferred)
mem ingest ./notes/ --recursive
mem segment search "pricing tiers"          # search inside ingested media
mem capabilities                            # what optional tooling is present
```

Type detection reads magic numbers rather than trusting the extension, so a
`.txt` that is really a PDF reaches the right processor. `--param KEY=VALUE`
tunes one (`ocr=false`, `model=small`, `keyframe_interval_s=10`).

| Processor | Extracts | Locator on each segment |
| --- | --- | --- |
| text | Markdown, plain text, CSV, JSON | — |
| document | PDF and docx text | `page`, or `meta.heading` for docx |
| image | dimensions, EXIF, optional OCR | — |
| audio | timestamped transcript | `start_ms`, `end_ms` |
| video | transcript from the audio track, keyframes | `start_ms`, `end_ms` |

**Originals are never modified.** Everything derived is written under
`data/derived/`, and each derived file embeds its own cache key, so
`mem rebuild` reconstructs the segment and derivation tables from the
filesystem alone — the same guarantee memories have.

### Cheap now, expensive later

Text, documents and images are processed inline. Transcription is not: a long
recording would block the command that imported it, so audio and video are
queued and `ingest` returns `status: "queued"`.

```bash
mem queue list                    # what is outstanding, failures first
mem queue run                     # drain ready work
mem queue retry --all-failed      # after installing a missing extra
```

The queue is SQLite-backed and survives process death. A claim takes a
*lease*, not a lock, so a worker killed mid-task has its work reclaimed once
the lease expires instead of being stranded; attempts are counted when the
task is claimed, so a crash that skips the failure handler cannot cause an
infinite retry. Failures back off exponentially and are capped. `mem rebuild`
deliberately leaves the queue alone, because a pending task is the one piece
of state the filesystem cannot reproduce.

### Optional media dependencies

```bash
uv sync --extra media            # Pillow, pypdf, python-docx
uv sync --extra transcription    # faster-whisper
uv sync --extra ocr              # pytesseract (also needs the tesseract binary)
```

**The core install needs none of them.** A missing dependency degrades to the
metadata that is still extractable and reports `status: "degraded"` with a
warning naming the extra that fixes it — never a crash, and never a silent
empty result. Video additionally wants `ffmpeg`/`ffprobe` on `PATH`.
`mem capabilities` reports exactly what is available; re-run
`mem reprocess <asset-id> --force` after installing something.

Third-party processors register through the `memory.processors` entry point
group, so a new media type can ship as a separate package.

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

Two skills teach any agent the system. They ship inside the wheel and
`mem install --skills` copies them to `~/.claude/skills/`:

- **memory-recall** — when and how to look things up.
- **memory-write** — what to record, and which mutation verb to use.

The CLI emits a JSON envelope on stdout (`{"ok": true, "data": {...}}`); logs go to
stderr. `mem context --format markdown` is the one command that prints prose, because
it is meant to be pasted into a prompt.

An optional MCP server (`mem install --mcp`, with the `mcp` extra) exposes the
same 18 operations to MCP clients over stdio via the `mem-mcp` command. That includes
`memory_ingest`, `memory_search_segments` and `memory_queue_run`, so a client
can extract and search inside media rather than only storing it. Set `MEMORY_ROOT` to
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
| Multimodal | Pluggable processors; media extras optional, degrade not crash |
| Retrieval | Segment-level, with timestamp/page provenance on every hit |
| Background work | SQLite-backed lease queue; survives process death, retries with backoff |
| Consolidation | Lexical duplicate/conflict detection; proposes, never writes |
| Intelligence | Optional, pluggable LLM provider; every feature works without one |
| Backup | tar of `data/` minus the index; explicit retention, lossless round trip |

Memory types: `fact`, `preference`, `event`, `decision`, `procedure`.

## Layout

```
data/memories/<type>/<uuid>.md   authoritative memory files
data/assets/blobs/ab/cd/<sha>    content-addressed media
data/derived/transcripts/        extracted text, timestamps, page numbers
data/derived/descriptions/       image metadata and OCR output
data/derived/keyframes/          frames sampled from video
data/index/memory.sqlite         rebuildable index
data/.journal/                   operation journal
data/.tombstones/                deletion records
data/config/llm.json             optional LLM provider settings
backups/                         archives, beside data/ not inside it
```

Set `$MEMORY_ROOT` or pass `--root` to use a data directory elsewhere.

## Development

```bash
uv sync --extra dev        # dependencies and test tooling
uv pip install .           # installs the `mem` and `mem-mcp` commands
```

```bash
uv run pytest                        # full suite
uv run mypy                          # strict, configured in pyproject
uv run ruff check . && uv run ruff format --check .
uv run python -m memory doctor       # environment + index health
uv run python -m memory verify       # filesystem vs. index divergence
```

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

## License

MIT — see [LICENSE](LICENSE).
