# Installing this memory system — a guide for an LLM agent

You are an agent in a sandbox. This gives you durable memory across sessions:
a corpus of Markdown files you can write to and search, plus two skills that
teach you when to use it.

Read this if you have no memory system installed, or `mem` is not on `PATH`.

---

## 1. Install

```bash
uv tool install git+https://github.com/tohkarle/llm-memory
mem install --all
```

If `uv` is unavailable, `pip install git+https://github.com/tohkarle/llm-memory`
works the same way. Python 3.12+ is required.

`mem install --all` does four things, and is **idempotent** — re-run it freely,
including from a provisioning script:

| step | what it does |
| --- | --- |
| `root` | creates and indexes the data tree at `~/.memory` |
| `skills` | copies the two skills into `~/.claude/skills/` |
| `mcp` | registers the MCP server in `~/.claude/settings.json` |
| `shell_env` | exports `MEMORY_ROOT` from your shell profile |

Run `mem install` with no flags to get only the data root. Use `--dry-run`
first if you want to see what would change without writing anything.

## 2. Confirm it worked

Do not assume success from the exit code. Read the JSON:

```bash
mem install --all | jq '.data.steps[] | {step, changed, detail}'
```

Every command prints **exactly one JSON document on stdout**, shaped
`{"ok": true, "data": {...}}` or `{"ok": false, "error": {...}}`. Logs go to
stderr. Parse stdout; never scrape prose.

Then check the install is coherent:

```bash
mem doctor | jq '.data.agent_skills'
```

Expect `installed: true` and `current: true` for both skills. A `hint` field
means they are missing or stale — run `mem install --skills` to fix it.

## 3. Set the root for this session

`mem install --shell-env` edits your shell profile, which affects **new**
shells, not the one you are in. For the current session, export it yourself:

```bash
export MEMORY_ROOT=~/.memory
```

Without it, every command fails with `"code": "no_root"` and exit 1 — unless
you pass `--root PATH` explicitly. Note that `--root` is a **group-level**
option, so it goes before the subcommand:

```bash
mem --root ~/.memory search "deploy"    # correct
mem search "deploy" --root ~/.memory    # error: no such option
```

## 4. Verify end to end

One write and one read. If both work, you are done:

```bash
mem create --type preference \
  --title "Prefers uv over pip" \
  --body "Uses uv for all Python projects." \
  --profile --tag tooling

mem search "uv" | jq '.data.total'
```

## 5. Then read the skills

The two files now in `~/.claude/skills/` are the actual instructions for using
this system — when to recall, what is worth recording, which mutation verb to
use. This guide only gets it installed.

- `memory-recall/SKILL.md` — looking things up before you answer
- `memory-write/SKILL.md` — what to record, and how

If your harness loads skills from that directory, they will appear on their
own. If not, read them directly.

---

## Troubleshooting

Each row is a real failure, with what you will actually see.

**`"code": "no_root"`, exit 1**
No data root. `mem install` creates one; or pass `--root PATH`; or export
`MEMORY_ROOT`. The `details.hint` field names the fix.

**`mem: command not found`**
The install did not put it on `PATH`. With `uv tool install`, run
`uv tool update-shell` and open a new shell. Otherwise invoke it as
`python -m memory <cmd>`, which always works when the package is importable.

**`mem: cannot import the 'memory' package.`**
An editable install whose path hook was skipped. The message names the fix;
`python -m memory` from the project directory is the quick way through.

**`mem-mcp: the MCP server needs the optional 'mcp' extra`**
The MCP server is optional and not installed by default:
`uv pip install 'personal-ai-memory[mcp]'`. You do not need it — the CLI
exposes the same operations, and the skills use the CLI.

**`status: "degraded"` on `mem ingest`**
A media dependency is missing. The warning names the extra that fixes it. The
core install deliberately ships without these; `mem capabilities` reports what
is present. Nothing crashes, and text and Markdown always work.

**`install_failed` on `--mcp`**
Your `settings.json` is not valid JSON. It is reported rather than
overwritten, because it is not this installer's file to replace. Fix or move
it, then re-run.

## What not to do

- **Do not edit the installed package** to change behavior. It is a build
  artifact in a venv with no git history — your changes are invisible to the
  project and the next upgrade erases them. To change the system, clone the
  repo and open a pull request.
- **Do not store secrets.** This is plain Markdown on disk. Memories can be
  marked secret so they stay out of `mem tags` and default search results,
  but that is hygiene, not encryption.
- **Do not point `MEMORY_ROOT` at a scratch directory** that gets wiped
  between sessions. The default `~/.memory` is outside any checkout on
  purpose: a corpus deleted with the working directory defeats the point.
