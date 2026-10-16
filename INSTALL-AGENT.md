# Installing this memory system — a guide for an LLM agent

You are an agent in a sandbox. This gives you durable memory across sessions:
a corpus of Markdown files you can write to and search, plus two skills that
teach you when to use it.

Read this if you have no memory system installed, or `mem` is not on `PATH`.

---

## 0. First: are you allowed to write outside your working directory?

This matters before you run anything. `mem install --all` writes to four
places, and three are under `$HOME`: the data root (`~/.memory`), the skills
(`~/.claude/skills/`), the MCP entry (`~/.claude/settings.json`), and your
shell profile.

**If you are confined to a working directory, or are not permitted to modify
your harness's configuration — which includes `~/.claude/` — do not run
`--all`.** That is the correct call, not an obstacle to work around. Use
[§1b, the confined install](#1b-confined-install-working-directory-only)
instead. It is fully supported and gives you a working memory system.

Use §1 only if you can write to `$HOME` freely — a container you own, a
provisioning script, or a human's shell.

## 1. Install (unrestricted)

```bash
uv tool install git+https://github.com/tohkarle/llm-memory
mem install --all
```

If `uv` is unavailable, `pip install git+https://github.com/tohkarle/llm-memory`
works the same way. **Python 3.12+ is required** — check with
`python3 --version` first, because `uv venv` may otherwise pick an older
interpreter and the install will fail with an unsatisfiable-requirements
error.

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

## 1b. Confined install (working directory only)

Everything stays inside the working directory. Nothing is written under
`$HOME`, and your harness's configuration is left alone.

```bash
# 1. A virtual environment here, on Python 3.12+.
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  "personal-ai-memory @ git+https://github.com/tohkarle/llm-memory"

# 2. Data root and skills, both local. No MCP, no shell edit.
.venv/bin/mem --root ./.memory install --skills --agent-dir ./.claude

# 3. Make both locations stick for every later command.
export MEMORY_ROOT="$PWD/.memory"
export MEMORY_AGENT_DIR="$PWD/.claude"
```

Then use `.venv/bin/mem` (or add `.venv/bin` to `PATH`). Read the two skills
at `./.claude/skills/*/SKILL.md` and follow them yourself — they will not be
auto-registered by your harness, so apply them manually.

**Export `MEMORY_AGENT_DIR`.** Without it, `mem doctor` looks under `$HOME`,
finds nothing, and reports the skills as missing even though they are
installed. `mem install` warns you if you forget.

If `uv` writes its cache to `$HOME` and that is blocked too, set
`UV_CACHE_DIR="$PWD/.uv-cache"` before installing. A pre-existing
`python3.12` avoids the download entirely.

Everything else in this guide applies unchanged, except that `--root` and the
local `mem` path are yours to supply.

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
- **Do not point `MEMORY_ROOT` at a directory that gets wiped** between
  sessions. The default `~/.memory` sits outside any checkout on purpose. A
  root inside the working directory (§1b) is correct when that is all you can
  write to — but say so plainly, because the corpus then lives and dies with
  that directory.
