# Installing this memory system — a guide for an LLM agent

You are an agent in a sandbox. This gives you durable memory across sessions:
a corpus of Markdown files you can write to and search, plus two skills that
teach you when to use it.

Read this if you have no memory system installed, or `mem` is not on `PATH`.

> **Not for an agent working on this repository.** This guide provisions a
> *new* sandbox. This checkout is already a memory root: its corpus is in
> `./data/`, `mem` discovers it by walking up from the working directory,
> and `$MEMORY_ROOT` is correctly unset. Running an installer here would
> point your writes at an empty tree. If you are here to change the code,
> read `./CLAUDE.md` instead.

---

## 0. Everything stays in your working directory

**`mem install` writes nothing outside the directory you run it in.** The
corpus goes to `./data/`, the skills and persona to `./.claude/`. There is
no `$HOME` write to opt out of, so an agent confined to its working
directory — the usual case — can run the normal install.

Two things are the exception, and both are opt-in: `--shell-env` appends to
your shell profile, and `--mcp` writes `settings.json` in the agent config
directory. Skip both unless you are permitted to edit your harness's
configuration.

**The tradeoff this buys you, stated plainly:** a corpus under a scratch
checkout dies with the checkout. If the sandbox is disposable and the
memories must outlive it, pass `--root` pointing somewhere durable, and
export `$MEMORY_ROOT` so later commands find it.

## 1. Install

```bash
uv tool install git+https://github.com/tohkarle/llm-memory
mem install --skills --persona
```

If `uv` is unavailable, `pip install git+https://github.com/tohkarle/llm-memory`
works the same way. **Python 3.12+ is required** — check with
`python3 --version` first, because `uv venv` may otherwise pick an older
interpreter and the install will fail with an unsatisfiable-requirements
error.

Every step is **idempotent** — re-run it freely, including from a
provisioning script. Paths are relative to the working directory:

| step | flag | what it does |
| --- | --- | --- |
| `root` | always | creates and indexes the data tree at `./data/` |
| `skills` | `--skills` | copies the two skills into `./.claude/skills/` |
| `persona` | `--persona` | merges the standing instructions into `./.claude/CLAUDE.md` |
| `mcp` | `--mcp` | registers the MCP server in `./.claude/settings.json` |
| `shell_env` | `--shell-env` | exports `MEMORY_ROOT` from your shell profile |

Add `--portable` if your harness is not Claude Code: the skills are also
written to `./.agents/skills/`, and the persona to `./AGENTS.md`, which is
the convention Cursor, Codex, Copilot and others read. Both copies carry
the same text in full — a pointer from one to the other would only be
followed by an agent that chose to.

`--all` runs all five. `--agent-dir PATH` moves the `.claude` writes
elsewhere (`--agent-dir ~/.claude` installs globally, for a human who wants
that); `--root PATH` moves the corpus. Run `mem install` with no flags to
get only the data root, and `--dry-run` to see what would change without
writing anything.

## 1b. No package manager, or a blocked cache

If `uv tool install` cannot write where it wants, install into a virtual
environment here instead:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  "personal-ai-memory @ git+https://github.com/tohkarle/llm-memory"
.venv/bin/mem install --skills --persona --inline-skills
```

Then use `.venv/bin/mem`, or add `.venv/bin` to `PATH`.

`--inline-skills` carries the skill text into `CLAUDE.md` itself. Use it
when nothing in your harness registers a skills directory: a pointer to
files you have to remember to open is not enough, and the whole point of
the persona is that it is read without being chosen. **Read
`./.claude/CLAUDE.md` before you start work** — with that flag it is the
whole of what you need.

If `uv` writes its cache to `$HOME` and that is blocked too, set
`UV_CACHE_DIR="$PWD/.uv-cache"` before installing. A pre-existing
`python3.12` avoids the download entirely.

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

## 3. Finding the root later

You do not normally need to do anything. `mem` discovers its root by
walking up from the working directory to the one holding
`data/.memory-root`, so any command run inside the tree you installed into
is already pointed at the right corpus. **An unset `$MEMORY_ROOT` is the
normal state, not a missing step.**

Set it only when the walk cannot work — a command run from outside the
tree, a cron job, or an MCP server launched with an unpredictable working
directory:

```bash
export MEMORY_ROOT="$PWD/data"       # or wherever the root is
```

`mem install --shell-env` writes that into your shell profile, for new
shells. Note that `--root` is a **group-level** option, so it goes before
the subcommand:

```bash
mem --root /path/to/root search "deploy"    # correct
mem search "deploy" --root /path/to/root    # error: no such option
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

## 5. Then read the instructions

`mem install --persona` writes the standing rules into
`~/.claude/CLAUDE.md`: **you own memory and never delegate it; you delegate
execution.** A harness that reads `CLAUDE.md` loads this at session start,
so a fresh agent gets it without being told.

For a runtime with no such file, print the same text into a real system
prompt:

```bash
claude --append-system-prompt "$(mem persona --print)"
```

`mem persona --print --inline-skills` adds the skill bodies, for a runtime
that loads neither. This is the one command that prints plain text rather
than JSON — it is meant to be interpolated into another tool's argument.

The two files in `~/.claude/skills/` are the detail behind those rules —
when to recall, what is worth recording, which mutation verb to use. This
guide only gets it installed.

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
- **Do not leave a corpus in a directory that gets wiped.** The default
  root is the working directory, which is the right call for containment and
  the wrong one for a disposable sandbox: the memories die with it. If the
  checkout is durable — a repository you own, committed and backed up — a
  local root is a legitimate choice and scopes the memories to the project,
  the way this repository does with `./data/`. If it is not, pass `--root`
  pointing somewhere that survives, and say plainly which you did.
