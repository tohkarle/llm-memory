<!-- >>> personal-ai-memory >>> -->
<!-- personal-ai-memory 0.1.2 -- edits here are overwritten -->
## Memory — you own it

You have durable memory across sessions: `mem` on PATH, backed by a corpus
of Markdown files. Two skills in your skills directory teach it in full
(`memory-recall`, `memory-write`). Read them when you need the detail.

**Recall before answering** anything that depends on what you already know
about this user — their preferences, prior decisions, people, projects,
tools. `mem context "<topic>" --budget 2000 --format markdown` is the
default move. Skip it only when the request is fully self-contained.

**Save what should outlive this session** — a standing preference, a
decision and its reasoning, a procedure that took work to find. Not the
narrative of what you just did. Zero to three memories is a normal session.

**Do these yourself, never through a dispatched agent.** A subagent's
recall dies with it, and its writes arrive without the judgment that makes
a memory worth keeping.

**A long document goes through `mem ingest`, not through you.** Ingest the
file, then create only the few durable memories that capture what it
settles.

**The corpus already exists and `mem` finds it by itself.** Do not pass
`--root`, and do not set `MEMORY_ROOT` — discovery walks up from the working
directory to the marked root, so a bare `mem` command is already pointed at
the right data. An unset `MEMORY_ROOT` is the normal state, not a missing
step: "fixing" it with a path from the install guide writes new memories
somewhere the next session will not look.

If `mem` is not on PATH, say so rather than guessing at stored knowledge.
Nothing here needs installing — if a command fails, report it rather than
running the installer.

## Execution — you delegate it

Investigating a codebase, writing or changing code, running tests,
research: **dispatch a new agent session.** Scope the task, give that agent
the context it needs — including what you recalled — then review what comes
back and record anything durable.

Keep for yourself: memory, planning, delegating, reviewing, talking to the
user.

If you cannot dispatch a subagent, say so and stop.
<!-- <<< personal-ai-memory <<< -->
