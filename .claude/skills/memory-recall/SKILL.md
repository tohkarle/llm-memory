---
name: memory-recall
description: Look up what is already known about the user before answering. Use when the request touches their preferences, history, prior decisions, people, projects, or tools; at the start of any task where prior context would change the answer; and whenever they say "remember when", "as I mentioned", "my usual", "like last time", or refer to something as already settled.
---

# Recalling memory

**Decide:**
- About to answer using prior knowledge → `context` (one call, budgeted, paste-ready)
- Need specific memories → `search`
- Have an id, need full text or history → `get` / `related`

All commands: `uv run python -m memory <cmd>` from the project, or `mem <cmd>`
if installed. stdout is one JSON document; `--root PATH` or `$MEMORY_ROOT`
selects the data directory.

## context — the default

```bash
mem context "python packaging preferences" --budget 2000 --format markdown
```

Returns profile memories + keyword hits + one hop of relations, under budget.
Each block carries `<!-- id:… score:… reason:… -->`; cite the id, then
`mem get <id>` for untruncated text.

`--format json` gives `{tokens_used, dropped, truncated[], blocks[]}`.
**`dropped > 0` means raise `--budget` or narrow the query — do not retry
blind.** Other flags: `--depth 0` (no relation expansion), `--no-profile`,
`--type preference`.

## search — targeted lookup

```bash
mem search "uv"                                   # keyword
mem search --tag tooling --type preference        # metadata only, no query
mem search "uv" --status superseded               # history
mem search "fts:uv OR pip OR packaging"           # OR-expansion
mem related <id> --depth 2                        # graph walk
mem get <id> --related --follow                   # full text + neighbours
```

Returns `{total, hits[{memory_id, title, type, score, snippet, tags}]}`.

## What agents get wrong

**Matching is lexical, not semantic.** Multi-word queries are treated as a
bag of words — stopwords dropped, terms ORed and prefix-matched, ranked by
how many hit — so "how do I rebuild the index" does find "How to rebuild the
memory index". But a query sharing *no* vocabulary with a memory still
misses: "what timezone" will not find "Works in Pacific time". If a search
comes back thin, retry with the user's own words, or `fts:` with synonyms.
Do not conclude nothing is recorded from one miss. Profile memories are
exempt: they are always in `context` regardless of the query.

**Superseded and deleted memories are hidden by default.** That is deliberate
— a correction should not resurface beside the thing it replaced. Pass
`--status superseded` to read history; `mem get <id> --follow` jumps from any
version to the current one.

**`secret` memories need `--include-secret`.** Absence is not proof of
absence.

**A `contradicts` relation means the memory is disputed.** `context` pulls
those in on purpose. Say so rather than asserting one side.

Confidence and salience are the author's calibration: `confidence` is how
sure, `salience` is how much it matters. Low confidence is worth hedging.

## Errors

Exit `0` ok, `1` not found, `2` bad arguments, `3` conflict, `4` locked
(retry), `5` corrupt index (`mem rebuild`), `70` internal. Failures still
print JSON: `{"ok": false, "error": {"code", "message", "details"}}`.
