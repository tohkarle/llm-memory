---
name: memory-write
description: Record, correct, or remove durable memories about the user. Use when they state a standing preference or fact, make a decision worth keeping, describe a reusable procedure, correct something recorded earlier, or say "remember this", "forget that", "I changed my mind". Also after substantial work whose outcome should outlive the conversation.
---

# Writing memory

**Decide:**
- New durable statement → `create`
- Same meaning, better wording/metadata → `update`
- Meaning changed → `replace`
- Duplicates found → `merge`
- Wrong or unwanted → `delete`

## Pick the type deliberately

Everything defaults to `fact` if you are not thinking. Don't.

| type | what it is | test |
|---|---|---|
| `preference` | a standing choice | "I prefer / I always / I'd rather" |
| `fact` | a stable truth about them or their world | true independent of time |
| `event` | something that happened | has a date; set `--occurred-at` |
| `decision` | a choice plus its reasoning | "we decided X because Y" |
| `procedure` | how to do something | steps someone could follow |

## Which verb

| situation | verb | why |
|---|---|---|
| typo, new tag, confidence tweak | `update` | same id, no history needed |
| they changed their mind | `replace` | old version kept, searchable as history |
| two memories say the same thing | `merge` | unions sources; provenance survives |
| recorded in error, or they ask to forget | `delete` | soft by default, recoverable |
| they want it *gone* | `delete --permanent --yes` | irreversible; blocks re-adding |

`update` cannot change `type` — that changes what the memory is, so use
`replace`. Use `--permanent` only on an explicit request; soft delete is the
default for a reason.

## Writing well

```bash
mem create --type preference --title "Prefers uv over pip for Python projects" \
  --body "Uses uv for every project; dislikes pip's resolver." \
  --tag tooling --entity uv --confidence 0.9 --profile \
  --source conversation:conversations/2026-10-15.md:L120-L142
```

- **Title is a standalone assertion**, not a topic. "Prefers uv over pip" —
  not "Python tooling". It is what search matches and what a context pack
  shows first.
- **Always `--source`** when one exists: `kind:ref[:locator]`, kinds
  `conversation|document|asset|manual|import`. Provenance is what lets a
  claim be checked later.
- **`--profile` only for identity-level facts** — the handful that should
  inform every conversation. These bypass keyword matching entirely, so each
  one spends budget on every recall. A dozen is a lot.
- **Calibrate `--confidence`**: 0.9+ stated outright, ~0.7 inferred, ~0.5
  guessed. **`--salience`** is importance, independently.
- `--sensitivity private|secret` for personal material; `secret` is excluded
  from search unless explicitly requested.
- `--review-after 2027-01-01` for anything that will go stale.

## Anti-patterns

- **Search first.** `mem search` before creating, or you add a near-duplicate
  someone has to merge later.
- **No ephemera.** What they asked you to do just now is not a memory. Record
  what will still be true next month.
- **No restating the obvious.** Nothing the code, README, or git history
  already says.
- **One claim per memory.** Two claims in one body cannot be superseded
  independently.

## Corrections and removal

```bash
mem replace <id> --title "Prefers uv, pinned to Python 3.12" --body "…" --reason "narrowed"
mem merge <id-a> <id-b> --title "Testing approach"     # new id, parents kept
mem merge <id-a> <id-b> --into <id-a>                  # keep a's id
mem delete <id> --reason "recorded in error"           # reversible
mem delete <id> --permanent --yes --reason "user asked"
```

**Exit 3 with `"code": "tombstoned"`** means this content was permanently
deleted before. That is the system refusing to resurrect something the user
removed. Do not retry with `--force-recreate` unless they explicitly ask for
it again — tell them it was previously deleted and let them decide.

## Finding duplicates before you add more

```bash
mem consolidate                                   # proposals, writes nothing
mem consolidate --apply --yes                     # execute duplicate merges only
```

Cheaper than `search` for a cleanup pass. `--apply` merges exact/high-overlap
duplicates; `contradiction` and `stale` stay advisory because a replacement
needs content only you have. One merge per pass per memory — rerun to cascade.

## Extraction

```bash
mem extract --file conversations/x.md --source conversation:conversations/x.md
mem extract --stdin --source conversation:x.md --commit --min-confidence 0.7
```

`--source` is required: a candidate with no provenance is refused. Review
before `--commit`; candidates carry `duplicate_of` and a `rationale`.
Confidence 0.5 = pattern-matched, verify it. `--llm` uses a configured
provider (`mem provider show`); without one, the deterministic path runs
anyway.

## Deleting source material

```bash
mem source plan-delete conversations/x.md         # full blast radius
mem source delete conversations/x.md --yes        # orphans citing memories
mem source delete conversations/x.md --memories purge --yes
```

Removes transcripts, segments, derivations and assets too. Default `orphan`
keeps memories and tags them `provenance-lost`; `purge` writes tombstones so
they can't be re-extracted. **Always plan first** — it is irreversible.

## Backup and export

```bash
mem backup create --label "before cleanup"
mem export --out mem.jsonl        &&  mem import mem.jsonl
mem backup prune --keep-days 90 --keep-count 10 --yes
```

Archives exclude the index (rebuildable) and always include tombstones.
Import is additive and **cannot resurrect a purged memory** — exporting first
is not a way around a deletion. Before any bulk operation, back up.

## Attachments

```bash
mem asset add ./paper.pdf --link-to <memory-id> --tag research
```

Stores the file content-addressed and attaches it. The asset id is stable
across `mem asset replace`, so links survive new versions.

## After writing

Nothing to flush — the file is the source of truth and the index updates in
the same operation. If search looks stale, `mem rebuild` reconstructs it from
the files; `mem verify` reports any divergence.
