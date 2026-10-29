---
name: context-hub
description: Lay out and maintain a project's documentation as a navigation hub (CONTEXT.md) plus docs/ and knowledge/ tiers, so a session that starts cold is oriented by one file and reaches the rest by pointer. Use when creating or reorganizing CONTEXT.md, a handoff or onboarding doc; when the hub has grown into tables of numbers; when deciding where a measurement, decision or account roster belongs; or when a shared skill has started accumulating one project's specifics (e.g. "tổ chức lại tài liệu", "CONTEXT quá dài", "đặt thông tin này ở đâu").
---

# Context hub

Work that spans many sessions dies of one failure: the next session cannot tell **what is
settled** from **what is still open**, or **which file to trust**. Documentation is how the
project survives its own context window.

The cure is a **hub**: one file that orients and routes, and nothing else. Every other document
sits behind a pointer from it.

## The four tiers

Each tier answers a different question, and each exists to stop a specific failure:

| tier | holds | the failure it prevents |
|---|---|---|
| `CONTEXT.md` — **the hub** | what the project is · current state · what is settled · what to open for anything else | a cold session guessing which file is authoritative |
| `docs/` | the project's own documents: method, decisions, measurements, operations | numbers buried in a file nobody re-reads |
| `knowledge/` (or equivalent) | facts that **do not change when the approach changes**: the data, the machine, the fixed architecture | one approach's chosen constant being read next session as a measured property of the world |
| shared skills | procedure reusable across projects | a skill that silently encodes one project's accounts, paths and dates |

The `knowledge/` boundary is the one that bites. A value that a *build* chose (a coefficient, a
batch size, a threshold) placed among *measured* facts will be read as ground truth by a later
session building something else — and nobody re-checks it, because it sits where checked facts
live. Keep chosen values in `docs/`, beside the reasoning that chose them.

## The hub's contract

**Belongs in the hub**

- What the project is: one paragraph a stranger can follow.
- Status: what works, what is running, what is blocked — dated.
- A **routing table**: `want to know X → open Y`. This is the hub's reason to exist.
- The decisions that change what someone does *today*, each one line, each pointing at the
  document carrying the reasoning.
- Whatever will look like a bug but is not — the predicted-and-explained surprises. A session
  that stops a healthy run has lost more than one that reads a stale table.
- Next actions, ordered.
- Where a fact goes (the tier table above), so the next session extends the layout instead of
  eroding it.

**Belongs one tier down**

Long result tables · every check a test performs · full command sequences · credentials and
quota detail · the complete list of deviations · anything a reader consults rather than reads.

**The test**: a hub someone reads top to bottom in about two minutes, and leaves knowing what the
project is, what state it is in, and which file to open next. Growing past that means a section
is due to move down, not that the hub needs trimming.

## Routing table

One row per question a future session will actually have. Phrase rows as the **question**, not
the filename — the reader arrives with a question, not a path.

```markdown
| want to know | open |
|---|---|
| every number measured, and what each test proves | `docs/tests.md` |
| why the build deviates from the source, and where | `docs/REBUILD.md` |
| how to run, monitor and recover a job | `docs/OPERATIONS.md` |
```

Give every document in the table a **single subject**, and let the filename say it. Split when a
file starts answering two unrelated questions; merge when two files are always opened together.
Four to six documents is the usual landing zone: fewer and the hub is doing their job, more and
the routing table becomes the thing that needs an index.

## Where a fact belongs

Ask, in order:

1. **Would this still be true if the approach changed?** Yes → `knowledge/`. No → `docs/`.
2. **Is it a number, a table, or a command sequence?** Yes → `docs/`, never the hub.
3. **Does a session need it to avoid a wrong action today?** Yes → one line in the hub *and* the
   full version in `docs/`. This is the one sanctioned duplication, and the hub's line must be
   the pointer, never the copy.
4. **Would another project use it unchanged?** Yes → a shared skill. Anything naming this
   project, its accounts, its dates or its paths → `docs/`, however tempting the skill looks.

## Keeping it honest

Run these whenever the docs change; each catches a failure that is otherwise silent:

- **Every relative link resolves.** A moved file leaves a dead pointer, and a dead pointer sends
  the next session to guess.
- **Each number lives in exactly one document.** Two copies drift, and the reader cannot tell
  which drifted.
- **No shared skill names this project.** Grep it for the project name, the account names and
  the dates.
- **The status line is dated**, and the date is the day it was last checked — not the day it was
  written.

```bash
python3 - <<'PY'
import re
from pathlib import Path
bad = [f"{md}: {t}" for md in list(Path(".").glob("*.md")) + list(Path(".").glob("docs/*.md"))
       for t in re.findall(r"\[[^\]]*\]\(([^)#]+)\)", md.read_text())
       if not t.startswith(("http", "mailto")) and not (md.parent / t).resolve().exists()]
print("\n".join(bad) if bad else "all relative markdown links resolve")
PY
```

## Language

Write each document in the language the project's owner works in, and keep shared skills in the
language the rest of the skill tree uses. A hub the owner skims quickly is worth more than one
that matches the skills around it.
