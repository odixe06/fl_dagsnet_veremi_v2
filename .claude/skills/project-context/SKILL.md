---
name: project-context
description: Read a project's whole context into a session and write it back so the next session resumes cleanly — hub, lessons, live handoff, routed docs, running-job ownership, handoff update in place, milestone snapshot, link checks. Use at session start, resume or after compaction, before stopping or compacting, when moving/merging docs, or when asked to "nắm context", "đọc toàn bộ dự án", "ghi lại/bàn giao", "tổng hợp ngữ cảnh".
---

# Project context

A long project survives its context window only through two documents: the **hub** (`CONTEXT.md` or the project's
equivalent: what the project is, dated status, decisions, routing table) and one **live handoff** the hub links as current
(what is running, what just finished, next actions, manual actions). Everything else is reached by pointer.
Layout rules (which tier a fact belongs to) live in the `context-hub` skill; research-loop rules in the project's workflow skill.

`scripts/docs_links.py` (next to this file) checks every relative markdown link and moves docs while rewriting links.

## Read — session start, resume, after compaction

1. **Hub.** Read it whole: problem, dated status, decisions, routing table, next actions.
2. **Lessons files** the hub names (often split into technical and operational): read each whole; they hold the mistakes that
   cost the most.
3. **Live handoff.** Read it whole. For each row of its *running* list, confirm against the live system
   (process list, remote job status, scheduler) and note the **owner**: this session, another agent/editor session, a background
   script. Files a running job or another session owns are read-only for you until it ends.
4. **Route.** Open only the documents the next action needs, chosen through the routing table. When you will act on a number or a
   claim, read the audited artifact or results document it points to, not a summary of it. Files under an archive folder are
   **snapshots**: history, never live state.
5. **Working tree.** Look at uncommitted changes and their likely authors; another session's edits stay as they are.

Done when you can state, without reopening files: what the project is; the status and its date; every running job with its owner;
the next action and the gates/constraints on it; the files you must not touch. If any of these is unknown, keep reading or ask.

## Write — while working and before stopping

- **One live handoff, edited in place.** Update its running list when you start or finish a job, record finished work with links to
  its artifacts/results, keep next actions ordered and manual actions exact. A second "current" handoff beside it splits the truth.
- **Each fact once, in its tier:** measured results → the method's results document; reusable procedure → operations docs or a skill;
  durable lesson → the matching lessons file (technical vs operational); facts independent of the approach → knowledge; current state → handoff; a one-line pointer in
  the hub only when a session needs it to avoid a wrong action today.
- **Hub status line** carries the time you last checked it; its routing rows are phrased as the question a reader arrives with.
- **Milestone or compaction → snapshot.** Copy the handoff into the archive folder with its date in the name and a first line saying
  it is a snapshot and where live state is; then shorten the live handoff to what is still live. Machine-readable evidence of a
  milestone goes in a dated evidence folder beside the snapshots.
- **Moving or merging docs:** `python3 <skill>/scripts/docs_links.py move <old> <new> [...] --skip <file owned by a running session>`
  moves and rewrites every relative link (including links inside the moved file); then regenerate any generated index the project has.
- Agent memory files only point to repo documents; the repo stays the source of truth.

## Checks before stopping

Run all; done when each passes or its failure is reported to the owner:

1. `python3 <skill>/scripts/docs_links.py check` from the repo root — every relative markdown link resolves.
2. The hub links exactly one live handoff, and that handoff's running list matches what is actually running now.
3. Generated indexes regenerated if docs moved; no credential or token text in any document you wrote.
4. Report the files you changed. Stage, commit or push only when the owner asked for it.
