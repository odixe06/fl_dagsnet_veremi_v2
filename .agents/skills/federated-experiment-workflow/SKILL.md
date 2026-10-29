---
name: federated-experiment-workflow
description: "Resume federated-learning research sessions and run the experiment loop around Kaggle builds: read the project hub, lessons and current handoff, inventory running and finished sessions, collect and audit completed runs against locked gates, record problem evidence → mechanism → falsifiable sign before proposing a method, and update per-method design/results/lessons and the handoff. Use at session start or after a summary/compaction, when collecting or auditing results, locking gates, proposing or comparing methods, or updating research docs (e.g. \"đọc CONTEXT\", \"bàn giao\", \"collect/audit kết quả\", \"đề xuất phương pháp\", \"cập nhật results/design\"). Does not authorize training launches, backup, deletion, Git or DVC pushes."
---

# Federated experiment workflow

Read the repository's `CONTEXT.md`, then `LESSONS.md`, then the linked current handoff. Later owner decisions override
older documents. Resolve uncertain scientific choices before dependent training; continue independent authorized work.

## Choose the branch

- **Collect/analyze:** use the adjacent Kaggle training skill, exact source/config/round audit and locked gates.
  Status COMPLETE/ERROR and W&B scalars are operational evidence, not verified scientific results.
- **Continue training:** require current launch authorization; use the training skill for quota, immutable notebooks,
  input access, online W&B and parent import. Read the project's kernel-error register before generation.
- **Organize/document:** use [references/build-layout.md](references/build-layout.md). Preserve all frozen bytes and
  remote-only DVC history; document physical moves and compatibility aliases. Do not infer permission for Git/DVC push.

## Evidence and decision

1. Inventory completed and active exact notebook versions. Check persisted markers/processes after crash; do not restart
   a collector whose outputs already passed audit. Restore only required missing DVC evidence.
2. Audit executed redacted source, fingerprints, data identity, committed rounds, weights/BN/resume, payloads,
   predictions → per-client confusion → all ten metrics. Distinguish bounded inference from whole-dataset proof.
3. Evaluate the prelocked primary rule, paired controls/seeds, utility/cost/byte gates. Identify what a failure actually
   isolates. Do not extend/tune a failed adaptation to find a win or select HP from official test.
4. Before a new improvement, record **problem evidence → mechanism → expected performance gain → falsifiable sign**.
   Source mathematical/codec feasibility alone does not establish classifier utility or finite-contact utility.
5. Update per-method design/results/lessons; one proposal catalog records tested/PASS/FAIL/pending with exact scope.
   `LESSONS.md` stores common training insights; `CONTEXT.md` stores current runtime warnings and next actions.

## Authorized continuation

Match each arm to its last **audited committed** round, source/config/data fingerprint and compatible parent.
Keep the originally locked total-round LR schedule. Import/check full parent state before expensive decode;
publish completion markers last. Never start r1 when require-resume fails.

Budget the whole staged batch from live owner-scoped quota, worst round, setup/import/compile/eval/artifact I/O and
the owner's margin. A partial quota batch is not the whole target completed. Verify actual input mounts; an output
listing does not prove Kaggle accepts a private ERROR kernel as `kernel_sources`. Use an authorized private checkpoint
dataset when required, preserving hashes and tabular bytes. A refused-input/lost push reply requires inspection before retry.

Follow current owner policy for credentials, private history and initial health checks. Do not print secrets.
W&B resume is not checkpoint resume; preflight must not advance the restored logging step. Stable runs need no repeated
health polling when the owner selected that policy. Report observed instability and the required owner stop action.

## Finish a milestone

Refresh the overview/file index, validate links and ignored credentials/artifacts, compare frozen SHA and DVC object
identities, and check Git HEAD/index unchanged unless Git work was requested. Save kernel IDs/versions, source/config,
verified rounds/results, quota and remaining work in the current handoff. State exact manual actions only when needed.

Repository tools are discovered through `scripts/README.md` and the project index. Read the current layout rather than
hardcoding historical build paths, owners, quotas or scientific constants into this skill.
