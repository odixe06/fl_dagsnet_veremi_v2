---
name: kaggle-training-notebook
description: Build, push, resume and collect PyTorch training notebooks on Kaggle (2x T4) -- generate notebooks and kernel metadata, stage source/checkpoint datasets, read quota and pick accounts, push per account with kaggle_as.py, continue across sessions, pull outputs, verify artifacts and report all 10 classification metrics. Use whenever work touches Kaggle kernels, quota, push/pull, session resume or Kaggle run artifacts (e.g. "push kernel", "chạy trên Kaggle", "kiểm quota", "kéo/collect kết quả", "nối tiếp session", "chạy 50 round"). Not for local-only data exploration; for training speed use federated-training-speed.
---

## Current checkout (2026-10-27)

Kaggle CLI is pinned to **2.2.5** in `environment.yml`. For a historical run, use
`<owner>/<slug>/<version>` consistently for status, files, output and source pulls.
The output helper preserves that version across every listing page. CLI 2.2.5
streams/resumes downloads; the helper still traverses all pages automatically and
checks missing files. Verify expected artifacts after either path.
`kernels push --no-run` saves a version without execution; it does not cancel a worker.

Read repository-root `knowledge/local-env.md` for the measured local hardware/runtime. Short tasks run locally; multi-hour tasks run on Kaggle. The current checkout is `/home/odixe/project/fl_dagsnet_veremi_v2`; use `nckh`.
Choose an account by live quota and available GPU session slots, not a fixed default. Use process-scoped credentials and leave the global login unchanged. Read `references/kaggle-credentials.md` before using token-only accounts; Quota/health/plan support token-only accounts; global OAuth switches still require snapshots. Environment setup does not authorize a new GPU launch.

## Paths

This file is the skill. Resolve every relative path below from this directory: helper scripts
are in `scripts/` next to it (`kaggle_account.py`, `kaggle_as.py`, `embed_wandb_key.py`,
`merge_sessions.py`, …) and the supporting notes in `references/`. Paths labelled as
repository-root paths belong to the calling project, not here.

Read the calling project's `CONTEXT.md` for its decisions, account roster, run status and
outstanding work. Nothing in this skill is project state: the historical build numbers,
accounts and results in the references are evidence from past runs, never authorization or
defaults for the project you are in.

# Kaggle training notebooks

Read the calling project's agent instructions (CLAUDE.md / AGENTS.md), its paper, architecture specification and dataset documentation.
Use `conda run -n nckh` for Python and Kaggle CLI in this repository. State assumptions and
verifiable success criteria. User decisions in this session govern the work.
Read current `CONTEXT.md` for approved settings, account policy and unresolved review findings.
Keep any mirrored copy of this skill in another tree synchronized instead of maintaining a second workflow.
Paths under `references/` and `scripts/` resolve from the skill directory unless labelled
repository-root paths. Keep mirrored references/helpers synchronized when modifying them.

Historical build numbers, accounts, W&B credentials and exceptions in supporting
references are evidence from other runs, not authorization or defaults for this project.
Check referenced paths/scripts exist. Never infer completed training from copied documentation.

For two training seeds, create one notebook/session per seed and push both in the same batch
so they run concurrently. Use available account/session capacity; preserve scientific settings
and distinct run IDs. Check the project context for selected seeds and account ownership.

**Where a job runs.** Estimate its runtime before starting it. The local machine takes short, light work: code tests,
one-client smoke runs, feasibility probes of a few minutes. Anything longer than roughly 30-60 minutes, multi-hour diagnostics
included, goes to Kaggle as its own notebook; so does any job when the local machine is already busy (backup transfers, other
projects). A long job started locally competes for RAM with everything else and stalls both.

## Route

1. Read [kaggle-mcp.md](references/kaggle-mcp.md), discover callable tools and verify an
   account-scoped read such as quota. Public dataset metadata does not authenticate the account.
   Record MCP and CLI results separately. Missing MCP does not prevent local audits or skill
   edits; respect a user requirement to verify MCP before building models. Never print tokens.
   Before requesting credentials, inspect the user-designated ~/.kaggle/ directory:
   credentials.json identifies the active CLI account; accounts/<username>.mcp-token
   stores its MCP bearer. Follow [saved credentials](references/kaggle-credentials.md).
   Never print tokens, silently switch accounts, or overwrite OAuth credentials with a
   bearer. The current Codex header helper does not require an environment export.
2. Extract the method and unresolved choices into `papers/<slug>/paper.md` and `rebuild.md`;
   see [paper-dossier.md](references/paper-dossier.md). Do not invent missing equations,
   optimizer settings or adaptive rules and attribute them to the paper. Resolve scientific
   choices with the user before implementing dependent training code when requested.
3. For any FL method on VeReMi/DAGSNet read [fl-veremi-dagsnet.md](references/fl-veremi-dagsnet.md)
   FIRST: it names the template to copy (GitHub `odixe06/fl_dagsnet_veremi`: `lwfednids/` for a
   global-model method, `perfedskd/` for per-client models), the measured T4 numbers, the CUDA-graph
   traps of multi-module steps, the sharded / cached eval, and the push/pull command sequence,
   so no sibling project has to be re-read.
   For FL read [federated-contract.md](references/federated-contract.md) before using any DDP template.
   **Before optimizing speed (workers, compile, fused kernels, host syncs) use the
   `federated-training-speed` skill**: it measures whether the loop is launch-bound, names the
   acceptance class (bit-exact or tolerance) and the equivalence tests, and holds the T4 playbooks.
   Audit actual data using [dataset-audit.md](references/dataset-audit.md) and
   [veremi-dataset-layout.md](references/veremi-dataset-layout.md). Distinguish full scans,
   samples and historical sidecar counts. Freeze feature order, class mapping and train-fitted
   preprocessing. Never standardize already standardized training data a second time.
4. Read [library-runtime.md](references/library-runtime.md) BEFORE writing or pushing a
   notebook: pin `docker_image` by digest, declare the accelerator in the notebook document as
   well as in `kernel-metadata.json`, and never reinstall torch on a Kaggle GPU image. The
   library stack on the worker is not the one you developed against, and Kaggle reports none of
   the differences as an error.
   Build using [notebook-template.md](references/notebook-template.md),
   [metrics.md](references/metrics.md), the `federated-training-speed` skill for performance,
   [long-run-stability.md](references/long-run-stability.md), and
   [observability.md](references/observability.md) as relevant.
   Historical speedups are not measurements of the new model. W&B is optional unless requested;
   read [wandb.md](references/wandb.md) only when using it. The owner of this environment has
   a standing authorization to embed the W&B key in their own `is_private: true` notebooks
   (wandb.md §1); state the permanence cost when you use it, and §1.1 when asked to undo it.
5. Test the exact notebook's generated modules: real-data forward/backward, aggregation,
   all metrics/artifacts, strict model reconstruction, interrupted-round recovery and next-round
   resume. Read [verifying-artifacts.md](references/verifying-artifacts.md) before trusting any
   verifier: checks built on freshly generated random inputs test the formula, not the run, and
   pass on tampered artifacts. It also carries the crash-safe write ordering, the atomic-import
   rules and the data-identity fingerprint that recovery depends on. Small fixtures can verify arithmetic/crash behavior; benchmarks use real data.
   State limits of one-GPU local checks for two-GPU execution.
6. Verify notebook metadata: correct code_file, kernel_type notebook, owner, dataset sources,
   and explicit is_private true. Request machine_shape NvidiaTeslaT4 and enable_gpu true;
   assert exactly two Tesla T4 devices with capability (7, 5) inside the notebook before train.
   Also set `nb.metadata["kaggle"]` in the notebook document and a pinned `docker_image`
   (see [library-runtime.md](references/library-runtime.md) §1-§2): notebooks that carried only
   the JSON metadata came up with zero GPUs. Zero GPUs can be account-scoped -- re-run unchanged
   on another account before editing code, and never respond by removing the assert.
   Read current quota/session limits. Inspect saved accounts with
   [kaggle_account.py](scripts/kaggle_account.py) `list`, `quota`, `health` and `plan`, none of
   which switch accounts. Size the whole training plan with `plan --gpu-hours`, not one session
   at a time, and read [multi-account.md](references/multi-account.md) before `ensure` or `use`.
   Under the **2026-09-10** policy the agent runs those two with `--confirm` itself, without
   asking; announce which account and why. The user's only account action is reconnecting
   the MCP client when a reconnect is actually needed. Budget free hours after used and
   reserved time, including all planned simultaneous runs. Reconnect MCP after a switch and verify
   alignment; use selected CLI OAuth while MCP identity remains uncertain. An account with quota
   is not an account with GPUs -- zero-GPU allocation has been account-scoped here before, and
   only a run proves it. Script creation does not itself authorize launching runs.
7. Pull bulk output with `scripts/pull_kernel_output.py` through `scripts/kaggle_as.py <owner>`
   (parallel, resumable, all-page listing, non-zero exit on gaps; one process per kernel, ~32 connections
   in total); account credentials stay process-scoped. Require exit 0 before artifact verification.
   For operation/recovery read [pull-outputs.md](references/pull-outputs.md),
   [stop-session.md](references/stop-session.md), and only if needed
   [multi-account.md](references/multi-account.md). Distinguish latest batch-version status,
   active session and fresh heartbeat. A replacement push does not prove cancellation.
   Recheck dated API limitations before asserting they still apply.
8. Build reports from verified artifacts with [reporting.md](references/reporting.md).
   A verifier certifies only the rounds that exist: require its explicit completion gate before
   any "the scenario finished" claim (see [verifying-artifacts.md](references/verifying-artifacts.md) §6).

Local tooling note: `conda run -n <env> python - <<'PY' ... PY` **discards the script's stdout**,
so an edit script that reports success prints nothing and a failed assertion looks like silence.
Write helper scripts to a file and run that file, or use the system `python3` for edits that need
no environment packages.
   Include all 10 metrics for every reported round; never fabricate results.

## Scientific and artifact contracts

- Confirm which model(s) are evaluated. Shared client/server architecture does not mean shared
  post-training weights. Global test scores are not personalized performance.
- Unless the user requests another schema, use this ordered METRIC_KEYS list:
  accuracy, precision_macro, precision_micro, precision_weighted, recall_macro,
  recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted.
  This differs from the paper's binary metric list. Resolve ambiguity explicitly.
- Evaluate the entire fixed test after every completed round without padding/dropped tails or
  test-driven adaptation. Compute metrics from full confusion counts, not batch metric averages.
  Save JSON, CSV, confusion matrices, per-class results, y_true and y_pred. Probabilities are
  optional and require a disk estimate. Keep all 10 columns even when values coincide.
- **A per-round checkpoint stores weights, not a model.** Write a `state_dict` of tensors
  including BatchNorm buffers; never a pickled `nn.Module`, and never the optimizer, scaler or
  scheduler state in the same file. Keep them as `weights/round_NNN.pt` and put the resume
  bundle in a separate `resume/round_NNN.pt`, so retaining every round costs weights alone.
  For DAGSNet that is 1.5 MB a round instead of 4.7 MB — and the weights file stays the thing
  a reader can load without trusting a pickle.
- **Weights are only a checkpoint if current code can rebuild the model at them.** Ship the
  `build_model(cfg)` used by the run, persist the architecture config, feature order, class
  names and scaler beside the weights, and make the rebuild path prove what it restored:
  `torch.load(..., weights_only=True)`, `load_state_dict(strict=True)`, an explicit
  parameter-count assert, and the config fingerprint. `strict=True` is what catches a filtered
  `running_mean`/`running_var`, which otherwise leaves `eval()` normalizing by 0/1 and reports
  no error at all. Exercise that path against a file you actually wrote, in the same test run
  that wrote it; a checkpoint no code reconstructs is not a checkpoint.
- **`weights_only=True` holds only if the file contains tensors and plain types.** One numpy
  RNG tuple, `Path`, or custom object anywhere in the dict forces every future reader back to
  `weights_only=False` — a code-execution path. Keep those in the resume bundle. Verify by
  loading with `weights_only=True` in the test, not by inspecting the dict.
- Resume needs the round, algorithm state, RNG and any persistent optimizer/scaler/scheduler
  states. If local optimizers deliberately reset every round, document this and omit their
  discarded state. Weights alone do not support arbitrary exact resume.
- Commit complete rounds atomically, completion marker last -- and the resume state must be
  inside that atomic unit, keyed by round and written before the marker. A latest-only
  resume_state written after the marker disagrees with it whenever a crash lands in between.
  Resume from the last complete round.
  Keep weights and metrics for every completed round. A crash during eval must not advance last
  or leave duplicate history rows. Estimate storage before launch.
- Persist effective config.json and a fingerprint covering model, preprocessing and scientific
  settings. Validate on resume. Search working output then unique attached outputs; require_resume
  must fail if absent. New sessions wipe working files. Notebook outputs use kernel_sources;
  datasets use dataset_sources.
- Flush logs and guard finite values/AMP skipped steps. Reject invalid client updates before
  aggregation. Finite weights are not enough: a client whose AMP steps were all skipped
  returns the global weights unchanged and finite. Report applied steps separately, compute
  loss/gradient means over applied steps only, and allow an absolute warm-up skip budget so a
  healthy client is not failed for the scaler calibrating itself. Do not silently discard
  failed clients or overwrite valid weights with NaNs -- dropping one changes the
  participation rate the specification fixed. Stop before the measured session budget,
  leaving time to finalize artifacts, and budget on the worst round observed including
  commit I/O rather than the last one.
- A worker killed by the OS puts nothing on the queue. Poll the result queue in short slices
  and check process liveness between them, or a single long timeout turns an OOM kill into a
  multi-hour hang.
- Cross-session resume needs an explicit import step and a require-resume gate that fails
  *before* the data decode; copy the previous output with `copyfile`, since a read-only
  mount's permissions otherwise carry across and break the first rewrite.
- Centralized DDP rules apply when ranks optimize one logical model: equal collective counts,
  device selection before process-group setup, appropriate NCCL settings. Different FL clients
  must not synchronize gradients with one another.

## Completion evidence

Report separately: account-authenticated MCP, measured dataset coverage, local verification,
remote two-GPU verification, training completion and pulled results. One does not prove another.
Name remaining manual actions with exact steps and observed failure. Complete independent
authorized work before handing back; do not describe a partial scaffold as ready to train.
