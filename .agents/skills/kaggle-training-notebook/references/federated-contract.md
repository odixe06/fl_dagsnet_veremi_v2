# Federated training on VeReMi / DAGSNet — the scientific contract

Generic rules for any FL method whose clients train the DAGSNet classifier on the α = 0.5
VeReMi partitions and are scored on the fixed 10,761,343-row test. Method-specific choices
(loss, aggregation weights, schedules, participation) belong to the project's `docs/REBUILD.md`
and are never taken from here. User decisions override anything below.

## Sources every project shares

- Model: `knowledge/architecture.md` — standalone DAGSNet, 66 features, 16 logits, 395,024
  learnable parameters, PyTorch default initialization, seed 42. Fresh initialization unless
  the project says otherwise.
- Train: local data root must be restored/configured and checked against `knowledge/dataset.md`;
  Kaggle `trietbackup/veremi-fl-20client`, `minhtriethihi/veremi-fl-50client`,
  `khanhmay0304/veremi-fl-100client` (public),
  relative roots `{20,50,100}_client/`.
  Resolve unique runtime sentinels, not a fixed mount prefix.
- Fixed test: restore/configure the local fixed-test root using `knowledge/dataset.md`; Kaggle
  `trietbackup/veremi-nextgen2026-test` (test only), `upload/test/` + `upload/scaler.json`.
- Train is standardized; test is raw. Reuse the matching `scaler.json`, feature order and
  labels; verify against `knowledge/meta.json`. Only `f_*` columns enter the model.
- 43,045,415 train rows per scenario and 10,761,343 test rows, 16 classes, Dirichlet α = 0.5
  (`knowledge/dataset.md`). Disclose global train-scaler access, receiver-unit clients and the
  dataset caveats in every report.

## What the project must fix before training

Rounds, local epochs, batch per client, participation, the local objective, the aggregation
rule (weights, which buffers, `num_batches_tracked` = max), the optimizer and its persistence
across rounds, the LR schedule, gradient clipping, AMP policy, and what is evaluated (the
post-aggregation global model, per-client models, or both) — with the skill's 10 metrics on
the full fixed test. Write them in `docs/REBUILD.md` as implementation choices where the paper
leaves them open; never attribute a completion to the paper.

## Two GPUs

Prefer two persistent spawned processes, one independent client task per GPU at a time.
Reuse model allocations and schedule large clients early. Reset/load states as the FL spec
requires. Never use one DDP process group to synchronize different clients' gradients.
Per-client batch does not double because two GPUs exist.

Keep the features resident on both GPUs when measured headroom allows (train 5.29 GiB +
test 1.32 GiB in fp16 fit a T4 next to the model); otherwise stream bounded batches. Avoid
host-RAM copies of all clients plus test. Global eval may use disjoint deterministic test
shards on both GPUs: sum integer confusion counts and restore prediction order. Rank-0-only
eval in the centralized template is not a scientific FL requirement.

Use fp16 AMP with scaling on T4 (never bf16: sm_75), loss and any regularizer in fp32.
Validate eager versus compile on this model; preserve weights, BN buffers and RNG around
probes. Do not transfer historical timing claims to a new model or treat spare VRAM as proof
of idle compute. Never use test metrics for adaptation or let GPU task completion order
determine floating-point aggregation order.

Keep global `state_dict` weights every round. Keep client states only when the chosen
algorithm requires persistence or the user requests them. If local optimizers reset each
round, boundary resume may omit their discarded state, but still needs the round and any
algorithm state.

Monitoring: stream progress to W&B (see `wandb.md`) alongside the committed artifacts. W&B
holds scalars only — every reported number still needs a pulled artifact.
