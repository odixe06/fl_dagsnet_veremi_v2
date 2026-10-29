# Rebuilding a federated method on VeReMi / DAGSNet — the procedure that already works

Read this before opening a sibling project for "how did the last rebuild do it". Everything
below was measured on 2×T4 or verified by a test in one of the rebuilds kept in the GitHub repo
`odixe06/fl_dagsnet_veremi` (clone it; the local sibling folders no longer exist); the paths name the copy to take.
Speed work (workers, compile, equivalence tests): the `federated-training-speed` skill. It covers **any** FL method whose clients train DAGSNet on the
α = 0.5 partitions and are scored on the fixed 10,761,343-row test.

## 1. Start from a template, not from scratch

Two generations exist, both with the same module layout, worker protocol, artifact contract
and test suite. Copy the one closer to the method and change only the method module:

| method shape | template | what it already handles |
|---|---|---|
| **global model** (FedAvg-like, one θ^t broadcast per round, one eval per round) | GitHub `odixe06/fl_dagsnet_veremi`: `lwfednids/papers/lwfednids-bouayad-2024/proj/` | eval of one model sharded across the two GPUs; a model whose architecture is derived from cfg (pruning plan in `prune.py`, rebuilt without the pruning library); `n_params` from cfg instead of a constant |
| **per-client models** (personalized θ_m, partial participation, N + 1 evals) | same repo: `perfedskd/papers/perfedskd-singh-2025/proj/` | resident per-client weight table on both workers, selection state in the checkpoint, per-client eval pinned to worker c % W |

| file | reuse | what changes per method |
|---|---|---|
| `model.py` | verbatim (DAGSNet, `build_model`) | only if the method changes the architecture |
| `data.py`, `metrics.py`, `evaluate.py` | verbatim | nothing |
| `<method>.py` | keep `layout/flatten/unflatten_into`, `amp`, `lr_at`, the `client_update` step loop with AMP skip accounting | the loss / update / aggregation formulas |
| `driver.py` | keep the worker protocol, gates, commit, budget | which state a worker holds, what `check_updates` rejects, what is evaluated |
| `ckpt.py` | keep everything; edit `FINGERPRINT_KEYS` and the weights-file schema | the keys that define the method |
| `verify.py` | keep the structure | per-artifact checks for the new schema |
| `scripts/gen_notebook.py`, `validate_notebooks.py`, `run_local_checked.py` | copy; edit the CFG block, `MODULES`, expected CFG values, `run_name` regex | |
| `tests/` | copy `test_units`, `test_smoke_real`, `test_two_workers`, `test_compile_gate`, `test_gpu_local`, `test_ckpt_verify`, `test_notebook_sim` and rewrite the method-specific assertions | |

## 2. The parts that must not be re-derived

* **Data path.** Parquet → resident fp16 tensors once per session (`data.load_clients` /
  `load_test`, cache under `/kaggle/temp`, manifest written LAST, `cache_ok` re-validated).
  train/ is already z-scored, test/ is raw and gets `scaler.json` exactly once. Both GPUs
  hold the whole train (5.29 GiB) and test (1.32 GiB). List parquet parts explicitly:
  the centralized test ships `.stats.json` sidecars that kill `ds.dataset(dir)`.
* **Mounts.** Never hard-code `/kaggle/input/<slug>`; resolve by sentinel
  (`find_root("train/client_id=000")`, `find_root("upload/test")`). Measured layout is
  `/kaggle/input/datasets/<owner>/<slug>/...`.
* **Workers.** One persistent process per GPU, spawned once; tasks by id over
  `mp.Queue`; payloads as **numpy** (torch tensors on the queue go through `/dev/shm`);
  LPT dispatch (longest client first); aggregation re-sorted by client id; every client
  re-seeds from `(seed, round, client)` so 1 worker ≡ 2 workers bit for bit
  (`tests/test_two_workers.py`). Never a process group between clients.
* **Compile.** `torch.compile(mode="reduce-overhead")` on the module, loss and optimizer
  outside the graph, tail batch on the eager module, weights swapped with `copy_`. Gate
  it on real rows with dropout off on BOTH sides and a decisive-row argmax rule
  (`driver._compile_train` / `_compile_eval`); fall back to eager loudly and publish the
  backend to W&B summary. Two traps of multi-module steps: call
  `torch.compiler.cudagraph_mark_step_begin()` at the top of every step when more than
  one captured graph runs per step (else an illegal memory access that only appears
  without `CUDA_LAUNCH_BLOCKING`), and raise `torch._dynamo.config.recompile_limit`
  when several module instances share one `forward` code object (past the default 8 the
  next variant silently runs eager).
* **Eval.** Fold BatchNorm into the convs (exact), load the weights into ONE compiled
  template, accumulate `bincount(y*C + pred)` on device, count non-finite logits
  explicitly. One global model: split the test ROWS between the workers
  (`driver.eval_bounds`) and sum the partial matrices — 1 worker ≡ 2 workers exactly.
  Per-client models: split clients across GPUs by index, cache the confusion matrix of any
  client whose weights did not change, and **pin client c to worker c % W** (two processes
  can pick different cuDNN algorithms and disagree on a handful of fp16 rows).
* **Artifacts.** Weights only, `weights_only=True`, one file per round holding every
  model the method keeps (state_dict tensors) plus the cfg that rebuilds the architecture;
  resume bundle keyed by round; marker absolutely last; history CSVs derived and rebuilt
  from the per-round JSON; y_true copied into the run; `verify.py` recomputes everything
  from the confusion matrices and rejects the tamper cases in `tests/test_ckpt_verify.py`.
* **Budget.** `T0 = time.monotonic()` in the first cell; driver stops when
  `elapsed + 1.15 × worst_round + reserve > max_seconds`; `max_hours` sized to the
  account's free quota, not to the 12 h cap. Read quota with `kaggle_account.py quota`
  right before pushing; reservations are returned when a session is cancelled.

## 3. Numbers to budget with (2×T4, torch 2.10, image digest in `knowledge/runtime.json`)

| quantity | value | source |
|---|---|---|
| plain DAGSNet step (395 k params), compiled+AMP | 6.2 ms @512, 5.7 ms @256 | calibration 2026-09-07 |
| two-module step (teacher forward + student fwd/bwd), compiled | 12.06 ms @512 (eager 42.7); compile+gate 264 s | probe 2026-09-11 |
| eval, one model, full test | 26 s/GPU compiled-folded @16384 (414 k rows/s); eager-folded 296 k; @8192 only 76 k — use 16384 | probe 2026-09-11 |
| per-client method, 20c round (20 clients train + 21 evals, 2 GPUs) | 1256 s = train 1000 + eval 256; startup 15.7 min; VRAM 7.3 GiB/GPU | probe 2026-09-11 |
| prepack 43 M rows | 90–130 s; startup incl. spawn + compile 12–17 min | measured 2026-09-10 |
| 100c at batch 256 | CPU-starved on 4 vCPU: util 58 %, round drifts +20 % — budget it separately | measured 2026-09-10 |
| eager fallback cost | ×2.9 on train | measured 2026-09-10 |
| weights per round | 1.6 MB per full DAGSNet (0.2 MB pruned to 35.9 k params); per-client methods: N × that | |

The lwfednids probe of 2026-09-22 (pruned DAGSNet, 20c) adds its own row to that project's
`docs/tests.md` §4 — read it there, not here.

## 4. Operating sequence (copy-paste)

```bash
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh && conda activate nckh
H=.agents/skills/kaggle-training-notebook/scripts
python $H/kaggle_account.py quota                      # who can pay; refresh time
python scripts/gen_notebook.py --owner <acct> --clients 20 --probe --max-hours 2   # 2-round probe + T4 micro-benchmark
python scripts/validate_notebooks.py
python $H/embed_wandb_key.py <nb.ipynb> --metadata <kernel-metadata.json> && python scripts/validate_notebooks.py
python $H/kaggle_as.py <acct> -- kaggle kernels push -p papers/<slug>/notebook/20c_probe/   # no account switch
python $H/kaggle_as.py <acct> -- kaggle kernels status <acct>/<kernel-slug>
# watch W&B: summary.backend, cal_*, history rows; Kaggle stdout is invisible while running
bash scripts/pull_output.sh <acct> <acct>/<slug> papers/<slug>/runs/pulls/<name>
python scripts/verify_run.py <pulled run dir> --require-rounds 50
```

Continuation on the SAME account: `--session 2 --require-resume --kernel-source <acct>/<slug>`
— its own slug `<slug>-s2` and dir `<K>c_s2`, same `run_name` (fingerprint, W&B id); a
notebook attaching its own output is unverified on Kaggle, so never reuse the slug. On ANOTHER
account: stage the pulled run tree (or a last-round handoff bundle) as a dataset owned by the
new account (`stage_ckpt_dataset.py`, `kaggle datasets create -p <dir> -r zip -t`), CPU-probe
the import (`gen_ckpt_probe.py`), then `--require-resume --dataset-source <acct>/<ds>` — see
[multi-account.md](multi-account.md).

## 5. Local test discipline

Read repository-root `knowledge/local-env.md` for the current local OS, RAM/swap,
GPU and environment. The 8 GB WSL limits below were measured on an older host;
they are historical fixture guidance, not the current machine's capacity.

Run local tests through repository-root `scripts/run_local_guarded.py`, one heavy
test process tree at a time. Start with 2–4 clients and 2–12k rows each, CPU,
`compile=False`; measure available RAM before increasing fixtures or workers.
Keep the documented watchdog threshold. Historical two-worker tests needed lazy
`pyarrow` imports and batch ≤ 128; remeasure on the current host before claiming
larger fixtures fit. The notebook simulation uses a fake `/kaggle/input` tree
(with sidecars): CPU for training, `--gpu` for calibration. A local sm_86 pass
certifies the graph scheme; the Kaggle probe certifies sm_75/T4 behavior.

## 6. Per-client models on the GLOBAL test peak early (measured 2026-09-13, sibling rebuild)

A per-client DAGSNet supervised only by its own labels reaches its global-test ceiling after
one to two local epochs and then erodes, because the fixed test set has the global prior and
the client does not. Predict the falling W&B curve in the plan before the owner watches it; a
per-round cosine schedule flattens the drift but does not recover the peak. A global-model
method (FedAvg-like) does not show this shape.
