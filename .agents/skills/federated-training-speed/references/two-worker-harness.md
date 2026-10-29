# One persistent process per GPU for federated rounds

Different FL clients are independent jobs inside a round: no gradient is shared between them, so they
must never form a DDP process group. Speed comes from running clients concurrently on separate GPUs
(and, on a CPU-bound step, from giving each GPU its own Python process). Two layouts cover the cases.

## Layout A — shared queue, dynamic longest-first (global-model methods)

Use when a client's training needs only the broadcast global state (FedAvg-like): any worker can take
any client, so scheduling can be dynamic.

```python
ctx = mp.get_context("spawn")                       # CUDA in children: spawn, never fork
task_qs = [ctx.Queue() for _ in range(W)]; res_q = ctx.Queue()
procs = [ctx.Process(target=worker, args=(r, cfg, task_qs[r], res_q)) for r in range(W)]
# per round
broadcast(("set_global", flat_params.numpy(), flat_ints.numpy()))   # numpy, one vector each
order = sorted(clients, key=rows, reverse=True)                      # longest first (LPT)
... hand out ids as workers free up ...
results = collect(res_q, procs)                                       # poll in short slices, check is_alive()
for c in sorted(results): aggregate(results[c])                       # fixed order, never completion order
```

## Layout B — clients pinned to processes (per-client state stays resident)

Use when each client keeps state across rounds (personal model, EMA, optimizer, control variates):
moving it between processes every round costs I/O and invites drift. Pin client sets by
**longest-first onto the least-loaded process** (balanced rows, deterministic):

```python
def assign(sizes, W):
    load, own = [0] * W, [[] for _ in range(W)]
    for k in sorted(range(len(sizes)), key=lambda k: (-sizes[k], k)):
        r = min(range(W), key=lambda w: (load[w], w)); load[r] += sizes[k]; own[r].append(k)
    return [sorted(o) for o in own]
```

Rank 0 is the server and also trains its own clients; ranks 1..W-1 are started with the same command
line plus `--rank/--address`, build the same deterministic setup, and serve commands:

```python
# rank 0, per round
state = {n: v.cpu() for n, v in global_state.items()}
for conn in conns: send(conn, ("round", r, state, version))
results = {k: client_round(k, r) for k in own_clients}
for conn in conns: results.update({x["k"]: x for x in recv(conn)})
for k in range(K): install(results[k])          # client order: same state/bytes/logs as a serial loop

def send(conn, obj): conn.send_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
def recv(conn): return pickle.loads(conn.recv_bytes())   # NOT conn.send(tensor): ForkingPickler
```

Return from a worker only what the server needs (upload payload, client log, any per-client weights the
server must checkpoint), as CPU tensors or numpy.

## Rules for both layouts

- **Every client epoch re-seeds from `(seed, round, client)`** (permutation, augmentation, synthetic
  draws, dropout via `torch.manual_seed`) so the result does not depend on which process ran it. This is
  what makes `1 worker == W workers` testable bit for bit.
- **GPU per process**: set `CUDA_VISIBLE_DEVICES` per child (round-robin by rank) before CUDA init;
  `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` in every process.
- **Evaluation**: split clients across processes and merge integer confusion matrices; for one shared
  model split test rows. Integer counts make the merge order-independent.
- **Liveness**: a process killed by the OS never answers. Poll in short slices and check
  `exitcode`; surface the child's traceback (worker logs per rank).
- **Budget guard** stays on rank 0: stop before a round when
  `elapsed + 1.15 x worst_round + reserve > max_seconds`.
- **Resume**: workers must load the same committed state as rank 0 (personal models, EMA) when a run is
  continued; test it ([equivalence-tests.md](equivalence-tests.md) §1).
- **Choosing W**: benchmark 1, 2 and 3 processes per GPU on the real device. On 2x T4 with 4 vCPU,
  one per GPU was +15 % total throughput; more processes added only 3-5 %.
- Never a process group or all-reduce between different clients.
