# Proving that faster code trains the same thing

A speed change is accepted only with evidence of the class named in SKILL.md §Contract. Write the
tests before the change, run them on a small real-data fixture (a few hundred to a few thousand rows per
client, all clients, the real model), and keep them in the project's test script as named parts.

## 1. Bit-exact class

Compare full state, not one metric. "Same" means `torch.equal` on every tensor and exact equality of
everything else.

| test | compare | catches |
|---|---|---|
| old code vs new code, same arm, 2+ epochs, eager | all weights (every model kept), shared/global state | any arithmetic or RNG-order change |
| `--workers 1` vs `--workers W` (W >= 3 if RAM allows, at least 2), compiled backend for the main arm | weights, prediction files, communication/byte log, per-round history (timings excluded), metrics CSV | state not reaching workers, completion-order effects, per-process RNG |
| uninterrupted run vs run continued from a committed epoch (serial and W workers) | weights, predictions, eval of the continued epoch; chain of hashes | resume state missing (EMA, server state, codec error feedback, counters) |
| every arm the change touches | at least one epoch of each through the same harness | a hook path the main arm does not exercise |

Load the old code from a frozen copy of the previous version (not from version control at test time),
import it as a separate module, and run both in the same process with the same fixture.
For a new option that must leave old behaviour intact, also assert the identities it implies
(e.g. "option on, epoch 1 == option off" when the option only acts after the first aggregation).

## 2. Tolerance class (owner's go required)

The trajectory diverges after a few steps (fp16 AMP amplifies rounding), so end-of-run equality is
impossible by construction. Evidence instead:

1. **Per-step agreement from identical state.** Same weights, batch, RNG and BN buffers; run K steps
   (e.g. 20) in both codes; compare loss and logits each step: `max|delta|` at rounding level
   (record the value), identical applied/skipped AMP steps, argmax agreement on decisive rows only
   (top-2 gap above 10 x delta), counted as integers. Compare logits, not weights: Adam's first update is about
   +-lr per coordinate, so parameters whose true gradient is ~0 (e.g. freshly zero BatchNorm biases)
   take the sign of rounding noise -- 0.5-0.7 relative tensor differences after one step are normal
   while logits move <1 %. Check the optimizer change alone (e.g. `fused`) separately: it should stay
   near 1e-5 relative; kernel-algorithm changes (cuDNN autotune) are larger but still decisive-row exact.
2. **Semantics checks** for every piece moved into a graph: tail batches run, skipped AMP steps leave
   weights unchanged, synthetic-row generators draw the same stream, BN buffers update once per step.
3. **End-to-end equivalence.** Re-run the reference arm (the control) in the new code for the same
   epochs and seed; the difference to the old run must lie inside the observed seed-to-seed spread of
   that metric. From then on compare new arms only against controls run in the new code.
4. Keep `1 worker == W workers` bit-exact inside the new code if determinism is kept; if
   nondeterministic kernels are enabled, state that runs are no longer reproducible bit for bit.

5. **Runtime gates that ship with a fast path** (a faster evaluator gated per model against the reference
   module) must use a bound **relative to the output scale** -- `max|delta| / max|reference logit|` -- not an
   absolute one: fp16 rounding error grows with logit magnitude, and logits grow through training. Calibrate the
   bound on the **latest, largest-magnitude checkpoints that will be scored**, not on the first rounds. Measured
   failure: an absolute 0.25 bound set from rounds 1-4 (max error 0.077) broke at round 34 of 50 (0.2508, logits
   ~3x larger, relative error still 0.2-0.7 %). A wrong weight load moves logits O(10 %) relative, so a relative
   bound of a few percent still catches it.
6. **A gate failure must not throw away the rest of a batch job.** One model over the bound should fall back to
   the reference path for that model (slower, exact) and be recorded, or at least end the job with every earlier
   unit committed and a collector that can audit the committed part; never let one model abort a multi-hour
   notebook with the remaining units unscored.

## 3. Local discipline

- Run tests through a memory guard, one CUDA job at a time; a guard kill (exit -15/137) is not a
  test failure -- reduce workers for memory-heavy setups and rerun.
- Output of `conda run` is buffered until exit; long tests belong in the background with a timeout
  well above their runtime (a short `timeout` kills a passing test).
- One local GPU certifies the graph scheme and the protocol, not the target architecture: the short
  benchmark on the target device decides adoption.
