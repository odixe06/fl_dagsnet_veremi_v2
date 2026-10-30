"""APFed — Almansour et al., Sci. Rep. 15:35604, 2025 (papers/apfed.pdf), Eq. (5)-(10).

Model w = (w_B base, w_C classifier); here base = DAGSNet up to the hidden layer, classifier = head.5 (knowledge/dagsnet.md
§5). Client i keeps its previous personalised model W̃_i and an element-wise fusion weight θ_i ∈ [0, 1] over the base.
Round t, client i:
  1. adaptive personalised update (Eq. 5-8): w_B = W̃_B + θ ⊙ (W_B^g − W̃_B), w_C = W_C^g;
  2. θ learned on local data, batch by batch, with W^g and W̃ frozen (Eq. 9-10): θ ← clip(θ − α (W_B^g − W̃_B) ⊙ ∇_w L, 0, 1),
     w_B recomputed after every batch;
  3. local training of w (the protocol's epoch) → the new W̃_i, uploaded; server FedAvg (Eq. 1).
The paper gives no α / θ init / data share for step 2; APFed builds on FedALA (its ref. 47), so FedALA's defaults are used:
θ⁰ = 1, α = 1. Step 2 is one pass over all local data (FedALA: a random 80 %, and until convergence in its first ALA round);
it is skipped while W̃ = W^g (round 1: every model is θ⁰). BN running statistics stay local (FedALA fuses parameters only).
Deviations: DAGSNet instead of LDwCBN (protocol); sample-weighted FedAvg (paper Eq. 1). θ (K × base) lives in pinned host
memory and is moved per group (with W̃ on the GPU it would not fit beside the data at W = 1); checkpointed as fp16 1 − θ.
Evaluation: W̃_i after local training on client i's test ("client", headline); W^g on the full test ("global").
"""
import torch

from core import local
from methods import fedavg

ALPHA = 1.0


class _Theta:
    """local_epoch optimizer for step 2: updates θ from the gradient at w, rewrites the base of w; no AdamW, no clip."""

    def __init__(self, theta, w0, diff, alpha):
        self.theta, self.w0, self.diff, self.alpha = theta, w0, diff, alpha

    def step(self, flat, grad, lr, active):
        if self.theta.shape[0] != flat.shape[0]:
            self.theta, self.w0, self.diff = (local.pad_rows(t, flat.shape[0]) for t in (self.theta, self.w0, self.diff))
        Pb = self.theta.shape[1]
        ok = active & torch.isfinite(grad).all(1)
        new = (self.theta - self.alpha * self.diff * grad[:, :Pb]).clamp_(0, 1)
        self.theta = torch.where(ok[:, None], new, self.theta)
        flat[:, :Pb] = self.w0 + self.theta * self.diff
        return None, ok


class Method(fedavg.Method):
    name = "apfed"
    per_client = True

    def __init__(self, env):
        super().__init__(env)
        self.alpha = env.hp.get("alpha", ALPHA)
        self.Pb = env.lay.P - int(env.lay.mask(["head.5"]).sum())
        assert bool(env.lay.mask(["head.5"])[self.Pb:].all()), "head.5 must be the last parameters"
        self.bank_p, self.bank_b = self.p.repeat(env.K, 1), self.b.repeat(env.K, 1)
        pin = env.dev.type == "cuda"
        self.theta = torch.ones(env.K, self.Pb, pin_memory=pin)
        self.fused = False                                            # W̃ = W^g until the first aggregation

    def round(self, r, lr):
        env, Pb = self.env, self.Pb
        acc_p, acc_b = torch.zeros_like(self.p), torch.zeros_like(self.b)
        loss = torch.zeros(env.K, device=env.dev)
        steps = torch.zeros(2, device=env.dev)
        th_mean = torch.zeros((), device=env.dev)
        for idx in env.groups_idx():
            Wt = self.bank_p[idx]
            flat = self.p.repeat(len(idx), 1)                          # w_C = W_C^g
            if self.fused:
                ic = idx.cpu()
                th = self.theta[ic].to(env.dev, non_blocking=True)
                diff = self.p[None, :Pb] - Wt[:, :Pb]
                flat[:, :Pb] = Wt[:, :Pb] + th * diff
                opt = _Theta(th, Wt[:, :Pb], diff, self.alpha)
                local.local_epoch(env.lay, flat, self.bank_b[idx].clone(), env.train, idx, env.cfg, env.hp, lr, env.gen,
                                  opt=opt)
                th = opt.theta[:len(idx)]
                th_mean += (th.mean(1) * env.n[idx]).sum()
                self.theta[ic] = th.cpu()
                del opt, diff
            bufs = self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            w = env.n[idx][:, None]
            acc_p += (w * flat).sum(0)
            acc_b += (w * bufs).sum(0)
            loss[idx] = st["loss"]
        tot = env.allreduce(env.n.sum().clone())
        self.p = env.allreduce(acc_p) / tot
        self.b = env.allreduce(acc_b) / tot
        self.fused = True
        steps = env.allreduce(steps)
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1)),
                "theta_mean": float(env.allreduce(th_mean) / tot)}

    def evaluate(self):
        env = self.env
        return {"client": env.eval_clients(self.bank_p, self.bank_b), "global": env.eval_global(self.p, self.b)}

    def state(self):
        # 1 − θ in fp16: θ stays near 1 for most coordinates, where fp16 of θ itself would round small moves away; halves
        # the checkpoint (the resumed θ differs by < 1e-3 relative to 1 − θ). Row blocks: a full fp32 1 − θ temporary
        # (~4 GB/rank at W = 1) on top of the pinned θ and the bank copy exceeded Kaggle host RAM at the r50 save (30/10).
        omt = torch.empty(self.theta.shape, dtype=torch.float16)
        for i in range(0, len(self.theta), 256):
            omt[i:i + 256] = 1 - self.theta[i:i + 256]
        return {**super().state(), "bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(),
                "one_minus_theta": omt, "fused": self.fused}

    def load_state(self, s):
        super().load_state(s)
        self.bank_p, self.bank_b = s["bank_p"].to(self.env.dev), s["bank_b"].to(self.env.dev)
        self.theta.copy_(1 - s["one_minus_theta"].float()); self.fused = s["fused"]
