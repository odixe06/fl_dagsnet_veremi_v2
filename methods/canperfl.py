"""CANPerFL — Nguyen et al., Appl. Sci. 13(11):6369, 2023 (papers/canperfl.pdf), §4.2, §5.5.1 ("Fed-FN").

FedAvg (Algorithm 1) for 50 rounds, then every client fine-tunes the global model independently on its own data for about
10 epochs and keeps it as its personalised IDS. Phase 1 is exactly methods/fedavg.py under the protocol, so it is not
re-run: this method starts from the final FedAvg state (hp "init", a glob for that run's state.pt, e.g. a Kaggle input
"/kaggle/input/**/fl2-fedavg/state.pt") and each "round" here is one fine-tuning epoch of every client from its own model
(rounds = 10). Evaluation: client k's fine-tuned model on its own test ("client", headline after round 10) and the FedAvg
model ("global", unchanged, for reference).
Deviations: AdamW with the protocol's cosine schedule over the 10 fine-tuning epochs (paper: SGD with momentum, local lr
0.05, federated lr 0.005); the whole model is fine-tuned (paper: "the global parameters are fine-tuned on each client").
Communication: phase 1 = FedAvg; phase 2 = none (one download of θ^50, counted in FedAvg's last round).
"""
import glob

import torch

from core import local


class Method:
    name = "canperfl"
    per_client = True

    def __init__(self, env):
        self.env = env
        hits = sorted(glob.glob(env.hp["init"], recursive=True))
        assert hits, f"no FedAvg state matches {env.hp['init']}"
        s = torch.load(hits[0], map_location="cpu", weights_only=True)
        assert s["method"] == "fedavg" and s["cfg"] == env.cfg and s["round"] == 50, (s["method"], s["round"])
        self.p, self.b = s["method_state"]["p"].to(env.dev), s["method_state"]["b"].to(env.dev)
        self.bank_p, self.bank_b = self.p.repeat(env.K, 1), self.b.repeat(env.K, 1)

    def payload_bytes(self):
        return {"down": 0, "up": 0, "note": "fine-tuning phase only; phase 1 = fedavg payload x 50 rounds"}

    def round(self, r, lr):
        env = self.env
        loss = torch.zeros(env.K, device=env.dev)
        steps = torch.zeros(2, device=env.dev)
        for idx in env.groups_idx():
            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            loss[idx] = st["loss"]
        tot = env.allreduce(env.n.sum().clone())
        steps = env.allreduce(steps)
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1))}

    def evaluate(self):
        env = self.env
        return {"client": env.eval_clients(self.bank_p, self.bank_b), "global": env.eval_global(self.p, self.b)}

    def state(self):
        return {"bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu()}

    def load_state(self, s):
        self.bank_p, self.bank_b = s["bank_p"].to(self.env.dev), s["bank_b"].to(self.env.dev)
