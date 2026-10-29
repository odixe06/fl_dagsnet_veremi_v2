"""Lightweight-Fed-NIDS — Bouayad et al., IEEE Access 2024 (papers/lwfednids.pdf), §III-B.

The server computes ONE data-free (zero-shot) structured pruning mask at initialisation and sends it with θ⁰ once; clients
train the pruned model, the server averages ("averaging all the clients' model weights", FedAvg). Pruning: per dependency
group of channels, importance = L1 norm of every parameter slice the channel touches (producer conv rows, its BN affine,
consumer input columns; Torch-Pruning/DepGraph GroupNormImportance with p = 1, which the paper uses), the lowest 70 % of
each group removed (70 % = the paper's best sparsity) and the pruned parameters physically dropped (DepGraph), so the
model trained and exchanged is a narrower DAGSNet whose initial weights are the kept slices of θ⁰.
Deviations: DAGSNet instead of the paper's CNN feature extractor + classifier; uniform 70 % per group (paper: a global
rate); FedAvg sample-weighted, the protocol rule (owner 29/10; paper's footnote: plain averaging).
"""
import torch

from core.dagsnet import DAGSNet
from core.flat import Layout
from methods import fedavg

RATIO = 0.7
KEYS = ("stem_ch", "dense_growth", "incep_ch", "alex_ch", "fire_sq", "fire_ex", "head_hidden")


def _keep(C, ratio):
    return C - int(ratio * C)


def prune_plan(cfg, sd, ratio=RATIO):
    """-> (pruned cfg, pruned state_dict) for DAGSNet cfg with initial state_dict sd."""
    W = {k: v.abs() for k, v in sd.items() if v.is_floating_point()}
    s, g, c, a, q, e, h = (cfg[k] for k in KEYS)
    L = cfg["dense_layers"]
    rows = lambda name: W[name + ".0.weight"].sum((1, 2)) + W[name + ".1.weight"] + W[name + ".1.bias"]
    cols = lambda name, lo, n: W[name + ".0.weight"][:, lo:lo + n].sum((0, 2))
    # combined feature layout: [dense: stem0, b0..b(L-1)] [google last module: b1 b3 b5 bp] [alex] [squeeze last: e1 e3]
    d_out = s + L * g
    goff, aoff, qoff = d_out, d_out + 4 * c, d_out + 4 * c + a
    comb = lambda lo, n: W["head.0.weight"][lo:lo + n] + W["head.0.bias"][lo:lo + n] + W["head.2.weight"][:, lo:lo + n].sum(0)
    keep = {}

    def top(score, n):
        return torch.sort(torch.topk(score, n).indices).values

    # stems
    sc = rows("stems.0") + comb(0, s) + sum(cols(f"dense.blocks.{i}", 0, s) for i in range(L))
    keep["stems.0"] = top(sc, _keep(s, ratio))
    entry = lambda m: [f"google.net.{m}.b1", f"google.net.{m}.b3.0", f"google.net.{m}.b5.0", f"google.net.{m}.bp.1"]
    sc = rows("stems.1") + sum(cols(n, 0, s) for n in entry(0))
    keep["stems.1"] = top(sc, _keep(s, ratio))
    keep["stems.2"] = top(rows("stems.2") + cols("alex.net.0", 0, s), _keep(s, ratio))
    keep["stems.3"] = top(rows("stems.3") + cols("squeeze.net.0.squeeze", 0, s), _keep(s, ratio))
    # dense
    for i in range(L):
        lo = s + i * g
        sc = rows(f"dense.blocks.{i}") + comb(lo, g) + sum(cols(f"dense.blocks.{j}", lo, g) for j in range(i + 1, L))
        keep[f"dense.blocks.{i}"] = top(sc, _keep(g, ratio))
    # google
    M = cfg["incep_modules"]
    for m in range(M):
        p = f"google.net.{m}"
        for inner, outer in (("b3.0", "b3.1"), ("b5.0", "b5.1")):
            keep[f"{p}.{inner}"] = top(rows(f"{p}.{inner}") + cols(f"{p}.{outer}", 0, c), _keep(c, ratio))
        for j, br in enumerate(("b1", "b3.1", "b5.1", "bp.1")):
            nxt = sum(cols(n, j * c, c) for n in entry(m + 1)) if m + 1 < M else comb(goff + j * c, c)
            keep[f"{p}.{br}"] = top(rows(f"{p}.{br}") + nxt, _keep(c, ratio))
    # alex
    keep["alex.net.0"] = top(rows("alex.net.0") + cols("alex.net.2", 0, a), _keep(a, ratio))
    keep["alex.net.2"] = top(rows("alex.net.2") + cols("alex.net.4", 0, a), _keep(a, ratio))
    keep["alex.net.4"] = top(rows("alex.net.4") + comb(aoff, a), _keep(a, ratio))
    # squeeze
    F_ = cfg["fire_modules"]
    for m in range(F_):
        p = f"squeeze.net.{m}"
        keep[f"{p}.squeeze"] = top(rows(f"{p}.squeeze") + cols(f"{p}.e1", 0, q) + cols(f"{p}.e3", 0, q), _keep(q, ratio))
        for j, br in enumerate(("e1", "e3")):
            nxt = cols(f"squeeze.net.{m + 1}.squeeze", j * e, e) if m + 1 < F_ else comb(qoff + j * e, e)
            keep[f"{p}.{br}"] = top(rows(f"{p}.{br}") + nxt, _keep(e, ratio))
    keep["head.2"] = top(W["head.2.weight"].sum(1) + W["head.2.bias"] + W["head.5.weight"].sum(0), _keep(h, ratio))

    # input index of every consumer, from the producers' kept channels
    def cat(parts):                                                   # parts: [(offset, kept indices)]
        return torch.cat([off + k for off, k in parts])
    inp = {f"stems.{j}": torch.arange(cfg["patch_len"]) for j in range(4)}
    for i in range(L):
        inp[f"dense.blocks.{i}"] = cat([(0, keep["stems.0"])] + [(s + j * g, keep[f"dense.blocks.{j}"]) for j in range(i)])
    prev = keep["stems.1"]
    for m in range(M):
        p = f"google.net.{m}"
        for n in entry(m):
            inp[n] = prev
        inp[f"{p}.b3.1"], inp[f"{p}.b5.1"] = keep[f"{p}.b3.0"], keep[f"{p}.b5.0"]
        prev = cat([(j * c, keep[f"{p}.{br}"]) for j, br in enumerate(("b1", "b3.1", "b5.1", "bp.1"))])
    google_out = prev
    inp["alex.net.0"], inp["alex.net.2"], inp["alex.net.4"] = keep["stems.2"], keep["alex.net.0"], keep["alex.net.2"]
    prev = keep["stems.3"]
    for m in range(F_):
        p = f"squeeze.net.{m}"
        inp[f"{p}.squeeze"] = prev
        inp[f"{p}.e1"] = inp[f"{p}.e3"] = keep[f"{p}.squeeze"]
        prev = cat([(j * e, keep[f"{p}.{br}"]) for j, br in enumerate(("e1", "e3"))])
    dense_out = cat([(0, keep["stems.0"])] + [(s + j * g, keep[f"dense.blocks.{j}"]) for j in range(L)])
    comb_keep = torch.cat([dense_out, goff + google_out, aoff + keep["alex.net.4"], qoff + prev])

    new = dict(cfg)
    for k in KEYS:
        new[k] = _keep(cfg[k], ratio)
    out = {}
    for name in keep:
        if name == "head.2":
            continue
        o, i = keep[name], inp[name]
        out[name + ".0.weight"] = sd[name + ".0.weight"][o][:, i].clone()
        for t in ("weight", "bias", "running_mean", "running_var"):
            out[f"{name}.1.{t}"] = sd[f"{name}.1.{t}"][o].clone()
        out[name + ".1.num_batches_tracked"] = sd[name + ".1.num_batches_tracked"].clone()
    out["head.0.weight"], out["head.0.bias"] = sd["head.0.weight"][comb_keep].clone(), sd["head.0.bias"][comb_keep].clone()
    hk = keep["head.2"]
    out["head.2.weight"] = sd["head.2.weight"][hk][:, comb_keep].clone()
    out["head.2.bias"] = sd["head.2.bias"][hk].clone()
    out["head.5.weight"], out["head.5.bias"] = sd["head.5.weight"][:, hk].clone(), sd["head.5.bias"].clone()
    return new, out


class Method(fedavg.Method):
    name = "lwfednids"

    def __init__(self, env):
        torch.manual_seed(env.hp["seed"])
        full = DAGSNet(env.cfg)
        cfg, sd = prune_plan(env.cfg, full.state_dict(), env.hp.get("prune", RATIO))
        m = DAGSNet(cfg)
        m.load_state_dict(sd)                                         # the pruned model is a narrower DAGSNet
        env.cfg, env.lay = cfg, Layout(cfg)                           # every client trains / exchanges the pruned model
        self.env = env
        self.p = torch.cat([sd[k].reshape(-1).float() for k, _ in env.lay.p]).to(env.dev)
        self.b = torch.cat([sd[k].reshape(-1).float() for k, _ in env.lay.b]).to(env.dev)
        self.full_params = sum(p.numel() for p in full.parameters())
