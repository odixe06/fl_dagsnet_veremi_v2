"""MobiHet — owner's method (repo ../mobihet, arm fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl, docs/methods/full-drift/design.md).

Every vehicle keeps a PRIVATE model that never leaves it (any architecture in principle; DAGSNet here) and trains a shared
PROXY that is the only thing exchanged, as a 3-bit stochastic-rounded delta + LZMA2 (q3_zl). Per local step on real rows
(x, y) and Gaussian rows (x_g, y_g) drawn from pooled class moments (gpool):

    private:  CE(z_p + log π_k − log π_g, y)                              la: logit-adjusted CE toward the global prior
            + w · KL(T(x) ‖ z_p(x))                                       frz: T = proxy snapshot received this round (frozen)
            + w · KL(T(x_g) ‖ z_p(x_g))       (round 1: CE(z_p(x_g), y_g))  gkd
    proxy:    CE(z_r, y) + min(w, 1) · KL(z_p ‖ z_r)                       kdpx: proxy <- private weight capped at 1
    w = 1 / (CE_private + CE_proxy)   (detached; DML adaptive weight)

KD on real rows, no KD in round 1 (no snapshot yet). Server: exact proxy kept, θ ← θ + mean_k Q3(Δ_k) (unweighted, as in
the original); downlink Q3(exact − replica), nearest rounding; BN buffers fp16, averaged. m4b (zeroing simulator noise and
absolute receiver/sender positions) is the identity on v4: the dataset has none of those columns.

One-time bootstrap (gpool + la): each client sends per-class counts, Σx, Σxxᵀ (77-d); the server returns the pooled class
means, Cholesky factors and the global prior. Deviations from the original: shared protocol (batch 64, wd 1e-4, cosine
1e-3 → 1e-5; original wd 0, batch 512), Gaussian rows per step = batch / 2 (original 256 per 512 real rows, same ratio),
LZMA2 byte count measured on a fixed sample of clients per rank (lossless stage only changes bytes, not training).
"""
import lzma
import math

import torch
import torch.nn.functional as F

from core import grouped, local

LZMA = [dict(id=lzma.FILTER_LZMA2, preset=9 | lzma.PRESET_EXTREME)]
BITS = 3
KD_CAP = 1.0
BYTES_SAMPLE = 64                                                   # clients per rank whose packets are LZMA-coded each round


def quantize(delta, gen=None):
    """antimem.quantize on (G, rows, -1): symmetric per-row scale (fp16), 3-bit codes in [-3, 3]; stochastic if gen."""
    levels = 2 ** (BITS - 1) - 1
    scale = (delta.abs().amax(2, keepdim=True) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    v = delta / safe
    q = torch.round(v) if gen is None else torch.floor(v + torch.rand(v.shape, device=v.device, generator=gen))
    q = q.clamp(-levels, levels) * (scale > 0)
    return q * scale, q.to(torch.int8), scale.half()


def packet_bytes(codes, scales, bufs_row):
    """Wire bytes of one client's packet: 3-bit codes packed (2 per byte, like the original 8//bits), fp16 scales, fp16
    BN buffers, LZMA2-coded."""
    body = []
    for q, s in zip(codes, scales):
        c = (q.reshape(-1).to(torch.int16) + 4).to(torch.uint8).cpu()
        if len(c) % 2:
            c = torch.cat([c, c.new_zeros(1)])
        body += [s.reshape(-1).cpu().numpy().tobytes(), (c[0::2] | (c[1::2] << 3)).numpy().tobytes()]
    body.append(bufs_row.half().cpu().numpy().tobytes())
    return 32 + len(lzma.compress(b"".join(body), format=lzma.FORMAT_RAW, filters=LZMA))   # 32: header as QHEADER


class Method:
    name = "mobihet"
    per_client = True

    def __init__(self, env):
        self.env = env
        lay, dev = env.lay, env.dev
        p0, b0 = lay.init(env.hp["seed"], 1, dev)
        self.priv_p = p0.repeat(env.K, 1)                            # private bank (this rank's clients)
        self.priv_b = b0.repeat(env.K, 1)
        self.g_p, self.g_b = p0[0].clone(), b0[0].clone()            # proxy replica every client holds
        self.exact = self.g_p.clone()                               # server's exact proxy (error feedback)
        self.version = 0
        self.aux = env.hp["batch"] // 2
        self.weighted = env.hp.get("agg", "mean") == "weighted"          # original: unweighted mean over clients
        self.gen_q = torch.Generator(device=dev).manual_seed(env.hp["seed"] * 7 + env.rank)
        self.gen_g = torch.Generator(device=dev).manual_seed(env.hp["seed"] * 11 + env.rank)
        self._bootstrap()

    def _bootstrap(self):
        env, X, Y = self.env, self.env.train.X, self.env.train.Y
        C, D = 16, X.shape[1]
        cnt = torch.zeros(env.K, C, device=env.dev)
        s1 = torch.zeros(C, D, dtype=torch.float64, device=env.dev)
        s2 = torch.zeros(C, D, D, dtype=torch.float64, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        cnt.index_put_((client, Y.long()), torch.ones(len(Y), device=env.dev), accumulate=True)
        for c in range(C):
            for i in range(0, len(Y), 1 << 21):
                x = X[i:i + (1 << 21)][Y[i:i + (1 << 21)] == c].double()
                s1[c] += x.sum(0); s2[c] += x.T @ x
        tot = env.allreduce(cnt.sum(0).double())
        s1, s2 = env.allreduce(s1), env.allreduce(s2)
        mu = s1 / tot[:, None]
        cov = s2 / tot[:, None, None] - mu[:, :, None] * mu[:, None, :]
        cov = cov + 1e-4 * torch.eye(D, device=env.dev, dtype=torch.float64)
        self.mu, self.L = mu.float(), torch.linalg.cholesky(cov).float()
        log_pk = ((cnt + 1) / (cnt.sum(1, keepdim=True) + C)).log()
        log_pg = ((tot + 1) / (tot.sum() + C)).log().float()
        self.la_shift = log_pk - log_pg                              # (K, 16)
        self.labels = torch.arange(self.aux, device=env.dev) % C
        tri = D * (D + 1) // 2
        self.bootstrap_bytes = {"up": 4 * C * (1 + D + tri), "down": 4 * C * (D + tri) + 4 * C}

    def payload_bytes(self):
        n = 4 * (self.env.lay.P + self.env.lay.B)
        return {"dense_reference": n, "bootstrap": self.bootstrap_bytes, "measured": "rounds.jsonl bytes_up/bytes_down"}

    def _loss(self, ctx):
        env, cfg = self.env, self.env.cfg
        G, b, a = ctx.Gfix, ctx.x.shape[1], self.aux
        Pp, Pr = ctx.P
        Bp, Br = ctx.B
        act = ctx.mask.any(1)
        z = torch.randn(G, a, self.mu.shape[1], device=ctx.x.device, generator=self.gen_g)
        xg = (self.mu[self.labels] + (self.L[self.labels] @ z[..., None]).squeeze(-1)).clamp(-4, 4).to(ctx.x.dtype)
        xp = torch.cat([ctx.x, xg], 1)
        mp = torch.cat([ctx.mask, act[:, None].expand(G, a)], 1)
        zp, nBp = grouped.TRAIN(Pp, Bp, xp, cfg, mask=mp, train=True)
        zr, nBr = grouped.TRAIN(Pr, Br, ctx.x, cfg, mask=ctx.mask, train=True)
        y, m = ctx.y, ctx.mask.float()
        nreal = m.sum(1).clamp_min(1)
        shift = self._shift[:, None, :]
        ce = (F.cross_entropy((zp[:, :b] + shift).reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b) * m).sum(1) / nreal
        pce = (F.cross_entropy(zr.reshape(-1, 16), y.reshape(-1), reduction="none").view(G, b) * m).sum(1) / nreal
        raw = 1 / (ce + pce).detach().clamp_min(1e-6)
        lsp, lsr = F.log_softmax(zp, -1), F.log_softmax(zr, -1)
        kl_proxy = ((lsp[:, :b].detach().exp() * (lsp[:, :b].detach() - lsr)).sum(-1) * m).sum(1) / nreal
        loss = ce + pce + raw.clamp(max=KD_CAP) * kl_proxy
        if self.version > 0:
            with torch.no_grad():
                zt, _ = grouped.TRAIN(self._teacher_p, self._teacher_b, xp, cfg, train=False)
                lt = F.log_softmax(zt.float(), -1)
            kl = (lt.exp() * (lt - lsp)).sum(-1)                         # KL(T ‖ private) per row
            kd = (kl[:, :b] * m).sum(1) / nreal
            gkd = kl[:, b:].mean(1)
            loss = loss + raw * (kd + gkd)
        else:
            aux_ce = F.cross_entropy(zp[:, b:].reshape(-1, 16), self.labels.repeat(G), reduction="none").view(G, a).mean(1)
            loss = loss + aux_ce
        return loss, [nBp, nBr]

    def round(self, r, lr):
        env, lay = self.env, self.env.lay
        P = lay.P
        dsum = torch.zeros_like(self.g_p, dtype=torch.float64)
        bsum = torch.zeros_like(self.g_b, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        nbytes, sampled = 0, 0
        for gi, idx in enumerate(env.groups_idx()):
            G = len(idx)
            flat = torch.cat([self.priv_p[idx], self.g_p.repeat(G, 1)], 1)
            bufs = torch.cat([self.priv_b[idx], self.g_b.repeat(G, 1)], 1)
            self._shift = local.pad_rows(self.la_shift[idx], local.group_size(env.train.off, idx, env.hp))
            if self.version > 0:
                Gf = self._shift.shape[0]
                self._teacher_p = lay.params(self.g_p[None].repeat(Gf, 1))
                self._teacher_b = lay.bufs(self.g_b[None].repeat(Gf, 1))
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss,
                                   models=2)
            self.priv_p[idx], self.priv_b[idx] = flat[:, :P], bufs[:, :lay.B]
            # uplink: Q3(proxy − replica) per tensor row, stochastic rounding
            delta = lay.params(flat[:, P:] - self.g_p[None])
            codes, scales = [], []
            for k, shape in lay.p:
                d = delta[k].reshape(G, shape[0] if len(shape) > 1 else 1, -1)
                sent, q, s = quantize(d, self.gen_q)
                wk = env.n[idx][:, None] if self.weighted else 1.0
                dsum[_slice(lay, k)] += (wk * sent.reshape(G, -1)).sum(0).double()
                codes.append(q); scales.append(s)
            wb = env.n[idx][:, None].double() if self.weighted else 1.0
            bsum += (wb * bufs[:, lay.B:].half().double()).sum(0)
            loss[idx] = st["loss"]
            if sampled < BYTES_SAMPLE:
                for i in range(min(G, BYTES_SAMPLE - sampled)):
                    nbytes += packet_bytes([c[i] for c in codes], [s[i] for s in scales], bufs[i, lay.B:])
                sampled += min(G, BYTES_SAMPLE - sampled)
        K = env.allreduce(env.n.sum().clone() if self.weighted else torch.tensor(float(env.K), device=env.dev))
        self.exact = self.exact + (env.allreduce(dsum) / K).float()
        self.g_b = (env.allreduce(bsum) / K).float()
        # downlink: nearest Q3 of (exact − replica); every client rebuilds the same replica
        down = self.exact - self.g_p
        dv = lay.params(down[None])
        dq, codes, scales = torch.zeros_like(down), [], []
        for k, shape in lay.p:
            d = dv[k].reshape(1, shape[0] if len(shape) > 1 else 1, -1)
            sent, q, s = quantize(d)
            dq[_slice(lay, k)] = sent.reshape(-1); codes.append(q[0]); scales.append(s[0])
        self.g_p = self.g_p + dq
        down_bytes = packet_bytes(codes, scales, self.g_b)
        up = env.allreduce(torch.tensor([float(nbytes), float(sampled)], device=env.dev))
        self.version = r
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / env.allreduce(env.n.sum().clone())),
                "bytes_up": float(up[0] / up[1]), "bytes_down": down_bytes,
                "q_residual_rel": float((self.exact - self.g_p).norm() / self.exact.norm())}

    def evaluate(self):
        return {"private": self.env.eval_clients(self.priv_p, self.priv_b), "proxy": self.env.eval_global(self.g_p, self.g_b)}

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("priv_p", "priv_b", "g_p", "g_b", "exact")} | {"version": self.version}

    def load_state(self, s):
        for k in ("priv_p", "priv_b", "g_p", "g_b", "exact"):
            setattr(self, k, s[k].to(self.env.dev))
        self.version = s["version"]


def _slice(lay, key):
    o = 0
    for k, s in lay.p:
        n = math.prod(s)
        if k == key:
            return slice(o, o + n)
        o += n
    raise KeyError(key)
