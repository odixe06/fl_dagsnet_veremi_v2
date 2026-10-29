"""DAGSNet — Khan et al. 2025 (Sci. Rep., DOI 10.1038/s41598-025-94445-9) §4.10, Eq. (38)-(48).

Reference nn.Module for v4: x (B, 77) -> view(B, 11, 7).transpose(1, 2) -> (B, 7, 11) -> 16 logits (no softmax).
Spec, widths and every choice the paper leaves open: knowledge/dagsnet.md. `core/grouped.py` is the same function
for G independent models at once and reads this module's state_dict names.
"""
import torch
import torch.nn as nn

FULL = {"patch_len": 7, "stem_ch": 96, "dense_growth": 32, "dense_layers": 3, "incep_ch": 32, "incep_modules": 2,
        "alex_ch": 128, "fire_sq": 32, "fire_ex": 48, "fire_modules": 3, "head_hidden": 256, "dropout": 0.1,
        "n_features": 77, "num_classes": 16}


def cbr(i, o, k):
    """Conv -> BatchNorm -> ReLU. bias=False because BatchNorm right after has its own shift."""
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class DenseNet1d(nn.Module):
    """Eq. (38)-(39): each layer reads the concatenation of every earlier feature map."""
    def __init__(self, cin, growth, layers):
        super().__init__()
        self.blocks = nn.ModuleList([cbr(cin + i * growth, growth, 3) for i in range(layers)])
        self.out_ch = cin + layers * growth

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([x, b(x)], dim=1)
        return x


class Inception1d(nn.Module):
    """Eq. (40): parallel 1x1 / 3x3 / 5x5 / pool branches, concatenated."""
    def __init__(self, cin, c):
        super().__init__()
        self.b1 = cbr(cin, c, 1)
        self.b3 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 3))
        self.b5 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 5))
        self.bp = nn.Sequential(nn.MaxPool1d(3, 1, 1), cbr(cin, c, 1))
        self.out_ch = 4 * c

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class GoogleNet1d(nn.Module):
    """Eq. (40)-(41): stacked inception modules."""
    def __init__(self, cin, c, modules_n):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Inception1d(ch, c); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class AlexNet1d(nn.Module):
    """Eq. (42)-(44). ceil_mode=True: the position axis is only 11 long (11 -> 6 -> 3)."""
    def __init__(self, cin, ch):
        super().__init__()
        self.net = nn.Sequential(cbr(cin, ch, 3), nn.MaxPool1d(2, ceil_mode=True),
                                 cbr(ch, ch, 3), nn.MaxPool1d(2, ceil_mode=True), cbr(ch, ch, 3))
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class Fire1d(nn.Module):
    """Eq. (45)-(46): 1x1 squeeze feeding 1x1 and 3x3 expand branches."""
    def __init__(self, cin, sq, ex):
        super().__init__()
        self.squeeze = cbr(cin, sq, 1)
        self.e1 = cbr(sq, ex, 1)
        self.e3 = cbr(sq, ex, 3)
        self.out_ch = 2 * ex

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], dim=1)


class SqueezeNet1d(nn.Module):
    def __init__(self, cin, sq, ex, modules_n):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Fire1d(ch, sq, ex); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class DAGSNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        assert cfg["n_features"] % self.patch_len == 0
        self.k = cfg["n_features"] // self.patch_len            # 11 positions
        s = cfg["stem_ch"]
        self.stems = nn.ModuleList([cbr(self.patch_len, s, 1) for _ in range(4)])
        self.dense = DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google = GoogleNet1d(s, cfg["incep_ch"], cfg["incep_modules"])
        self.alex = AlexNet1d(s, cfg["alex_ch"])
        self.squeeze = SqueezeNet1d(s, cfg["fire_sq"], cfg["fire_ex"], cfg["fire_modules"])
        self.comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch
        h = cfg["head_hidden"]
        self.head = nn.Sequential(                               # Eq. (48), logits
            nn.LayerNorm(self.comb), nn.Dropout(cfg["dropout"]), nn.Linear(self.comb, h), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(h, cfg["num_classes"]))

    def combined(self, x):                                       # (B, 77) -> (B, comb), Eq. (47)
        # view THEN transpose: 7 consecutive columns form one patch, patches become the position axis.
        fm = x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)
        nets = [self.dense, self.google, self.alex, self.squeeze]
        return torch.cat([net(stem(fm)).mean(dim=-1) for stem, net in zip(self.stems, nets)], dim=1)

    def hidden(self, x):                                         # head's hidden layer (prototype/representation methods)
        return self.head[:4](self.combined(x))

    def forward(self, x):
        return self.head(self.combined(x))


def width(scale, **over):
    """FULL with every channel count multiplied by `scale` (rounded, >= 4). Layer counts and kernels unchanged."""
    cfg = dict(FULL)
    for k in ("stem_ch", "dense_growth", "incep_ch", "alex_ch", "fire_sq", "fire_ex", "head_hidden"):
        cfg[k] = max(4, int(round(FULL[k] * scale)))
    cfg.update(over)
    return cfg


def build_model(cfg):
    return DAGSNet(cfg)


def n_params(cfg):
    return sum(p.numel() for p in DAGSNet(cfg).parameters())
