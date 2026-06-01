"""Spatial-latent explosion surrogate: a conv feature map (C x 14 x 14) instead of a single vector,
so decoded predictions are crisp. Action-conditioned (FiLM) conv predictor over the latent grid.
Shared by train / viz / planning / timing scripts.
"""
import torch
import torch.nn.functional as F
from torch import nn


def gn(c):
    return nn.GroupNorm(8, c)


class SpatialEnc(nn.Module):
    def __init__(self, C=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, 2, 1), gn(48), nn.GELU(),     # 112 -> 56
            nn.Conv2d(48, 96, 3, 2, 1), gn(96), nn.GELU(),    # 56 -> 28
            nn.Conv2d(96, C, 3, 2, 1), gn(C), nn.GELU(),      # 28 -> 14
            nn.Conv2d(C, C, 3, 1, 1), gn(C), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialDec(nn.Module):
    def __init__(self, C=128):
        super().__init__()

        def up(ci, co):
            return [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ci, co, 3, 1, 1), gn(co), nn.GELU()]

        self.net = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1), gn(C), nn.GELU(),
            *up(C, 96), *up(96, 48), *up(48, 32),             # 14 -> 28 -> 56 -> 112
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class FiLMBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.c1 = nn.Conv2d(C, C, 3, 1, 1); self.n1 = gn(C)
        self.c2 = nn.Conv2d(C, C, 3, 1, 1); self.n2 = gn(C)

    def forward(self, x, gamma, beta):
        h = self.n1(self.c1(x)); h = h * (1 + gamma) + beta; h = F.gelu(h)
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class ConvPredictor(nn.Module):
    """Predict next latent grid from K history latents + the 8 params (FiLM conditioning)."""

    def __init__(self, C=128, K=3, action_dim=8, nblk=3):
        super().__init__()
        self.K, self.C, self.nblk = K, C, nblk
        self.inp = nn.Conv2d(K * C, C, 1)
        self.act = nn.Sequential(nn.Linear(action_dim, 128), nn.GELU(), nn.Linear(128, nblk * 2 * C))
        self.blocks = nn.ModuleList([FiLMBlock(C) for _ in range(nblk)])
        self.out = nn.Conv2d(C, C, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start as identity (residual)

    def forward(self, zhist, a):
        B, K, C, h, w = zhist.shape
        x = self.inp(zhist.reshape(B, K * C, h, w))
        gb = self.act(a).view(B, self.nblk, 2, C, 1, 1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, gb[:, i, 0], gb[:, i, 1])
        return zhist[:, -1] + self.out(x)


class SpatialSurrogate(nn.Module):
    def __init__(self, C=128, K=3, action_dim=8):
        super().__init__()
        self.enc = SpatialEnc(C); self.dec = SpatialDec(C)
        self.pred = ConvPredictor(C, K, action_dim)
        self.K, self.C = K, C

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def predict_next(self, zhist, a):
        return self.pred(zhist, a)

    def rollout(self, zctx, a, R):
        """zctx: (B,K,C,h,w) encoded context, a: (B,action_dim) -> predicted (B,R,C,h,w)."""
        zs = [zctx[:, i] for i in range(zctx.size(1))]
        for _ in range(R):
            zs.append(self.pred(torch.stack(zs[-self.K:], 1), a))
        return torch.stack(zs[self.K:], 1)


def load_spatial(ckpt_path, device):
    m = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpatialSurrogate(m["C"], m["K"]).to(device)
    model.load_state_dict(m["state_dict"]); model.eval()
    pmean = torch.tensor(m["p_mean"], device=device)
    pstd = torch.tensor(m["p_std"], device=device)
    return model, pmean, pstd, m["K"]
