#!/usr/bin/env python3
"""Abstract art-direction: specify a TARGET shape (where the fireball should be), and plan the 8
inputs with CEM through the spatial surrogate to achieve it. Verified by running the true sim.

    ~/miniconda3/envs/torch251/bin/python examples/plan_explosion_art.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explosion3d_sim as ex
from explosion_spatial import load_spatial
from gif_util import save_gif, chw_to_hwc_u8

CKPT = Path(__file__).resolve().parent / "explosion_ckpt" / "spatial.pt"
OUT = Path(__file__).resolve().parent / "explosion_viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = {                              # (cx, cy, sx, sy)  cy: 0=top, ~0.86=ground
    "tall high fireball": (0.50, 0.30, 0.11, 0.20),
    "wide low blast":     (0.50, 0.58, 0.28, 0.10),
    "left-leaning blast": (0.34, 0.40, 0.12, 0.17),
}


def mask(spec, N=112):
    ys = torch.linspace(0, 1, N, device=DEV).view(N, 1)
    xs = torch.linspace(0, 1, N, device=DEV).view(1, N)
    cx, cy, sx, sy = spec
    m = torch.exp(-(((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2) / 2)
    return (m / m.max()).clamp(0, 1)


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, pmean, pstd, K = load_spatial(CKPT, DEV)
    T = 32
    lo = (ex.PARAM_LO.to(DEV) - pmean) / pstd
    hi = (ex.PARAM_HI.to(DEV) - pmean) / pstd

    # canonical context (object/ignition is ~param-independent); params drive the rest
    mid = ((ex.PARAM_LO + ex.PARAM_HI) / 2)[None]
    cframes, _ = ex.simulate(mid, H=112, W=112, steps=T, device=DEV)
    zctx = model.encode((cframes[0, :K].float() / 255.0).to(DEV))[None]      # (1,K,C,14,14)

    key = [t - K for t in (4, 7, 10, 13)]                                    # fireball-peak pred steps
    S, iters, topk = 160, 9, 22
    gen = torch.Generator(device=DEV).manual_seed(0)

    results = []
    for name, spec in TARGETS.items():
        tmask = mask(spec)
        mean, std = torch.zeros(8, device=DEV), torch.ones(8, device=DEV)
        for it in range(iters):
            cand = (mean + std * torch.randn(S, 8, generator=gen, device=DEV)).clamp(lo, hi)
            zpred = model.rollout(zctx.expand(S, -1, -1, -1, -1), cand, T - K)
            dec = model.decode(zpred[:, key].reshape(S * len(key), model.C, 14, 14))
            env = dec.mean(1).reshape(S, len(key), 112, 112).amax(1)         # brightness envelope
            cost = (env - tmask).pow(2).mean(dim=(1, 2))
            mean, std = cand[cost.topk(topk, largest=False).indices].mean(0), \
                cand[cost.topk(topk, largest=False).indices].std(0) + 1e-4
        rec = (mean * pstd + pmean).clamp(ex.PARAM_LO.to(DEV), ex.PARAM_HI.to(DEV))
        rframes, _ = ex.simulate(rec[None].cpu(), H=112, W=112, steps=T, device=DEV)
        results.append((name, tmask.cpu().numpy(), rframes[0].numpy(), rec.cpu().numpy()))
        save_gif([chw_to_hwc_u8(rframes[0, t].numpy()) for t in range(T)],
                 OUT / f"art_{name.split()[0]}.gif", scale=3, fps=10)
        print(f"  [{name}] best cost {cost.min().item():.4f} -> art_{name.split()[0]}.gif")

    # combined figure: per target -> target mask + resulting explosion frames
    cols = 9
    idx = np.linspace(2, T - 1, cols - 1).astype(int)
    fig, ax = plt.subplots(len(results), cols, figsize=(1.5 * cols, 1.6 * len(results)))
    for r, (name, tmask, rframes, rec) in enumerate(results):
        ax[r, 0].imshow(tmask, cmap="inferno"); ax[r, 0].axis("off")
        ax[r, 0].set_title("target", fontsize=8)
        ax[r, 0].set_ylabel(name, fontsize=9)
        fig.text(0.085, 1 - (r + 0.5) / len(results), name, rotation=90, va="center", ha="right", fontsize=9)
        for j, t in enumerate(idx):
            ax[r, j + 1].imshow(chw_to_hwc_u8(rframes[t])); ax[r, j + 1].axis("off")
            if r == 0:
                ax[r, j + 1].set_title(f"t{t}", fontsize=8)
    fig.suptitle("art-direction: target shape (left) -> explosion from surrogate-planned inputs", fontsize=12)
    fig.savefig(OUT / "art_direction.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {OUT / 'art_direction.png'}")


if __name__ == "__main__":
    main()
