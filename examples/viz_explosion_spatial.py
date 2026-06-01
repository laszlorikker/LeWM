#!/usr/bin/env python3
"""Crisp forward-rollout demo with the spatial-latent surrogate: montage + side-by-side GIF
(true sim vs surrogate prediction) on HELD-OUT params.

    ~/miniconda3/envs/torch251/bin/python examples/viz_explosion_spatial.py
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
from gif_util import save_gif, hstack, chw_to_hwc_u8

CKPT = Path(__file__).resolve().parent / "explosion_ckpt" / "spatial.pt"
OUT = Path(__file__).resolve().parent / "explosion_viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def rollout_full(model, true, act_n, K):
    """true: (T,3,H,W) float on DEV. -> decoded full sequence (T,3,H,W) + latent MSE per pred step."""
    T = true.shape[0]
    z = model.encode(true[:K])                                   # (K,C,14,14)
    z_true = model.encode(true)                                  # (T,C,14,14) for the error metric
    zpred = model.rollout(z[None], act_n[None], T - K)[0]        # (T-K,C,14,14)
    full = torch.cat([z, zpred], 0)
    dec = model.decode(full)                                     # (T,3,H,W)
    lat_mse = (zpred - z_true[K:]).pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
    return dec, lat_mse


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, pmean, pstd, K = load_spatial(CKPT, DEV)
    print(f"loaded spatial surrogate (K={K}) on {DEV}")
    params_raw = ex.sample_params(4, seed=999)                   # held-out
    frames, _ = ex.simulate(params_raw, H=112, W=112, steps=32, device=DEV)

    for i in range(2):
        true = (frames[i].float() / 255.0).to(DEV)
        act_n = (params_raw[i].to(DEV) - pmean) / pstd
        dec, lat_mse = rollout_full(model, true, act_n, K)
        true_u8 = frames[i].numpy()
        pred_u8 = (dec.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        T = true_u8.shape[0]

        # montage
        cols = 11
        idx = np.linspace(0, T - 1, cols).astype(int)
        fig = plt.figure(figsize=(1.7 * cols, 4.2))
        gs = fig.add_gridspec(2, cols, hspace=0.12, wspace=0.05)
        for j, t in enumerate(idx):
            a0 = fig.add_subplot(gs[0, j]); a0.imshow(chw_to_hwc_u8(true_u8[t])); a0.axis("off")
            a0.set_title(f"t{t}", fontsize=8)
            a1 = fig.add_subplot(gs[1, j]); a1.imshow(chw_to_hwc_u8(pred_u8[t])); a1.axis("off")
        fig.text(0.09, 0.70, "true sim", rotation=90, va="center", fontsize=11, color="C3")
        fig.text(0.09, 0.30, "surrogate", rotation=90, va="center", fontsize=11, color="C2")
        fig.suptitle("spatial LeWM explosion surrogate — held-out rollout (top: true | bottom: surrogate)",
                     fontsize=12)
        fig.savefig(OUT / f"spatial_rollout_{i}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

        # side-by-side GIF
        gif = [hstack(chw_to_hwc_u8(true_u8[t]), chw_to_hwc_u8(pred_u8[t])) for t in range(T)]
        save_gif(gif, OUT / f"spatial_rollout_{i}.gif", scale=3, fps=10)
        print(f"  sim {i}: mean latent MSE {lat_mse.mean():.3f} -> spatial_rollout_{i}.png/.gif")


if __name__ == "__main__":
    main()
