#!/usr/bin/env python3
"""Art-direction demo: given a TARGET explosion, use CEM over the 8 params — evaluated with the
fast LeWM surrogate — to recover the inputs that reproduce it, then verify by running the true sim
once with the recovered inputs.

    ~/miniconda3/envs/torch251/bin/python examples/plan_explosion.py
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
from explosion_model import build_surrogate, build_decoder
from viz_explosion_rollout import load, hwc, CKPT  # reuse loader

OUT = Path(__file__).resolve().parent / "explosion_viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def batch_rollout(model, ctx_emb, act_n, T, HS):
    """ctx_emb:(1,HS,D) shared context; act_n:(S,8) normalized params -> pred latents (S,T-HS,D)."""
    S = act_n.size(0)
    emb = ctx_emb.expand(S, -1, -1).clone()
    a_emb = model.action_encoder(act_n.view(S, 1, -1).expand(S, HS, -1))
    preds = []
    for _ in range(HS, T):
        p = model.predict(emb[:, -HS:], a_emb)[:, -1:]
        emb = torch.cat([emb, p], 1)
        preds.append(p)
    return torch.cat(preds, 1)


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, dec, pmean, pstd, HS = load()
    T = 32

    # ---- target (held-out) ----
    target_raw = ex.sample_params(1, seed=777)
    tgt_frames, _ = ex.simulate(target_raw, H=112, W=112, steps=T, device=DEV)
    true = (tgt_frames[0].float() / 255.0).to(DEV)
    tgt_emb = model.encode({"pixels": true[None]})["emb"]          # (1,T,D)
    ctx_emb = tgt_emb[:, :HS]
    tgt_pred = tgt_emb[:, HS:]                                     # latent trajectory to match

    # ---- CEM over normalized params (0 = dataset mean) ----
    lo = ((ex.PARAM_LO.to(DEV) - pmean) / pstd)
    hi = ((ex.PARAM_HI.to(DEV) - pmean) / pstd)
    mean = torch.zeros(8, device=DEV); std = torch.ones(8, device=DEV)
    S, iters, topk = 400, 10, 40
    gen = torch.Generator(device=DEV).manual_seed(0)
    for it in range(iters):
        cand = (mean + std * torch.randn(S, 8, generator=gen, device=DEV)).clamp(lo, hi)
        pred = batch_rollout(model, ctx_emb, cand, T, HS)         # (S,T-HS,D)
        cost = (pred - tgt_pred).pow(2).mean(dim=(-1, -2))        # (S,)
        elite = cand[cost.topk(topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-4
        if it % 3 == 0 or it == iters - 1:
            print(f"  CEM iter {it:2d} | best cost {cost.min().item():.4f}")

    recovered_raw = (mean * pstd + pmean).clamp(ex.PARAM_LO.to(DEV), ex.PARAM_HI.to(DEV))

    # ---- verify: run the TRUE sim with recovered params ----
    rec_frames, _ = ex.simulate(recovered_raw[None].cpu(), H=112, W=112, steps=T, device=DEV)

    print("\n  param recovery (target vs recovered):")
    for k, nm in enumerate(ex.PARAM_NAMES):
        print(f"    {nm:11s} {target_raw[0, k].item():7.3f}  ->  {recovered_raw[k].item():7.3f}")

    # ---- figure: target true (top) vs explosion from recovered inputs (bottom) ----
    cols = 11
    idx = np.linspace(0, T - 1, cols).astype(int)
    fig, ax = plt.subplots(2, cols, figsize=(1.7 * cols, 3.7))
    for j, t in enumerate(idx):
        ax[0, j].imshow(hwc(tgt_frames[0, t].numpy() / 255.0)); ax[0, j].axis("off")
        ax[0, j].set_title(f"t{t}", fontsize=8)
        ax[1, j].imshow(hwc(rec_frames[0, t].numpy() / 255.0)); ax[1, j].axis("off")
    fig.text(0.09, 0.70, "TARGET", rotation=90, va="center", fontsize=11, color="C3")
    fig.text(0.09, 0.30, "recovered", rotation=90, va="center", fontsize=11, color="C2")
    fig.suptitle("art-direction: target explosion (top) vs sim from surrogate-recovered inputs (bottom)",
                 fontsize=12)
    out = OUT / "planning.png"
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
