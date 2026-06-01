#!/usr/bin/env python3
"""Render the same held-out LeWM rollout at a series of checkpoints (e.g. every 10 epochs) and
build a summary showing the prediction sharpening over training.

    STABLEWM_HOME=$HOME/.stable-wm \
    ~/miniconda3/envs/torch251/bin/python examples/viz_over_epochs.py --epochs 10,20,...,100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm
import rollout_lib as RL

OUTDIR = Path(__file__).resolve().parent / "viz_progress"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--run", default="lewm", help="checkpoint subdir under $STABLEWM_HOME/checkpoints")
    ap.add_argument("--cache", default=str(OUTDIR / "rollout_cache.npz"))
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = [int(e) for e in args.epochs.split(",") if e.strip()]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cache or not Path(args.cache).exists():
        RL.build_cache(args.cache)
    cache = RL.load_cache(args.cache)
    print(f"window clip {int(cache['clip'])}, motion {float(cache['motion']):.0f}px; epochs {epochs}")

    results = {}
    for e in epochs:
        ckpt = f"{args.run}/weights_epoch_{e}.pt"
        try:
            model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        except Exception as ex:
            print(f"  epoch {e}: SKIP ({type(ex).__name__}: {str(ex)[:60]})")
            continue
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        m = RL.render(model, cache, device, e, OUTDIR / f"viz_epoch_{e:03d}.png")
        results[e] = m
        print(f"  epoch {e:3d}: mean latent MSE {m['latent_mse'].mean():.3f} -> viz_epoch_{e:03d}.png")

    if not results:
        print("no checkpoints found — nothing to summarize.")
        return

    # ---------- summary figure ----------
    eps = sorted(results)
    colors = cm.viridis(np.linspace(0, 1, len(eps)))
    proprio = cache["window_proprio"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) latent MSE vs horizon, one line per epoch
    for c, e in zip(colors, eps):
        ax[0].plot(range(1, RL.K + 1), results[e]["latent_mse"], "-o", color=c, label=f"ep{e}")
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("latent MSE (pred vs true)")
    ax[0].set_title("rollout error vs horizon, per epoch"); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7, ncol=2)

    # (b) predicted trajectory per epoch vs the true path
    ax[1].plot(proprio[:, 0], proprio[:, 1], "-o", color="k", lw=2.5, ms=4, label="true", zorder=5)
    ax[1].plot(proprio[:RL.HS, 0], proprio[:RL.HS, 1], "o", color="0.5", ms=8, label="context", zorder=6)
    for c, e in zip(colors, eps):
        pp = np.vstack([proprio[RL.HS - 1], results[e]["pos_pred"]])
        ax[1].plot(pp[:, 0], pp[:, 1], "--s", color=c, ms=3, alpha=0.9)
    ax[1].set_xlim(0, 224); ax[1].set_ylim(224, 0); ax[1].set_aspect("equal")
    ax[1].set_title("predicted trajectory converging to true"); ax[1].legend(fontsize=8)

    # (c) mean latent MSE vs epoch
    means = [results[e]["latent_mse"].mean() for e in eps]
    ax[2].plot(eps, means, "-o", color="C0")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("mean rollout latent MSE")
    ax[2].set_title("rollout error vs training epoch"); ax[2].grid(alpha=0.3, which="both")
    ax[2].set_yscale("log")
    np.savez(OUTDIR / "metrics.npz", epochs=np.array(eps),
             mse=np.array([results[e]["latent_mse"] for e in eps]))

    fig.suptitle("LeWM two-room: validation rollout sharpening over 100 epochs", fontsize=13)
    fig.tight_layout()
    out = OUTDIR / "summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
