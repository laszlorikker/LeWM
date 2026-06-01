#!/usr/bin/env python3
"""Forward-rollout demo: the LeWM surrogate gets the first few frames + the 8 params and predicts
the whole explosion (decoded to pixels), shown vs the ground-truth sim. Uses HELD-OUT params.

    ~/miniconda3/envs/torch251/bin/python examples/viz_explosion_rollout.py
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

CKPT = Path(__file__).resolve().parent / "explosion_ckpt"
OUT = Path(__file__).resolve().parent / "explosion_viz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    m = torch.load(CKPT / "model.pt", map_location=DEV, weights_only=False)
    model = build_surrogate(m["embed_dim"], m["history_size"]).to(DEV)
    model.load_state_dict(m["state_dict"]); model.eval()
    d = torch.load(CKPT / "decoder.pt", map_location=DEV, weights_only=False)
    dec = build_decoder(m["embed_dim"]).to(DEV); dec.load_state_dict(d["state_dict"]); dec.eval()
    pmean = torch.tensor(m["p_mean"], device=DEV); pstd = torch.tensor(m["p_std"], device=DEV)
    return model, dec, pmean, pstd, m["history_size"]


def hwc(a):
    return np.clip(np.transpose(np.asarray(a), (1, 2, 0)), 0, 1)


@torch.no_grad()
def rollout(model, dec, true, act_n, HS):
    """true: (T,3,H,W) float on DEV. act_n: (8,) normalized params. Returns decoded pred + latent MSE."""
    T = true.shape[0]
    true_emb = model.encode({"pixels": true[None]})["emb"]            # (1,T,D)
    act_seq = act_n.view(1, 1, -1).expand(1, HS, act_n.numel())
    a_emb = model.action_encoder(act_seq)                            # (1,HS,D) (params constant)
    emb_roll = true_emb[:, :HS].clone()
    for _ in range(HS, T):
        pred = model.predict(emb_roll[:, -HS:], a_emb)[:, -1:]
        emb_roll = torch.cat([emb_roll, pred], 1)
    pred_emb = emb_roll[:, HS:]                                      # (1,T-HS,D)
    latent_mse = ((pred_emb - true_emb[:, HS:]) ** 2).mean(-1)[0].cpu().numpy()
    dec_ctx = dec(true_emb[0, :HS]).cpu().numpy()
    dec_pred = dec(pred_emb[0]).cpu().numpy()
    return dec_ctx, dec_pred, latent_mse


def figure(true_u8, dec_ctx, dec_pred, HS, latent_mse, out_png, title):
    T = true_u8.shape[0]
    cols = 11
    idx = np.linspace(0, T - 1, cols).astype(int)
    fig = plt.figure(figsize=(1.7 * cols, 5.6))
    gs = fig.add_gridspec(3, cols, height_ratios=[1, 1, 0.9], hspace=0.28, wspace=0.05)
    for j, t in enumerate(idx):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(hwc(true_u8[t] / 255.0)); ax.axis("off")
        ax.set_title(f"t{t}", fontsize=8, color=("0.4" if t < HS else "C3"))
    for j, t in enumerate(idx):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        img = dec_ctx[t] if t < HS else dec_pred[t - HS]
        ax.imshow(hwc(img))
        ax.set_title("(ctx)" if t < HS else "pred", fontsize=8, color=("0.6" if t < HS else "C2"))
    axm = fig.add_subplot(gs[2, :])
    axm.plot(range(HS, T), latent_mse, "-o", color="C0", ms=3)
    axm.set_xlabel("rollout step"); axm.set_ylabel("latent MSE"); axm.grid(alpha=0.3)
    axm.set_title("surrogate rollout error vs horizon", fontsize=9)
    fig.text(0.09, 0.81, "true sim", rotation=90, va="center", fontsize=10, color="C3")
    fig.text(0.09, 0.55, "surrogate", rotation=90, va="center", fontsize=10, color="C2")
    fig.suptitle(title, fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, dec, pmean, pstd, HS = load()
    print(f"loaded surrogate (HS={HS}) on {DEV}")

    params_raw = ex.sample_params(4, seed=999)                       # HELD-OUT params
    frames, _ = ex.simulate(params_raw, H=112, W=112, steps=32, device=DEV)
    for i in range(2):
        true = (frames[i].float() / 255.0).to(DEV)
        act_n = (params_raw[i].to(DEV) - pmean) / pstd
        dec_ctx, dec_pred, mse = rollout(model, dec, true, act_n, HS)
        figure(frames[i].numpy(), dec_ctx, dec_pred, HS, mse, OUT / f"rollout_{i}.png",
               f"LeWM explosion surrogate — held-out rollout (top: true sim | bottom: surrogate prediction)")
        print(f"  sim {i}: mean latent MSE {mse.mean():.3f} -> rollout_{i}.png")


if __name__ == "__main__":
    main()
