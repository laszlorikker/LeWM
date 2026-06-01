#!/usr/bin/env python3
"""Render a LeWM rollout with frames produced DIRECTLY by the trained pixel decoder
(examples/train_decoder.py). Reuses the held-out window cached by rollout_lib.

    STABLEWM_HOME=$HOME/.stable-wm \
    ~/miniconda3/envs/torch251/bin/python examples/viz_decoded_rollout.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm
import rollout_lib as RL
from train_decoder import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HS, K, T = RL.HS, RL.K, RL.T
VP = Path(__file__).resolve().parent / "viz_progress"


def hwc(a):
    return np.clip(np.transpose(np.asarray(a), (1, 2, 0)), 0, 1)


@torch.no_grad()
def main():
    ckpt = "lewm/weights_epoch_100.pt"
    cache = RL.load_cache(str(VP / "rollout_cache.npz"))
    model = swm.wm.utils.load_pretrained(ckpt).to(DEVICE).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    d = torch.load(VP / "decoder.pt", map_location=DEVICE)
    dec = Decoder()
    dec.load_state_dict(d["state_dict"])
    dec.to(DEVICE).eval()

    pix = torch.as_tensor(cache["window_pix"])[None].float()
    act = torch.as_tensor(cache["window_act"])[None].float()

    true_emb = model.encode({"pixels": pix.to(DEVICE)})["emb"]          # (1,T,192)
    emb_roll = true_emb[:, :HS].clone()
    aw = act[:, :HS].to(DEVICE)
    preds = []
    for k in range(K):
        if k > 0:
            aw = torch.cat([aw, act[:, HS - 1 + k:HS + k].to(DEVICE)], 1)
        a_emb = model.action_encoder(aw[:, -HS:])
        p = model.predict(emb_roll[:, -HS:], a_emb)[:, -1:]
        emb_roll = torch.cat([emb_roll, p], 1)
        preds.append(p)
    pred_emb = torch.cat(preds, 1)                                      # (1,K,192)

    dec_true = dec(true_emb[0]).cpu().numpy()                           # (T,3,112,112)
    dec_pred = dec(pred_emb[0]).cpu().numpy()                           # (K,3,112,112)
    frames = cache["window_frames"]                                     # (T,3,224,224) uint8

    rows = [("true frame", "C3"), ("decode(true latent)", "0.5"), ("decode(PREDICTED latent)", "C2")]
    fig = plt.figure(figsize=(2.0 * T, 6.4))
    gs = fig.add_gridspec(3, T, hspace=0.16, wspace=0.05)
    for j in range(T):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(RL.chw_to_hwc(frames[j])); ax.axis("off")
        ax.set_title(("ctx " if j < HS else "true ") + f"t{j}", fontsize=9, color=("0.4" if j < HS else "C3"))
    for j in range(T):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(hwc(dec_true[j])); ax.axis("off")
    for j in range(T):
        ax = fig.add_subplot(gs[2, j]); ax.axis("off")
        if j < HS:
            ax.imshow(hwc(dec_true[j])); ax.set_title("(given)", fontsize=8, color="0.6")
        else:
            ax.imshow(hwc(dec_pred[j - HS])); ax.set_title(f"pred t{j}", fontsize=9, color="C2")
    # row labels on the left margin
    for i, (lab, col) in enumerate(rows):
        fig.text(0.085, 0.78 - i * 0.29, lab, va="center", ha="right", rotation=90, fontsize=10, color=col)

    fig.suptitle("LeWM two-room rollout — frames rendered DIRECTLY by a trained pixel decoder "
                 "(epoch 100)\nrow2 = decoder reconstruction (latent ceiling) · row3 = decoded "
                 "predicted latents vs row1 truth", fontsize=11)
    out = VP / "decoded_rollout.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
