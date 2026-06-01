#!/usr/bin/env python3
"""Visualize a LeWM latent rollout on two-room: predicted vs true.

LeWM predicts in *latent* space (there is no pixel decoder), so "predicted frames" are
produced by **nearest-neighbor decoding**: we roll the predictor forward over a real action
sequence, and for each predicted latent retrieve the real frame whose embedding is closest.
We also fit a tiny latent->(x,y) probe to show the predicted vs true agent trajectory, and plot
latent MSE vs rollout horizon.

Run (needs the trained checkpoint at $STABLEWM_HOME/checkpoints/lewm/weights_epoch_8.pt):
    STABLEWM_HOME=$HOME/.stable-wm \
    ~/miniconda3/envs/torch251/bin/python examples/visualize_rollout.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm
from stable_pretraining import data as dt
from utils import get_img_preprocessor, get_column_normalizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HS = 3          # predictor context length (num_frames)
K = 6           # rollout horizon (future steps to predict)
FRAMESKIP = 5
T = HS + K
CKPT = "lewm/weights_epoch_8.pt"
BANK = 1200     # frames in the nearest-neighbor / probe bank


def chw_to_hwc(a):
    a = np.asarray(a).squeeze()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.uint8)
    return a


@torch.no_grad()
def encode(model, pix):  # pix: (B, T, 3, H, W) float -> (B, T, D)
    return model.encode({"pixels": pix.to(DEVICE)})["emb"]


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ----- model -----
    model = swm.wm.utils.load_pretrained(CKPT).to(DEVICE).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    print(f"loaded {CKPT} on {DEVICE}")

    img_pre = get_img_preprocessor(source="pixels", target="pixels", img_size=224)

    # ----- action normalizer: z-score on the RAW 2-dim action (applied per-frame BEFORE the
    # frameskip-stacking to 10-dim), exactly as train.py does it -----
    ds_raw = swm.data.load_dataset("tworoom.h5", num_steps=T, frameskip=FRAMESKIP,
                                   keys_to_load=["pixels", "action", "proprio"])
    act_norm = get_column_normalizer(ds_raw, "action", "action")

    ds_t = swm.data.load_dataset("tworoom.h5", num_steps=T, frameskip=FRAMESKIP,
                                 keys_to_load=["pixels", "action", "proprio"])
    ds_t.transform = dt.transforms.Compose(img_pre, act_norm)

    # ----- pick a clip (window) with clear agent motion. The dataset indexes by clips, each
    # already within one episode (clip_indices[idx] = (ep_idx, start)). -----
    cand = rng.choice(len(ds_raw), size=100, replace=False)
    r, best_d = int(cand[0]), -1.0
    for c in cand:
        pr = np.asarray(ds_raw[int(c)]["proprio"]).reshape(T, 2)
        d = float(np.linalg.norm(pr[-1] - pr[0]))
        if d > best_d:
            best_d, r = d, int(c)
    epid = ds_raw.clip_indices[r][0] if hasattr(ds_raw, "clip_indices") else "?"
    print(f"rollout clip {r} (episode {epid}), agent moves {best_d:.0f}px over the window")

    win_t = ds_t[r]
    win_raw = ds_raw[r]
    pix = torch.as_tensor(np.asarray(win_t["pixels"]))[None].float()        # (1,T,3,224,224)
    act = torch.nan_to_num(torch.as_tensor(np.asarray(win_t["action"])))[None].float()  # (1,T,10)
    proprio = np.asarray(win_raw["proprio"]).reshape(T, 2)                  # (T,2) pixel coords
    frames_raw = [chw_to_hwc(f) for f in np.asarray(win_raw["pixels"])]      # list of HWC uint8

    # ----- true latents + autoregressive rollout using the TRUE actions -----
    true_emb = encode(model, pix)                       # (1, T, D)
    emb_roll = true_emb[:, :HS].clone()                 # seed with real context latents
    aw = act[:, :HS].clone().to(DEVICE)
    preds = []
    for k in range(K):
        if k > 0:
            aw = torch.cat([aw, act[:, HS - 1 + k:HS + k].to(DEVICE)], dim=1)
        a_emb = model.action_encoder(aw[:, -HS:])
        p = model.predict(emb_roll[:, -HS:], a_emb)[:, -1:]   # (1,1,D)
        emb_roll = torch.cat([emb_roll, p], dim=1)
        preds.append(p)
    pred_emb = torch.cat(preds, dim=1)                   # (1, K, D)  predicted future latents
    true_future = true_emb[:, HS:HS + K]                 # (1, K, D)
    latent_mse = ((pred_emb - true_future) ** 2).mean(-1)[0].cpu().numpy()   # (K,)

    # ----- bank for NN-decode + latent->position probe (single-frame reads) -----
    ds1 = swm.data.load_dataset("tworoom.h5", num_steps=1, frameskip=1,
                                keys_to_load=["pixels", "proprio"])
    bank_idx = np.sort(rng.choice(len(ds1), size=BANK, replace=False))
    bank_pix, bank_img, bank_pos = [], [], []
    for i in bank_idx:
        w = ds1[i]
        f0 = np.asarray(w["pixels"])[0]                  # single frame, CHW uint8
        bank_img.append(chw_to_hwc(f0))
        bank_pix.append(img_pre({"pixels": torch.as_tensor(f0)[None]})["pixels"][0])
        bank_pos.append(np.asarray(w["proprio"]).reshape(-1, 2)[0])
    bank_pix = torch.stack(bank_pix)[:, None].float()    # (BANK,1,3,224,224)
    bank_emb = torch.cat([encode(model, bank_pix[j:j + 128]) for j in range(0, BANK, 128)])[:, 0]
    bank_pos = np.stack(bank_pos)

    # nearest-neighbor decode: each predicted latent -> closest bank frame
    d = torch.cdist(pred_emb[0], bank_emb)               # (K, BANK)
    nn = d.argmin(1).cpu().numpy()

    # latent -> (x,y) probe (ridge), fit on the bank
    from sklearn.linear_model import Ridge
    probe = Ridge(alpha=1.0).fit(bank_emb.cpu().numpy(), bank_pos)
    pos_pred = probe.predict(pred_emb[0].cpu().numpy())  # (K,2) predicted agent positions

    # ===================== figure =====================
    ncol = T
    fig = plt.figure(figsize=(2.0 * ncol, 7.2))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1, 1, 1.5], hspace=0.32, wspace=0.06)

    for j in range(ncol):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(frames_raw[j]); ax.axis("off")
        ax.set_title(("ctx " if j < HS else "true ") + f"t{j}", fontsize=9,
                     color=("0.4" if j < HS else "C3"))
    for j in range(ncol):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        if j < HS:
            ax.imshow(frames_raw[j]); ax.set_title("(given)", fontsize=8, color="0.6")
        else:
            ax.imshow(bank_img[nn[j - HS]]); ax.set_title(f"pred t{j}", fontsize=9, color="C2")

    # trajectory: true (red) vs predicted (green)
    axt = fig.add_subplot(gs[2, : ncol // 2])
    axt.plot(proprio[:, 0], proprio[:, 1], "-o", color="C3", ms=4, label="true path")
    axt.plot(proprio[:HS, 0], proprio[:HS, 1], "o", color="0.4", ms=7, label="context")
    pp = np.vstack([proprio[HS - 1], pos_pred])
    axt.plot(pp[:, 0], pp[:, 1], "--s", color="C2", ms=4, label="predicted (probe)")
    axt.set_xlim(0, 224); axt.set_ylim(224, 0); axt.set_aspect("equal")
    axt.set_title("agent trajectory (pixels)", fontsize=10); axt.legend(fontsize=8)

    axm = fig.add_subplot(gs[2, ncol // 2:])
    axm.plot(range(1, K + 1), latent_mse, "-o", color="C0")
    axm.set_xlabel("rollout step"); axm.set_ylabel("latent MSE (pred vs true)")
    axm.set_title("prediction error vs horizon", fontsize=10); axm.grid(alpha=0.3)

    fig.suptitle("LeWM two-room rollout — top: true frames | middle: NN-decoded predictions | "
                 f"{HS} context + {K} predicted steps (frameskip {FRAMESKIP})", fontsize=11)
    out = Path(__file__).resolve().parent / "rollout_viz.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"latent MSE per step: {latent_mse.round(2)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
