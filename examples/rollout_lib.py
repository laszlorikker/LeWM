"""Reusable latent-rollout visualization for LeWM (shared by viz_over_epochs.py).

Builds a fixed held-out rollout window + a nearest-neighbor bank once (cached to .npz so the
disk reads happen a single time), then `render(model, cache, ...)` re-encodes with whatever
checkpoint you pass and draws: true frames vs NN-decoded predicted frames, the predicted-vs-true
agent trajectory (latent->position probe), and latent MSE vs horizon.
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
from stable_pretraining import data as dt
from utils import get_img_preprocessor, get_column_normalizer

HS = 3          # predictor context length
K = 6           # rollout horizon
FRAMESKIP = 5
T = HS + K


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
def _encode(model, pix, device):
    return model.encode({"pixels": pix.to(device)})["emb"]


def build_cache(out_npz, seed=0, bank=1200):
    """Pick a fixed high-motion window + NN bank, cache raw reads to npz (done once)."""
    rng = np.random.default_rng(seed)
    img_pre = get_img_preprocessor("pixels", "pixels", 224)

    ds_raw = swm.data.load_dataset("tworoom.h5", num_steps=T, frameskip=FRAMESKIP,
                                   keys_to_load=["pixels", "action", "proprio"])
    act_norm = get_column_normalizer(ds_raw, "action", "action")
    ds_t = swm.data.load_dataset("tworoom.h5", num_steps=T, frameskip=FRAMESKIP,
                                 keys_to_load=["pixels", "action", "proprio"])
    ds_t.transform = dt.transforms.Compose(img_pre, act_norm)

    # high-motion window (the dataset indexes by within-episode clips)
    cand = rng.choice(len(ds_raw), size=100, replace=False)
    r, best = int(cand[0]), -1.0
    for c in cand:
        pr = np.asarray(ds_raw[int(c)]["proprio"]).reshape(T, 2)
        d = float(np.linalg.norm(pr[-1] - pr[0]))
        if d > best:
            best, r = d, int(c)

    win_t, win_raw = ds_t[r], ds_raw[r]
    window_pix = np.asarray(win_t["pixels"]).astype(np.float32)            # (T,3,224,224) normalized
    window_act = np.nan_to_num(np.asarray(win_t["action"])).astype(np.float32)  # (T,10)
    window_proprio = np.asarray(win_raw["proprio"]).reshape(T, 2).astype(np.float32)
    window_frames = np.asarray(win_raw["pixels"]).astype(np.uint8)        # (T,3,224,224) raw

    # NN bank: single frames
    ds1 = swm.data.load_dataset("tworoom.h5", num_steps=1, frameskip=1,
                                keys_to_load=["pixels", "proprio"])
    bidx = np.sort(rng.choice(len(ds1), size=bank, replace=False))
    bank_raw, bank_pos = [], []
    for i in bidx:
        w = ds1[int(i)]
        bank_raw.append(np.asarray(w["pixels"])[0].astype(np.uint8))      # (3,224,224)
        bank_pos.append(np.asarray(w["proprio"]).reshape(-1, 2)[0])

    np.savez(out_npz,
             window_pix=window_pix, window_act=window_act, window_proprio=window_proprio,
             window_frames=window_frames, bank_raw=np.stack(bank_raw),
             bank_pos=np.stack(bank_pos).astype(np.float32),
             motion=np.float32(best), clip=np.int64(r))
    print(f"cache built: clip {r}, agent moves {best:.0f}px, bank={bank} -> {out_npz}")
    return out_npz


def load_cache(npz):
    return {k: v for k, v in np.load(npz, allow_pickle=False).items()}


@torch.no_grad()
def rollout_metrics(model, cache, device):
    img_pre = get_img_preprocessor("pixels", "pixels", 224)
    pix = torch.as_tensor(cache["window_pix"])[None].float()
    act = torch.as_tensor(cache["window_act"])[None].float()

    true_emb = _encode(model, pix, device)                 # (1,T,D)
    emb_roll = true_emb[:, :HS].clone()
    aw = act[:, :HS].to(device)
    preds = []
    for k in range(K):
        if k > 0:
            aw = torch.cat([aw, act[:, HS - 1 + k:HS + k].to(device)], 1)
        a_emb = model.action_encoder(aw[:, -HS:])
        p = model.predict(emb_roll[:, -HS:], a_emb)[:, -1:]
        emb_roll = torch.cat([emb_roll, p], 1)
        preds.append(p)
    pred_emb = torch.cat(preds, 1)                          # (1,K,D)
    true_future = true_emb[:, HS:HS + K]
    latent_mse = ((pred_emb - true_future) ** 2).mean(-1)[0].cpu().numpy()

    # encode NN bank with the current model
    raw = cache["bank_raw"]
    bank_emb = []
    for j in range(0, len(raw), 128):
        bp = img_pre({"pixels": torch.as_tensor(raw[j:j + 128])})["pixels"]
        bank_emb.append(_encode(model, bp[:, None].float(), device)[:, 0])
    bank_emb = torch.cat(bank_emb)
    nn = torch.cdist(pred_emb[0], bank_emb).argmin(1).cpu().numpy()

    from sklearn.linear_model import Ridge
    probe = Ridge(alpha=1.0).fit(bank_emb.cpu().numpy(), cache["bank_pos"])
    pos_pred = probe.predict(pred_emb[0].cpu().numpy())     # (K,2)
    return dict(latent_mse=latent_mse, nn=nn, pos_pred=pos_pred)


def render(model, cache, device, epoch, out_png):
    m = rollout_metrics(model, cache, device)
    frames = [chw_to_hwc(f) for f in cache["window_frames"]]
    bank_img = cache["bank_raw"]
    proprio = cache["window_proprio"]
    nn, pos_pred, latent_mse = m["nn"], m["pos_pred"], m["latent_mse"]

    fig = plt.figure(figsize=(2.0 * T, 7.2))
    gs = fig.add_gridspec(3, T, height_ratios=[1, 1, 1.5], hspace=0.32, wspace=0.06)
    for j in range(T):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(frames[j]); ax.axis("off")
        ax.set_title(("ctx " if j < HS else "true ") + f"t{j}", fontsize=9,
                     color=("0.4" if j < HS else "C3"))
    for j in range(T):
        ax = fig.add_subplot(gs[1, j]); ax.axis("off")
        if j < HS:
            ax.imshow(frames[j]); ax.set_title("(given)", fontsize=8, color="0.6")
        else:
            ax.imshow(chw_to_hwc(bank_img[nn[j - HS]])); ax.set_title(f"pred t{j}", fontsize=9, color="C2")

    axt = fig.add_subplot(gs[2, : T // 2])
    axt.plot(proprio[:, 0], proprio[:, 1], "-o", color="C3", ms=4, label="true path")
    axt.plot(proprio[:HS, 0], proprio[:HS, 1], "o", color="0.4", ms=7, label="context")
    pp = np.vstack([proprio[HS - 1], pos_pred])
    axt.plot(pp[:, 0], pp[:, 1], "--s", color="C2", ms=4, label="predicted (probe)")
    axt.set_xlim(0, 224); axt.set_ylim(224, 0); axt.set_aspect("equal")
    axt.set_title("agent trajectory (pixels)", fontsize=10); axt.legend(fontsize=8)

    axm = fig.add_subplot(gs[2, T // 2:])
    axm.plot(range(1, K + 1), latent_mse, "-o", color="C0")
    axm.set_xlabel("rollout step"); axm.set_ylabel("latent MSE (pred vs true)")
    axm.set_title("prediction error vs horizon", fontsize=10); axm.grid(alpha=0.3)

    fig.suptitle(f"LeWM two-room validation rollout @ epoch {epoch} — "
                 f"top: true frames | middle: NN-decoded predictions", fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return m
