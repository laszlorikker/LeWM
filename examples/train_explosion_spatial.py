#!/usr/bin/env python3
"""Train the spatial-latent explosion surrogate: joint autoencoder + action-conditioned conv
predictor, with reconstruction + multi-step rollout pixel loss (-> crisp predicted frames).

    ~/miniconda3/envs/torch251/bin/python examples/train_explosion_spatial.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explosion_spatial import SpatialSurrogate

DATA = Path(__file__).resolve().parent / "explosion_data"
CKPT = Path(__file__).resolve().parent / "explosion_ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--K", type=int, default=3)        # context frames
    ap.add_argument("--R", type=int, default=4)        # train rollout horizon
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    K, R, W = args.K, args.R, args.K + args.R
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)

    frames = torch.from_numpy(np.load(DATA / "frames.npy")).to(dev)          # (N,T,3,H,W) uint8
    p_raw = torch.from_numpy(np.load(DATA / "params.npy")).to(dev).float()
    p_mean, p_std = p_raw.mean(0), p_raw.std(0) + 1e-6
    params = (p_raw - p_mean) / p_std
    N, T, _, H, Wd = frames.shape
    print(f"data {tuple(frames.shape)} on {dev}; window K={K} R={R}")

    model = SpatialSurrogate(args.C, K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()

    for step in range(args.steps):
        seq = torch.randint(0, N, (args.batch,), generator=gen, device=dev)
        start = torch.randint(0, T - W + 1, (args.batch,), generator=gen, device=dev)
        idx = start[:, None] + torch.arange(W, device=dev)
        x = frames[seq[:, None], idx].float() / 255.0                        # (B,W,3,H,W)
        act = params[seq]                                                    # (B,8)

        z = model.encode(x.reshape(-1, 3, H, Wd)).reshape(args.batch, W, args.C, 14, 14)
        recon = F.l1_loss(model.decode(z.reshape(-1, args.C, 14, 14)), x.reshape(-1, 3, H, Wd))

        zs = [z[:, i] for i in range(K)]
        for _ in range(R):
            zs.append(model.predict_next(torch.stack(zs[-K:], 1), act))
        zpred = torch.stack(zs[K:], 1)                                       # (B,R,C,14,14)
        pred_lat = F.mse_loss(zpred, z[:, K:K + R].detach())
        pred_pix = F.l1_loss(model.decode(zpred.reshape(-1, args.C, 14, 14)),
                             x[:, K:K + R].reshape(-1, 3, H, Wd))
        loss = recon + pred_pix + 0.25 * pred_lat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} | loss {loss.item():.4f} | recon {recon.item():.4f} "
                  f"| pred_pix {pred_pix.item():.4f} | pred_lat {pred_lat.item():.4f}")

    torch.save({"state_dict": model.state_dict(), "C": args.C, "K": K,
                "p_mean": p_mean.cpu().numpy(), "p_std": p_std.cpu().numpy()},
               CKPT / "spatial.pt")
    print(f"saved -> {CKPT / 'spatial.pt'}")


if __name__ == "__main__":
    main()
