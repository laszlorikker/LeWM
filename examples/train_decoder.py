#!/usr/bin/env python3
"""Fit a pixel decoder on FROZEN LeWM latents, so predicted latents can be rendered as images.

LeWM is a latent world model (no decoder). Here we freeze a trained LeWM encoder+projector and
learn a small conv decoder  emb(192) -> image(112x112x3)  by reconstructing real frames. Because
the predictor's outputs live in the same latent space as `emb` (the loss matches them), the same
decoder renders predicted latents directly. Tied to one checkpoint's latent space.

    STABLEWM_HOME=$HOME/.stable-wm \
    ~/miniconda3/envs/torch251/bin/python examples/train_decoder.py --ckpt lewm/weights_epoch_100.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 112


class Decoder(nn.Module):
    """emb (192) -> 7x7 -> (x2)^4 -> 112x112x3 in [0,1]."""

    def __init__(self, dim=192, ch=256):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(dim, ch * 7 * 7)

        def up(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            )

        self.net = nn.Sequential(
            up(ch, 128), up(128, 64), up(64, 32), up(32, 16),   # 7 -> 14 -> 28 -> 56 -> 112
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.ch, 7, 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lewm/weights_epoch_100.pt")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "viz_progress" / "decoder.pt"))
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = swm.wm.utils.load_pretrained(args.ckpt).to(DEVICE).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    img_pre = get_img_preprocessor("pixels", "pixels", 224)

    # ---- pre-encode a random set of frames with the frozen encoder ----
    ds1 = swm.data.load_dataset("tworoom.h5", num_steps=1, frameskip=1, keys_to_load=["pixels"])
    idx = np.sort(rng.choice(len(ds1), size=args.n, replace=False)).tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds1, idx),
                                         batch_size=128, num_workers=8, pin_memory=False)
    print(f"pre-encoding {args.n} frames with frozen encoder ({args.ckpt})...")
    embs, targs = [], []
    with torch.no_grad():
        for b in loader:
            raw = b["pixels"]
            raw = (torch.as_tensor(np.asarray(raw)) if not torch.is_tensor(raw) else raw).squeeze(1)  # (B,3,224,224) uint8
            norm = img_pre({"pixels": raw})["pixels"].float().to(DEVICE)
            embs.append(model.encode({"pixels": norm[:, None]})["emb"][:, 0].cpu())
            t = F.interpolate(raw.float() / 255.0, size=RES, mode="area")
            targs.append((t * 255).to(torch.uint8))
    embs = torch.cat(embs)
    targs = torch.cat(targs)
    print(f"pairs: emb {tuple(embs.shape)}  target {tuple(targs.shape)}")

    # ---- train the decoder ----
    dec = Decoder().to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    N, bs = embs.size(0), 128
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot = nb = 0
        for i in range(0, N - bs + 1, bs):
            ii = perm[i:i + bs]
            z = embs[ii].to(DEVICE)
            tgt = (targs[ii].float() / 255.0).to(DEVICE)
            loss = F.mse_loss(dec(z), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:2d} | recon MSE {tot / nb:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dec.state_dict(), "res": RES, "ckpt": args.ckpt}, args.out)
    print(f"saved decoder -> {args.out}")


if __name__ == "__main__":
    main()
