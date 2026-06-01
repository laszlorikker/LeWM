#!/usr/bin/env python3
"""Generate a cached dataset of (8-param -> 32-frame explosion video) sequences for training the
LeWM surrogate. Frames are uint8 (N,T,3,H,W); params are float32 (N,8).

    STABLEWM_HOME unused here. Run:
    ~/miniconda3/envs/torch251/bin/python examples/gen_explosion_dataset.py --n 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explosion3d_sim as ex

OUT = Path(__file__).resolve().parent / "explosion_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    params = ex.sample_params(args.n, seed=12345)               # (N,8) deterministic
    frames = np.empty((args.n, args.steps, 3, args.res, args.res), np.uint8)
    for i in range(0, args.n, args.batch):
        p = params[i:i + args.batch]
        f, _ = ex.simulate(p, H=args.res, W=args.res, steps=args.steps, device=args.device)
        frames[i:i + args.batch] = f.numpy()
        print(f"  {min(i + args.batch, args.n)}/{args.n}")
    np.save(OUT / "frames.npy", frames)
    np.save(OUT / "params.npy", params.numpy().astype(np.float32))
    gb = frames.nbytes / 1e9
    print(f"saved frames {frames.shape} ({gb:.1f} GB) + params {tuple(params.shape)} -> {OUT}")


if __name__ == "__main__":
    main()
