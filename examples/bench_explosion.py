#!/usr/bin/env python3
"""Timing: surrogate (full rollout+decode, and latent-only for planning) vs the sim.

    ~/miniconda3/envs/torch251/bin/python examples/bench_explosion.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explosion3d_sim as ex
from explosion_spatial import load_spatial

CKPT = Path(__file__).resolve().parent / "explosion_ckpt" / "spatial.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = 32


def timed(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


@torch.no_grad()
def main():
    model, pmean, pstd, K = load_spatial(CKPT, DEV)
    C = model.C

    # --- sim ---
    p1 = ex.sample_params(1); p64 = ex.sample_params(64)
    sim1 = timed(lambda: ex.simulate(p1, H=112, W=112, steps=T, device=DEV))
    sim64 = timed(lambda: ex.simulate(p64, H=112, W=112, steps=T, device=DEV)) / 64

    # --- surrogate ---
    true = (ex.simulate(p1, H=112, W=112, steps=T, device=DEV)[0][0].float() / 255).to(DEV)
    zctx = model.encode(true[:K])[None]
    act = ((p1[0].to(DEV) - pmean) / pstd)[None]

    def full(S):
        zp = model.rollout(zctx.expand(S, -1, -1, -1, -1), act.expand(S, -1), T - K)
        return model.decode(zp.reshape(S * (T - K), C, 14, 14))

    def latent(S):
        return model.rollout(zctx.expand(S, -1, -1, -1, -1), act.expand(S, -1), T - K)

    full1 = timed(lambda: full(1))
    full64 = timed(lambda: full(64)) / 64
    lat1 = timed(lambda: latent(1))
    lat256 = timed(lambda: latent(256)) / 256

    ms = lambda s: s * 1000
    print("\n================ timing (RTX 5000, 112x112, 32 frames) ================")
    print(f"  sim (Stable-Fluids/particle):    {ms(sim1):7.1f} ms/seq (B=1) | {ms(sim64):6.2f} ms/seq (B=64)")
    print(f"  surrogate rollout + decode:      {ms(full1):7.1f} ms/seq (B=1) | {ms(full64):6.2f} ms/seq (B=64)")
    print(f"  surrogate latent-only (planning):{ms(lat1):7.1f} ms/seq (B=1) | {ms(lat256):6.3f} ms/seq (B=256)")
    n_eval = 160 * 9
    print(f"\n  planning cost = {n_eval} surrogate latent rollouts:")
    print(f"    via surrogate : {ms(lat256) * n_eval / 1000:6.2f} s")
    print(f"    via sim       : {ms(sim64) * n_eval / 1000:6.2f} s   (and the sim is not differentiable)")
    print("\n  context: a production Houdini pyro sim is ~seconds-minutes PER FRAME (minutes-hours")
    print("  per sequence). The surrogate runs an explosion in milliseconds and is batchable +")
    print("  plannable — that gap is the real value; here both are fast only because our sim is toy.")


if __name__ == "__main__":
    main()
