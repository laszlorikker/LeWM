#!/usr/bin/env python3
"""Synthetic pyro engine — a 2D grid fluid solver in the same family as Houdini's pyro solver,
used to generate action/param-conditioned explosion data for a LeWM surrogate (no Houdini needed).

Physics (per step): buoyancy + wind + vorticity-confinement forces -> semi-Lagrangian velocity
advection -> pressure projection (incompressible) -> advect temperature & density -> dissipate.
An "explosion" = a t0 injection of hot, dense material + an outward radial velocity impulse.

Inputs (the conditioning vector) map onto Houdini pyro knobs:
    ex, ey       emitter position            (Houdini: source position)
    blast        ignition heat + blast speed (Houdini: temperature / fuel ignition)
    buoyancy     hot-gas lift                 (Houdini: buoyancy lift)
    wind_x,wind_y ambient force               (Houdini: wind / forces)
    vorticity    turbulence / curl            (Houdini: turbulence / disturbance)
    dissipation  smoke + heat decay           (Houdini: dissipation / cooling)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PARAM_NAMES = ["ex", "ey", "blast", "buoyancy", "wind_x", "wind_y", "vorticity", "dissipation"]
PARAM_LO = torch.tensor([0.30, 0.66, 3.0, 0.6, -0.6, -0.2, 0.05, 0.003])
PARAM_HI = torch.tensor([0.70, 0.80, 7.0, 2.2, 0.6, 0.2, 0.45, 0.011])

# scene constants (ground plane + ballistic debris under gravity)
GROUND = 0.86          # ground height as a fraction of frame (from top)
GRAVITY = 0.22         # debris downward acceleration (cells/step^2)
DRAG = 0.99            # debris air drag per step
RESTITUTION = 0.35     # debris bounce off the ground
FRICTION = 0.7         # debris horizontal damping on bounce
N_DEBRIS = 70
GROUND_COLOR = (0.16, 0.11, 0.07)
DEBRIS_COLOR = (1.0, 0.6, 0.22)


def _nb(x):
    """left,right,up,down neighbors with replicate (Neumann) boundaries. x: (B,1,N,N)."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return xp[..., 1:-1, 0:-2], xp[..., 1:-1, 2:], xp[..., 0:-2, 1:-1], xp[..., 2:, 1:-1]


def _advect(field, u, v, ig, N):
    """Semi-Lagrangian backtrace. field,u,v: (B,1,N,N); ig: (1,N,N,2) identity grid in [-1,1]."""
    gx = ig[..., 0] - u[:, 0] * (2.0 / (N - 1))
    gy = ig[..., 1] - v[:, 0] * (2.0 / (N - 1))
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _project(u, v, iters=50):
    """Make velocity divergence-free via Jacobi pressure solve."""
    lu, ru, _, _ = _nb(u)
    _, _, uv, dv = _nb(v)
    div = 0.5 * ((ru - lu) + (dv - uv))
    p = torch.zeros_like(div)
    for _ in range(iters):
        lp, rp, up_, dp = _nb(p)
        p = 0.25 * (lp + rp + up_ + dp - div)
    lp, rp, up_, dp = _nb(p)
    u = u - 0.5 * (rp - lp)
    v = v - 0.5 * (dp - up_)
    return u, v


def _zero_walls(u, v):
    for a in (u, v):
        a[..., 0, :] = 0; a[..., -1, :] = 0; a[..., :, 0] = 0; a[..., :, -1] = 0
    return u, v


def _splat(pos, heat, color, N):
    """Additively splat K particles (2x2 brush) into a (B,3,N,N) buffer. pos:(B,K,2) heat:(B,K)."""
    B, K, _ = pos.shape
    buf = torch.zeros(B, 3, N * N, device=pos.device)
    px = pos[..., 0].round().long(); py = pos[..., 1].round().long()
    for oy in (0, 1):
        for ox in (0, 1):
            idx = (py + oy).clamp(0, N - 1) * N + (px + ox).clamp(0, N - 1)
            for c in range(3):
                buf[:, c].scatter_add_(1, idx, heat * float(color[c]))
    return buf.view(B, 3, N, N)


def render(d, T, pos, heat, N, ground_row):
    """density/temperature fields + debris particles -> RGB (B,3,N,N) in [0,1]."""
    Tn = (T / 1.3).clamp(0, 1)
    em = (Tn ** 1.3)                                  # emission strength
    fire = torch.cat([(2.2 * Tn).clamp(0, 1),
                      (2.2 * Tn - 0.75).clamp(0, 1),
                      (2.2 * Tn - 1.5).clamp(0, 1)], dim=1)   # red -> yellow -> white
    smoke = (d * 0.9).clamp(0, 1)
    gray = torch.cat([smoke * 0.50, smoke * 0.50, smoke * 0.55], dim=1)   # lingering smoke
    out = fire * em + gray * (1 - em)
    # ground plane (earth band + thin top highlight)
    rows = torch.arange(N, device=d.device).view(1, 1, N, 1)
    gmask = (rows >= ground_row).float()
    gcol = torch.tensor(GROUND_COLOR, device=d.device).view(1, 3, 1, 1)
    out = out * (1 - gmask) + gcol * gmask
    out = out + ((rows >= ground_row) & (rows <= ground_row + 1)).float() * 0.06
    # debris on top: warm body + brighter hot core
    out = out + _splat(pos, heat, DEBRIS_COLOR, N) + _splat(pos, heat ** 2, (0.45, 0.4, 0.3), N)
    return out.clamp(0, 1)


@torch.no_grad()
def simulate(params, N=96, steps=32, device="cuda", render_every=1):
    """params: (B,8) in real units. Returns frames (B,F,3,N,N) uint8 and the params (B,8)."""
    params = params.to(device).float()
    B = params.size(0)
    ex, ey, blast, buoy, wx, wy, vort, diss = [params[:, i].view(B, 1, 1, 1) for i in range(8)]
    grow = int(GROUND * (N - 1))

    ys, xs = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing="ij")
    xs = xs.float()[None, None]; ys = ys.float()[None, None]
    ig = torch.stack([(2 * xs[0, 0] / (N - 1) - 1), (2 * ys[0, 0] / (N - 1) - 1)], dim=-1)[None]  # (1,N,N,2)

    u = torch.zeros(B, 1, N, N, device=device)
    v = torch.zeros_like(u); d = torch.zeros_like(u); T = torch.zeros_like(u)

    # --- t0 explosion injection: hot dense blob + outward radial velocity impulse ---
    cx = ex * (N - 1); cy = ey * (N - 1)
    r = 0.06 * N
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    blob = torch.exp(-dist2 / (2 * r ** 2))
    d += 1.5 * blob; T += 1.8 * blob
    dd = torch.sqrt(dist2) + 1e-3
    u += blast * blob * (xs - cx) / dd
    v += blast * blob * (ys - cy) / dd
    for f in (d, T, u, v):
        f[..., grow + 1:, :] = 0                     # nothing below ground

    # --- ballistic debris (deterministic fan -> params fully determine the sim) ---
    K = N_DEBRIS
    ang = torch.linspace(-math.pi, math.pi, K, device=device)
    diru = torch.stack([torch.cos(ang), torch.sin(ang) - 1.1], -1)        # upward-biased fan
    diru = diru / (diru.norm(dim=-1, keepdim=True) + 1e-6)
    ramp = (0.4 + 0.5 * torch.linspace(0, 1, K, device=device)).view(1, K, 1)
    vel = diru.view(1, K, 2) * ramp * blast.view(B, 1, 1)                 # (B,K,2)
    pos = torch.stack([cx.view(B, 1).expand(B, K), cy.view(B, 1).expand(B, K)], -1).clone()
    heat = torch.ones(B, K, device=device)

    frames = []
    for t in range(steps):
        # --- fluid ---
        v = v - buoy * T * 0.5 + 0.06 * d            # buoyant rise (up = -y) minus smoke weight
        u = u + wx * 0.5; v = v + wy * 0.5
        lu, ru, uu, du = _nb(u); lv, rv, uv, dv = _nb(v)
        curl = 0.5 * ((rv - lv) - (du - uu))
        lc, rc, uc, dc = _nb(curl.abs())
        gx = 0.5 * (rc - lc); gy = 0.5 * (dc - uc)
        mag = torch.sqrt(gx * gx + gy * gy) + 1e-5
        u = u + vort * (gy / mag) * curl
        v = v - vort * (gx / mag) * curl
        u, v = _zero_walls(u, v)
        u2 = _advect(u, u, v, ig, N); v2 = _advect(v, u, v, ig, N)
        u, v = _zero_walls(*_project(u2, v2))
        d = _advect(d, u, v, ig, N) * (1 - diss)
        T = _advect(T, u, v, ig, N) * (1 - diss * 1.3)
        v[..., grow:, :] = v[..., grow:, :].clamp(max=0)     # solid floor (no inflow)
        d[..., grow + 1:, :] = 0; T[..., grow + 1:, :] = 0

        # --- debris ballistics + ground bounce ---
        vel[..., 1] = vel[..., 1] + GRAVITY
        vel = vel * DRAG
        pos = pos + vel
        below = pos[..., 1] >= grow
        pos[..., 1] = torch.where(below, torch.full_like(pos[..., 1], float(grow)), pos[..., 1])
        vel[..., 1] = torch.where(below, -vel[..., 1].abs() * RESTITUTION, vel[..., 1])
        vel[..., 0] = torch.where(below, vel[..., 0] * FRICTION, vel[..., 0])
        offx = (pos[..., 0] < 0) | (pos[..., 0] > N - 1)
        vel[..., 0] = torch.where(offx, -vel[..., 0] * 0.5, vel[..., 0])
        pos[..., 0] = pos[..., 0].clamp(0, N - 1)
        heat = heat * 0.93

        if t % render_every == 0:
            frames.append((render(d, T, pos, heat, N, grow) * 255).to(torch.uint8).cpu())
    return torch.stack(frames, dim=1), params.cpu()        # (B,F,3,N,N), (B,8)


def sample_params(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return PARAM_LO + (PARAM_HI - PARAM_LO) * torch.rand(n, 8, generator=g)


def _montage(frames, params, cols, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B, Fn = frames.shape[:2]
    idx = np.linspace(0, Fn - 1, cols).astype(int)
    fig, ax = plt.subplots(B, cols, figsize=(1.5 * cols, 1.6 * B))
    for i in range(B):
        for j, t in enumerate(idx):
            a = ax[i, j] if B > 1 else ax[j]
            a.imshow(np.transpose(frames[i, t].numpy(), (1, 2, 0))); a.axis("off")
            if i == 0:
                a.set_title(f"t{t}", fontsize=8)
        (ax[i, 0] if B > 1 else ax[0]).set_ylabel(f"sim {i}", fontsize=9)
    fig.suptitle("synthetic pyro engine — sample explosions (Houdini-style inputs)", fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "explosion_samples.png"))
    args = ap.parse_args()
    p = sample_params(args.n)
    frames, params = simulate(p, N=args.N, steps=args.steps, device=args.device)
    print(f"generated {tuple(frames.shape)} uint8 on {args.device}")
    for i in range(args.n):
        print("  sim", i, "|", "  ".join(f"{n}={params[i, k]:.2f}" for k, n in enumerate(PARAM_NAMES)))
    _montage(frames, params, args.cols, args.out)
    print(f"saved montage -> {args.out}")


if __name__ == "__main__":
    main()
