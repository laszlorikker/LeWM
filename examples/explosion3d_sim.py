#!/usr/bin/env python3
"""Synthetic 3D explosion engine: a cube object is blown into tumbling fragments + sparks + smoke,
rendered through a perspective camera as depth-faded additive splats on a dark sky. Cheap (no
voxel fluid), genuinely 3D (perspective + depth), and a clean params->video function so it stays a
valid, plannable surrogate target for a LeWM neural-sim demo (no Houdini needed).

Inputs (conditioning vector), analogous to a Houdini RBD/pyro setup:
    ex, ez       blast epicenter offset under the object   (source position)
    blast        explosion energy (fragment + spark speed) (ignition / fuel)
    buoyancy     smoke rise rate                            (buoyancy lift)
    wind_x,wind_z ambient wind on smoke/sparks              (wind / forces)
    scatter      upward bias of the debris fan              (disturbance)
    dissipation  smoke + spark fade                         (dissipation / cooling)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

PARAM_NAMES = ["ex", "ez", "blast", "buoyancy", "wind_x", "wind_z", "scatter", "dissipation"]
PARAM_LO = torch.tensor([-0.4, -0.4, 3.0, 0.03, -0.05, -0.05, 0.5, 0.020])
PARAM_HI = torch.tensor([0.4, 0.4, 8.0, 0.09, 0.05, 0.05, 1.6, 0.060])

# scene / physics constants (world units; y is up, ground at y=0)
L = 2.0                 # object cube side
M = 7                   # fragments per axis -> M^3 fragments
N_SPARK = 200
N_SMOKE = 90
N_FIRE = 90             # fireball gas particles
GRAVITY = 0.018         # units / step^2
DRAG = 0.992
REST = 0.32             # fragment ground bounce
FRIC = 0.7
VEL_FRAG = 0.060        # blast -> fragment speed
VEL_SPARK = 0.11        # blast -> spark speed
VEL_FIRE = 0.045        # blast -> fireball expansion speed


def _fib_sphere(n, device):
    """n deterministic ~uniform unit directions (fibonacci sphere)."""
    i = torch.arange(n, device=device).float() + 0.5
    phi = torch.acos(1 - 2 * i / n)
    gold = math.pi * (1 + 5 ** 0.5)
    theta = gold * i
    return torch.stack([torch.sin(phi) * torch.cos(theta),
                        torch.cos(phi),
                        torch.sin(phi) * torch.sin(theta)], -1)


class Camera:
    def __init__(self, H, W, eye, target, up=(0, 1, 0), fov=50.0, device="cuda"):
        self.H, self.W = H, W
        eye = torch.tensor(eye, device=device).float()
        target = torch.tensor(target, device=device).float()
        up = torch.tensor(up, device=device).float()
        f = target - eye; f = f / f.norm()
        r = torch.cross(f, up, dim=-1); r = r / r.norm()
        u = torch.cross(r, f, dim=-1)
        self.eye, self.R = eye, torch.stack([r, u, f])      # rows: right, up, forward
        self.focal = 0.5 * W / math.tan(math.radians(fov) / 2)

    def project(self, P):
        """P: (...,3) world -> sx, sy (pixels), depth (forward). """
        rel = P - self.eye
        cam = rel @ self.R.T                                # (...,3): xc, yc, zc
        zc = cam[..., 2].clamp(min=0.05)
        sx = self.W / 2 + self.focal * cam[..., 0] / zc
        sy = self.H / 2 - self.focal * cam[..., 1] / zc
        return sx, sy, cam[..., 2]


def _splat(img, sx, sy, sig, col, inten, k):
    """Additive Gaussian splat of E elements into img (B,3,H,W). sx,sy,sig,inten:(B,E) col:(B,E,3)."""
    B, _, H, W = img.shape
    rng = torch.arange(-(k // 2), k // 2 + 1, device=img.device)
    oy, ox = torch.meshgrid(rng, rng, indexing="ij")
    oy = oy.reshape(-1).float(); ox = ox.reshape(-1).float()
    px = sx.round()[..., None] + ox; py = sy.round()[..., None] + oy          # (B,E,k2)
    dx = px - sx[..., None]; dy = py - sy[..., None]
    s = sig[..., None].clamp(0.6, k / 3.2)            # keep the gaussian soft within the patch
    w = torch.exp(-(dx * dx + dy * dy) / (2 * s * s)) * inten[..., None]
    w = w * ((px >= 0) & (px < W) & (py >= 0) & (py < H))
    pxl = px.long().clamp(0, W - 1); pyl = py.long().clamp(0, H - 1)
    bidx = torch.arange(B, device=img.device).view(B, 1, 1).expand_as(px)
    flat = (bidx * (H * W) + pyl * W + pxl).reshape(-1)
    for c in range(3):
        buf = torch.zeros(B * H * W, device=img.device)
        buf.scatter_add_(0, flat, (w * col[..., c][..., None]).reshape(-1))
        img[:, c] += buf.view(B, H, W)
    return img


def _fog(depth, k=0.045):
    return torch.exp(-(depth - 4).clamp(min=0) * k)


@torch.no_grad()
def simulate(params, H=112, W=112, steps=32, device="cuda"):
    params = params.to(device).float()
    B = params.size(0)
    ex, ez, blast, buoy, wx, wz, scat, diss = [params[:, i].view(B, 1) for i in range(8)]

    cam = Camera(H, W, eye=(0.0, 1.9, -6.8), target=(0.0, 1.8, 0.0), fov=52.0, device=device)

    # ground grid (static -> project once); points on y=0
    gx, gz = torch.meshgrid(torch.linspace(-7, 7, 64, device=device),
                            torch.linspace(-3.5, 11, 64, device=device), indexing="ij")
    gp = torch.stack([gx, torch.zeros_like(gx), gz], -1).reshape(1, -1, 3).expand(B, -1, 3)
    gsx, gsy, gdep = cam.project(gp)
    gcol = torch.tensor([0.10, 0.10, 0.13], device=device).view(1, 1, 3).expand(B, gp.shape[1], 3)
    ginten = _fog(gdep) * (gdep > 0.1).float() * 0.5

    # object fragments (M^3 cube on the ground, centered at x=z=0)
    lin = (torch.arange(M, device=device).float() + 0.5) / M * L - L / 2
    fy = (torch.arange(M, device=device).float() + 0.5) / M * L
    FX, FY, FZ = torch.meshgrid(lin, fy, lin, indexing="ij")
    frag0 = torch.stack([FX, FY, FZ], -1).reshape(1, -1, 3).expand(B, -1, 3).clone()
    Mf = frag0.shape[1]
    fsize = L / M

    E = torch.stack([ex[:, 0], torch.full_like(ex[:, 0], 0.3), ez[:, 0]], -1).view(B, 1, 3)
    d = frag0 - E
    dist = d.norm(dim=-1, keepdim=True) + 1e-3
    dirn = d / dist
    fvel = VEL_FRAG * blast.view(B, 1, 1) * dirn * (0.8 + 0.5 * torch.exp(-dist))   # all shatter
    fvel[..., 1] += (VEL_FRAG * blast * scat * 0.5).view(B, 1)      # upward bias
    fpos = frag0
    fheat = torch.ones(B, Mf, device=device)

    # sparks
    sdir = _fib_sphere(N_SPARK, device).view(1, N_SPARK, 3).expand(B, -1, 3).clone()
    sdir[..., 1] = sdir[..., 1].abs() * 1.1 + 0.2
    sramp = (0.5 + 0.7 * torch.linspace(0, 1, N_SPARK, device=device)).view(1, N_SPARK, 1)
    svel = VEL_SPARK * blast.view(B, 1, 1) * sdir * sramp
    spos = E.expand(B, N_SPARK, 3).clone()
    sheat = torch.ones(B, N_SPARK, device=device)

    # smoke puffs
    mdir = _fib_sphere(N_SMOKE, device).view(1, N_SMOKE, 3).expand(B, -1, 3) * 0.45
    mpos = E.expand(B, N_SMOKE, 3) + mdir
    msize = torch.full((B, N_SMOKE), 0.45, device=device)
    malpha = torch.ones(B, N_SMOKE, device=device)

    # fireball: hot gas that blooms outward then rises and cools
    fidir = _fib_sphere(N_FIRE, device).view(1, N_FIRE, 3).expand(B, -1, 3).clone()
    fidir[..., 1] = fidir[..., 1] * 0.6 + 0.5
    fivel = VEL_FIRE * blast.view(B, 1, 1) * fidir
    fipos = E.expand(B, N_FIRE, 3) + _fib_sphere(N_FIRE, device).view(1, N_FIRE, 3) * 0.15
    fiheat = torch.ones(B, N_FIRE, device=device)
    fisize = torch.full((B, N_FIRE), 0.35, device=device)

    frames = []
    for t in range(steps):
        # --- physics ---
        fvel[..., 1] -= GRAVITY; fvel *= DRAG; fpos = fpos + fvel
        gmask = fpos[..., 1] < fsize / 2
        fpos[..., 1] = torch.where(gmask, torch.full_like(fpos[..., 1], fsize / 2), fpos[..., 1])
        fvel[..., 1] = torch.where(gmask, fvel[..., 1].abs() * REST, fvel[..., 1])
        fvel[..., 0] = torch.where(gmask, fvel[..., 0] * FRIC, fvel[..., 0])
        fvel[..., 2] = torch.where(gmask, fvel[..., 2] * FRIC, fvel[..., 2])
        fheat = fheat * 0.90

        svel[..., 1] -= GRAVITY; svel *= DRAG
        svel[..., 0] += wx * 0.02; svel[..., 2] += wz * 0.02
        spos = spos + svel
        spos[..., 1] = spos[..., 1].clamp(min=0.0)
        sheat = sheat * (1 - diss * 6).clamp(0.7, 0.99)

        mpos[..., 1] += buoy + 0.06
        mpos[..., 0] += wx; mpos[..., 2] += wz
        msize = msize + 0.09
        malpha = malpha * (1 - diss)

        fivel[..., 1] += buoy * 1.5 - GRAVITY * 0.3
        fivel = fivel * 0.92
        fipos = fipos + fivel
        fipos[..., 1] = fipos[..., 1].clamp(min=0.0)
        fisize = fisize + 0.08
        fiheat = fiheat * 0.80

        # --- render (additive splats, back-to-front-ish via fog only) ---
        img = torch.zeros(B, 3, H, W, device=device)
        img += torch.tensor([0.02, 0.02, 0.05], device=device).view(1, 3, 1, 1)
        _splat(img, gsx, gsy, torch.full_like(gsx, 0.8), gcol, ginten, k=3)

        msx, msy, mdep = cam.project(mpos)
        msig = (msize * cam.focal / mdep.clamp(min=0.3)).clamp(2, 11)
        mcol = torch.tensor([0.55, 0.42, 0.32], device=device).view(1, 1, 3).expand(B, N_SMOKE, 3)
        _splat(img, msx, msy, msig, mcol, malpha.clamp(0, 1) * _fog(mdep) * 0.12, k=33)

        fisx, fisy, fidep = cam.project(fipos)
        fisig = (fisize * cam.focal / fidep.clamp(min=0.3)).clamp(2, 14)
        hot = torch.tensor([1.0, 0.85, 0.55], device=device).view(1, 1, 3)
        cool = torch.tensor([0.80, 0.22, 0.05], device=device).view(1, 1, 3)
        ficol = hot * fiheat[..., None] + cool * (1 - fiheat[..., None])
        _splat(img, fisx, fisy, fisig, ficol, fiheat ** 1.2 * _fog(fidep) * 0.85, k=23)

        fsx, fsy, fdep = cam.project(fpos)
        fsig = (fsize * 0.8 * cam.focal / fdep.clamp(min=0.3)).clamp(1.2, 4.0)
        ember = torch.tensor([1.0, 0.45, 0.12], device=device).view(1, 1, 3)
        rock = torch.tensor([0.30, 0.26, 0.24], device=device).view(1, 1, 3)
        fcol = ember * fheat[..., None] + rock * (1 - fheat[..., None])
        _splat(img, fsx, fsy, fsig, fcol, _fog(fdep) * 0.9, k=13)

        ssx, ssy, sdep = cam.project(spos)
        scol = torch.tensor([1.0, 0.75, 0.35], device=device).view(1, 1, 3).expand(B, N_SPARK, 3)
        _splat(img, ssx, ssy, torch.full_like(ssx, 0.9), scol, sheat ** 1.5 * _fog(sdep) * 0.9, k=3)

        if t < 6:                                            # ignition flash
            flsx, flsy, fldep = cam.project(E[:, 0])
            fl = torch.tensor([1.0, 0.8, 0.5], device=device).view(1, 1, 3).expand(B, 1, 3)
            _splat(img, flsx[:, None], flsy[:, None], torch.full((B, 1), 18.0, device=device),
                   fl, torch.full((B, 1), float(1.0 - t / 6) ** 2, device=device), k=41)

        frames.append((img.clamp(0, 1) * 255).to(torch.uint8).cpu())
    return torch.stack(frames, dim=1), params.cpu()


def sample_params(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return PARAM_LO + (PARAM_HI - PARAM_LO) * torch.rand(n, 8, generator=g)


def _montage(frames, cols, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B, Fn = frames.shape[:2]
    idx = np.linspace(0, Fn - 1, cols).astype(int)
    fig, ax = plt.subplots(B, cols, figsize=(2.1 * cols, 2.1 * B))
    for i in range(B):
        for j, t in enumerate(idx):
            a = ax[i, j] if B > 1 else ax[j]
            a.imshow(np.transpose(frames[i, t].numpy(), (1, 2, 0))); a.axis("off")
            if i == 0:
                a.set_title(f"t{t}", fontsize=8)
        (ax[i, 0] if B > 1 else ax[0]).set_ylabel(f"sim {i}", fontsize=9)
    fig.suptitle("synthetic 3D explosion engine — object shatters into fragments + sparks + smoke", fontsize=11)
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "explosion3d_samples.png"))
    args = ap.parse_args()
    p = sample_params(args.n)
    frames, params = simulate(p, H=args.res, W=args.res, steps=args.steps, device=args.device)
    print(f"generated {tuple(frames.shape)} uint8 on {args.device}")
    for i in range(args.n):
        print("  sim", i, "|", "  ".join(f"{n}={params[i, k]:.2f}" for k, n in enumerate(PARAM_NAMES)))
    _montage(frames, args.cols, args.out)
    print(f"saved montage -> {args.out}")


if __name__ == "__main__":
    main()
