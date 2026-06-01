#!/usr/bin/env python3
"""Train the LeWM explosion surrogate (param-conditioned latent dynamics) + a pixel decoder.

    ~/miniconda3/envs/torch251/bin/python examples/train_explosion_surrogate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from module import SIGReg                       # noqa: E402
from explosion_model import build_surrogate, build_decoder  # noqa: E402

DATA = Path(__file__).resolve().parent / "explosion_data"
CKPT = Path(__file__).resolve().parent / "explosion_ckpt"


def sample_windows(frames, params, bs, num_steps, gen, device):
    N, T = frames.shape[:2]
    seq = torch.randint(0, N, (bs,), generator=gen, device=device)
    start = torch.randint(0, T - num_steps + 1, (bs,), generator=gen, device=device)
    idx_t = start[:, None] + torch.arange(num_steps, device=device)
    pix = frames[seq[:, None], idx_t].float() / 255.0          # (bs,num_steps,3,H,W)
    act = params[seq][:, None, :].expand(bs, num_steps, params.size(1))
    return {"pixels": pix, "action": act}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dec-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--num-preds", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--sigreg-weight", type=float, default=0.09)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    HS, NP = args.history_size, args.num_preds
    num_steps = HS + NP
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)

    # ---- data ----
    frames = torch.from_numpy(np.load(DATA / "frames.npy")).to(dev)          # (N,T,3,H,W) uint8
    params_raw = torch.from_numpy(np.load(DATA / "params.npy")).to(dev).float()
    p_mean = params_raw.mean(0); p_std = params_raw.std(0) + 1e-6
    params = (params_raw - p_mean) / p_std                                   # z-scored action
    print(f"data: frames {tuple(frames.shape)}  params {tuple(params_raw.shape)}  on {dev}")

    # ---- model ----
    model = build_surrogate(args.embed_dim, HS).to(dev)
    sigreg = SIGReg(knots=17, num_proj=256).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    model.train()

    print("=== training LeWM surrogate ===")
    for step in range(args.steps):
        batch = sample_windows(frames, params, args.batch, num_steps, gen, dev)
        out = model.encode(batch)
        emb, act_emb = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :HS], act_emb[:, :HS])
        pred_loss = (pred - emb[:, NP:]).pow(2).mean()
        sig = sigreg(emb.transpose(0, 1))
        loss = pred_loss + args.sigreg_weight * sig
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            std = emb.reshape(-1, emb.size(-1)).std(0).mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | pred {pred_loss.item():.4f} "
                  f"| sigreg {sig.item():.3f} | emb_std {std:.3f}")

    torch.save({"state_dict": model.state_dict(), "embed_dim": args.embed_dim,
                "history_size": HS, "num_preds": NP,
                "p_mean": p_mean.cpu().numpy(), "p_std": p_std.cpu().numpy()}, CKPT / "model.pt")
    print(f"  saved -> {CKPT / 'model.pt'}")

    # ---- decoder on frozen latents ----
    print("=== training decoder ===")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dec = build_decoder(args.embed_dim).to(dev)
    dopt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    N, T = frames.shape[:2]
    for step in range(args.dec_steps):
        seq = torch.randint(0, N, (args.batch,), generator=gen, device=dev)
        tt = torch.randint(0, T, (args.batch,), generator=gen, device=dev)
        pix = frames[seq, tt].float() / 255.0                                # (B,3,H,W)
        with torch.no_grad():
            emb = model.encode({"pixels": pix[:, None]})["emb"][:, 0]
        loss = F.mse_loss(dec(emb), pix)
        dopt.zero_grad(set_to_none=True)
        loss.backward()
        dopt.step()
        if step % 300 == 0 or step == args.dec_steps - 1:
            print(f"  dec step {step:4d} | recon MSE {loss.item():.5f}")
    torch.save({"state_dict": dec.state_dict(), "embed_dim": args.embed_dim}, CKPT / "decoder.pt")
    print(f"  saved -> {CKPT / 'decoder.pt'}")


if __name__ == "__main__":
    main()
