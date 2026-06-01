#!/usr/bin/env python3
"""Self-contained example training for LeWorldModel (LeWM) on a toy "moving ball" world.

Why this exists
---------------
The real pipeline (``train.py``) needs ``stable-worldmodel`` + ``stable-pretraining`` +
HuggingFace datasets + Hydra/Lightning. This script instead trains the **real** ``JEPA``
model and the **real** LeWM objective (next-embedding prediction + SIGReg) from the repo's
own ``jepa.py`` / ``module.py`` on a tiny synthetic world. The only substitution is the
encoder: the paper's from-scratch ViT-tiny is swapped for a small from-scratch CNN that
exposes the same interface JEPA expects (``encoder(x, interpolate_pos_encoding=...)`` ->
object with ``.last_hidden_state[:, 0]`` as the CLS summary). Everything else -- the
predictor, the action conditioning, the projector/pred_proj heads, and the loss -- is the
genuine code path used for the benchmarks.

The world
---------
A ball lives at position ``(x, y) in [0, 1]^2``. Each step an action ``a = (dx, dy)`` moves
it: ``pos_{t+1} = clip(pos_t + a_t)``. Each frame renders the ball as a Gaussian blob. The
action is *required* to predict the next frame (the walk is non-deterministic without it),
so the world model must learn to encode position and integrate the action.

What it demonstrates (the paper's thesis, on a toy)
---------------------------------------------------
Run with ``--compare`` to train twice -- with SIGReg (lambda=0.09) and without (lambda=0):

* **Anti-collapse.** With a *shared* encoder and *no* stop-gradient/EMA, the trivial optimum
  is to map every frame to a constant (prediction loss -> 0, representation useless). SIGReg
  forbids that by forcing embeddings toward an isotropic Gaussian. We track ``emb_std`` (mean
  per-dim std of the latent): ~0 means collapsed, ~1 means healthy.
* **Usefulness.** A linear probe decodes the true ball ``(x, y)`` from the frozen latent and
  reports R^2 on held-out data. Collapsed latent -> R^2 ~ 0; SIGReg latent -> high R^2. This
  mirrors the paper's "probing physical quantities".

Single-run mode (default) additionally runs a small latent-space CEM planner to show the
trained world model can pick actions that drive the ball to a goal image.

Usage
-----
    python examples/train_toy.py --compare          # the headline contrast
    python examples/train_toy.py --plan             # train once + planning demo
    python examples/train_toy.py --steps 1200       # train longer
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

# import the repo's real model code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder, SIGReg  # noqa: E402

MAX_STEP = 0.15  # max per-step ball displacement (also the action clamp for planning)


# --------------------------------------------------------------------------------------
# Encoder: lightweight stand-in for the paper's vit_hf, same interface JEPA.encode needs.
# --------------------------------------------------------------------------------------
class TinyEncoder(nn.Module):
    """From-scratch CNN exposing the vit_hf interface used by ``JEPA.encode``.

    ``JEPA.encode`` calls ``self.encoder(pixels, interpolate_pos_encoding=True)`` and reads
    ``output.last_hidden_state[:, 0]`` as the CLS token. We return a 1-token sequence whose
    single token is the pooled CNN feature, so token 0 *is* the summary.
    """

    def __init__(self, in_ch: int = 3, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),   # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),       # 16 -> 8
            nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.GroupNorm(8, dim), nn.GELU(),      # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # (N, dim)
        )

    def forward(self, x, interpolate_pos_encoding: bool = False):
        h = self.net(x)
        return SimpleNamespace(last_hidden_state=h.unsqueeze(1))  # (N, 1, dim)


# --------------------------------------------------------------------------------------
# Synthetic "moving ball" dataset.
# --------------------------------------------------------------------------------------
def make_dataset(n, seq_len, img=32, sigma=0.06, seed=0):
    """Return (pixels, action, state) windows.

    pixels: (n, seq_len, 3, img, img) float32 in [0, 1]
    action: (n, seq_len, 2)  -- action[:, t] is applied at frame t (frame_t -> frame_{t+1})
    state:  (n, seq_len, 2)  -- ground-truth ball (x, y), used only for the linear probe
    """
    rng = np.random.default_rng(seed)
    pos = np.empty((n, seq_len, 2), np.float32)
    act = np.empty((n, seq_len, 2), np.float32)
    p = rng.uniform(0.1, 0.9, size=(n, 2)).astype(np.float32)
    for t in range(seq_len):
        pos[:, t] = p
        a = rng.uniform(-MAX_STEP, MAX_STEP, size=(n, 2)).astype(np.float32)
        act[:, t] = a
        p = np.clip(p + a, 0.05, 0.95)

    pixels = _render(pos, img, sigma)
    return (
        torch.from_numpy(pixels),
        torch.from_numpy(act),
        torch.from_numpy(pos),
    )


def _render(pos, img, sigma):
    """Vectorized Gaussian-blob renderer. pos: (..., 2) in [0, 1] -> (..., 3, img, img)."""
    coord = np.linspace(0, 1, img, dtype=np.float32)
    yy, xx = np.meshgrid(coord, coord, indexing="ij")  # (img, img)
    cx = pos[..., 0][..., None, None]
    cy = pos[..., 1][..., None, None]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-d2 / (2 * sigma ** 2))  # (..., img, img) in [0, 1]
    return np.repeat(blob[..., None, :, :], 3, axis=-3)  # gray ball -> 3 channels


# --------------------------------------------------------------------------------------
# Model assembly (real JEPA + real modules from the repo).
# --------------------------------------------------------------------------------------
def build_model(embed_dim, history_size, action_dim):
    encoder = TinyEncoder(in_ch=3, dim=embed_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
        depth=4, heads=4, dim_head=32, mlp_dim=256, dropout=0.0,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    head = lambda: MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=512, norm_fn=nn.BatchNorm1d)
    return JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=head(), pred_proj=head(),
    )


def lewm_loss(model, sigreg, batch, history_size, num_preds, lambd):
    """Faithful re-implementation of train.py::lejepa_forward (the LeWM objective)."""
    out = model.encode(batch)
    emb = out["emb"]            # (B, T, D)
    act_emb = out["act_emb"]    # (B, T, D)

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]                 # label (NOT detached -> end-to-end)
    pred_emb = model.predict(ctx_emb, ctx_act)   # prediction

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    if lambd > 0:
        sig_loss = sigreg(emb.transpose(0, 1))   # SIGReg wants (T, B, D)
    else:
        sig_loss = torch.zeros((), device=emb.device)
    loss = pred_loss + lambd * sig_loss
    return loss, pred_loss.detach(), sig_loss.detach(), emb.detach()


# --------------------------------------------------------------------------------------
# Training.
# --------------------------------------------------------------------------------------
def iterate_minibatches(n, batch_size, generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n - batch_size + 1, batch_size):
        yield perm[i : i + batch_size]


def train(pixels, action, embed_dim, history_size, num_preds, lambd, steps, batch_size,
          lr, weight_decay, num_proj, knots, device, seed, log_every=100):
    torch.manual_seed(seed)  # identical init across configs for a fair comparison
    model = build_model(embed_dim, history_size, action_dim=action.size(-1)).to(device)
    sigreg = SIGReg(knots=knots, num_proj=num_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pixels, action = pixels.to(device), action.to(device)
    gen = torch.Generator().manual_seed(seed)
    model.train()

    step = 0
    while step < steps:
        for idx in iterate_minibatches(pixels.size(0), batch_size, gen):
            batch = {"pixels": pixels[idx], "action": action[idx]}
            loss, pred_loss, sig_loss, emb = lewm_loss(
                model, sigreg, batch, history_size, num_preds, lambd
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % log_every == 0 or step == steps - 1:
                emb_std = emb.reshape(-1, emb.size(-1)).std(0).mean().item()
                print(f"  step {step:4d} | loss {loss.item():.4f} | pred {pred_loss.item():.4f} "
                      f"| sigreg {sig_loss.item():.4f} | emb_std {emb_std:.4f}")
            step += 1
            if step >= steps:
                break
    return model


# --------------------------------------------------------------------------------------
# Evaluation: collapse metric, linear probe, planning.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode_frames(model, pixels, device, chunk=4096):
    """Encode every frame independently -> (N*T, D) numpy latents (eval mode)."""
    model.eval()
    n, t = pixels.shape[:2]
    flat = pixels.reshape(n * t, 1, *pixels.shape[2:])
    outs = []
    for i in range(0, flat.size(0), chunk):
        emb = model.encode({"pixels": flat[i : i + chunk].to(device)})["emb"][:, 0]
        outs.append(emb.cpu())
    return torch.cat(outs, 0).numpy()


@torch.no_grad()
def val_prediction_loss(model, val_data, history_size, num_preds, device):
    """The LeWM next-embedding prediction loss on held-out data. A collapsed model drives
    this near zero (predicting a constant is trivial) -- low loss != useful representation."""
    model.eval()
    px, act, _ = val_data
    out = model.encode({"pixels": px.to(device), "action": act.to(device)})
    emb, act_emb = out["emb"], out["act_emb"]
    pred = model.predict(emb[:, :history_size], act_emb[:, :history_size])
    return float((pred - emb[:, num_preds:]).pow(2).mean())


def evaluate(model, train_data, val_data, history_size, num_preds, device):
    """Collapse metric (emb_std) + linear-probe R^2 decoding ball position from latent."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    tr_px, _, tr_state = train_data
    va_px, _, va_state = val_data

    emb_tr = encode_frames(model, tr_px, device)
    emb_va = encode_frames(model, va_px, device)
    y_tr = tr_state.reshape(-1, 2).numpy()
    y_va = va_state.reshape(-1, 2).numpy()

    emb_std = float(emb_va.std(0).mean())
    reg = LinearRegression().fit(emb_tr, y_tr)
    probe_r2 = float(r2_score(y_va, reg.predict(emb_va)))
    pred_loss = val_prediction_loss(model, val_data, history_size, num_preds, device)
    return {"emb_std": emb_std, "val_pred_loss": pred_loss, "probe_r2": probe_r2}


@torch.no_grad()
def plan_to_goal(model, history_size, device, horizon=6, n_samples=400, iters=6, topk=40,
                 sigma0=0.12, seed=0):
    """Latent-space CEM: optimize an action sequence so the imagined final latent matches a
    goal image, then execute the plan in the *true* simulator and report goal distance."""
    model.eval()
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.15, 0.35, size=2).astype(np.float32)
    goal = rng.uniform(0.65, 0.85, size=2).astype(np.float32)

    # history = ball sitting still at the start for `history_size` frames (zero actions)
    start_hist = np.broadcast_to(start, (1, history_size, 2))
    start_px = torch.from_numpy(_render(start_hist, 32, 0.06)).to(device)        # (1,H,3,32,32)
    start_act = torch.zeros(1, history_size, 2, device=device)
    goal_px = torch.from_numpy(_render(goal[None, None], 32, 0.06)).to(device)   # (1,1,3,32,32)
    goal_emb = model.encode({"pixels": goal_px})["emb"][:, 0]                     # (1, D)

    HS = history_size
    mean = torch.zeros(horizon, 2, device=device)
    std = torch.full((horizon, 2), sigma0, device=device)
    cem_gen = torch.Generator(device=device).manual_seed(seed)

    for _ in range(iters):
        noise = torch.randn(n_samples, horizon, 2, generator=cem_gen, device=device)
        cand = (mean + std * noise).clamp(-MAX_STEP, MAX_STEP)             # (S, horizon, 2)

        ctx = model.encode({
            "pixels": start_px.expand(n_samples, -1, -1, -1, -1),
            "action": start_act.expand(n_samples, -1, -1),
        })
        emb = ctx["emb"]                                                  # (S, HS, D)
        act = start_act.expand(n_samples, -1, -1).clone()
        for k in range(horizon):
            a_emb = model.action_encoder(act[:, -HS:])
            pred = model.predict(emb[:, -HS:], a_emb)[:, -1:]             # (S, 1, D)
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, cand[:, k : k + 1]], dim=1)
        cost = (emb[:, -1] - goal_emb).pow(2).sum(-1)                     # (S,)
        elite = cand[cost.topk(topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-6

    # execute the mean plan in the true simulator
    pos = start.copy()
    for k in range(horizon):
        pos = np.clip(pos + mean[k].cpu().numpy(), 0.05, 0.95)
    return {
        "start": start.tolist(),
        "goal": goal.tolist(),
        "reached": pos.tolist(),
        "dist_before": float(np.linalg.norm(goal - start)),
        "dist_after": float(np.linalg.norm(goal - pos)),
    }


# --------------------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true", help="train with and without SIGReg and contrast")
    ap.add_argument("--plan", action="store_true", help="run the CEM planning demo (single-run mode)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--num-preds", type=int, default=1)
    ap.add_argument("--sigreg-weight", type=float, default=0.09)
    ap.add_argument("--num-proj", type=int, default=512)
    ap.add_argument("--knots", type=int, default=17)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    seq_len = args.history_size + args.num_preds
    print(f"device={args.device} | seq_len={seq_len} | steps={args.steps} | embed_dim={args.embed_dim}")

    train_data = make_dataset(args.n_train, seq_len, seed=args.seed)
    val_data = make_dataset(args.n_val, seq_len, seed=args.seed + 1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        embed_dim=args.embed_dim, history_size=args.history_size, num_preds=args.num_preds,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_proj=args.num_proj, knots=args.knots, device=args.device, seed=args.seed,
    )

    configs = [("sigreg", args.sigreg_weight)]
    if args.compare:
        configs.append(("no_sigreg", 0.0))

    results = {}
    for name, lambd in configs:
        print(f"\n=== training [{name}] (sigreg weight = {lambd}) ===")
        model = train(train_data[0], train_data[1], lambd=lambd, **common)
        metrics = evaluate(model, train_data, val_data, args.history_size, args.num_preds, args.device)
        results[name] = metrics
        print(f"  -> emb_std={metrics['emb_std']:.4f}  val_pred_loss={metrics['val_pred_loss']:.5f}  "
              f"probe_r2={metrics['probe_r2']:.4f}")

        ckpt = out_dir / f"lewm_toy_{name}_object.ckpt"
        torch.save(model, ckpt)
        print(f"  -> saved checkpoint: {ckpt}")

        if args.plan:  # plan with every model so a collapsed one can be shown to fail
            plan = plan_to_goal(model, args.history_size, args.device, seed=args.seed)
            metrics["plan"] = plan
            print(f"  -> plan: goal distance {plan['dist_before']:.3f} -> {plan['dist_after']:.3f} "
                  f"({100*(1-plan['dist_after']/plan['dist_before']):.0f}% closer)")

    # summary
    collapsed = lambda m: m["emb_std"] < 0.1  # emb_std is the honest collapse metric
    print("\n" + "=" * 78)
    head = f"{'config':<12}{'emb_std':>10}{'pred_loss':>11}{'probe_r2':>10}"
    if args.plan:
        head += f"{'plan_closer':>13}"
    print(head + "   verdict")
    print("-" * 78)
    for name, m in results.items():
        row = f"{name:<12}{m['emb_std']:>10.4f}{m['val_pred_loss']:>11.5f}{m['probe_r2']:>10.4f}"
        if args.plan and "plan" in m:
            p = m["plan"]
            row += f"{100*(1-p['dist_after']/p['dist_before']):>12.0f}%"
        verdict = "COLLAPSED" if collapsed(m) else "healthy"
        print(row + f"   {verdict}")
    print("=" * 78)
    if args.compare:
        print("Takeaway: without SIGReg the shared-encoder JEPA collapses (emb_std ~ 0) and drives\n"
              "prediction loss to ~0 by mapping every frame to a near-constant -- a useless model\n"
              "that cannot plan. SIGReg alone keeps the latent isotropic, decodable, and plannable.\n"
              "That is the LeWM thesis: one regularizer replaces stop-gradient / EMA / 6-term losses.")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
