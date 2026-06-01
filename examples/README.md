# Toy example: training LeWM end-to-end on a "moving ball" world

`train_toy.py` is a self-contained example that trains the **real** LeWorldModel — the actual
`JEPA` from `jepa.py` and the actual LeWM objective (next-embedding prediction + `SIGReg`) —
without needing `stable-worldmodel`, `stable-pretraining`, Hydra, Lightning, or any dataset
download. The only substitution is the encoder: the paper's from-scratch ViT-tiny is replaced
by a small from-scratch CNN that exposes the same interface `JEPA.encode` expects. It runs on a
single GPU in ~90 s.

## The world

A ball at position `(x, y) ∈ [0,1]²`. Each step an action `a = (dx, dy)` moves it
(`pos_{t+1} = clip(pos_t + a_t)`); each frame renders the ball as a Gaussian blob. The action is
*needed* to predict the next frame, so the model must learn to encode position **and** integrate
the action — the minimal ingredients of a world model.

## Run it

Use the CUDA-enabled env (`torch251`: Python 3.10, torch + CUDA, einops, numpy, sklearn):

```bash
# the headline result: train with vs. without SIGReg and contrast
~/miniconda3/envs/torch251/bin/python examples/train_toy.py --compare --plan

# just train once (with SIGReg) and run the planning demo
~/miniconda3/envs/torch251/bin/python examples/train_toy.py --plan

# train longer / tweak
~/miniconda3/envs/torch251/bin/python examples/train_toy.py --steps 1500 --embed-dim 192
```

Checkpoints (`*_object.ckpt`, full pickled model like the real pipeline) and `metrics.json` are
written to `examples/runs/` (git-ignored).

## What it demonstrates — the LeWM thesis

A typical `--compare --plan` run:

| config       | emb_std | val pred_loss | probe R² | plan (goal closer) | verdict   |
|--------------|--------:|--------------:|---------:|-------------------:|-----------|
| **sigreg**   |   1.005 |        0.137  |   0.994  |              ~84 % | healthy   |
| **no_sigreg**|   0.007 |      0.00009  |   0.55   |              ~11 % | COLLAPSED |

Read it as four lenses on the same point:

* **`emb_std`** (mean per-dim std of the latent) is the honest **collapse metric**. With a *shared*
  encoder and *no* stop-gradient/EMA, the trivial optimum is to map every frame to a constant.
  Without SIGReg the latent collapses (`emb_std → 0`); SIGReg keeps it isotropic (`≈ 1`, since it
  targets `N(0,1)`).
* **`val pred_loss`** — the collapsed model's prediction loss is ~1500× *lower*. **Low prediction
  loss ≠ useful model**: predicting a constant is trivial. This is precisely why naive end-to-end
  JEPAs need stop-gradient/EMA, and what SIGReg replaces.
* **probe R²** — a linear probe decoding the true `(x, y)` from the frozen latent. The probe is
  scale-invariant so it still scrapes ~0.55 off the collapsed residual, but the SIGReg latent is
  near-perfectly decodable (0.99). Mirrors the paper's "probing physical quantities".
* **planning** — a latent-space CEM optimizes an action sequence to match a goal image, then
  executes it in the true simulator. The SIGReg model drives the ball ~84 % closer to the goal;
  the collapsed model can't (~11 %, ≈ chance) because its cost landscape is flat.

## How it maps to the real pipeline

| toy script                        | real pipeline (`train.py`)                              |
|-----------------------------------|---------------------------------------------------------|
| `TinyEncoder` (CNN)               | `stable_pretraining.backbone.utils.vit_hf` (ViT-tiny)   |
| `lewm_loss(...)`                  | `lejepa_forward(...)`                                    |
| in-memory ball windows            | HDF5/Lance datasets via `swm.data.load_dataset`         |
| manual AdamW loop                 | Lightning `spt.Module` / `spt.Manager` + Hydra config   |
| `plan_to_goal` (latent CEM)       | `swm.solver.CEMSolver` + `JEPA.get_cost` / `rollout`    |
| `torch.save(model, *_object.ckpt)`| `SaveCkptCallback` / `swm.wm.utils.save_pretrained`     |

`JEPA`, `ARPredictor`, `Embedder`, `MLP`, and `SIGReg` are imported unchanged from the repo, so
the architecture and objective exercised here are the genuine ones.

## Extending toward a real task

* Swap the synthetic dataset for real `(pixels, action, state)` windows.
* Increase `--embed-dim`, predictor `depth`/`heads`, and `--steps` toward the paper's settings.
* Replace `TinyEncoder` with `vit_hf` once `stable-pretraining` is installed to match the paper.
