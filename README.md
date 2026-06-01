
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Examples & demos (`examples/`)

The `examples/` directory contains self-contained scripts built on top of LeWM — from a minimal toy
to a full "neural simulator" demo. The LeWM-core demos import the model straight from
`jepa.py`/`module.py`; the explosion demo is **fully standalone** and needs only
`torch`, `numpy`, `einops`, `matplotlib`, `Pillow`, `scikit-learn` (no `stable-worldmodel`).

Generated datasets, checkpoints, and media land under
`examples/{explosion_data,explosion_ckpt,explosion_viz,viz_progress,runs}/` and are **git-ignored** —
regenerate them locally with the commands below. Heavy training is best run on a workstation GPU.

### 1. Toy LeWM — synthetic "moving ball" world

`examples/train_toy.py` trains the real `JEPA` + LeWM objective (next-embedding prediction + `SIGReg`)
on a tiny controllable world, with only the heavy ViT encoder swapped for a small CNN. Runs in ~2 min
and demonstrates the paper's thesis directly: **SIGReg alone prevents collapse**.

```bash
python examples/train_toy.py --compare --plan
```

`--compare` trains with vs. without SIGReg (without → the latent collapses, `emb_std → 0`, prediction
loss → ~0 but the representation is useless); `--plan` runs a latent-space CEM to reach a goal image.

### 2. Genuine `train.py` on real data + visualization

See **`examples/real_data_run.md`** for the exact working recipe to run the real pipeline on the
two-room dataset (installing `stable-worldmodel[train]`, downloading the data, and the memory-safe
dataloader flags). After a run, visualize it:

```bash
# render the held-out rollout at every checkpoint + a summary of how predictions sharpen over epochs
python examples/viz_over_epochs.py
# fit a pixel decoder on the frozen LeWM latents, then render true / decode(true) / decode(pred)
python examples/train_decoder.py --ckpt lewm/weights_epoch_100.pt
python examples/viz_decoded_rollout.py
```

(`rollout_lib.py` is the shared rollout/decoder library; `visualize_rollout.py` is a single-checkpoint
nearest-neighbour-decoded variant.)

### 3. Explosion neural-sim surrogate (no Houdini needed)

A complete *"fast neural surrogate for an explosion sim"* demo: a synthetic pyro engine produces
`params → video` data, a LeWM-style latent dynamics model learns to predict the explosion from the
inputs, and those inputs can be **planned** to art-direct the result.

**a. The synthetic pyro engines** — a grid-based **2D** solver (`explosion_sim.py`, a real
Stable-Fluids fluid sim) and a **3D** particle/fragment engine (`explosion3d_sim.py`: an object bursts
into fragments + sparks + a fireball + smoke, perspective camera). Preview either:

```bash
python examples/explosion3d_sim.py --n 6      # 3D (used for the surrogate)
python examples/explosion_sim.py   --n 6      # 2D grid fluid
```

Both are driven by **8 Houdini-style inputs**, deterministic so the sim is a clean `params → video`
function (which is what makes it a learnable, plannable surrogate target):

| input | Houdini pyro / RBD analog |
|---|---|
| `ex, ez` | source / emitter position |
| `blast` | ignition temperature / fuel (+ debris energy) |
| `buoyancy` | buoyancy lift |
| `wind_x, wind_z` | wind / forces |
| `scatter` | turbulence / disturbance (debris up-bias) |
| `dissipation` | dissipation / cooling |

**b. Generate the dataset:**

```bash
python examples/gen_explosion_dataset.py --n 2000
# -> examples/explosion_data/frames.npy (uint8, N×32×3×112×112) + params.npy (N×8)
```

**c. Train the surrogate** (the "action" is the 8 params, broadcast to every timestep):

```bash
# single-vector latent — faithful to LeWM's CLS-token design (coarse decoded frames)
python examples/train_explosion_surrogate.py     # -> explosion_ckpt/model.pt + decoder.pt
# spatial latent — conv feature map + FiLM conv predictor (much crisper frames)
python examples/train_explosion_spatial.py       # -> explosion_ckpt/spatial.pt
```

The single-vector model is the LeWM-faithful version (CNN encoder → one 256-d token, `ARPredictor`,
`SIGReg`, + a separate pixel decoder). The spatial model keeps a `C×14×14` feature map and trains an
autoencoder + action-conditioned conv predictor jointly with a rollout pixel loss, so decoded
predictions are sharp. If the 16 GB defaults are tight, trim `--batch` (40) or `--R` (rollout
horizon, 4).

**d. Forward rollout — surrogate vs. ground truth** (the surrogate gets the first 3 frames + the
params, then imagines the rest; writes a montage **and** a side-by-side GIF for slides):

```bash
python examples/viz_explosion_spatial.py    # spatial (crisp) -> explosion_viz/spatial_rollout_*.png + .gif
python examples/viz_explosion_rollout.py    # single-vector variant
```

**e. Planning / art-direction** — optimize the 8 inputs with CEM *evaluated through the fast
surrogate*, then verify by running the true sim once:

```bash
# recover the inputs of a target (held-out) explosion
python examples/plan_explosion.py
# abstract art-direction: specify a target SHAPE (tall / wide / left) -> plan inputs -> GIFs
python examples/plan_explosion_art.py       # -> explosion_viz/art_direction.png + art_*.gif
```

**f. Timing** — surrogate (full rollout+decode, and latent-only for planning) vs. the sim:

```bash
python examples/bench_explosion.py
```

**Honest caveats.** The 3D engine is a particle/RBD-style system (not a voxel fluid) with an additive
"night explosion" render, and the surrogate is an **approximate previz** model — not, and not meant to
be, Houdini-identical. Its value is a fast, batchable, *plannable* approximation; the single-vector
latent yields coarse decoded frames (bloom / spread / timing, not crisp debris), while the spatial
latent is markedly sharper.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
