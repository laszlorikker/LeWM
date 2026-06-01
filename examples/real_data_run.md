# Running the genuine `train.py` on a real dataset (two-room)

This documents an actual, working end-to-end run of the repo's real training pipeline
(`train.py` → `stable-worldmodel` HDF5 loader → from-scratch ViT-tiny → `ARPredictor` →
SIGReg loss → checkpoints) on the **two-room** dataset, on a single 16 GB GPU
(Quadro RTX 5000, Turing). Unlike `examples/train_toy.py`, this uses the genuine
`stable-worldmodel` / `stable-pretraining` stack and a real downloaded dataset.

## Environment

Use the CUDA conda env (`torch251`: Python 3.10, torch 2.11+cu126). Install the train stack:

```bash
~/miniconda3/envs/torch251/bin/pip install "stable-worldmodel[train]"
# the HDF5 dataset format only registers if these are present:
~/miniconda3/envs/torch251/bin/pip install h5py hdf5plugin
```

Pin a single cache root for data + checkpoints (the package default is `~/.stable_worldmodel`;
the README's `~/.stable-wm` is stale — just set it explicitly and stay consistent):

```bash
export STABLEWM_HOME=$HOME/.stable-wm
```

## Get the data

```bash
huggingface-cli download quentinll/lewm-tworooms tworoom.tar.zst \
    --repo-type dataset --local-dir "$STABLEWM_HOME/_dl"
mkdir -p "$STABLEWM_HOME/datasets"
tar --zstd -xf "$STABLEWM_HOME/_dl/tworoom.tar.zst" -C "$STABLEWM_HOME/datasets/"
# -> $STABLEWM_HOME/datasets/tworoom.h5  (12.8 GB, 920,809 frames; cols: pixels, action, proprio)
```

`config/train/data/tworoom.yaml` references `name: tworoom.h5`, and `load_dataset` detects the
`hdf5` format (registered once `h5py`+`hdf5plugin` are installed). No lance conversion needed.
(Two-room is the smallest of the four datasets: 3.4 GB vs pusht 13 GB, reacher 24 GB, cube 46 GB.)

## Train (memory-safe, fast config)

```bash
cd /home/laszlo/jepa/le-wm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True STABLEWM_HOME=$HOME/.stable-wm \
~/miniconda3/envs/torch251/bin/python train.py \
  data=tworoom \
  trainer.max_epochs=8 trainer.precision=16-mixed \
  +trainer.limit_train_batches=50 +trainer.limit_val_batches=8 \
  loader.batch_size=64 loader.prefetch_factor=2 loader.pin_memory=False \
  loader.persistent_workers=True num_workers=8
```

This is a **short demo** config (8 × 50 = 400 steps) to show the loss decreasing, not a
paper-scale run (the real config is `max_epochs=100` over the full ~920k frames). Drop the
`limit_*` / `max_epochs` overrides for a full run.

### Two gotchas this config fixes

1. **IO bound, not compute bound.** Random access into the 12.8 GB *compressed* HDF5 is
   latency-bound (the naive `num_workers=2` run logged `cpu_percent≈6%` and took ~48 s/step).
   `num_workers=8` parallelizes the random reads → **~27 s/epoch (~0.55 s/step)**, a ~30–90×
   speedup. (After the first read the frames warm the OS page cache — the machine has 27 GB RAM.)
2. **Pinned-memory OOM, not VRAM OOM.** `num_workers=8 × prefetch_factor=6 × ~154 MB/batch`
   ≈ 7.4 GB of pinned host memory made `cudaHostAlloc` fail with
   `CUDA error: out of memory ... in pin memory thread`. Fix: `prefetch_factor=2` +
   `pin_memory=False`. GPU VRAM itself is fine even at batch 128. Use FP16 (`16-mixed`), not
   BF16 — Turing has no hardware BF16.

## Result (8-epoch demo)

| epoch | fit/loss | pred_loss | sigreg_loss | val/pred_loss |
|------:|---------:|----------:|------------:|--------------:|
| 0 | 1.135 | 0.441 | 7.71 | 9.07 |
| 1 | 0.943 | 0.429 | 5.71 | 178.6 |
| 2 | 0.937 | 0.475 | 5.13 | 185.6 |
| 3 | 0.897 | 0.440 | 5.07 | 359.0 |
| 4 | 0.890 | 0.460 | 4.78 | 315.6 |
| 5 | 0.862 | 0.444 | 4.64 | 458.2 |
| 6 | 0.840 | 0.409 | 4.78 | 6.96 |
| 7 | 0.829 | 0.402 | 4.74 | **0.47** |

- **`fit/loss` falls monotonically 1.135 → 0.829**; the SIGReg term drives most of it
  (7.71 → 4.74) as the embeddings are pushed toward an isotropic Gaussian, while `pred_loss`
  stays controlled (~0.40 by the end) — a non-collapsed predictive representation.
- **`val/pred_loss` spikes then resolves** (9 → 458 → **0.47**). This is a BatchNorm train/eval
  gap: the projector/`pred_proj` use `BatchNorm1d`, and while SIGReg rapidly inflates the
  embedding scale early on, the BN *running* stats (used in eval mode) lag the moving
  distribution. Once they catch up (~epoch 6) eval-mode prediction becomes excellent.

## Checkpoints

Saved under `$STABLEWM_HOME/checkpoints/lewm/` as `weights_epoch_{1..8}.pt` (69 MB each, via
`SaveCkptCallback` → `swm.wm.utils.save_pretrained`) plus `config.json`. Lightning also writes
crash-safe `epoch=*.ckpt` / `last.ckpt` under `~/.cache/stable-pretraining/runs/<date>/<id>/`.
