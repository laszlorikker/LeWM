"""Tiny GIF helpers (Pillow) for the explosion demo slides."""
from pathlib import Path

import numpy as np
from PIL import Image


def save_gif(frames, path, scale=3, fps=12):
    """frames: list of (H,W,3) uint8 -> animated GIF."""
    imgs = []
    for f in frames:
        im = Image.fromarray(np.ascontiguousarray(f.astype(np.uint8)))
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        imgs.append(im)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def hstack(*imgs, gap=4, bg=30):
    """horizontally concat (H,W,3) uint8 frames with a separator."""
    H = imgs[0].shape[0]
    sep = np.full((H, gap, 3), bg, np.uint8)
    out = []
    for i, a in enumerate(imgs):
        out.append(a)
        if i < len(imgs) - 1:
            out.append(sep)
    return np.concatenate(out, axis=1)


def chw_to_hwc_u8(x):
    """x: (3,H,W) float[0,1] or uint8 -> (H,W,3) uint8."""
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return np.transpose(a, (1, 2, 0))
