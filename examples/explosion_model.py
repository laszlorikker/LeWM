"""Shared model definition for the explosion LeWM surrogate (imported by train + viz scripts).

The real LeWM pieces (JEPA, ARPredictor, Embedder, MLP, SIGReg) come straight from the repo; only
the encoder is a small from-scratch CNN for 112x112 frames (single-vector latent, like the toy).
The "action" is the 8 sim params, broadcast to every timestep.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jepa import JEPA                       # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_decoder import Decoder           # noqa: E402  (emb -> image)

ACTION_DIM = 8


class Enc(nn.Module):
    """112x112x3 -> single latent vector, exposing the vit_hf interface JEPA.encode expects."""

    def __init__(self, dim=256):
        super().__init__()

        def blk(ci, co):
            return [nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co), nn.GELU()]

        self.net = nn.Sequential(
            *blk(3, 32), *blk(32, 64), *blk(64, 128), *blk(128, dim),   # 112->56->28->14->7
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x, interpolate_pos_encoding=False):
        return SimpleNamespace(last_hidden_state=self.net(x).unsqueeze(1))   # (N,1,dim)


def build_surrogate(embed_dim=256, history_size=3, action_dim=ACTION_DIM):
    pred = ARPredictor(num_frames=history_size, input_dim=embed_dim, hidden_dim=embed_dim,
                       output_dim=embed_dim, depth=4, heads=4, dim_head=64, mlp_dim=512, dropout=0.1)
    head = lambda: MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=512, norm_fn=nn.BatchNorm1d)
    return JEPA(encoder=Enc(embed_dim), predictor=pred,
                action_encoder=Embedder(input_dim=action_dim, emb_dim=embed_dim),
                projector=head(), pred_proj=head())


def build_decoder(embed_dim=256):
    return Decoder(dim=embed_dim)
