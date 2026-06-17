"""Lightweight wandb visualizations for JEPA pretraining.

Neuro-JEPA predicts in *latent* space (no pixel decoder), so there is no
"reconstructed image" the way an MAE has. What we *can* show is the input
volume and the masking the model is asked to predict through. This logs three
orthogonal central slices of a sample with the predicted-(target) patches
highlighted, so you can watch what the encoder sees vs. what the predictor must
fill in.
"""

from typing import Sequence

import numpy as np
import torch


def _norm_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def log_jepa_masking(wandb_run, volume, masks_pred, patch_size, step, tag="train"):
    """Log input slices + target-patch overlay to wandb.

    Args:
        wandb_run: the wandb run (rank 0 only).
        volume: dense batch tensor ``[B, 1, H, W, D]``.
        masks_pred: list (per mask scale) of ``[B, K]`` long tensors of predicted
            token ids over the patch grid (row-major over ``(gH, gW, gD)``).
        patch_size: ``(pH, pW, pD)``.
        step: global step (x-axis in wandb).
        tag: log-key prefix, e.g. "train" / "val".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    pH, pW, pD = (int(p) for p in patch_size)
    vol = volume[0, 0].detach().float().cpu().numpy()  # [H, W, D]
    H, W, D = vol.shape
    gH, gW, gD = H // pH, W // pW, D // pD

    # Union of predicted (target) token ids for sample 0 across mask scales.
    occ = np.zeros((gH, gW, gD), dtype=bool)
    n_tokens = gH * gW * gD
    for m in masks_pred:
        row = m[0] if m.dim() == 2 else m
        for tid in row.flatten().tolist():
            if 0 <= tid < n_tokens:
                ih, iw, idd = np.unravel_index(int(tid), (gH, gW, gD))
                occ[ih, iw, idd] = True

    def up(grid2d, a, b):  # patch grid -> voxel resolution
        return np.kron(grid2d, np.ones((a, b), dtype=bool))

    views = [
        ("axial",    vol[:, :, D // 2], up(occ[:, :, (D // 2) // pD], pH, pW)[:H, :W]),
        ("coronal",  vol[:, W // 2, :], up(occ[:, (W // 2) // pW, :], pH, pD)[:H, :D]),
        ("sagittal", vol[H // 2, :, :], up(occ[(H // 2) // pH, :, :], pW, pD)[:W, :D]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for c, (name, sl, ov) in enumerate(views):
        base = _norm_u8(sl).T
        axes[0, c].imshow(base, cmap="gray", origin="lower")
        axes[0, c].set_title(name)
        axes[0, c].axis("off")
        axes[1, c].imshow(base, cmap="gray", origin="lower")
        axes[1, c].imshow(np.ma.masked_where(~ov.T, ov.T), cmap="autumn", alpha=0.45, origin="lower")
        axes[1, c].set_title(f"{name} — target patches")
        axes[1, c].axis("off")
    fig.suptitle(f"{tag}: input (top) + predicted patches (bottom) — step {step}")
    fig.tight_layout()
    wandb_run.log({f"{tag}/masking": wandb.Image(fig)}, step=step)
    plt.close(fig)
