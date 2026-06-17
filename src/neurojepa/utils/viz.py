"""Lightweight wandb visualizations for JEPA pretraining.

Neuro-JEPA predicts in *latent* space (no pixel decoder), so there is no
"reconstructed image" the way an MAE has. We log instead:

* **input + masking** -- orthogonal central slices with the predicted (target)
  patches highlighted: what the encoder sees vs. what the predictor must fill in.
* **latent prediction error** -- per-target-patch ``|z - h|`` (the JEPA analog of
  reconstruction quality) mapped back onto the slices; it should sharpen/shrink
  over the run as the predictor improves.

Both are logged for train and (held-out) val batches.
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


@torch.no_grad()
def per_patch_error(z, h, masks_pred, num_tokens: int):
    """Per-target-patch latent L1 error, scattered onto the full patch grid.

    Mirrors ``jepa_loss.loss_fn``'s masking exactly. ``z`` is the predictor
    output (list -> per-scale ``[B, K, Dd]``), ``h`` the target-encoder output
    (list of ``[B, N, Dd]``), ``masks_pred`` the predicted token ids
    (list -> per-scale ``[B, K]``). Returns ``err`` ``[B, num_tokens]`` with NaN
    at non-predicted patches.
    """
    from neurojepa.masks.utils import apply_masks

    B = h[0].shape[0]
    device = h[0].device
    err_sum = torch.zeros(B, num_tokens, device=device)
    cnt = torch.zeros(B, num_tokens, device=device)
    for hi, zi, mi in zip(h, z, masks_pred):
        h_masked = apply_masks(hi, mi, concat=False)  # list per scale [B, K, Dd]
        for zij, hij, ids in zip(zi, h_masked, mi):
            e = (zij.float() - hij.float()).abs().mean(dim=-1)  # [B, K]
            err_sum.scatter_add_(1, ids, e)
            cnt.scatter_add_(1, ids, torch.ones_like(e))
    err = err_sum / cnt.clamp(min=1)
    err[cnt == 0] = float("nan")
    return err


def log_jepa_masking(wandb_run, volume, masks_pred, patch_size, step, tag="train", err_tokens=None):
    """Log input slices + target-patch overlay (+ optional error heatmap) to wandb.

    Args:
        volume: dense batch tensor ``[B, 1, H, W, D]``.
        masks_pred: list (per scale) of ``[B, K]`` predicted token ids.
        patch_size: ``(pH, pW, pD)``.
        err_tokens: optional ``[num_tokens]`` per-patch error for sample 0
            (NaN where not predicted) -> adds a heatmap row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    pH, pW, pD = (int(p) for p in patch_size)
    vol = volume[0, 0].detach().float().cpu().numpy()  # [H, W, D]
    H, W, D = vol.shape
    gH, gW, gD = H // pH, W // pW, D // pD
    n_tokens = gH * gW * gD

    occ = np.zeros((gH, gW, gD), dtype=bool)
    for m in masks_pred:
        row = m[0] if m.dim() == 2 else m
        for tid in row.flatten().tolist():
            if 0 <= tid < n_tokens:
                ih, iw, idd = np.unravel_index(int(tid), (gH, gW, gD))
                occ[ih, iw, idd] = True

    err_grid = None
    if err_tokens is not None:
        err_grid = np.asarray(err_tokens.detach().float().cpu().numpy()).reshape(gH, gW, gD)

    def up_bool(g, a, b):
        return np.kron(g, np.ones((a, b), dtype=bool))

    def up_f(g, a, b):
        return np.kron(g, np.ones((a, b), dtype=np.float32))

    planes = [
        ("axial", vol[:, :, D // 2], occ[:, :, (D // 2) // pD], None if err_grid is None else err_grid[:, :, (D // 2) // pD], pH, pW, H, W),
        ("coronal", vol[:, W // 2, :], occ[:, (W // 2) // pW, :], None if err_grid is None else err_grid[:, (W // 2) // pW, :], pH, pD, H, D),
        ("sagittal", vol[H // 2, :, :], occ[(H // 2) // pH, :, :], None if err_grid is None else err_grid[(H // 2) // pH, :, :], pW, pD, W, D),
    ]

    nrows = 3 if err_grid is not None else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows))
    if err_grid is not None:
        finite = err_grid[np.isfinite(err_grid)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    for c, (name, sl, og, eg, a, b, s0, s1) in enumerate(planes):
        base = _norm_u8(sl).T
        axes[0, c].imshow(base, cmap="gray", origin="lower"); axes[0, c].set_title(name); axes[0, c].axis("off")
        ov = up_bool(og, a, b)[:s0, :s1]
        axes[1, c].imshow(base, cmap="gray", origin="lower")
        axes[1, c].imshow(np.ma.masked_where(~ov.T, ov.T), cmap="autumn", alpha=0.45, origin="lower")
        axes[1, c].set_title("target patches"); axes[1, c].axis("off")
        if err_grid is not None:
            eup = up_f(eg, a, b)[:s0, :s1]
            axes[2, c].imshow(base, cmap="gray", origin="lower")
            im = axes[2, c].imshow(np.ma.masked_invalid(eup.T), cmap="magma", alpha=0.75, origin="lower", vmin=vmin, vmax=vmax)
            axes[2, c].set_title("latent |z-h|"); axes[2, c].axis("off")
    if err_grid is not None:
        fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.02)
    fig.suptitle(f"{tag} — step {step}")
    fig.tight_layout()
    wandb_run.log({f"{tag}/masking": wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_jepa_val(wandb_run, batch_data, model, device, patch_size, step, amp_dtype, use_amp, tag="val"):
    """Forward a held-out batch (no grad) and log its masking + latent error."""
    import torch.nn.functional as F

    encoder, target_encoder, predictor = model["encoder"], model["target_encoder"], model["predictor"]
    all_packed_data, all_masks_enc, all_masks_pred = batch_data[:3]
    all_data = all_packed_data[0]

    if isinstance(all_data, dict) and all_data.get("__sparse__"):
        from neurojepa.data.wds_pretrain import gpu_densify_batch
        data = [gpu_densify_batch(all_data, device)]
    else:
        from monai.utils.type_conversion import convert_to_tensor
        if not isinstance(all_data, list):
            all_data = [all_data]
        data = [convert_to_tensor(d, track_meta=False).to(device, non_blocking=True) for d in all_data]
    masks_enc = [[m.to(device, non_blocking=True) for m in all_masks_enc]]
    masks_pred = [[m.to(device, non_blocking=True) for m in all_masks_pred]]

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        h = target_encoder(data)
        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
        z, _ = encoder(data, masks_enc)
        z = predictor(z, masks_enc, masks_pred)

    pH, pW, pD = (int(p) for p in patch_size)
    H, W, D = data[0].shape[2:]
    num_tokens = (H // pH) * (W // pW) * (D // pD)
    err = per_patch_error(z, h, masks_pred, num_tokens)
    log_jepa_masking(wandb_run, data[0], masks_pred[0], patch_size, step, tag=tag, err_tokens=err[0])
