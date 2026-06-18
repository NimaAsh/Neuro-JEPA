"""WebDataset-backed pretraining data for FOMO300 sparse shards.

Bridges the MedARC smri-fm sparse WebDataset shards into Neuro-JEPA's
``MaskCollator`` + JEPA engine, so we can pretrain on FOMO300(k) without
re-caching the data into MONAI's PersistentDataset.

Each shard sample stores a *sparse* brain volume:
  - ``image_values.npy`` : float16, the in-brain voxel intensities, flattened
  - ``img_mask.npy``     : uint8, bit-packed (MSB-first) brain-presence mask
  - ``meta.json``        : per-scan metadata (modality, source dataset, ...)

Two data paths (selected by ``data.gpu_densify``):

* **gpu_densify: true (default, fast)** -- workers yield the *compact sparse*
  arrays (~4 MB/sample vs ~21 MB dense). ``SparseMaskCollator`` stacks them,
  derives the foreground patch grid straight from the bit-packed brain mask
  (no dense volume on the CPU), and runs the unchanged mask generators. The
  engine scatters to a dense volume on the GPU (``gpu_densify_batch``). This
  keeps the dataloader workers cheap (no unpack+scatter of a 10.4 M-voxel
  float volume) and shrinks the shared-memory payload ~5x so ``prefetch`` /
  ``num_workers`` can be raised to hide /data read latency -> smooth GPU util.

* **gpu_densify: false (fallback)** -- workers densify on the CPU and yield a
  dense ``[C, D, H, W]`` tensor (original behaviour).

Densify / bit-unpack mirrors smri-fm ``src/data/mri_data.py``
(``unpack_img_mask_batch`` / ``densify_sparse_image_batch``).
"""

import math
from glob import glob
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

from neurojepa.masks.masking import MaskCollator

# Modality string (from meta.json) -> integer flag. Unknown -> -1.
# `modality` is currently unused by the JEPA pretraining loss; plumbed through
# only to match the (image, modality) sample contract used elsewhere.
# Covers the FOMO300 modality set (see metadata.json modality_counts).
MOD_MAP = {
    "t1w": 0, "t1": 0, "t1c": 1, "t1ce": 1, "t1map": 2, "mp2rage": 3, "unit1": 4,
    "t2w": 5, "t2": 5, "flair": 6, "t2flair": 6, "t2starw": 7, "r2starmap": 8,
    "gre": 9, "flash": 10, "swi": 11, "angio": 12, "pdw": 13, "mese": 14,
    "dwi": 15, "adc": 16, "asl": 17, "cbf": 18, "m0scan": 19, "r1map": 20,
    "ute": 21, "scan": 22,
}


def expand_urls(urls: "str | list[str]") -> list[str]:
    """Expand brace/glob shard URLs (adapted from smri-fm.mri_data.expand_urls)."""
    import braceexpand

    if isinstance(urls, str):
        urls = [urls]
    results: list[str] = []
    for url in urls:
        chars = set(url)
        if chars.intersection("[*?"):
            results.extend(sorted(glob(url)))
        elif "{" in chars:
            results.extend(braceexpand.braceexpand(url))
        else:
            results.append(url)
    return results


def _warn_and_continue(exn):
    print(f"WARNING (wds) {exn!r}")
    return True


def _extract_sparse_sample(sample: dict) -> dict:
    return {
        "image_values": np.asarray(sample["image_values.npy"], dtype=np.float16),
        "img_mask": np.asarray(sample["img_mask.npy"], dtype=np.uint8),
        "meta": sample.get("meta.json", {}),
    }


def _check_packed(packed_size: int, numel: int, shape) -> None:
    expected = math.ceil(numel / 8)
    if packed_size != expected:
        raise ValueError(
            f"packed img_mask has {packed_size} bytes, expected {expected} for "
            f"image_shape {tuple(shape)}; check that data.source_img_size matches the shard geometry."
        )


def _densify(
    image_values: np.ndarray,
    packed_mask: np.ndarray,
    shape: Sequence[int],
    dtype: np.dtype = np.float16,
) -> np.ndarray:
    """CPU densify (gpu_densify=false path). Reconstruct a dense ``shape`` volume."""
    shape = tuple(int(d) for d in shape)
    numel = int(math.prod(shape))
    _check_packed(packed_mask.size, numel, shape)
    # MSB-first unpacking matches smri-fm's `unpack_img_mask_batch` (shifts 7..0).
    flat_mask = np.unpackbits(packed_mask, count=numel, bitorder="big").astype(bool)
    dense = np.zeros(numel, dtype=dtype)
    dense[flat_mask] = image_values.astype(dtype)
    return dense.reshape(shape)


def _foreground_from_packed(packed_mask: np.ndarray, shape, patch_size, min_fraction: float) -> np.ndarray:
    """Per-patch foreground grid straight from the bit-packed brain mask.

    Returns a bool array [nD, nH, nW] (True = patch has >= ``min_fraction`` brain
    voxels). Equivalent to ``compute_foreground_patches`` but uses the exact
    brain mask instead of an intensity threshold, and never builds a float volume.
    """
    shape = tuple(int(d) for d in shape)
    D, H, W = shape
    numel = D * H * W
    _check_packed(packed_mask.size, numel, shape)
    pD, pH, pW = (int(p) for p in patch_size)
    bits = np.unpackbits(packed_mask, count=numel, bitorder="big").reshape(D, H, W)
    nD, nH, nW = D // pD, H // pH, W // pW
    # strided mean over each patch block == avg_pool3d(kernel=stride=patch)
    frac = bits.reshape(nD, pD, nH, pH, nW, pW).mean(axis=(1, 3, 5), dtype=np.float32)
    return frac >= min_fraction


def _foreground_grid_from_packed(
    packed_mask: np.ndarray, source_shape, target_grid, min_fraction: float
) -> np.ndarray:
    """Per-patch foreground grid at an arbitrary ``target_grid`` (downsample path).

    Unpacks the bit-packed brain mask at the *source* (shard) geometry and
    average-pools the brain-presence fraction onto ``target_grid`` -- the model's
    post-downsample patch grid, which need not divide the source evenly
    (e.g. 208x240x208 -> 8x9x8). Returns a bool array [gD, gH, gW]
    (True = patch has >= ``min_fraction`` brain). Used when source != target;
    the cheap strided ``_foreground_from_packed`` is used when they match.

    Exact: pools the full source-resolution brain mask directly. (A gcd-strided
    pre-pool is ~3x cheaper but shifts ~4% of boundary patches across the
    threshold, so we keep the exact form -- the run is I/O- not CPU-bound anyway;
    see PERF NOTE in the faithful config for the real fix.)
    """
    import torch.nn.functional as F

    D, H, W = (int(s) for s in source_shape)
    numel = D * H * W
    _check_packed(packed_mask.size, numel, source_shape)
    bits = np.unpackbits(packed_mask, count=numel, bitorder="big").reshape(D, H, W)
    vol = torch.from_numpy(bits.astype(np.float32))[None, None]          # [1, 1, D, H, W]
    frac = F.adaptive_avg_pool3d(vol, tuple(int(g) for g in target_grid))[0, 0]
    return (frac >= min_fraction).numpy()


def gpu_densify_batch(payload: dict, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Scatter a stacked sparse batch to a dense ``[B, 1, D, H, W]`` volume on GPU.

    ``payload`` carries ``packed_mask`` [B, P] uint8 and ``image_values`` [N]
    (the per-sample in-brain values concatenated in batch order). Mirrors
    smri-fm ``densify_sparse_image_batch`` but on the GPU, off the dataloader.

    If ``payload["target_shape"]`` differs from the (source) ``image_shape``, the
    dense volume is trilinearly resampled to it -- e.g. to match the original
    Neuro-JEPA 96x108x96 geometry from the 208x240x208 FOMO300 shards.
    """
    shape = tuple(int(d) for d in payload["image_shape"])
    D, H, W = shape
    numel = D * H * W
    packed = payload["packed_mask"].to(device, non_blocking=True)        # [B, P] uint8
    values = payload["image_values"].to(device, non_blocking=True)       # [N]
    B = packed.shape[0]
    shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)   # MSB-first
    bits = (packed.unsqueeze(-1).bitwise_right_shift(shifts) & 1).reshape(B, -1)[:, :numel].bool()
    dense = torch.zeros((B, numel), dtype=dtype, device=device)
    dense[bits] = values.to(dtype)
    dense = dense.view(B, 1, D, H, W)

    target = payload.get("target_shape")
    if target is not None:
        target = tuple(int(t) for t in target)
        if target != (D, H, W):
            dense = torch.nn.functional.interpolate(
                dense, size=target, mode="trilinear", align_corners=False
            )
    return dense


def _modality_flag(meta: Any) -> int:
    if isinstance(meta, dict):
        for key in ("modality", "modality_flag", "contrast", "sequence"):
            val = meta.get(key)
            if isinstance(val, str):
                return MOD_MAP.get(val.strip().lower(), -1)
            if isinstance(val, int):
                return val
    return -1


class FomoWdsPretrainDataset(IterableDataset):
    """Iterable dataset over sparse FOMO300 WebDataset shards.

    When ``sparse=True`` yields ``(image_values[f16], packed_mask[u8], modality)``
    (densified on the GPU later); otherwise yields a dense
    ``(image[C, D, H, W], modality)``. Shards are resampled with replacement when
    ``shuffle=True`` (standard WDS large-scale SSL), so the stream is effectively
    infinite and the epoch length is set by ``optimization.ipe``, not shard count.
    """

    def __init__(
        self,
        url: "str | list[str]",
        source_img_size: Sequence[int],
        target_img_size: "Sequence[int] | None" = None,
        in_chans: int = 1,
        samples_per_epoch: int = 1,
        num_workers: int = 1,
        shuffle: bool = True,
        buffer_size: int = 1500,
        image_dtype: np.dtype = np.float16,
        sparse: bool = True,
    ):
        super().__init__()
        self.url = url
        # source = shard storage geometry (used to unpack/densify); target =
        # post-downsample geometry the model sees (defaults to source = no resize).
        self.source_img_size = tuple(int(d) for d in source_img_size)
        self.target_img_size = tuple(int(d) for d in (target_img_size or source_img_size))
        self.in_chans = int(in_chans)
        self._samples_per_epoch = int(samples_per_epoch)
        self._num_workers = max(1, int(num_workers))
        self.shuffle = shuffle
        self.buffer_size = int(buffer_size)
        self.image_dtype = image_dtype
        self.sparse = sparse

    def _per_worker(self) -> int:
        return max(1, self._samples_per_epoch // self._num_workers)

    def __len__(self) -> int:
        # Samples drawn per epoch (DataLoader divides by batch_size for len()).
        # Capping __iter__ to this many gives a clean per-epoch StopIteration and
        # avoids the IterableDataset length-mismatch warning.
        return self._per_worker() * self._num_workers

    def _build_pipeline(self):
        import webdataset as wds

        nodesplitter = wds.split_by_node if dist.is_available() and dist.is_initialized() else None
        dataset = wds.WebDataset(
            expand_urls(self.url),
            handler=_warn_and_continue,
            resampled=self.shuffle,
            shardshuffle=False,
            nodesplitter=nodesplitter,
        )
        dataset = dataset.decode().map(_extract_sparse_sample, handler=_warn_and_continue)
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
        return dataset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        nw = worker_info.num_workers if worker_info is not None else 1
        per_worker = max(1, self._samples_per_epoch // nw)
        numel = int(math.prod(self.source_img_size))
        count = 0
        for sample in self._build_pipeline():
            if count >= per_worker:
                return
            try:
                values = np.ascontiguousarray(sample["image_values"], dtype=np.float16)
                packed = np.ascontiguousarray(sample["img_mask"], dtype=np.uint8)
                _check_packed(packed.size, numel, self.source_img_size)
                modality = torch.tensor(_modality_flag(sample["meta"]), dtype=torch.long)
                if self.sparse:
                    out = (torch.from_numpy(values), torch.from_numpy(packed), modality)
                else:
                    dense = _densify(values, packed, self.source_img_size, self.image_dtype)
                    image = torch.from_numpy(dense).unsqueeze(0)         # [1, D, H, W]
                    if self.target_img_size != self.source_img_size:
                        import torch.nn.functional as F
                        image = F.interpolate(
                            image[None].float(), size=self.target_img_size,
                            mode="trilinear", align_corners=False,
                        )[0].to(image.dtype)
                    if self.in_chans > 1:
                        image = image.repeat(self.in_chans, 1, 1, 1)
                    out = (image, modality)
            except Exception as exn:  # noqa: BLE001 - skip corrupt samples, keep streaming
                _warn_and_continue(exn)
                continue
            yield out
            count += 1


class SparseMaskCollator(MaskCollator):
    """MaskCollator variant for the sparse (GPU-densify) path.

    Stacks the sparse arrays, derives the foreground patch grid from the
    bit-packed brain mask (no dense volume on the CPU), runs the same mask
    generators, and returns ``((sparse_payload, modality), masks_enc,
    masks_pred, fg_flat)``. The engine densifies ``sparse_payload`` on the GPU.
    """

    def __init__(self, *args, source_shape, target_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        # source = shard storage geometry (densify); target = post-downsample
        # geometry the model sees (== crop_size). Foreground is at the model grid.
        self.source_shape = tuple(int(d) for d in source_shape)
        self.target_shape = tuple(int(d) for d in (target_shape or source_shape))

    def __call__(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("Detect batch size of 0 in SparseMaskCollator")

        values = torch.cat([b[0] for b in batch])                 # [sum_nonzero] f16
        packed = torch.stack([b[1] for b in batch])               # [B, P] uint8
        modality = torch.stack([b[2] for b in batch])             # [B]
        payload = {
            "__sparse__": True,
            "image_values": values,
            "packed_mask": packed,
            "image_shape": self.source_shape,   # densify at shard geometry
            "target_shape": self.target_shape,  # then resample to model geometry on GPU
        }

        fg_mask = None
        if self.foreground_aware:
            packed_np = packed.numpy()
            if self.source_shape == self.target_shape:
                # No resize: cheap exact strided fraction at the patch grid.
                fg = np.stack([
                    _foreground_from_packed(packed_np[i], self.source_shape, self.patch_size, self.min_foreground_fraction)
                    for i in range(batch_size)
                ])
            else:
                # Downsample: pool the brain mask onto the model patch grid
                # (target_shape // patch_size), which may not divide the source.
                target_grid = tuple(t // p for t, p in zip(self.target_shape, self.patch_size))
                fg = np.stack([
                    _foreground_grid_from_packed(packed_np[i], self.source_shape, target_grid, self.min_foreground_fraction)
                    for i in range(batch_size)
                ])
            fg_mask = torch.from_numpy(fg)                        # [B, gD, gH, gW] bool

        collated_masks_enc, collated_masks_pred = [], []
        for mask_generator in self.mask_generators:
            masks_enc, masks_pred = mask_generator(batch_size, foreground_mask=fg_mask)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        fg_flat = fg_mask.flatten(1).float() if fg_mask is not None else None
        return ((payload, modality), collated_masks_enc, collated_masks_pred, fg_flat)


def get_pretrain_dataloaders_wds(cfg: Any, augs: Any = None):
    """Drop-in replacement for ``datasets.get_pretrain_dataloaders`` (WDS source).

    Selected from ``scripts/pretrain.py`` when ``cfg.data.loader == "wds"``.
    FOMO300 is already RAS-registered, brain-masked and intensity-normalized, so
    no MONAI loading transforms are applied (``augs`` ignored).
    """
    img_size = tuple(cfg.model.img_size)               # model (post-downsample) geometry
    # Shard storage geometry; defaults to model img_size (no resize / full-res path).
    source_img_size = tuple(cfg.data.get("source_img_size", img_size))
    patch_size = tuple(cfg.model.patch_size)
    foreground_aware = getattr(cfg.model, "foreground_aware", False)
    gpu_densify = bool(cfg.data.get("gpu_densify", True))

    mask_kwargs = dict(
        cfgs_mask=cfg.mask,
        crop_size=img_size,
        patch_size=patch_size,
        foreground_aware=foreground_aware,
        foreground_threshold=cfg.data.get("foreground_threshold", 0.0),
        min_foreground_fraction=cfg.data.get("min_foreground_fraction", 0.1),
    )
    mask_collator = (
        SparseMaskCollator(source_shape=source_img_size, target_shape=img_size, **mask_kwargs)
        if gpu_densify
        else MaskCollator(**mask_kwargs)
    )

    num_workers = int(cfg.data.num_workers)
    # One epoch = ipe optimizer steps * batch_size samples.
    samples_per_epoch = int(cfg.optimization.ipe) * int(cfg.data.batch_size)
    image_dtype = np.dtype(cfg.data.get("image_dtype", "float16"))
    dataset = FomoWdsPretrainDataset(
        url=cfg.data.train_url,
        source_img_size=source_img_size,
        target_img_size=img_size,
        in_chans=cfg.model.in_chans,
        samples_per_epoch=samples_per_epoch,
        num_workers=num_workers,
        shuffle=True,
        buffer_size=cfg.data.get("buffer_size", 1500),
        image_dtype=image_dtype,
        sparse=gpu_densify,
    )
    loader_kwargs: dict[str, Any] = dict(
        batch_size=cfg.data.batch_size,
        collate_fn=mask_collator,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
        worker_init_fn=mask_collator.worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.data.get("prefetch_factor", 6)

    # Plain DataLoader over the IterableDataset: its sampler is an
    # _InfiniteConstantSampler (no set_epoch), which InfiniteLoader handles
    # safely -- unlike wds.WebLoader.
    train_loader = DataLoader(dataset, **loader_kwargs)
    return train_loader, mask_collator


def make_val_loader_wds(cfg: Any):
    """Small held-out loader for periodic viz (``data.val_url``); None if unset.

    Single-process (num_workers=0) and tiny batch -- it's only sampled every
    ``log.viz_freq`` steps on rank 0, so it must not compete with the train
    loader for CPU/IO.
    """
    val_url = cfg.data.get("val_url", None)
    if not val_url:
        return None

    img_size = tuple(cfg.model.img_size)
    source_img_size = tuple(cfg.data.get("source_img_size", img_size))
    patch_size = tuple(cfg.model.patch_size)
    foreground_aware = getattr(cfg.model, "foreground_aware", False)
    gpu_densify = bool(cfg.data.get("gpu_densify", True))
    batch_size = int(cfg.data.get("val_batch_size", 4))

    mask_kwargs = dict(
        cfgs_mask=cfg.mask,
        crop_size=img_size,
        patch_size=patch_size,
        foreground_aware=foreground_aware,
        foreground_threshold=cfg.data.get("foreground_threshold", 0.0),
        min_foreground_fraction=cfg.data.get("min_foreground_fraction", 0.1),
    )
    collator = (
        SparseMaskCollator(source_shape=source_img_size, target_shape=img_size, **mask_kwargs)
        if gpu_densify
        else MaskCollator(**mask_kwargs)
    )
    dataset = FomoWdsPretrainDataset(
        url=val_url,
        source_img_size=source_img_size,
        target_img_size=img_size,
        in_chans=cfg.model.in_chans,
        samples_per_epoch=10 ** 9,  # effectively unbounded; we pull batches on demand
        num_workers=1,
        shuffle=True,
        buffer_size=cfg.data.get("val_buffer_size", 256),
        image_dtype=np.dtype(cfg.data.get("image_dtype", "float16")),
        sparse=gpu_densify,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
    )
