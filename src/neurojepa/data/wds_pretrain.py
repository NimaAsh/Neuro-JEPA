"""WebDataset-backed pretraining data for FOMO300 sparse shards.

Bridges the MedARC smri-fm sparse WebDataset shards into Neuro-JEPA's
``MaskCollator`` + JEPA engine, so we can pretrain on FOMO300(k) without
re-caching the data into MONAI's PersistentDataset.

Each shard sample stores a *sparse* brain volume:
  - ``image_values.npy`` : float16, the in-brain voxel intensities, flattened
  - ``img_mask.npy``     : uint8, bit-packed (MSB-first) brain-presence mask
  - ``meta.json``        : per-scan metadata (modality, source dataset, ...)

We densify each sample back to a ``[C, D, H, W]`` float tensor (zeros outside
the brain) and yield ``(image, modality)`` -- the exact tuple
``PretrainDataset.__getitem__`` returns -- so the rest of the pipeline
(``MaskCollator``, foreground-aware masking, the JEPA loss) is unchanged.

The densify / bit-unpack logic mirrors smri-fm ``src/data/mri_data.py``
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
# `modality` is currently unused by the JEPA pretraining loss, so a missing /
# unknown value is harmless; it is plumbed through only to match the
# (image, modality) sample contract used elsewhere in the codebase.
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


def _densify(image_values: np.ndarray, packed_mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Reconstruct a dense ``shape`` volume from packed mask + in-brain values."""
    shape = tuple(int(d) for d in shape)
    numel = int(math.prod(shape))
    expected_packed = math.ceil(numel / 8)
    if packed_mask.size != expected_packed:
        raise ValueError(
            f"packed img_mask has {packed_mask.size} bytes, expected {expected_packed} "
            f"for image_shape {shape}; check that data.img_size matches the shard geometry."
        )
    # MSB-first unpacking matches smri-fm's `unpack_img_mask_batch` (shifts 7..0).
    flat_mask = np.unpackbits(packed_mask, count=numel, bitorder="big").astype(bool)
    dense = np.zeros(numel, dtype=np.float32)
    dense[flat_mask] = image_values.astype(np.float32)
    return dense.reshape(shape)


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

    Yields ``(image[C, D, H, W] float32, modality long)``. Shards are resampled
    with replacement when ``shuffle=True`` (the standard WebDataset large-scale
    SSL pattern), so the stream is effectively infinite and the epoch length is
    controlled by ``optimization.ipe`` in the engine, not by the shard count.
    """

    def __init__(
        self,
        url: "str | list[str]",
        img_size: Sequence[int],
        in_chans: int = 1,
        samples_per_epoch: int = 1,
        num_workers: int = 1,
        shuffle: bool = True,
        buffer_size: int = 8000,
    ):
        super().__init__()
        self.url = url
        self.img_size = tuple(int(d) for d in img_size)
        self.in_chans = int(in_chans)
        self._samples_per_epoch = int(samples_per_epoch)
        self._num_workers = max(1, int(num_workers))
        self.shuffle = shuffle
        self.buffer_size = int(buffer_size)

    def _per_worker(self) -> int:
        return max(1, self._samples_per_epoch // self._num_workers)

    def __len__(self) -> int:
        # Number of samples drawn per epoch (PyTorch divides this by batch_size
        # to report len(loader)). The resampled WDS stream is infinite, so we
        # cap __iter__ to exactly this many per epoch -- matching __len__ avoids
        # the IterableDataset length-mismatch warning and gives InfiniteLoader a
        # clean per-epoch StopIteration to reset on.
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
        count = 0
        for sample in self._build_pipeline():
            if count >= per_worker:
                return
            try:
                dense = _densify(sample["image_values"], sample["img_mask"], self.img_size)
            except Exception as exn:  # noqa: BLE001 - skip corrupt samples, keep streaming
                _warn_and_continue(exn)
                continue
            image = torch.from_numpy(dense).unsqueeze(0)  # [1, D, H, W]
            if self.in_chans > 1:
                image = image.repeat(self.in_chans, 1, 1, 1)
            modality = torch.tensor(_modality_flag(sample["meta"]), dtype=torch.long)
            yield image, modality
            count += 1


def get_pretrain_dataloaders_wds(cfg: Any, augs: Any = None):
    """Drop-in replacement for ``datasets.get_pretrain_dataloaders`` (WDS source).

    Selected from ``scripts/pretrain.py`` when ``cfg.data.loader == "wds"``.
    Reuses the unchanged ``MaskCollator``; FOMO300 is already RAS-registered,
    brain-masked and intensity-normalized, so no MONAI loading transforms are
    applied (``augs`` is accepted for signature parity and ignored).
    """
    img_size = tuple(cfg.model.img_size)
    patch_size = tuple(cfg.model.patch_size)
    foreground_aware = getattr(cfg.model, "foreground_aware", False)

    mask_collator = MaskCollator(
        cfgs_mask=cfg.mask,
        crop_size=img_size,
        patch_size=patch_size,
        foreground_aware=foreground_aware,
        foreground_threshold=cfg.data.get("foreground_threshold", 0.0),
        min_foreground_fraction=cfg.data.get("min_foreground_fraction", 0.1),
    )

    num_workers = int(cfg.data.num_workers)
    # One epoch = ipe optimizer steps * batch_size samples. Capping the stream
    # to this many per epoch keeps len(loader) == ipe and silences the
    # IterableDataset length warning.
    samples_per_epoch = int(cfg.optimization.ipe) * int(cfg.data.batch_size)
    dataset = FomoWdsPretrainDataset(
        url=cfg.data.train_url,
        img_size=img_size,
        in_chans=cfg.model.in_chans,
        samples_per_epoch=samples_per_epoch,
        num_workers=num_workers,
        shuffle=True,
        buffer_size=cfg.data.get("buffer_size", 8000),
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
        loader_kwargs["prefetch_factor"] = cfg.data.get("prefetch_factor", 4)

    # Plain DataLoader over the IterableDataset: its sampler is an
    # _InfiniteConstantSampler (no set_epoch), which InfiniteLoader handles
    # safely -- unlike wds.WebLoader, whose internals InfiniteLoader assumes a
    # `.sampler` on.
    train_loader = DataLoader(dataset, **loader_kwargs)
    return train_loader, mask_collator
