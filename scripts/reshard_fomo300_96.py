#!/usr/bin/env python
"""Re-shard the FOMO300 sparse WebDataset to the original Neuro-JEPA geometry.

Reads the 208x240x208 sparse shards, densifies each scan (zero-fill outside
brain), trilinearly downsamples to 96x108x96 (identical to the on-the-fly
``gpu_densify_batch`` resample), re-masks at >= ``--mask-threshold`` brain and
re-packs as a *sparse* 96^3 WebDataset sample. One output shard per input shard
(same basename) so the 90/10 train/val split is preserved.

Why: training the faithful 96^3 run directly off the 208^3 shards is I/O-bound
(shared-FS reads + per-sample 208^3 unpack). The 96^3 shards are ~5-6x smaller
and need no unpack, so trainings become GPU-bound. Run ONCE.

Usage (see scripts/slurm/reshard_fomo300_96.sh):
  python scripts/reshard_fomo300_96.py \
    --in-glob '/data/smri-datasets/FOMO300/shard.{000000..001134}.tar' \
    --out-dir /path/to/your/writable/FOMO300_96 \
    --num-workers 64
"""
import argparse
import math
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds


def _parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))


def expand(glob_or_brace: str):
    import braceexpand
    from glob import glob

    if any(c in glob_or_brace for c in "[*?"):
        return sorted(glob(glob_or_brace))
    if "{" in glob_or_brace:
        return list(braceexpand.braceexpand(glob_or_brace))
    return [glob_or_brace]


def transform(values: np.ndarray, packed: np.ndarray, src, tgt, mask_threshold: float):
    """208^3 sparse sample -> 96^3 sparse (values_f16, packed_u8)."""
    D, H, W = src
    numel = D * H * W
    if packed.size != math.ceil(numel / 8):
        raise ValueError(f"packed {packed.size} != ceil({numel}/8)")
    flat = np.unpackbits(packed, count=numel, bitorder="big").astype(bool)
    img = np.zeros(numel, dtype=np.float32)
    img[flat] = values.astype(np.float32)
    timg = torch.from_numpy(img.reshape(1, 1, D, H, W))
    tmsk = torch.from_numpy(flat.astype(np.float32).reshape(1, 1, D, H, W))
    img_t = F.interpolate(timg, size=tgt, mode="trilinear", align_corners=False)[0, 0].reshape(-1)
    msk_t = F.interpolate(tmsk, size=tgt, mode="trilinear", align_corners=False)[0, 0].reshape(-1)
    brain = msk_t >= mask_threshold                                   # majority-brain voxels
    values96 = img_t[brain].to(torch.float16).numpy()
    packed96 = np.packbits(brain.numpy().astype(np.uint8), bitorder="big")
    return values96, packed96


def process_shard(in_path: str, out_dir: str, src, tgt, mask_threshold: float, overwrite: bool):
    base = os.path.basename(in_path)
    out_path = os.path.join(out_dir, base)
    if os.path.exists(out_path) and not overwrite:
        return f"skip {base} (exists)"
    tmp_path = out_path + ".tmp"
    n = 0
    src_ds = wds.WebDataset(in_path, shardshuffle=False, nodesplitter=None)
    with wds.TarWriter(tmp_path) as sink:
        for sample in src_ds.decode():
            values = np.asarray(sample["image_values.npy"], dtype=np.float16)
            packed = np.asarray(sample["img_mask.npy"], dtype=np.uint8)
            v96, p96 = transform(values, packed, src, tgt, mask_threshold)
            sink.write({
                "__key__": sample["__key__"],
                "image_values.npy": v96,
                "img_mask.npy": p96,
                "meta.json": sample.get("meta.json", {}),
            })
            n += 1
    os.replace(tmp_path, out_path)                                    # atomic -> resumable
    return f"done {base} ({n} samples)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-size", default="208,240,208")
    ap.add_argument("--target-size", default="96,108,96")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)                                          # one thread/proc; parallelism is over shards
    src, tgt = _parse_shape(args.source_size), _parse_shape(args.target_size)
    os.makedirs(args.out_dir, exist_ok=True)
    shards = expand(args.in_glob)
    print(f"re-sharding {len(shards)} shards {src} -> {tgt} (mask>={args.mask_threshold}) "
          f"into {args.out_dir} with {args.num_workers} workers", flush=True)

    fn = partial(process_shard, out_dir=args.out_dir, src=src, tgt=tgt,
                 mask_threshold=args.mask_threshold, overwrite=args.overwrite)
    with Pool(args.num_workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(fn, shards), 1):
            print(f"[{i}/{len(shards)}] {msg}", flush=True)
    print("ALL SHARDS DONE", flush=True)


if __name__ == "__main__":
    main()
