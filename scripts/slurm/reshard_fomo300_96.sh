#!/bin/bash
# One-time: re-shard the 208^3 FOMO300 sparse shards to the faithful 96x108x96
# geometry (see scripts/reshard_fomo300_96.py). Produces ~5-6x smaller sparse
# shards with no 208^3 unpack, so the faithful training becomes GPU-bound.
#
# CPU-only job (the transform runs on CPU). Set FOMO300_96_DIR to a WRITABLE
# output dir (the same value the pre96 training configs read via ${oc.env}).
# If your QOS requires a GPU on `main`, add: #SBATCH --gres=gpu:1 (unused).
#SBATCH --job-name=fomo300-reshard-96
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/fomo300-reshard-96-%j.out
#SBATCH --error=./log/fomo300-reshard-96-%j.err

set -euo pipefail
mkdir -p log
: "${FOMO300_96_DIR:?set FOMO300_96_DIR to a writable output dir for the 96^3 shards}"

REPO=${REPO:-$HOME/smri-proj/Neuro-JEPA}
cd "$REPO"
source "$REPO/.venv/bin/activate"

# Resumable: shards already present in FOMO300_96_DIR are skipped (add --overwrite to redo).
python scripts/reshard_fomo300_96.py \
  --in-glob '/data/smri-datasets/FOMO300/shard.{000000..001134}.tar' \
  --out-dir "$FOMO300_96_DIR" \
  --source-size 208,240,208 \
  --target-size 96,108,96 \
  --mask-threshold 0.5 \
  --num-workers "${SLURM_CPUS_PER_TASK:-64}"

echo "Re-shard complete. Train with: --config-name pretrain_fomo300_faithful_pre96"
