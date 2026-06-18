#!/bin/bash
# Neuro-JEPA pretraining on FOMO300 -- FAITHFUL recipe (200 epochs, ipe 1600,
# warmup 40), geometry matched to the original (208^3 shards GPU-downsampled to
# 96x108x96 / patch 12, batch 48). Reproduces the original 320k-step schedule.
# The model is ~4.4x cheaper than the full-res run but the run is DATALOADER-BOUND
# (208^3 unpack/downsample per sample), so util sags; expect ~1.5-2.5 days.
# Checkpoints every 5 epochs (save_checkpoint_freq), so it is resumable: if the
# job hits the time limit, set meta.load_checkpoint=true (loads latest.pt +
# resumes the step counter because is_anneal=false) and resubmit.
#SBATCH --job-name=neurojepa-fomo300-faithful
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128   # 16 CPU/GPU x 8 -> feeds num_workers=16 per rank
#SBATCH --time=3-00:00:00     # 320k steps @ 96^3; lower if your QOS caps it (run is resumable)
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/neurojepa-fomo300-faithful-%j.out
#SBATCH --error=./log/neurojepa-fomo300-faithful-%j.err

set -euo pipefail
mkdir -p log

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50000)))
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

REPO=${REPO:-$HOME/smri-proj/Neuro-JEPA}
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/src"

srun --cpu_bind=v --accel-bind=gn bash -c '
  export WORLD_SIZE=$SLURM_NTASKS
  export RANK=$SLURM_PROCID
  echo "NODE $SLURMD_NODENAME: RANK $RANK / $WORLD_SIZE"
  python -m torch.distributed.run \
    --nproc_per_node "$SLURM_GPUS_ON_NODE" \
    --nnodes "$SLURM_NNODES" \
    --node_rank "$SLURM_NODEID" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    '"$REPO"'/scripts/pretrain.py \
    -cd '"$REPO"'/configs/pretrain \
    --config-name pretrain_fomo300_faithful
'
