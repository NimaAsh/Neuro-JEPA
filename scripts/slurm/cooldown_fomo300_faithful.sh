#!/bin/bash
# Neuro-JEPA cooldown (LR anneal) -- FAITHFUL recipe (40 epochs, ipe 1600),
# resuming the faithful 200-epoch checkpoint. ~64k steps -> under a day.
#SBATCH --job-name=neurojepa-fomo300-faithful-cooldown
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128   # 16 CPU/GPU x 8 -> feeds num_workers=16 per rank
#SBATCH --time=1-12:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/neurojepa-fomo300-faithful-cooldown-%j.out
#SBATCH --error=./log/neurojepa-fomo300-faithful-cooldown-%j.err

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
    --config-name cooldown_fomo300_faithful
'
