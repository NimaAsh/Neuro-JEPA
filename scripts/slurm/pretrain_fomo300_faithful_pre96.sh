#!/bin/bash
# Neuro-JEPA pretraining -- FAITHFUL recipe on PRE-RESHARDED 96^3 shards.
# Same 320k-step schedule as pretrain_fomo300_faithful.sh, but GPU-bound (no
# 208^3 reads/unpack) -> high util, ~1-1.5 days. Run scripts/slurm/reshard_fomo300_96.sh
# first and export FOMO300_96_DIR to the output dir it wrote.
# Resumable: set meta.load_checkpoint=true and resubmit if it hits the time limit.
#SBATCH --job-name=neurojepa-fomo300-faithful-pre96
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=1-12:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/neurojepa-fomo300-faithful-pre96-%j.out
#SBATCH --error=./log/neurojepa-fomo300-faithful-pre96-%j.err

set -euo pipefail
mkdir -p log
: "${FOMO300_96_DIR:?set FOMO300_96_DIR to the dir you re-sharded into (scripts/slurm/reshard_fomo300_96.sh)}"
export FOMO300_96_DIR

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
    --config-name pretrain_fomo300_faithful_pre96
'
