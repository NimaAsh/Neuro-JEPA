#!/bin/bash
# Neuro-JEPA pretraining on FOMO300 sparse WebDataset shards.
# Single-node 8xH100 example. Set ipe in configs/pretrain/pretrain_fomo300.yaml
# to match (world_size * batch_size): here world_size=8.
#
# On the MedARC/Sophont cluster, prefer your enroot/pyxis container + the
# project's env setup (see AGENTS.md); the `conda activate` line below is the
# upstream default -- swap it for your launcher.
#SBATCH --job-name=neurojepa-fomo300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/neurojepa-fomo300-%j.out
#SBATCH --error=./log/neurojepa-fomo300-%j.err

set -euo pipefail
mkdir -p log

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50000)))
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online   # stream to wandb (override a stale offline `wandb/settings`)

# Repo + environment.
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
    --config-name pretrain_fomo300
'
