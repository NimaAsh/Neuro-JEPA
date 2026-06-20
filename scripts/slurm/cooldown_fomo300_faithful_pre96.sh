#!/bin/bash
# Neuro-JEPA cooldown -- FAITHFUL recipe on PRE-RESHARDED 96^3 shards (40 epochs,
# ipe 1600), resuming the pre96 200-epoch checkpoint. GPU-bound, ~3-5 h.
# Export FOMO300_96_DIR (same dir as the reshard / main run).
#SBATCH --job-name=neurojepa-fomo300-faithful-pre96-cooldown
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=./log/neurojepa-fomo300-faithful-pre96-cooldown-%j.out
#SBATCH --error=./log/neurojepa-fomo300-faithful-pre96-cooldown-%j.err

set -euo pipefail
mkdir -p log
: "${FOMO300_96_DIR:?set FOMO300_96_DIR to the dir you re-sharded into}"
export FOMO300_96_DIR

# Raise the open-file limit (DataLoader shm tensors; see scripts/pretrain.py).
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)"

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
  # Clear OUR OWN stale /dev/shm torch_* files a prior job left on this node
  # (file_system strategy leaks named shm files); they cause a cold-start unlink
  # race. Safe: gpu:8 = whole-node alloc, runs before any rank is spawned, scoped
  # to $USER, so it cannot touch another job/user.
  find /dev/shm -maxdepth 1 -user "$USER" -name "torch_*" -delete 2>/dev/null || true
  python -m torch.distributed.run \
    --nproc_per_node "$SLURM_GPUS_ON_NODE" \
    --nnodes "$SLURM_NNODES" \
    --node_rank "$SLURM_NODEID" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    '"$REPO"'/scripts/pretrain.py \
    -cd '"$REPO"'/configs/pretrain \
    --config-name cooldown_fomo300_faithful_pre96
'
