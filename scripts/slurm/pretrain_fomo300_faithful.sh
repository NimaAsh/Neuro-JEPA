#!/bin/bash
# Neuro-JEPA pretraining on FOMO300 -- FAITHFUL recipe (200 epochs, ipe 1600,
# warmup 40). This reproduces the original 320k-step main schedule, so it is
# ~2.7x longer than the 100-epoch run. Checkpoints every 5 epochs
# (save_checkpoint_freq), so it is resumable: if the job hits the time limit,
# set meta.load_checkpoint=true (loads latest.pt + resumes the step counter
# because is_anneal=false) and resubmit.
#SBATCH --job-name=neurojepa-fomo300-faithful
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128   # 16 CPU/GPU x 8 -> feeds num_workers=16 per rank
#SBATCH --time=7-00:00:00     # 320k steps; lower if your QOS caps it (run is resumable)
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
