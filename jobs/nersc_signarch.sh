#!/bin/bash
# High-capacity architectures on the x-only sign problem: scripts/sign_learn_arch.py.
# Fresh GPU-generated batches, held-out x patterns, one architecture per job.
#
# Env: ARCH (mlp|cnn|tf), LX, LY, SAMPLES (plain integer, default 1000000000), MAX_MIN (graceful
#      stop after this many minutes; keep >= 8 below the walltime), WIDTH/LAYERS/HEADS/LR/BATCH
#      (0 = architecture/size default), SEED, TAG.
#
# Submit (debug queue, quick look):
#   bash scripts/cluster.sh submit jobs/nersc_signarch.sh --export=ALL,ARCH=cnn,LX=5,LY=5,MAX_MIN=24
# Submit (shared queue, long):
#   bash scripts/cluster.sh submit jobs/nersc_signarch.sh -q shared -t 6:00:00 \
#     --export=ALL,ARCH=tf,LX=8,LY=8,MAX_MIN=340
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_signarch
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
export PYTHONPATH=$REPO
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
ARGS=(--arch "${ARCH:?ARCH}" --Lx "${LX:-5}" --Ly "${LY:-5}" --samples "${SAMPLES:-1000000000}"
      --batch "${BATCH:-0}" --seed "${SEED:-0}" --width "${WIDTH:-0}" --layers "${LAYERS:-0}" --lr "${LR:-0}"
      --max_minutes "${MAX_MIN:-0}" --out_dir "$REPO/results/signarch")
[ -n "${HEADS:-}" ] && ARGS+=(--heads "$HEADS")
[ -n "${TAG:-}" ] && ARGS+=(--tag "$TAG")
python scripts/sign_learn_arch.py "${ARGS[@]}"
rc=$?
echo "=== signarch done rc=$rc $(date) ==="
exit $rc
