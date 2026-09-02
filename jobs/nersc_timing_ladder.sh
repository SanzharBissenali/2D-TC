#!/bin/bash
# In-vivo TIMING ladder (decoder-scaling study, 2026-09-02): total VMC step wall-clock
# vs system size for the GPU-only baseline (arm `cnn` = same positive Combo network,
# same DS Hamiltonian, NO sign head) and the five QEC-head decoder arms (cnnqB +
# --decoder). Same recipe as the accuracy ladder (minSR lr 0.01, 8192 samples, custom
# sampler, seed 0) but SHORT: SIM_TIME 1.0 => 100 steps -- the plateau median of
# step_wall / t_head (steps 50-100) is the measurement (tie_sum's early transient is
# over by ~step 50). Runs log per-step `step_wall`, `t_head`, `n_head_configs`.
# jobid: hc{Lx}x{Ly}_ds_hx{hx}_hz{hz}_tim_{arm}[_{decoder}]  (distinct from the
# 350-step accuracy runs; skip-if-complete on .mpack).
# Threads: the head's numba kernels use NUMBA_NUM_THREADS (default 32 = the shared-node
# core count); THREADS=1 reproduces the serial head.
#
# Env: SIZES ('+'-sep, default 1x2+2x2+2x3), ARMS ('+'-sep from cnn,mwpm,anchor,greedy,
#      unionfind,tie_sum; default all six), POINTS (default 0.8:0.4 -- stresses the head: more defects per sample than the accuracy point (0.4,0)), SIM_TIME (1.0),
#      SEED (0), THREADS (32), WANDB (1), WANDB_GROUP (hc-timing).
# Submit: ... submit jobs/nersc_timing_ladder.sh -t 1:00:00 --export=ALL,SIZES=3x3,ARMS=cnn+mwpm
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_timing
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

SIZES="${SIZES:-1x2+2x2+2x3}"; SIZES="${SIZES//+/ }"
ARMS="${ARMS:-cnn+mwpm+anchor+greedy+unionfind+tie_sum}"; ARMS="${ARMS//+/ }"
POINTS="${POINTS:-0.8:0.4}"; POINTS="${POINTS//+/ }"
SIM_TIME="${SIM_TIME:-1.0}"
SEED="${SEED:-0}"
export NUMBA_NUM_THREADS="${THREADS:-32}"
echo "=== NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  code $(git -C "$REPO" rev-parse --short HEAD) ==="

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-timing}"
fi

echo "=== timing ladder sizes=[$SIZES] arms=[$ARMS] points=[$POINTS] sim_time=$SIM_TIME seed=$SEED ==="
for sz in $SIZES; do
  LX="${sz%%x*}"; LY="${sz##*x}"
  for pt in $POINTS; do
    hx="${pt%%:*}"; hz="${pt##*:}"
    for arm in $ARMS; do
      case "$arm" in
        cnn)   HEAD_FLAGS=""; SUFFIX="cnn" ;;
        mwpm)  HEAD_FLAGS="--sign_head qec --sign_impl operator"; SUFFIX="cnnqB" ;;
        anchor|greedy|unionfind|tie_sum)
               HEAD_FLAGS="--sign_head qec --sign_impl operator --decoder $arm"; SUFFIX="cnnqB_${arm}" ;;
        *) echo "!!! unknown arm '$arm'"; continue ;;
      esac
      jobid="hc${LX}x${LY}_ds_hx${hx}_hz${hz}_tim_${SUFFIX}"
      if [ -f "$OUTDIR/G-equiv_1_${jobid}.mpack" ]; then echo "=== skip $jobid (complete) ==="; continue; fi
      echo "=== NQS $jobid start $(date) ==="
      ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
          --outindex 1 --jobid "$jobid" \
          --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" \
          --hx "$hx" --hy 0.0 --hz "$hz" \
          --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
          --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
          --n_samples_fin 8192 --use_custom_sampler \
          --channels_noninv 1,16 --channels_inv 16,8,1 \
          $HEAD_FLAGS $WB_FLAGS ) \
          || echo "!!! $jobid FAILED (exit $?) -- continuing"
      echo "=== NQS $jobid done $(date) ==="
    done
  done
done
echo "=== timing ladder driver done ==="
