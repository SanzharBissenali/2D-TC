#!/bin/bash
# Decoder-ladder NQS runs vs system size (decoder-scaling study, 2026-09-02).
# Same recipe as Phase 3/4/4c (jobs/nersc_phase4_fields.sh: minSR lr 0.01,
# 350 steps, seed 0, 8192 samples, custom sampler, float64) with the production
# sign-head arm cnnqB (--sign_head qec --sign_impl operator) and a LIST of
# decoders x a LIST of sizes in one job. jobid naming matches Phase 4c so
# existing runs are reused (skip-if-complete):
#   hc{Lx}x{Ly}_ds_hx{hx}_hz{hz}_cnnqB[_{decoder}]   (mwpm => no suffix)
#
# Env: SIZES ('+'-sep LxxLy, default 1x2+2x2), DECODERS ('+'-sep, default all 5),
#      POINTS (hx:hz pairs, '+' between, default 0.4:0), SIM_TIME (3.5), SEED (0),
#      WANDB (1), WANDB_GROUP (hc-decoder-scaling).
# Submit (shared): ... submit jobs/nersc_decoder_ladder.sh -t 1:00:00 \
#     --export=ALL,SIZES=1x2+2x2,DECODERS=anchor+greedy
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_dladder
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

SIZES="${SIZES:-1x2+2x2}"; SIZES="${SIZES//+/ }"
DECODERS="${DECODERS:-mwpm+anchor+greedy+unionfind+tie_sum}"; DECODERS="${DECODERS//+/ }"
POINTS="${POINTS:-0.4:0}"; POINTS="${POINTS//+/ }"
SIM_TIME="${SIM_TIME:-3.5}"
SEED="${SEED:-0}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-decoder-scaling}"
    echo "=== W&B ON (offline, group ${WANDB_GROUP:-hc-decoder-scaling}) ==="
fi

echo "=== decoder ladder sizes=[$SIZES] decoders=[$DECODERS] points=[$POINTS] sim_time=$SIM_TIME seed=$SEED ==="
echo "=== code: $(git -C "$REPO" rev-parse --short HEAD) ==="
for sz in $SIZES; do
  LX="${sz%%x*}"; LY="${sz##*x}"
  for pt in $POINTS; do
    hx="${pt%%:*}"; hz="${pt##*:}"
    for dec in $DECODERS; do
      DEC_FLAG=""; DEC_SUFFIX=""
      if [ "$dec" != "mwpm" ]; then DEC_FLAG="--decoder $dec"; DEC_SUFFIX="_${dec}"; fi
      jobid="hc${LX}x${LY}_ds_hx${hx}_hz${hz}_cnnqB${DEC_SUFFIX}"
      if [ -f "$OUTDIR/G-equiv_1_${jobid}.mpack" ]; then
          echo "=== skip $jobid (complete) ==="; continue
      fi
      echo "=== NQS $jobid start $(date) ==="
      ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
          --outindex 1 --jobid "$jobid" \
          --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" \
          --hx "$hx" --hy 0.0 --hz "$hz" \
          --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
          --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
          --n_samples_fin 8192 --use_custom_sampler \
          --channels_noninv 1,16 --channels_inv 16,8,1 \
          --sign_head qec --sign_impl operator $DEC_FLAG $WB_FLAGS ) \
          || echo "!!! $jobid FAILED (exit $?) -- continuing"
      echo "=== NQS $jobid done $(date) ==="
    done
  done
done
echo "=== decoder ladder driver done ==="
