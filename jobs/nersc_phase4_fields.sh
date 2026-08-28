#!/bin/bash
# Phase 4 field arms: DS + QEC sign head (production impl 'operator') at
# hz != 0 (head provably EXACT at any hz -- sector theorem) and hx != 0
# (head ceiling = gate-0 F_s; energy cost of tie-broken signs ~(1-F_s)*spread,
# negligible <= hx 0.3). Same recipe as Phase 3/4 h=0 (minSR lr 0.01, 350
# steps, seed 0, 8192 samples, custom sampler); dtype stays float64 (honeycomb
# H exactly real -- no complex-JIT tax at fields).
# Exact references: results/diagnostics/signfid_hc*_ds*.json E0s (2x2: all
# points incl. hx 0.5 + mixed) and results/ed/ed_hc2x3_ds_*.json (2x3).
# Env: LX (2), LY (2), ARMS (cnnqB), POINTS -- hx:hz pairs, ':' inner comma,
#      '+' between pairs (e.g. POINTS=0.1:0+0:0.2), SIM_TIME (3.5), SEED (0),
#      WANDB (1).
# Submit (shared queue, ~14 min/pt @2x2, ~30-40 min/pt @2x3):
#   ... submit jobs/nersc_phase4_fields.sh -t 1:30:00 \
#       --export=ALL,LX=2,LY=2,POINTS=0:0.1+0:0.2+0:0.5+0.1:0.1
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_p4f
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

LX="${LX:-2}"
LY="${LY:-2}"
ARMS="${ARMS:-cnnqB}"; ARMS="${ARMS//+/ }"
POINTS="${POINTS:-0:0.1}"; POINTS="${POINTS//+/ }"
SIM_TIME="${SIM_TIME:-3.5}"
SEED="${SEED:-0}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-phase4-fields}"
    echo "=== W&B ON (offline, group ${WANDB_GROUP:-hc-phase4-fields}) ==="
fi

declare -A ARM_FLAGS
ARM_FLAGS[cnnqB]="--sign_head qec --sign_impl operator"
ARM_FLAGS[cnnqA]="--sign_head qec --sign_impl model"
ARM_FLAGS[cnnqC]="--complex_ansatz --sign_head qec --sign_impl operator"  # complex trunk + head
ARM_FLAGS[cnnqR]="--sign_head qec --sign_impl residual"                   # + tie-gated phase MLP
ARM_FLAGS[cnn]=""    # sign-free positive control (expected to FAIL the DS floors)
for arm in $ARMS; do
    [ -n "${ARM_FLAGS[$arm]+set}" ] || { echo "!!! unknown arm '$arm' (known: ${!ARM_FLAGS[*]})"; exit 1; }
done

echo "=== phase4-fields: ${LX}x${LY} ds points=[$POINTS] arms=[$ARMS] sim_time=$SIM_TIME seed=$SEED ==="
for pt in $POINTS; do
  hx="${pt%%:*}"; hz="${pt##*:}"
  for arm in $ARMS; do
    jobid="hc${LX}x${LY}_ds_hx${hx}_hz${hz}_${arm}"
    if [ -f "$OUTDIR/G-equiv_1_${jobid}.mpack" ]; then
        echo "=== skip $jobid (complete) ==="
        continue
    fi
    echo "=== NQS $jobid ==="
    ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" \
        --hx "$hx" --hy 0.0 --hz "$hz" \
        --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
        --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler \
        --channels_noninv 1,16 --channels_inv 16,8,1 \
        ${ARM_FLAGS[$arm]} $WB_FLAGS ) \
        || echo "!!! $jobid FAILED (exit $?) -- continuing"
  done
done
echo "=== phase4-fields driver done ==="
