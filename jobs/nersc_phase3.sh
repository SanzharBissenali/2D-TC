#!/bin/bash
# Phase 3: honeycomb fixed-point experiment (h=0) -- the DS-failure measurement.
# Arms x models at one patch size per job:
#   cnn      = approximately-symmetric Combo, float64 (positive)  -> TC exact;
#              DS must plateau at the EXACT positive-state (Hastings) floor:
#              1x2 = -10-sqrt(2) = -11.41421356 (observed to 8 decimals in smoke),
#              2x2 = -19.5699527513 (exact combinatorial prediction).
#   cnnc     = same Combo with --complex_ansatz (complex CNN) -- the empirical
#              "can a generic phase head learn (-1)^{#loops}" arm (no theorem).
#   plaincnn = unconstrained baseline (falls to the trivial |+> state in smoke).
# minSR (lr 0.01) for ALL arms: dense-SR tdvp is O(P^2-3)/step at option-(a)
# param counts (12.5k-19k). Skip-if-.mpack-exists => resumable.
# Env: LX (2), LY (2), MODELS ("tc ds"), ARMS ("cnn plaincnn"), SIM_TIME (3.5),
#      SEED (0), WANDB (1). '+' works as a separator in any env list.
#
# Submit (grouping = size x dtype-class; complex separated for the JIT tax):
#   ... submit jobs/nersc_phase3.sh -t 1:30:00 --export=ALL,LX=2,LY=2            # A1
#   ... submit jobs/nersc_phase3.sh -t 1:30:00 --export=ALL,LX=2,LY=3            # A2
#   ... submit jobs/nersc_phase3.sh -t 2:30:00 --export=ALL,LX=2,LY=2,ARMS=cnnc  # B1
#   ... submit jobs/nersc_phase3.sh -t 2:30:00 --export=ALL,LX=2,LY=3,ARMS=cnnc  # B2
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_p3
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

LX="${LX:-2}"
LY="${LY:-2}"
MODELS="${MODELS:-tc ds}"; MODELS="${MODELS//+/ }"
ARMS="${ARMS:-cnn plaincnn}"; ARMS="${ARMS//+/ }"
SIM_TIME="${SIM_TIME:-3.5}"
SEED="${SEED:-0}"

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-phase3-${LX}x${LY}}"
    echo "=== W&B ON (offline, group ${WANDB_GROUP:-hc-phase3-${LX}x${LY}}) ==="
fi

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--channels_noninv 1,16 --channels_inv 16,8,1"
ARM_FLAGS[cnnc]="--channels_noninv 1,16 --channels_inv 16,8,1 --complex_ansatz"
ARM_FLAGS[plaincnn]="--architecture PlainCNN --channels_noninv 1,32,24,8,2 --channels_inv 16,8,1"
for arm in $ARMS; do
    [ -n "${ARM_FLAGS[$arm]+set}" ] || { echo "!!! unknown arm '$arm' (known: ${!ARM_FLAGS[*]})"; exit 1; }
done

echo "=== phase3: ${LX}x${LY} models=[$MODELS] arms=[$ARMS] sim_time=$SIM_TIME seed=$SEED ==="
for m in $MODELS; do
  for arm in $ARMS; do
    jobid="hc${LX}x${LY}_${m}_h0_${arm}"
    if [ -f "$OUTDIR/G-equiv_1_${jobid}.mpack" ]; then
        echo "=== skip $jobid (complete) ==="
        continue
    fi
    echo "=== NQS $jobid ==="
    ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --lattice honeycomb --model "$m" --Lx "$LX" --Ly "$LY" \
        --hx 0.0 --hy 0.0 --hz 0.0 \
        --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
        --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler \
        ${ARM_FLAGS[$arm]} $WB_FLAGS ) \
        || echo "!!! $jobid FAILED (exit $?) -- continuing"
  done
done
echo "=== phase3 driver done ==="
