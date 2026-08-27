#!/bin/bash
# Phase 4: DS h=0 with the QEC sign head -- the Hastings-floor-smashing runs,
# both formulations together (user requirement: verify equivalence in vivo).
#   cnnqB = --sign_head qec --sign_impl operator  (production: H~ = SHS,
#           network/sampler/dtype byte-identical to the Phase-3 cnn arm)
#   cnnqA = --sign_head qec --sign_impl model     (equivalence witness:
#           log psi += i*pi*s via pure_callback; R->C, slower sampling)
# Same recipe as Phase 3 exactly (minSR lr 0.01, 350 steps, seed 0, 8192
# samples, custom sampler); controls = the committed Phase-3 runs.
# Pre-registered: E -> -(V+F) at ~1e-9 rel-err THROUGH the floors
# (2x2: -19.5699527513, 2x3: -27.2189973793); <plaq_term> -> -1.
# Env: LX (2), LY (2), ARMS ("cnnqA cnnqB"; '+' works), SIM_TIME (3.5),
#      SEED (0), WANDB (1).
# Submit (debug fits: 2x2 both arms ~<15 min; 2x3 cnnqB alone):
#   ... submit jobs/nersc_phase4_ab.sh --export=ALL,LX=2,LY=2
#   ... submit jobs/nersc_phase4_ab.sh --export=ALL,LX=2,LY=3,ARMS=cnnqB
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_p4ab
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

LX="${LX:-2}"
LY="${LY:-2}"
ARMS="${ARMS:-cnnqA cnnqB}"; ARMS="${ARMS//+/ }"
SIM_TIME="${SIM_TIME:-3.5}"
SEED="${SEED:-0}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-phase4-h0}"
    echo "=== W&B ON (offline, group ${WANDB_GROUP:-hc-phase4-h0}) ==="
fi

declare -A ARM_FLAGS
ARM_FLAGS[cnnqB]="--sign_head qec --sign_impl operator"
ARM_FLAGS[cnnqA]="--sign_head qec --sign_impl model"
for arm in $ARMS; do
    [ -n "${ARM_FLAGS[$arm]+set}" ] || { echo "!!! unknown arm '$arm' (known: ${!ARM_FLAGS[*]})"; exit 1; }
done

echo "=== phase4-ab: ${LX}x${LY} ds h=0 arms=[$ARMS] sim_time=$SIM_TIME seed=$SEED ==="
for arm in $ARMS; do
    jobid="hc${LX}x${LY}_ds_h0_${arm}"
    if [ -f "$OUTDIR/G-equiv_1_${jobid}.mpack" ]; then
        echo "=== skip $jobid (complete) ==="
        continue
    fi
    echo "=== NQS $jobid ==="
    ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" \
        --hx 0.0 --hy 0.0 --hz 0.0 \
        --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
        --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler \
        --channels_noninv 1,16 --channels_inv 16,8,1 \
        ${ARM_FLAGS[$arm]} $WB_FLAGS ) \
        || echo "!!! $jobid FAILED (exit $?) -- continuing"
done
echo "=== phase4-ab driver done ==="
