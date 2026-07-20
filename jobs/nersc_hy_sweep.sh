#!/bin/bash
# Pure-Y-field (hx=hz=0) sweep to map the first-order transition (theory: hy=1.0
# in the thermodynamic limit; shifts at finite L). One L per job; the 9 hy points
# are split into `SLURM_ARRAY_TASK_COUNT` contiguous chunks, so the array SIZE
# alone picks the batching regime (as in nersc_sweep.sh):
#   - one job, all points   (L=4):  no --array          -> 1 task,  9 points
#   - batched points/job    (L=6):  --array=0-2         -> chunks of 3
#   - one point per job     (L=8):  --array=0-8         -> 9 tasks, 1 point each
#
# hy!=0 => dtype=complex (utils/config.py); the CNN complex path is the paper's
# trusted one. Arm dispatch (cnn/tf/tfg) is kept for a later transformer follow-up,
# but the transformer's complex path has known blockers => default ARMS="cnn".
#
# Submit (per-L spec; walltime/name/array/env at submit time, not in #SBATCH):
#   sbatch -J tc_hyL4 -t 1:30:00              --export=ALL,LX=4 jobs/nersc_hy_sweep.sh
#   sbatch -J tc_hyL6 -t 1:30:00 --array=0-2  --export=ALL,LX=6 jobs/nersc_hy_sweep.sh
#   sbatch -J tc_hyL8 -t 1:30:00 --array=0-8  --export=ALL,LX=8 jobs/nersc_hy_sweep.sh
#
# Resumable: a point is skipped only if its run is COMPLETE (full training + L>=6
# end observables) per scripts/is_complete.py, so a walltime kill mid-point re-runs it.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -o logs/%x_%A_%a.out

set -uo pipefail
module load conda
conda activate 2dtc

: "${LX:?set LX via --export=ALL,LX=<n>}"
HX=0.0
HZ=0.0
SEED="${SEED:-0}"
SIM_TIME="${SIM_TIME:-2.0}"
ARMS="${ARMS:-cnn}"

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

# --- optional W&B logging (offline; `wandb sync` from a login node afterward) ---
# Enable per-submit:  --export=ALL,LX=..,WANDB=1[,WANDB_GROUP=...,WANDB_PROJECT=...]
# For the sign-full (hy) runs this captures energy / Vscore / grad-norms / QGT cond AND the
# phase circular-variance per step (optimizer.py folds wb_extra into each wandb.log row).
WANDB="${WANDB:-0}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO"   # wandb appends its own /wandb => runs at $REPO/wandb/offline-run-*
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hy_L${LX}}"
    echo "=== W&B ON (mode=$WANDB_MODE dir=$WANDB_DIR group=${WANDB_GROUP:-hy_L${LX}}) ==="
fi

HY_VALUES=(0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20)
N=${#HY_VALUES[@]}

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[tfg]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 6 --tf_heads 2 --tf_ffn_mult 4 --tf_activation gelu"
# v3 sign-full: v1 (CNN Block1 + Wilson + transformer Block3) with the COMPLEX shallow readout
# (Viteritti et al. 2311.16889). --tf_complex_output keeps the encoder real (float64) and injects
# amplitude+phase only at the readout, arming qgt_mode='complex'. Trailing --diag_shift 1e-3 (paper's
# value; last-wins over the base 6e-5) + cosine lr->0.25x settle the non-holomorphic SR.
ARM_FLAGS[v1cx8]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --tf_complex_output --diag_shift 1e-3 --lr_schedule cosine --lr_final_frac 0.25"
ARM_FLAGS[v1cx16]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 16 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --tf_complex_output --diag_shift 1e-3 --lr_schedule cosine --lr_final_frac 0.25"

# Contiguous chunk of the points for this array task (defaults => all points when
# run without --array, e.g. the L=4 single-task job or a manual test).
T=${SLURM_ARRAY_TASK_COUNT:-1}
i=${SLURM_ARRAY_TASK_ID:-0}
ppt=$(( (N + T - 1) / T ))       # points per task (ceil)
start=$(( i * ppt ))
end=$(( start + ppt )); (( end > N )) && end=N

echo "=== L=$LX task $i/$T -> points [$start,$end) of $N | ARMS=[$ARMS] sim_time=$SIM_TIME seed=$SEED ==="
for (( k=start; k<end; k++ )); do
    hy=${HY_VALUES[$k]}
    for arm in ${ARMS//[:,]/ }; do   # ':' / ',' / space separated (':' survives sbatch --export)
        jobid=$(printf "L%d_hx%.2f_hy%.2f_%s" "$LX" "$HX" "$hy" "$arm")
        base="$OUTDIR/G-equiv_1_${jobid}"
        if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
            echo "=== skip $jobid (complete) ==="
            continue
        fi
        echo "=== NQS $jobid (sim_time=$SIM_TIME) ==="
        # Run from OUTDIR so main.py writes G-equiv_*.{json,mpack} there, not the repo root.
        ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
            --outindex 1 --jobid "$jobid" \
            --Lx "$LX" --hx "$HX" --hy "$hy" --hz "$HZ" \
            --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" --seed "$SEED" \
            --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
            --n_samples_fin 8192 --use_custom_sampler \
            ${ARM_FLAGS[$arm]} $WB_FLAGS ) || echo "!!! $jobid FAILED (exit $?) — continuing"
    done
done
echo "=== L=$LX task $i done ==="
