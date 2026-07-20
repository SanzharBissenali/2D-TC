#!/bin/bash
# v2 full-transformer vs CNN/v1 comparison. Env-parameterized so ONE script covers the
# per-arm / per-L jobs (kept small to schedule fast on the shared queue).
#   Arms:  cnn  (baseline for the wall-clock ratio)
#          v2m  (full_transformer, d=8: param-matched@L4)
#          v2h  (full_transformer, d=4: ~half@L4)
#          v1tf (v1 Block-3 transformer, optional baseline)
#   Env:   LX (4), HZ_LIST ("0.15 0.30"), SIM_TIME (3.5), ARMS ("cnn v2m v2h"), SEED (0).
#          Walltime / name / env are set at SUBMIT time via sbatch overrides.
#
# Submit (gated -- consult first):
#   bash scripts/cluster.sh submit jobs/nersc_v2.sh -J tc_v2_L4 -t 1:00:00 --export=ALL
#   bash scripts/cluster.sh submit jobs/nersc_v2.sh -J tc_v2_L6 -t 2:00:00 --export=ALL,LX=6,HZ_LIST=0.15,SIM_TIME=2.0
#
# Gate 2 (L=4 vs ED at hz in {0.15, 0.33}): ED exists to hz=0.30 -> generate ED@0.33 first
#   or run the runnable {0.15, 0.30}. Gate 3 (L=6): wall-clock-to-eps ratio v2/v1 (needs cnn arm).
# Resumable: skip-if-COMPLETE via scripts/is_complete.py (.mpack => training done;
#   L>=6 also needs end observables), so a walltime kill re-runs only unfinished points.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_v2
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

HX=0.0
SEED="${SEED:-0}"
LX="${LX:-4}"
SIM_TIME="${SIM_TIME:-2.0}"                       # 200 TDVP steps (dt=0.01)
HZ_LIST="${HZ_LIST:-0.10 0.15 0.20 0.25 0.30}"    # 5 L=4 ED points (match analysis/01, 05)
ARMS="${ARMS:-cnn v2}"                            # cnn skipped if already complete (skip-if-complete)

# --- optional W&B logging (offline; `wandb sync` from a login node afterward) ---
# Enable per-submit:  --export=ALL,WANDB=1[,WANDB_GROUP=...,WANDB_PROJECT=...]
WANDB="${WANDB:-0}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-L${LX}}"
    echo "=== W&B ON (mode=$WANDB_MODE dir=$WANDB_DIR group=${WANDB_GROUP:-L${LX}}) ==="
fi

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[v1tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
# Primary v2 arm: n1=2, n2=2, d=8, h=2, FFN 2d, ReLU (~2688 params @L4).
ARM_FLAGS[v2]="--symmetric_block full_transformer --tf1_layers 2 --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[v2h]="--symmetric_block full_transformer --tf1_layers 1 --tf_layers 2 --tf_dmodel 4 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
# diag_shift retune arms: stronger SR regularization to damp the late-training QGT blow-up
# (cond ~1e9 at 6e-5). Trailing --diag_shift overrides the base 6e-5 (argparse: last wins).
ARM_FLAGS[v2ds3e4]="${ARM_FLAGS[v2]} --diag_shift 3e-4"
ARM_FLAGS[v2ds1e3]="${ARM_FLAGS[v2]} --diag_shift 1e-3"
# lr-schedule retune: 1e-3 diag_shift + cosine dt-decay to 0.1x (settle the late-training wander).
ARM_FLAGS[v2cos]="${ARM_FLAGS[v2]} --diag_shift 1e-3 --lr_schedule cosine --lr_final_frac 0.1"
# Campaign recipe: 3e-4 diag_shift (deepest convergence) + gentle cosine lr to 0.25x (settles the
# tail without starving fine-convergence). d=8 and d=16 residual streams.
ARM_FLAGS[v2c8]="--symmetric_block full_transformer --tf1_layers 2 --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --diag_shift 3e-4 --lr_schedule cosine --lr_final_frac 0.25"
ARM_FLAGS[v2c16]="--symmetric_block full_transformer --tf1_layers 2 --tf_layers 2 --tf_dmodel 16 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --diag_shift 3e-4 --lr_schedule cosine --lr_final_frac 0.25"
# v1 (Block-3-only transformer) campaign arms: IDENTICAL Block-3 config + IDENTICAL recipe to
# v2c8/v2c16, but Block 1 stays the CNN. So v1cN vs v2cN isolates v2's transformer Block-1 +
# Wilson-fusion as the ONLY difference => localizes any accuracy gap. Same 3e-4 diag_shift +
# cosine lr->0.25x, 200 steps, seed 0.
ARM_FLAGS[v1c8]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --diag_shift 3e-4 --lr_schedule cosine --lr_final_frac 0.25"
ARM_FLAGS[v1c16]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 16 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu --diag_shift 3e-4 --lr_schedule cosine --lr_final_frac 0.25"
# Timing arms: 10-step s/step benchmark at large L (residual-stream scaling knob d).
# Distinct names => never collide with accuracy runs. t8 = d=8, t16 = d=16.
ARM_FLAGS[t8]="--symmetric_block full_transformer --tf1_layers 2 --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[t16]="--symmetric_block full_transformer --tf1_layers 2 --tf_layers 2 --tf_dmodel 16 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"

echo "=== driver v2: LX=$LX SIM_TIME=$SIM_TIME ARMS=[$ARMS] HZ=[$HZ_LIST] SEED=$SEED ==="
for hz in ${HZ_LIST//[:,]/ }; do   # ':' , ',' or space separated. ':' survives sbatch --export
    for arm in ${ARMS//[:,]/ }; do  # (no spaces => no ssh re-tokenize; no commas => not eaten by --export)
        jobid=$(printf "L%d_hx%.2f_hz%.2f_%s" "$LX" "$HX" "$hz" "$arm")
        base="$OUTDIR/G-equiv_1_${jobid}"
        if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
            echo "=== skip $jobid (complete) ==="
            continue
        fi
        echo "=== NQS $jobid (sim_time=$SIM_TIME) ==="
        ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
            --outindex 1 --jobid "$jobid" \
            --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
            --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" --seed "$SEED" \
            --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
            --n_samples_fin 8192 --use_custom_sampler \
            ${ARM_FLAGS[$arm]} $WB_FLAGS ) || echo "!!! $jobid FAILED (exit $?) — continuing"
    done
done
echo "=== driver v2 done ==="
