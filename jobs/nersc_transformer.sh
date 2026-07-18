#!/bin/bash
# CNN-vs-transformer Block-3 comparison, env-parameterized so ONE script covers the
# per-arm / per-L jobs (split small so they schedule fast on the shared queue).
#   Arms:  cnn / tf (n_l2 d8 h2 FFN2d ReLU) / tfg (n_l2 d6 h2 FFN4d GELU)
#   Env:   LX (4), HZ_LIST ("0.10 0.15 0.20 0.25 0.30"), SIM_TIME (3.5),
#          ARMS ("cnn tf tfg"), SEED (0).  Walltime/name/env set at SUBMIT time.
#
# Submit the four jobs (walltime/name/env via sbatch overrides):
#   bash scripts/cluster.sh submit jobs/nersc_transformer.sh -J tc_L4cnn -t 1:00:00 --export=ALL,ARMS=cnn
#   bash scripts/cluster.sh submit jobs/nersc_transformer.sh -J tc_L4tf  -t 1:00:00 --export=ALL,ARMS=tf
#   bash scripts/cluster.sh submit jobs/nersc_transformer.sh -J tc_L4tfg -t 1:00:00 --export=ALL,ARMS=tfg
#   bash scripts/cluster.sh submit jobs/nersc_transformer.sh -J tc_L6 -t 1:30:00 --export=ALL,LX=6,HZ_LIST=0.15,SIM_TIME=2.0
#
# Resumable: skip-if-COMPLETE via scripts/is_complete.py (.mpack => training done;
# L>=6 also needs end observables), so a walltime kill re-runs only unfinished points.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_tf
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
SIM_TIME="${SIM_TIME:-3.5}"
HZ_LIST="${HZ_LIST:-0.10 0.15 0.20 0.25 0.30}"
ARMS="${ARMS:-cnn tf tfg}"

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[tfg]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 6 --tf_heads 2 --tf_ffn_mult 4 --tf_activation gelu"
# Convergence-tweak arms (used only if the primary tf fails to hit ED at L=4):
ARM_FLAGS[tfbig]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 16 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"   # 2x residual stream (d=16)
ARM_FLAGS[tfwide]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 4 --tf_activation relu"   # 2x FFN hidden (4d)

echo "=== driver: LX=$LX SIM_TIME=$SIM_TIME ARMS=[$ARMS] HZ=[$HZ_LIST] SEED=$SEED ==="
for hz in $HZ_LIST; do
    for arm in $ARMS; do
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
            ${ARM_FLAGS[$arm]} ) || echo "!!! $jobid FAILED (exit $?) — continuing"
    done
done
echo "=== driver done ==="
