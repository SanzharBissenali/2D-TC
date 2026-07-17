#!/bin/bash
# L=4 validation driver, meant to run INSIDE an interactive allocation, e.g.:
#   salloc -A m5340_g -C gpu -N1 -G1 -t 60 -q interactive bash jobs/interactive_L4.sh
#
# Runs the CNN-vs-transformer arms at L=4 for one or more hz points (env-overridable),
# writing G-equiv_* to results/nqs and printing jax devices + live per-step timing
# (main.py tqdm bar + JSON t_sample/t_grad/t_sr). Arms default to tf,cnn,tfg with the
# transformer arm listed FIRST so a shape bug (or the s/step number) shows up immediately.
set -uo pipefail
module load conda
conda activate 2dtc

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/results/nqs"
mkdir -p "$OUTDIR"

HX=0.0
SEED="${SEED:-0}"
SIM_TIME="${SIM_TIME:-3.5}"
# shellcheck disable=SC2206
HZ_VALUES=(${HZ:-0.15})
# shellcheck disable=SC2206
ARMS=(${ARMS:-tf cnn tfg})

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[tfg]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 6 --tf_heads 2 --tf_ffn_mult 4 --tf_activation gelu"

echo "=== jax devices (expect a CudaDevice if on the GPU node) ==="
python -c "import jax; print(jax.devices())"

for hz in "${HZ_VALUES[@]}"; do
    for arm in "${ARMS[@]}"; do
        jobid=$(printf "L4_hx%.2f_hz%.2f_%s" "$HX" "$hz" "$arm")
        echo "=== RUN $jobid  (sim_time=$SIM_TIME, seed=$SEED) ==="
        ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
            --outindex 1 --jobid "$jobid" \
            --Lx 4 --hx "$HX" --hy 0.0 --hz "$hz" \
            --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" --seed "$SEED" \
            --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
            --n_samples_fin 8192 --use_custom_sampler \
            ${ARM_FLAGS[$arm]} ) || echo "!!! $jobid FAILED (exit $?) — continuing to next arm"
    done
done
echo "=== interactive_L4 done ==="
