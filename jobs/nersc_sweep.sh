#!/bin/bash
# Phase-transition hz sweep for one L, as a SLURM job array.
#
# The 15 hz points are split into `SLURM_ARRAY_TASK_COUNT` contiguous chunks, so
# the array SIZE alone picks the batching regime — ONE script covers both:
#   - one job per hz point (L=10,12):  --array=0-14   -> 15 tasks, 1 point each
#   - batched points per job (L=6,8):  --array=0-1 / 0-2 -> chunks of the 15
#
# Submit (per-L spec; walltime/name/array set here, not in #SBATCH):
#   sbatch -J tc_L6  -t 1:30:00 --array=0-1  --export=ALL,LX=6  jobs/nersc_sweep.sh
#   sbatch -J tc_L8  -t 3:00:00 --array=0-2  --export=ALL,LX=8  jobs/nersc_sweep.sh
#   sbatch -J tc_L10 -t 2:00:00 --array=0-14 --export=ALL,LX=10 jobs/nersc_sweep.sh
#   sbatch -J tc_L12 -t 5:00:00 --array=0-14 --export=ALL,LX=12 jobs/nersc_sweep.sh
#
# Resumable: a point is skipped only if its run is COMPLETE (full training + end
# observables) per scripts/is_complete.py, so a walltime kill mid-point re-runs it.
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
STEPS=${STEPS:-200}
HX=0.0
SIM_TIME=$(awk "BEGIN{print $STEPS*0.01}")

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

HZ_VALUES=(0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.325 0.35 0.375 0.40 0.425 0.45 0.475 0.50)
N=${#HZ_VALUES[@]}

# Contiguous chunk of the points for this array task (defaults => all points when
# run without --array, e.g. a manual test).
T=${SLURM_ARRAY_TASK_COUNT:-1}
i=${SLURM_ARRAY_TASK_ID:-0}
ppt=$(( (N + T - 1) / T ))       # points per task (ceil)
start=$(( i * ppt ))
end=$(( start + ppt )); (( end > N )) && end=N

echo "=== L=$LX task $i/$T -> points [$start,$end) of $N | steps=$STEPS (sim_time=$SIM_TIME) ==="
for (( k=start; k<end; k++ )); do
    hz=${HZ_VALUES[$k]}
    jobid=$(printf "L%d_hx%.2f_hz%.3f" "$LX" "$HX" "$hz")
    base="$OUTDIR/G-equiv_1_${jobid}"
    if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
        echo "=== skip hz=$hz (complete) ==="
        continue
    fi
    echo "=== NQS L=$LX hz=$hz ==="
    # Run from OUTDIR so main.py writes G-equiv_*.{json,mpack} there, not the repo root.
    ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
        --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" \
        --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler )
done
echo "=== L=$LX task $i done ==="
