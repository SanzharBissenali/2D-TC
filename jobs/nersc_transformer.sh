#!/bin/bash
# Unattended CNN-vs-transformer comparison for Block 3 (the exactly-symmetric block).
# Batch job (runs on the scheduler, independent of any laptop). One changed variable:
#   cnn : original global-kernel invariant CNN          (channels_inv 16,8,1)
#   tf  : factored-attention transformer, primary       (n_l=2 d=8 h=2 FFN 2d ReLU  ~1236 params @L4)
#   tfg : factored-attention transformer, GELU/4d bonus (n_l=2 d=6 h=2 FFN 4d GELU  ~1108 params @L4)
#
#   L=4: rel-err vs ED at MULTIPLE hz (ED in results/ed/ covers hz 0.10..0.30). 350 steps.
#   L=6: a single full run (hz=0.15, no ED -> compare tf vs cnn converged energy/timing). 200 steps.
#
# Resumable: skip-if-COMPLETE via scripts/is_complete.py (.mpack => training done;
# L>=6 also needs end observables), so a walltime kill re-runs only unfinished points.
#
# Submit:  bash scripts/cluster.sh submit jobs/nersc_transformer.sh
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_tf
#SBATCH -o logs/tf_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

HX=0.0
SEED=0

declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[tfg]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 6 --tf_heads 2 --tf_ffn_mult 4 --tf_activation gelu"

run_point () {
    local LX="$1" hz="$2" sim_time="$3" arm="$4"
    local jobid base
    jobid=$(printf "L%d_hx%.2f_hz%.2f_%s" "$LX" "$HX" "$hz" "$arm")
    base="$OUTDIR/G-equiv_1_${jobid}"
    if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
        echo "=== skip $jobid (complete) ==="
        return
    fi
    echo "=== NQS L=$LX hz=$hz arm=$arm (sim_time=$sim_time) ==="
    ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
        --dt 0.01 --diag_shift 6e-5 --sim_time "$sim_time" --seed "$SEED" \
        --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler \
        ${ARM_FLAGS[$arm]} ) || echo "!!! $jobid FAILED (exit $?) — continuing"
}

# L=4: multi-point rel-err validation (all hz have ED in results/ed/). 350 steps.
for hz in 0.10 0.15 0.20 0.25 0.30; do
    for arm in cnn tf tfg; do
        run_point 4 "$hz" 3.5 "$arm"
    done
done

# L=6: single full run per arm at hz=0.15 (wall-clock + energy comparison). 200 steps.
for arm in cnn tf tfg; do
    run_point 6 0.15 2.0 "$arm"
done

echo "=== transformer comparison done ==="
