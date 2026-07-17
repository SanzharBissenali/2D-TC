#!/bin/bash
# Paired CNN-vs-transformer comparison for Block 3 (the exactly-symmetric block).
# Runs the same physics (L in {4,6}, hz in {0.15,0.30}) with three arms, all sharing
# an identical seed / sampler / optimizer so the ONLY changed variable is Block 3:
#   cnn : original global-kernel invariant CNN          (channels_inv 16,8,1)
#   tf  : factored-attention transformer, primary       (n_l=2 d=8 h=2 FFN 2d ReLU  ~1236 params @L4)
#   tfg : factored-attention transformer, GELU/4d bonus (n_l=2 d=6 h=2 FFN 4d GELU  ~1108 params @L4)
#
# Kill criteria (see analysis): L=4 must reach rel-err <= 1e-5 vs ED at BOTH hz
# (results/ed/ed_L4_hx0.00_hz{0.15,0.30}.json). L=6 decision = wall-clock to
# rel-err 1e-5, transformer/CNN ratio > 3 => abandon.
#
# Submit:  sbatch jobs/nersc_transformer.sh
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_tf
#SBATCH -o logs/tf_%j.out

set -euo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

HX=0.0
SEED=0
HZ_VALUES=(0.15 0.30)

# Per-arm extra flags. cnn uses --channels_inv (the real Block 3); the transformer
# arms still pass --channels_inv (required flag, unused by their code path).
declare -A ARM_FLAGS
ARM_FLAGS[cnn]="--symmetric_block cnn"
ARM_FLAGS[tf]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 8 --tf_heads 2 --tf_ffn_mult 2 --tf_activation relu"
ARM_FLAGS[tfg]="--symmetric_block transformer --tf_layers 2 --tf_dmodel 6 --tf_heads 2 --tf_ffn_mult 4 --tf_activation gelu"

for LX in 4 6; do
    # L=4 uses the validated 350-step budget; L=6 the 200-step sweep budget.
    if [ "$LX" -eq 4 ]; then SIM_TIME=3.5; else SIM_TIME=2.0; fi
    for hz in "${HZ_VALUES[@]}"; do
        for arm in cnn tf tfg; do
            jobid=$(printf "L%d_hx%.2f_hz%.2f_%s" "$LX" "$HX" "$hz" "$arm")
            if [ -f "$OUTDIR/G-equiv_1_${jobid}.json" ]; then
                echo "=== skip $jobid (results exist) ==="
                continue
            fi
            echo "=== NQS L=$LX hz=$hz arm=$arm ==="
            # Run from OUTDIR so main.py writes G-equiv_*.{json,mpack} there.
            ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
                --outindex 1 --jobid "$jobid" \
                --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
                --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" --seed "$SEED" \
                --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
                --n_samples_fin 8192 --use_custom_sampler \
                ${ARM_FLAGS[$arm]} )
        done
    done
done
echo "=== transformer comparison done ==="
