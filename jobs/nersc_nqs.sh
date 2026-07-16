#!/bin/bash
# NQS training over the full hz sweep, sequentially, in ONE GPU job (~5 min/run).
# Submit:  sbatch jobs/nersc_nqs.sh
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -J tc_nqs
#SBATCH -o logs/nqs_%j.out

set -euo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

HZ_VALUES=(0.0 0.05 0.10 0.15 0.20 0.25 0.30)
HX=0.0
LX=4

for hz in "${HZ_VALUES[@]}"; do
    jobid=$(printf "L%d_hx%.2f_hz%.2f" "$LX" "$HX" "$hz")
    if [ -f "$OUTDIR/G-equiv_1_${jobid}.json" ]; then
        echo "=== skip hz=$hz (results exist) ==="
        continue
    fi
    echo "=== NQS hz=$hz ==="
    # Run from OUTDIR so main.py writes G-equiv_*.{json,mpack} there, not the repo root.
    ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
        --dt 0.01 --diag_shift 6e-5 --sim_time 3.5 \
        --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler )
done
echo "=== NQS sweep done ==="
