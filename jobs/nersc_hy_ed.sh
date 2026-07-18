#!/bin/bash
# Exact-diagonalization ground truth for the L=4 pure-Y-field (hx=hz=0) sweep.
# ED is scipy/sparse (CPU-bound) but runs on the GPU node's CPUs. hy!=0 => complex
# (dtype auto-selected), ~2x the float64 cost, so ~15 min/point with observables at
# 2^24. The 9 hy points split into `SLURM_ARRAY_TASK_COUNT` contiguous chunks (array
# SIZE picks batching); writes one file per hy as it finishes (skip-if-exists).
#
# Submit:  sbatch -J tc_hyED -t 1:30:00 --array=0-2 jobs/nersc_hy_ed.sh   # 3 pts/task
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

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/ed
mkdir -p "$OUTDIR"

HX=0.0
LX=4
HY_VALUES=(0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20)
N=${#HY_VALUES[@]}

T=${SLURM_ARRAY_TASK_COUNT:-1}
i=${SLURM_ARRAY_TASK_ID:-0}
ppt=$(( (N + T - 1) / T ))
start=$(( i * ppt ))
end=$(( start + ppt )); (( end > N )) && end=N

echo "=== ED L=$LX task $i/$T -> points [$start,$end) of $N ==="
for (( k=start; k<end; k++ )); do
    hy=${HY_VALUES[$k]}
    out=$(printf "%s/ed_L%d_hx%.2f_hy%.2f.json" "$OUTDIR" "$LX" "$HX" "$hy")
    if [ -f "$out" ]; then
        echo "=== skip hy=$hy (results exist) ==="
        continue
    fi
    echo "=== ED hy=$hy ==="
    python -m exact.lanczos_ed --Lx "$LX" --hx "$HX" --hy "$hy" --k 4 --out "$out" \
        || echo "!!! ED hy=$hy FAILED (exit $?) — continuing"
done
echo "=== ED L=$LX task $i done ==="
