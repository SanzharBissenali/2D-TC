#!/bin/bash
# ED benchmark over the full hz sweep, sequentially, in ONE GPU job.
# ED is scipy/sparse (CPU-bound) but runs on the GPU node's CPUs; each L=4 point
# is ~8 min with observables (~4 min with --no-observables). Writes one file per
# hz as it finishes, so a wall-time kill only loses the in-progress point.
# Submit:  sbatch jobs/nersc_ed.sh
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -J tc_ed
#SBATCH -o logs/ed_%j.out

set -euo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/ed
mkdir -p "$OUTDIR"

HZ_VALUES=(0.0 0.05 0.10 0.15 0.20 0.25 0.30)
HX=0.0
LX=4

for hz in "${HZ_VALUES[@]}"; do
    out=$(printf "%s/ed_L%d_hx%.2f_hz%.2f.json" "$OUTDIR" "$LX" "$HX" "$hz")
    if [ -f "$out" ]; then
        echo "=== skip hz=$hz (results exist) ==="
        continue
    fi
    echo "=== ED hz=$hz ==="
    python -m exact.lanczos_ed --Lx "$LX" --hx "$HX" --hz "$hz" --k 4 --out "$out"
done
echo "=== ED sweep done ==="
