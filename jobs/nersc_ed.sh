#!/bin/bash
# ED benchmark sweep on a NERSC Perlmutter GPU node (m5340_g is GPU-only).
# ED is scipy/sparse (CPU-bound) but runs on a GPU node's CPUs; 24-qubit L=4 is
# memory-heavy (~tens of GB) and fits in the node's 256 GB.
# Submit as a job array over the hz sweep:  sbatch --array=0-6 jobs/nersc_ed.sh
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J tc_ed
#SBATCH -o logs/ed_%A_%a.out

set -euo pipefail

HZ_VALUES=(0.0 0.05 0.10 0.15 0.20 0.25 0.30)
HX=0.0
LX=4
hz=${HZ_VALUES[$SLURM_ARRAY_TASK_ID]}

# TODO(nersc): confirm env name (create once via requirements.txt).
module load conda
conda activate 2dtc

out=$(printf "results/ed/ed_L%d_hx%.2f_hz%.2f.json" "$LX" "$HX" "$hz")
python -m exact.lanczos_ed --Lx "$LX" --hx "$HX" --hz "$hz" --k 4 --out "$out"
