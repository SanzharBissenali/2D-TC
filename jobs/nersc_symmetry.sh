#!/bin/bash
# A_v (vertex/star) symmetry-breaking check over the trained L=4 hz sweep, in ONE GPU job.
# Loads each trained NQS, samples Born + uniform configs, applies every vertex flip, and
# records delta_v = log psi(A_v sigma) - log psi(sigma). Lightweight (L=4: forward eval +
# short MCMC), so a short walltime suffices. Env-parameterized (LX / SUFFIX / N_SAMPLES).
# Submit:  sbatch jobs/nersc_symmetry.sh
#          sbatch --export=ALL,LX=4,SUFFIX=tf jobs/nersc_symmetry.sh   # a transformer arm
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_sym
#SBATCH -o logs/sym_%j.out

set -euo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
LX=${LX:-4}
SUFFIX=${SUFFIX:-}
N_SAMPLES=${N_SAMPLES:-8192}

mkdir -p "$REPO/results/symmetry"
echo "=== A_v symmetry check: L=$LX suffix='${SUFFIX}' n_samples=$N_SAMPLES ==="
PYTHONPATH="$REPO" python "$REPO/scripts/symmetry_check.py" \
    --Lx "$LX" --suffix "$SUFFIX" --n_samples "$N_SAMPLES" \
    --nqs-dir "$REPO/results/nqs" --out-dir "$REPO/results/symmetry"
echo "=== symmetry check done ==="
