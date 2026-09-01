#!/bin/bash
# Exact trained-NQS fidelity vs ED (scripts/nqs_fidelity.py). Needs the ED
# memory profile of nersc_signfid.sh (2^27 hx/hy CSR ~60-91 GB => regular
# node) PLUS the GPU for the network forward passes. Self-validating per run
# (energy cross-check). One invocation covers POINTS x ARMS at one (size, HY).
#
# Env: LX, LY, HY (0), POINTS (hx:hz, '+' between pairs), ARMS ('+'-separated
# arm tokens, e.g. cnnqC+cnnqC_anchor+cnnqC_tie_sum), TAG (output filename tag).
# Submit: bash scripts/cluster.sh submit jobs/nersc_fidelity.sh -q regular \
#   -t 3:00:00 --export=ALL,LX=2,LY=3,HY=0.4,POINTS=0:0+0:0.4,ARMS=cnnqC,TAG=c1
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH -J hc_fid
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
LX="${LX:-2}"; LY="${LY:-3}"; HY="${HY:-0}"
PTS="${POINTS:-0:0}"; PTS="${PTS//+/;}"; PTS="${PTS//:/,}"
ARMS="${ARMS:-cnnqB}"; ARMS="${ARMS//+/,}"
TAG="${TAG:-x}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc"; exit 1; }

OUT=$REPO/results/diagnostics
mkdir -p "$OUT"
echo "=== fidelity ${LX}x${LY} hy=$HY points=[$PTS] arms=[$ARMS] ==="
PYTHONPATH=$REPO python "$REPO/scripts/nqs_fidelity.py" \
    --Lx "$LX" --Ly "$LY" --hy "$HY" --points "$PTS" --arms "$ARMS" \
    --out "$OUT/fidelity_hc${LX}x${LY}_hy${HY}_${TAG}.json"
echo "=== fidelity driver done ==="
