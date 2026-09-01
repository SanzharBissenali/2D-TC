#!/bin/bash
# Decoder-ladder EXACT ceilings vs system size (decoder-scaling study, 2026-09-02).
# Runs scripts/sign_fidelity.py with all 5 decoders on the Phase-4c 9-point field
# grid {0,0.4,0.8}^2 for a list of SMALL sizes in one shared-queue job (the
# per-size debug-node script jobs/nersc_signfid.sh is overkill below 2^27).
# Largest intended size here is 1x5 = 2^26 (hx CSR ~30 GB fits the shared 55 GB).
# Output: results/diagnostics/signfid_hc{Lx}x{Ly}_ds_grid_dl.json (same naming
# as the committed 2x2 grid file; skip-if-exists).
#
# Env: SIZES ('+'-separated LxxLy list, default 1x1+1x2+1x3+1x4+1x5), K (1),
#      TOL (1e-8), DECODERS (mwpm+anchor+greedy+unionfind+tie_sum),
#      POINTS (grid, '+' between pairs, ':' inner).
# Submit: bash scripts/cluster.sh submit jobs/nersc_signfid_ladder.sh -t 2:00:00
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_sfladder
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
SIZES="${SIZES:-1x1+1x2+1x3+1x4+1x5}"; SIZES="${SIZES//+/ }"
K="${K:-1}"; TOL="${TOL:-1e-8}"
DECODERS="${DECODERS:-mwpm+anchor+greedy+unionfind+tie_sum}"; DECODERS="${DECODERS//+/,}"
PTS="${POINTS:-0:0+0:0.4+0:0.8+0.4:0+0.4:0.4+0.4:0.8+0.8:0+0.8:0.4+0.8:0.8}"
PTS="${PTS//+/;}"; PTS="${PTS//:/,}"
TIESUM_DMAX="${TIESUM_DMAX:-10}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

OUT=$REPO/results/diagnostics
mkdir -p "$OUT"
echo "=== signfid ladder sizes=[$SIZES] decoders=[$DECODERS] points=[$PTS] k=$K tol=$TOL ==="
for sz in $SIZES; do
    LX="${sz%%x*}"; LY="${sz##*x}"
    f="$OUT/signfid_hc${LX}x${LY}_ds_grid_dl.json"
    if [ -f "$f" ]; then echo "=== skip ${LX}x${LY} (exists) ==="; continue; fi
    echo "=== ${LX}x${LY} start $(date) ==="
    PYTHONPATH=$REPO python "$REPO/scripts/sign_fidelity.py" \
        --Lx "$LX" --Ly "$LY" --model ds --points "$PTS" \
        --k "$K" --tol "$TOL" \
        --decoders "$DECODERS" --tiesum_dmax "$TIESUM_DMAX" \
        --out "$f" \
        || echo "!!! ${LX}x${LY} FAILED (exit $?) -- continuing"
    echo "=== ${LX}x${LY} done $(date) ==="
done
echo "=== signfid ladder driver done ==="
