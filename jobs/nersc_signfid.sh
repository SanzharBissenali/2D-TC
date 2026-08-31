#!/bin/bash
# Sign-fidelity diagnostic (Phase-4 head ceiling) at 2x3 = 2^27 -- the one
# size the laptop can't do: the ED campaign never stored ground VECTORS
# (~1 GB each), so each point recomputes ED here. hx builds need ~60 GB CSR
# => exclusive-node memory; debug QOS (30 min) fits with --tol 1e-8, k=1.
# One point per submission.
#
# Env: LX (2), LY (3), MODEL (ds), K (1), TOL (1e-8), POINTS -- hx:hz pairs,
# ':' for the inner comma and '+' between pairs (cluster.sh submit flattens
# ssh args and --export splits on commas), e.g. POINTS=0.1:0
#
# Submit:  bash scripts/cluster.sh submit jobs/nersc_signfid.sh \
#              --export=ALL,POINTS=0.1:0
# Needs pymatching in the 2dtc env (pip install on a login node, like wandb).
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH -J hc_signfid
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
LX="${LX:-2}"; LY="${LY:-3}"; MODEL="${MODEL:-ds}"
K="${K:-1}"; TOL="${TOL:-1e-8}"
PTS="${POINTS:-0:0}"; PTS="${PTS//+/;}"; PTS="${PTS//:/,}"
# Phase-4c decoder ladder: DECODERS='+'-separated list (default mwpm =
# legacy single-decoder run); output filename gains a _dl tag when active.
DECODERS="${DECODERS:-mwpm}"; DECODERS="${DECODERS//+/,}"
TIESUM_DMAX="${TIESUM_DMAX:-10}"
# Phase 4d: HY != 0 => complex ED + phase-optimized ceilings (~2x memory/time:
# merged single-flip channel keeps the 2^27 hx+hy CSR at ~91 GB -- override
# the debug header with -q regular and a bigger -t for 2x3 hy points).
HY="${HY:-0}"
awk "BEGIN{exit !($HY == 0)}" && HY=0   # canonicalize zero spellings

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

OUT=$REPO/results/diagnostics
mkdir -p "$OUT"
TAG=$(echo "$PTS" | tr ';,' '__')
[ "$DECODERS" != "mwpm" ] && TAG="${TAG}_dl"
[ "$HY" != "0" ] && TAG="${TAG}_hy${HY}"
echo "=== signfid ${LX}x${LY} $MODEL points=[$PTS] hy=$HY k=$K tol=$TOL decoders=[$DECODERS] ==="
PYTHONPATH=$REPO python "$REPO/scripts/sign_fidelity.py" \
    --Lx "$LX" --Ly "$LY" --model "$MODEL" --points "$PTS" --hy "$HY" \
    --k "$K" --tol "$TOL" \
    --decoders "$DECODERS" --tiesum_dmax "$TIESUM_DMAX" \
    --out "$OUT/signfid_hc${LX}x${LY}_${MODEL}_${TAG}.json"
echo "=== signfid driver done ==="
