#!/bin/bash
# QEC sign-head decoder microbenchmark (Phase 4c scaling): time per head evaluation
# and per-VMC-step equivalent vs qubit number N, one curve per decoder in the ladder
# (mwpm, anchor, greedy, unionfind, tie_sum) -- scripts/bench_decoders.py.
# CPU-only work (the head is host numpy/pymatching/numba), but the allocation is
# GPU-only so -G 1 is mandatory; shared queue needs exactly -c 32 per GPU and NO
# --mem. The script pins BLAS/numba to 1 thread itself (reproducible timings).
#
# Env ('+' separates list items: cluster.sh submit flattens ssh args, so ',' inside
# --export values is unsafe; converted to ',' below):
#   SIZES      (1x2+2x2+2x3+3x3+4x4+5x5+6x6+8x8+10x10+12x12)
#   DECODERS   (mwpm+anchor+greedy+unionfind+tie_sum)
#   DENSITIES  (0.005+0.02+0.08)     link-flip densities, neighborhood workload
#   BUDGET     (120)  timed seconds per (size, decoder, workload, density) cell
#   CONSTRUCT_BUDGET (=BUDGET) hard cap on head construction per (size, decoder)
#   N_BASE (64)  REPEATS (3)  CHUNK (4096)  SEED (0)  HARD_CAP (1.5)
#   TAG        (perlmutter)  -> results/diagnostics/bench_decoders_${TAG}.json
#   EXTRA      extra CLI flags passed verbatim (e.g. --no_uniform)
#
# Wall budget: 240 cells at the defaults; fast decoders finish each cell in
# seconds, only budget-saturating cells (tie_sum / un-optimized Python paths at
# 6x6+) cost ~BUDGET each, capped at HARD_CAP*BUDGET+5 s. If a run risks the
# 1:30 wall, split DECODERS across submissions or lower BUDGET.
#
# Submit:
#   bash scripts/cluster.sh submit jobs/nersc_bench_decoders.sh \
#       --export=ALL,TAG=perlmutter,BUDGET=120
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_bench
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
SIZES="${SIZES:-1x2+2x2+2x3+3x3+4x4+5x5+6x6+8x8+10x10+12x12}"; SIZES="${SIZES//+/,}"
DECODERS="${DECODERS:-mwpm+anchor+greedy+unionfind+tie_sum}"; DECODERS="${DECODERS//+/,}"
DENSITIES="${DENSITIES:-0.005+0.02+0.08}"; DENSITIES="${DENSITIES//+/,}"
BUDGET="${BUDGET:-120}"
CONSTRUCT_BUDGET="${CONSTRUCT_BUDGET:-$BUDGET}"
N_BASE="${N_BASE:-64}"; REPEATS="${REPEATS:-3}"; CHUNK="${CHUNK:-4096}"
SEED="${SEED:-0}"; HARD_CAP="${HARD_CAP:-1.5}"
TAG="${TAG:-perlmutter}"
EXTRA="${EXTRA:-}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }
python -c "import numba" 2>/dev/null || echo "!!! numba missing: kernels fall back to Python"

echo "=== host: $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ) ==="
lscpu | grep 'Model name'
echo "nproc=$(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "git: $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" status --porcelain -- model exact | tr '\n' ' ')"

OUT=$REPO/results/diagnostics
mkdir -p "$OUT" "$REPO/logs"
echo "=== bench_decoders tag=$TAG sizes=[$SIZES] decoders=[$DECODERS] densities=[$DENSITIES]" \
     "budget=${BUDGET}s construct=${CONSTRUCT_BUDGET}s n_base=$N_BASE repeats=$REPEATS chunk=$CHUNK ==="
# shellcheck disable=SC2086
PYTHONPATH=$REPO python "$REPO/scripts/bench_decoders.py" \
    --sizes "$SIZES" --decoders "$DECODERS" --densities "$DENSITIES" \
    --budget "$BUDGET" --construct_budget "$CONSTRUCT_BUDGET" --hard_cap "$HARD_CAP" \
    --n_base "$N_BASE" --repeats "$REPEATS" --chunk "$CHUNK" --seed "$SEED" \
    --tag "$TAG" --out "$OUT/bench_decoders_${TAG}.json" $EXTRA
rc=$?
echo "=== bench_decoders done (exit $rc) $(date -u +%FT%TZ) ==="
exit $rc
