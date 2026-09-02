#!/bin/bash
# QEC sign-head decoder microbenchmark (Phase 4c scaling): time per head evaluation
# and per-VMC-step equivalent vs qubit number N, one curve per decoder in the ladder
# (mwpm, anchor, greedy, unionfind, tie_sum) -- scripts/bench_decoders.py.
# CPU-only work (the head is host numpy/pymatching/numba), but the allocation is
# GPU-only so -G 1 is mandatory; shared queue needs exactly -c 32 per GPU and NO
# --mem. The script sets the BLAS/numba thread env vars to --threads itself
# (THREADS=1 is the reproducible single-thread mode; THREADS>1 lets the head's
# numba prange kernels use the cores -- nothing is pinned to 1 then).
#
# Env ('+' separates list items: cluster.sh submit flattens ssh args, so ',' inside
# --export values is unsafe; converted to ',' below):
#   SIZES      (1x2+2x2+2x3+3x3+4x4+5x5+6x6+8x8+10x10+12x12)
#   DECODERS   (mwpm+anchor+greedy+unionfind+tie_sum)
#   DENSITIES  (0.005+0.02+0.08)     link-flip densities, neighborhood workload
#   THREADS    (32)   numba thread count(s), '+'-list => one run per value
#   BLAS_THREADS (1)  OMP/MKL/OpenBLAS pool size (pinned separately; 32+32 starves numba)
#   API        (conn) flat | conn, '+'-list => one run per value
#                     (conn = head.sign_pm1_conn on (n_base,1+N+F,N) neighbourhoods,
#                      us/config over n_base*(2+N+F); flat = head.s01 on flat rows)
#   BUDGET     (120)  timed seconds per (size, decoder, workload, density) cell
#   CONSTRUCT_BUDGET (=BUDGET) hard cap on head construction per (size, decoder)
#   N_BASE (64)  REPEATS (3)  CHUNK (4096)  SEED (0)  HARD_CAP (1.5)
#   TAG        (perlmutter)  -> results/diagnostics/bench_decoders_${TAG}_${api}_t${threads}.json
#                             (one file per (API, THREADS) combination)
#   EXTRA      extra CLI flags passed verbatim (e.g. --no_uniform)
#
# Wall budget: 240 cells at the defaults; fast decoders finish each cell in
# seconds, only budget-saturating cells (tie_sum / un-optimized Python paths at
# 6x6+) cost ~BUDGET each, capped at HARD_CAP*BUDGET+5 s. If a run risks the
# 1:30 wall, split DECODERS across submissions or lower BUDGET.
#
# Submit:
#   bash scripts/cluster.sh submit jobs/nersc_bench_decoders.sh \
#       --export=ALL,TAG=perlmutter,BUDGET=120,THREADS=1+32,API=flat+conn
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
THREADS="${THREADS:-32}"; THREADS="${THREADS//+/,}"
API="${API:-conn}"; API="${API//+/,}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }
python -c "import numba" 2>/dev/null || echo "!!! numba missing: kernels fall back to Python"

echo "=== host: $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ) ==="
lscpu | grep 'Model name'
echo "nproc=$(nproc)  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"
echo "git: $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" status --porcelain -- model exact | tr '\n' ' ')"

OUT=$REPO/results/diagnostics
mkdir -p "$OUT" "$REPO/logs"
rc_all=0
for api in ${API//,/ }; do
  for thr in ${THREADS//,/ }; do
    echo "=== bench_decoders tag=$TAG api=$api threads=$thr sizes=[$SIZES] decoders=[$DECODERS]" \
         "densities=[$DENSITIES] budget=${BUDGET}s construct=${CONSTRUCT_BUDGET}s" \
         "n_base=$N_BASE repeats=$REPEATS chunk=$CHUNK ==="
    # --threads sets OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS inside the script (before numpy)
    # shellcheck disable=SC2086
    PYTHONPATH=$REPO python "$REPO/scripts/bench_decoders.py" \
        --sizes "$SIZES" --decoders "$DECODERS" --densities "$DENSITIES" \
        --budget "$BUDGET" --construct_budget "$CONSTRUCT_BUDGET" --hard_cap "$HARD_CAP" \
        --n_base "$N_BASE" --repeats "$REPEATS" --chunk "$CHUNK" --seed "$SEED" \
        --threads "$thr" --blas_threads "${BLAS_THREADS:-1}" --api "$api" \
        --tag "${TAG}_${api}_t${thr}" --out "$OUT/bench_decoders_${TAG}_${api}_t${thr}.json" $EXTRA
    rc=$?
    echo "=== bench_decoders api=$api threads=$thr done (exit $rc) $(date -u +%FT%TZ) ==="
    [ $rc -ne 0 ] && rc_all=$rc
  done
done
exit $rc_all
