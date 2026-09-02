#!/bin/bash
# Sampled decoder-disagreement proxy (scripts/decoder_disagreement.py): rebuild a
# TRAINED honeycomb run from its .json sim_params + .mpack, draw sigma ~ |psi|^2,
# evaluate all five QEC heads on the same samples and report the pairwise
# disagreement matrix D(dec, mwpm) ~ 1 - F_s(dec) (the head-ceiling proxy that
# scales past ED), in-vivo per-decoder timings, syndrome + recovery statistics,
# and (EXACT=1, N <= 24 only) the exact 2^N |psi|^2-weighted matrix + the ED
# ground truth for validation. GPU for the sampling forward passes; shared queue.
#
# Env: RUNS  -- '+'-separated run prefixes relative to the repo (cluster.sh
#               submit flattens ssh args, so '+' is the list separator), e.g.
#               RUNS=results/nqs/G-equiv_1_hc2x3_ds_hx0.4_hz0_cnnqB+results/nqs/G-equiv_1_hc2x2_ds_hx0.4_hz0_cnnqB
#      NSAMP (65536)  EXACT (0; 1 adds --exact -- the script itself skips the 2^N
#      route when N > 24, so it is safe on mixed lists)  SEED (0)
#      DECODERS (mwpm,anchor,greedy,unionfind,tie_sum)  EXTRA (extra script flags)
# Output: results/diagnostics/disagree_<jobid>.json (skip-if-exists). Budget:
# ~5-10 min/run at 2x3 (JIT + 65k samples + tie_sum class enumeration); the
# exact route adds ~5 min at 2x2 (ED 2^19 + decoder tables + enumeration).
# Submit (shared queue):
#   bash scripts/cluster.sh submit jobs/nersc_decoder_disagree.sh \
#       --export=ALL,RUNS=<prefix1>+<prefix2>,EXACT=1
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:45:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_disagree
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/diagnostics
mkdir -p "$OUTDIR"

RUNS="${RUNS:?set RUNS='+'-separated run prefixes (relative to the repo)}"
RUNS="${RUNS//+/ }"
NSAMP="${NSAMP:-65536}"
EXACT="${EXACT:-0}"
SEED="${SEED:-0}"
DECODERS="${DECODERS:-mwpm,anchor,greedy,unionfind,tie_sum}"
EXTRA="${EXTRA:-}"

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

EXACT_FLAG=""
if [ "$EXACT" = "1" ]; then EXACT_FLAG="--exact"; fi

echo "=== decoder-disagree: runs=[$RUNS] nsamp=$NSAMP exact=$EXACT seed=$SEED decoders=$DECODERS ==="
for run in $RUNS; do
    case "$run" in
        /*) prefix="$run" ;;
        *)  prefix="$REPO/$run" ;;
    esac
    prefix="${prefix%.json}"; prefix="${prefix%.mpack}"
    name=$(basename "$prefix")
    jobid="${name#G-equiv_1_}"
    out="$OUTDIR/disagree_${jobid}.json"
    if [ -f "$out" ]; then
        echo "=== skip $jobid (output exists: $out) ==="
        continue
    fi
    if [ ! -f "$prefix.mpack" ] || [ ! -f "$prefix.json" ]; then
        echo "!!! $jobid: missing $prefix.{json,mpack} (run not complete?) -- skipping"
        continue
    fi
    echo "=== disagree $jobid ==="
    ( cd "$REPO" && PYTHONPATH=$REPO python "$REPO/scripts/decoder_disagreement.py" \
        --run "$prefix" --n_samples "$NSAMP" --decoders "$DECODERS" --seed "$SEED" \
        --out "$out" $EXACT_FLAG $EXTRA ) \
        || echo "!!! $jobid FAILED (exit $?) -- continuing"
done
echo "=== decoder-disagree driver done ==="
