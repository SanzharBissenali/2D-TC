#!/bin/bash
# Quick fTC ED amplitude-structure check on the gpu debug queue (exclusive node,
# fast start, 30 min hard cap). Runs `lanczos_ed --ftc` per HX point and records
# the amplitude-DISTRIBUTION diagnostics (n_nonzero / amp_rel_spread /
# amp_values_scaled / signed histogram) added for the "is the GS 0-or-identical
# on every Z string?" question. `--no-observables` skips only the 48-op
# magnetization pass; the eigenvector (and hence the diagnostics) still computes
# for real dtype. Outputs use an `_amp` suffix so the observable-carrying
# Phase-4 ED files are neither clobbered nor pre-empted.
# After the ED points, evaluates the TRAINED h=0 NQS wavefunctions (cnn +
# plaincnn .mpack from the fixed-point comparison) over the full 2^24 basis via
# scripts/nqs_amp_hist.py -> *_amp.json next to each run JSON (same histogram
# schema, split by exact-GS support), so the same plot answers "does the NQS
# show the spike?".
#   Env: LX (4), HX_LIST ("0.00"), NQS_JSONS (the two h=0 run JSONs; "" skips)
#
# Submit:
#   bash scripts/cluster.sh submit jobs/nersc_amp_ed.sh --export=ALL,HX_LIST="0.00 0.30"
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -J tc_amp_ed
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
EDDIR=$REPO/results/ed
mkdir -p "$EDDIR"

LX="${LX:-4}"
HX_LIST="${HX_LIST:-0.00}"
NQS_JSONS="${NQS_JSONS:-results/nqs/G-equiv_1_ftc_L4_hx0.00_hz0.00_cnn.json results/nqs/G-equiv_1_ftc_L4_hx0.00_hz0.00_plaincnn.json}"

for hx in $HX_LIST; do
    out=$(printf "%s/ed_ftc_L%d_hx%.2f_hz0.00_amp.json" "$EDDIR" "$LX" "$hx")
    echo "=== ED (ftc, amp diagnostics) L$LX hx=$hx -> $out ==="
    ( cd "$REPO" && python -m exact.lanczos_ed --Lx "$LX" --hx "$hx" --hz 0.0 --k 4 \
        --ftc --no-observables --out "$out" ) || echo "!!! hx=$hx FAILED (exit $?) — continuing"
done

if [ -n "$NQS_JSONS" ]; then
    echo "=== NQS full-basis amplitude histograms ==="
    ( cd "$REPO" && python scripts/nqs_amp_hist.py --json $NQS_JSONS ) \
        || echo "!!! nqs_amp_hist FAILED (exit $?)"
fi
echo "=== done ==="
