#!/bin/bash
# Honeycomb (Levin-Gu, arXiv:1202.3120 Sec. IV) ED companions: toric code +
# doubled semion on the smooth-OBC brick-wall patch. ED only -- NQS is Phase 3.
#
# Default: 2x3 patch (27 qubits, the largest ED-anchored NQS size), h=0 + hz cut.
# Hard gates at h=0 (predicted analytically + by exact/loops.py, must match):
#   E0 = -(V+F) = -28 exact, gap = 2, unique GS, support = 2^6 = 64,
#   tc neg_amp_fraction_support = 0, ds = 48/64 = 0.75.
# hz-only points: TC and DS spectra must COINCIDE (hz preserves vertex sectors
# on a simply-connected patch); hx points are the model-discriminating ones.
#
# MEMORY: h=0 / hz-only points have ~7 connected flip patterns => sparse H
# ~12 GB at 2^27, fits the shared node. hx != 0 adds 27 single-flip patterns
# => ~60-70 GB sparse + build intermediates => needs an exclusive node:
#   bash scripts/cluster.sh submit jobs/nersc_honeycomb_ed.sh -q regular -t 3:00:00 \
#       --export=ALL,POINTS="ds:0.10:0.00 tc:0.10:0.00 ds:0.20:0.00 tc:0.20:0.00"
#
# Env: LX (2), LY (3), K (6), POINTS ("model:hx:hz" list; default = h=0 + hz cut).
# Resumable: skip-if-exists (ED writes its JSON once, at the end -- safe test).
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_ed
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
EDDIR=$REPO/results/ed
mkdir -p "$EDDIR"

LX="${LX:-2}"
LY="${LY:-3}"
K="${K:-6}"
POINTS="${POINTS:-tc:0.00:0.00 ds:0.00:0.00 tc:0.00:0.10 ds:0.00:0.10 tc:0.00:0.20 ds:0.00:0.20}"
# '+' works as a separator too: --export values with SPACES don't survive the
# cluster.sh ssh flattening (remote() rebuilds the command as one string), so
# pass e.g. POINTS=ds:0.10:0.00+tc:0.10:0.00 -- no quoting needed anywhere.
POINTS="${POINTS//+/ }"

echo "=== honeycomb ED: ${LX}x${LY}, k=$K, points=[$POINTS] ==="
for pt in $POINTS; do
    IFS=: read -r model hx hz <<< "$pt"
    out=$(printf "%s/ed_hc%dx%d_%s_hx%.2f_hz%.2f.json" "$EDDIR" "$LX" "$LY" "$model" "$hx" "$hz")
    if [ -f "$out" ]; then
        echo "=== skip $out (exists) ==="
        continue
    fi
    echo "=== ED honeycomb ${LX}x${LY} $model hx=$hx hz=$hz ==="
    ( cd "$REPO" && python -m exact.lanczos_ed --lattice honeycomb --model "$model" \
        --Lx "$LX" --Ly "$LY" --hx "$hx" --hz "$hz" --k "$K" --out "$out" ) \
        || echo "!!! $model hx=$hx hz=$hz FAILED (exit $?) -- continuing"
done
echo "=== driver done ==="
