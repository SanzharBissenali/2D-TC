#!/bin/bash
# Arm M-pre supervised warm start (scripts/pretrain_sign_mlp.py): ED psi on
# all 2^N configs -> features_ex (host, ~18 min at 2x3) -> optax fit of the
# sign MLP -> results/pretrain/mlp_hc{Lx}x{Ly}_hx*_hz*_h{H}d{D}.npz (+ json
# with the CEILING CHECK). 2x3 hx points need the signfid memory profile
# (~60 GB CSR => -q regular); 1x2/2x2 fit the debug QOS in minutes.
# Sequential over POINTS ('hx:hz+hx:hz', same encoding as nersc_signfid.sh).
#
# Submit: bash scripts/cluster.sh submit jobs/nersc_pretrain.sh -q regular \
#   -t 5:00:00 --export=ALL,LX=2,LY=3,POINTS=0.4:0+0.4:0.2+0.4:0.4
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 128
#SBATCH -J hc_pretrain
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
LX="${LX:-1}"; LY="${LY:-2}"
HIDDEN="${HIDDEN:-64}"; DEPTH="${DEPTH:-2}"
EPOCHS="${EPOCHS:-3000}"; TARGET="${TARGET:-1e-5}"; SEED="${SEED:-0}"
PTS="${POINTS:-0.4:0}"; PTS="${PTS//+/;}"

python -c "import pymatching, optax" 2>/dev/null \
    || { echo "!!! pymatching/optax missing in 2dtc"; exit 1; }
OUT=$REPO/results/pretrain
mkdir -p "$OUT"
echo "=== pretrain ${LX}x${LY} points=[$PTS] h=$HIDDEN d=$DEPTH epochs=$EPOCHS target=$TARGET ==="
IFS=';' read -ra PAIRS <<< "$PTS"
for pair in "${PAIRS[@]}"; do
    hx="${pair%%:*}"; hz="${pair##*:}"
    tag=$(LC_ALL=C awk -v a="$hx" -v b="$hz" 'BEGIN{printf "hx%g_hz%g", a, b}')
    if [ -f "$OUT/mlp_hc${LX}x${LY}_${tag}_h${HIDDEN}d${DEPTH}.npz" ]; then
        echo "--- skip (exists): $tag"; continue
    fi
    echo "--- pretrain $tag  $(date)"
    PYTHONPATH=$REPO python "$REPO/scripts/pretrain_sign_mlp.py" \
        --Lx "$LX" --Ly "$LY" --hx "$hx" --hz "$hz" \
        --hidden "$HIDDEN" --depth "$DEPTH" --epochs "$EPOCHS" \
        --target "$TARGET" --seed "$SEED" --out_dir "$OUT" \
        || echo "!!! pretrain failed at $tag"
done
echo "=== pretrain driver done $(date) ==="
