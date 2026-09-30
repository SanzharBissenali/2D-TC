#!/bin/bash
# Supervised sign-learnability test (scripts/sign_learn_split.py): MLP on (eps, x)
# -> sign, 80/20 split, random vs pattern-held-out. No VMC.
#
# Env: SOURCE (ed|head), LX, LY, HIDDEN (64), DEPTH (2), STEPS (3000),
#      N_DATA (200000), FOLDS (5);
#      SOURCE=ed:   POINTS "hx:hz+hx:hz+..." (one ED solve per point);
#      SOURCE=head: SIZES "LxxLy+LxxLy+..." (overrides LX/LY), K_MAX (-1 => 2F).
#
# Submit (2x3 ED points; 2^27 ED needs the ~110 GB of a 2-GPU shared slice):
#   bash scripts/cluster.sh submit jobs/nersc_signlearn.sh -q shared -G 2 -c 64 -t 2:30:00 \
#     --export=ALL,SOURCE=ed,LX=2,LY=3,POINTS=0.4:0+0.4:0.4+0.8:0.2
# Submit (head labels, several sizes, debug):
#   (one size per debug job: 10 fits of 10k steps do not leave room for four sizes)
#   bash scripts/cluster.sh submit jobs/nersc_signlearn.sh \
#     --export=ALL,SOURCE=head,SIZES=4x4,HIDDEN=128,STEPS=10000
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_signlearn
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
SOURCE="${SOURCE:-head}"
HIDDEN="${HIDDEN:-64}"; DEPTH="${DEPTH:-2}"; STEPS="${STEPS:-3000}"
N_DATA="${N_DATA:-200000}"; FOLDS="${FOLDS:-5}"
COMMON=(--hidden "$HIDDEN" --depth "$DEPTH" --steps "$STEPS" --n_data "$N_DATA"
        --folds "$FOLDS" --out_dir "$REPO/results/signlearn")
cd "$REPO"

if [ "$SOURCE" = "ed" ]; then
    LX="${LX:-2}"; LY="${LY:-3}"
    PTS="${POINTS:-0.4:0}"; PTS="${PTS//+/;}"
    IFS=';' read -ra PAIRS <<< "$PTS"
    for pair in "${PAIRS[@]}"; do
        hx="${pair%%:*}"; hz="${pair##*:}"
        echo "--- signlearn ed ${LX}x${LY} ($hx,$hz)  $(date)"
        PYTHONPATH=$REPO python scripts/sign_learn_split.py --source ed \
            --Lx "$LX" --Ly "$LY" --hx "$hx" --hz "$hz" "${COMMON[@]}" \
            || echo "!!! failed at ($hx,$hz)"
    done
else
    SZ="${SIZES:-${LX:-2}x${LY:-3}}"; SZ="${SZ//+/;}"
    IFS=';' read -ra SS <<< "$SZ"
    for s in "${SS[@]}"; do
        echo "--- signlearn head $s  $(date)"
        PYTHONPATH=$REPO python scripts/sign_learn_split.py --source head \
            --Lx "${s%%x*}" --Ly "${s##*x}" --k_max "${K_MAX:--1}" "${COMMON[@]}" \
            || echo "!!! failed at $s"
    done
fi
echo "=== signlearn driver done $(date) ==="
