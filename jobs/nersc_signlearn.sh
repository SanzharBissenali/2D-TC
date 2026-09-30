#!/bin/bash
# Supervised sign-learnability test (scripts/sign_learn_split.py): MLP on (eps, x)
# -> sign, 80/20 split, random vs pattern-held-out. No VMC.
#
# Env: SOURCE (ed|head), LX, LY, HIDDEN (64), DEPTH (2), STEPS (3000),
#      N_DATA (200000) or N_LIST "n+n+..." (one run each, file tagged _n<N>_s<STEPS>),
#      FOLDS (5), MAX_FOLDS (0 => all), SPLITS (random,pattern; pass as random+pattern in --export);
#      SOURCE=ed:   POINTS "hx:hz+hx:hz+..." (one ED solve per point);
#      SOURCE=xsyn: as head (SIZES, N_LIST, ...) plus DEGREE (3|2|1); x-only inputs, no decoder;
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
MAX_FOLDS="${MAX_FOLDS:-0}"; SPLITS="${SPLITS:-random,pattern}"; SPLITS="${SPLITS//+/,}"   # --export cannot carry commas
N_LIST="${N_LIST:-$N_DATA}"; N_LIST="${N_LIST//+/;}"
IFS=';' read -ra NS <<< "$N_LIST"
COMMON=(--hidden "$HIDDEN" --depth "$DEPTH" --steps "$STEPS" --folds "$FOLDS"
        --max_folds "$MAX_FOLDS" --splits "$SPLITS" --log_every "${LOG_EVERY:-10}" --degree "${DEGREE:-3}" --out_dir "$REPO/results/signlearn")
cd "$REPO"

if [ "$SOURCE" = "ed" ]; then
    LX="${LX:-2}"; LY="${LY:-3}"
    PTS="${POINTS:-0.4:0}"; PTS="${PTS//+/;}"
    IFS=';' read -ra PAIRS <<< "$PTS"
    for pair in "${PAIRS[@]}"; do
        hx="${pair%%:*}"; hz="${pair##*:}"
        echo "--- signlearn ed ${LX}x${LY} ($hx,$hz)  $(date)"
        PYTHONPATH=$REPO python scripts/sign_learn_split.py --source ed \
            --Lx "$LX" --Ly "$LY" --hx "$hx" --hz "$hz" --n_data "${NS[0]}" "${COMMON[@]}" \
            || echo "!!! failed at ($hx,$hz)"
    done
else
    SZ="${SIZES:-${LX:-2}x${LY:-3}}"; SZ="${SZ//+/;}"
    IFS=';' read -ra SS <<< "$SZ"
    for s in "${SS[@]}"; do
        for n in "${NS[@]}"; do
            TAGARG=()
            [ "$n" != "200000" ] || [ "$STEPS" != "10000" ] && TAGARG=(--tag "_n${n}_s${STEPS}")
            [ "$SOURCE" = "xsyn" ] && TAGARG=(--tag "_n${n}_s${STEPS}")
            echo "--- signlearn $SOURCE $s n=$n steps=$STEPS  $(date)"
            PYTHONPATH=$REPO python scripts/sign_learn_split.py --source "$SOURCE" \
                --Lx "${s%%x*}" --Ly "${s##*x}" --k_max "${K_MAX:--1}" --n_data "$n" \
                ${TAGARG[@]+"${TAGARG[@]}"} "${COMMON[@]}" || echo "!!! failed at $s n=$n"
        done
    done
fi
echo "=== signlearn driver done $(date) ==="
