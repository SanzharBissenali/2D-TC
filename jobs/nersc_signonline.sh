#!/bin/bash
# Online (streaming) sign-learnability test: scripts/sign_learn_online.py.
# Fresh minibatches every step, fixed held-out validation, no VMC.
#
# Env: SOURCE (head|ed), LX, LY, SAMPLES (fresh samples per split, default 400000000),
#      SPLITS (random+pattern), BATCH (4096), LR (1e-3), HIDDEN (128), SEEDS (3),
#      WORKERS (default: cpus per task - 4; use 24 for ED jobs), SAMPLES as a plain integer, EVALS (160), TAG;
#      SOURCE=ed: POINTS "hx:hz+hx:hz+..." (ED solve ~25 min each, cached in $SCRATCH).
#
# Submit (head labels; 32 cores per GPU on the shared queue):
#   bash scripts/cluster.sh submit jobs/nersc_signonline.sh -q shared -t 2:00:00 \
#     --export=ALL,SOURCE=head,LX=4,LY=4
# Submit (ED labels; the 2^27 ED solve needs ~110 GB => 2 GPU slices):
#   bash scripts/cluster.sh submit jobs/nersc_signonline.sh -q shared -G 2 -c 64 -t 2:30:00 \
#     --export=ALL,SOURCE=ed,LX=2,LY=3,POINTS=0.4:0
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_signonline
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
SOURCE="${SOURCE:-head}"
LX="${LX:-4}"; LY="${LY:-4}"
SAMPLES="${SAMPLES:-400000000}"
SPLITS="${SPLITS:-random+pattern}"; SPLITS="${SPLITS//+/,}"     # --export cannot carry commas
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} - 4 ))}"
export PYTHONPATH=$REPO
[ "$SOURCE" != "ed" ] || : "${SCRATCH:?SCRATCH unset: the ED cache needs it}"
ARGS=(--source "$SOURCE" --Lx "$LX" --Ly "$LY" --samples "$SAMPLES" --splits "$SPLITS"
      --batch "${BATCH:-4096}" --lr "${LR:-1e-3}" --hidden "${HIDDEN:-128}" --seeds "${SEEDS:-3}"
      --workers "$WORKERS" --evals "${EVALS:-160}" --out_dir "$REPO/results/signonline"
      --n_val "${N_VAL:-200000}")
[ -n "${TAG:-}" ] && ARGS+=(--tag "$TAG")
rc=0
if [ "$SOURCE" = "ed" ]; then
    PTS="${POINTS:-0.4:0}"; PTS="${PTS//+/;}"
    IFS=';' read -ra PAIRS <<< "$PTS"
    for pair in "${PAIRS[@]}"; do
        hx="${pair%%:*}"; hz="${pair##*:}"
        echo "--- signonline ed ${LX}x${LY} ($hx,$hz) samples=$SAMPLES workers=$WORKERS  $(date)"
        python scripts/sign_learn_online.py "${ARGS[@]}" --hx "$hx" --hz "$hz" || { echo "!!! failed at ($hx,$hz)"; rc=1; }
    done
else
    echo "--- signonline head ${LX}x${LY} samples=$SAMPLES workers=$WORKERS  $(date)"
    python scripts/sign_learn_online.py "${ARGS[@]}" || { echo "!!! failed"; rc=1; }
fi
echo "=== signonline driver done rc=$rc $(date) ==="
exit $rc
