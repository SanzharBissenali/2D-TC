#!/bin/bash
# scripts/pretrain_sign_mlp.py driver, two modes selected by TARGET:
#
#   TARGET=ed (default, byte-identical to the original single-mode script):
#     supervised warm start at specific ED points. ED psi on all 2^N configs
#     -> features_ex (host, ~18 min at 2x3) -> optax fit of the sign MLP ->
#     results/pretrain/mlp_hc{Lx}x{Ly}_hx*_hz*_h{H}d{D}.npz (+ json with the
#     CEILING CHECK). 2x3 hx points need the signfid memory profile (~60 GB
#     CSR => -q regular); 1x2/2x2 fit the debug QOS in minutes. Sequential
#     over POINTS ('hx:hz+hx:hz', same encoding as nersc_signfid.sh).
#
#   TARGET=head (docs/signhead_benchmark_plan.md addendum, 2026-09-24):
#     field-independent, ED-free warm start -- fits the SAME MLP to the
#     CLOSED-FORM head sign on synthetic data (no ED, no POINTS loop; one
#     call regardless of field). Fast debug-queue job by itself. Writes
#     results/pretrain/mlp_hc{Lx}x{Ly}_head_h{H}d{D}_k{K}.npz (+
#     _snapshots.npz + _val.npz + .json with the learning curve). Set
#     ED_POINTS (same '+'-separated hx:hz encoding as POINTS) to ALSO grade
#     the fitted snapshots against exact ED inline -- this pulls in the ED
#     memory profile, so prefer a separate EVAL_ONLY invocation instead (see
#     below) unless the run is small (1x2/2x2).
#
#   EVAL_ONLY=<path to a *_snapshots.npz> (any TARGET; takes priority):
#     grades a previously-fit TARGET=head snapshot set against ED_POINTS,
#     decoupled from the (debug-queue) fitting run -- no jax/flax/optax
#     needed, only netket (ED) + pymatching/numba (the head). This is the
#     recommended way to do the 2x3 ED grading: fit on debug, then eval on a
#     job with the ED memory profile (-q shared -G 2 -c 64 -t 2:00, or
#     -q regular for larger hx points -- see jobs/nersc_signfid.sh).
#
# Env (TARGET=ed): LX (1), LY (2), HIDDEN (64), DEPTH (2), EPOCHS (3000),
#   FS_TARGET (1e-5, the 1-F_s early-stop threshold -- this env var was named
#   TARGET before 2026-09-24; renamed because TARGET now selects the
#   ed/head mode), SEED (0), POINTS ('hx:hz+hx:hz', default 0.4:0).
# Env (TARGET=head): LX, LY, HIDDEN, DEPTH, SEED as above; SYNTHETIC
#   (200000, N_TRAIN), N_VAL (20000), K_MAX (12), STEPS (20000), ED_POINTS
#   ('' = no inline ED grading, else 'hx:hz+hx:hz' as in POINTS).
# Env (EVAL_ONLY): EVAL_ONLY (path), LX, LY (must match the fitting run),
#   ED_POINTS (required).
#
# Submit (debug, fast field-independent fit):
#   ... submit jobs/nersc_pretrain.sh -q debug -t 0:30 \
#       --export=ALL,TARGET=head,LX=2,LY=3,HIDDEN=64,K_MAX=12
# Submit (shared, decoupled ED grading of the fit above at 3 points):
#   ... submit jobs/nersc_pretrain.sh -q shared -G 2 -c 64 -t 2:00 \
#       --export=ALL,EVAL_ONLY=results/pretrain/mlp_hc2x3_head_h64d2_k12_snapshots.npz,LX=2,LY=3,ED_POINTS=0.4:0+0.8:0.2+1.2:0
# Submit (original TARGET=ed path, unchanged):
#   ... submit jobs/nersc_pretrain.sh -q regular -t 5:00:00 \
#       --export=ALL,LX=2,LY=3,POINTS=0.4:0+0.4:0.2+0.4:0.4
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
HIDDEN="${HIDDEN:-64}"; DEPTH="${DEPTH:-2}"; SEED="${SEED:-0}"
TARGET="${TARGET:-ed}"
EVAL_ONLY="${EVAL_ONLY:-}"
ED_POINTS="${ED_POINTS:-}"; ED_POINTS="${ED_POINTS//+/,}"

python -c "import pymatching, optax" 2>/dev/null \
    || { echo "!!! pymatching/optax missing in 2dtc"; exit 1; }
OUT=$REPO/results/pretrain
mkdir -p "$OUT"

if [ -n "$EVAL_ONLY" ]; then
    [ -n "$ED_POINTS" ] || { echo "!!! EVAL_ONLY needs ED_POINTS"; exit 1; }
    echo "=== eval_only ${LX}x${LY} snapshots=$EVAL_ONLY points=[$ED_POINTS] ==="
    PYTHONPATH=$REPO python "$REPO/scripts/pretrain_sign_mlp.py" \
        --eval_only "$EVAL_ONLY" --Lx "$LX" --Ly "$LY" \
        --ed_points "$ED_POINTS" \
        || echo "!!! eval_only failed"
    echo "=== eval_only driver done $(date) ==="
    exit 0
fi

if [ "$TARGET" = "head" ]; then
    SYNTHETIC="${SYNTHETIC:-200000}"; N_VAL="${N_VAL:-20000}"
    K_MAX="${K_MAX:-12}"; STEPS="${STEPS:-20000}"
    tag="mlp_hc${LX}x${LY}_head_h${HIDDEN}d${DEPTH}_k${K_MAX}"
    echo "=== pretrain --target head ${LX}x${LY} n_train=$SYNTHETIC n_val=$N_VAL " \
         "k_max=$K_MAX steps=$STEPS h=$HIDDEN d=$DEPTH ed_points=[$ED_POINTS] ==="
    if [ -f "$OUT/${tag}.npz" ]; then
        echo "--- skip (exists): $tag"
    else
        EXTRA=()
        [ -n "$ED_POINTS" ] && EXTRA=(--ed_points "$ED_POINTS")
        PYTHONPATH=$REPO python "$REPO/scripts/pretrain_sign_mlp.py" \
            --target head --Lx "$LX" --Ly "$LY" \
            --hidden "$HIDDEN" --depth "$DEPTH" --seed "$SEED" \
            --synthetic "$SYNTHETIC" --n_val "$N_VAL" --k_max "$K_MAX" \
            --steps "$STEPS" --out_dir "$OUT" "${EXTRA[@]}" \
            || echo "!!! pretrain --target head failed"
    fi
    echo "=== pretrain driver done $(date) ==="
    exit 0
fi

# TARGET=ed (original path, byte-identical)
EPOCHS="${EPOCHS:-3000}"; FS_TARGET="${FS_TARGET:-1e-5}"
PTS="${POINTS:-0.4:0}"; PTS="${PTS//+/;}"
echo "=== pretrain --target ed ${LX}x${LY} points=[$PTS] h=$HIDDEN d=$DEPTH epochs=$EPOCHS fs_target=$FS_TARGET ==="
IFS=';' read -ra PAIRS <<< "$PTS"
for pair in "${PAIRS[@]}"; do
    hx="${pair%%:*}"; hz="${pair##*:}"
    tag=$(LC_ALL=C awk -v a="$hx" -v b="$hz" 'BEGIN{printf "hx%g_hz%g", a, b}')
    if [ -f "$OUT/mlp_hc${LX}x${LY}_${tag}_h${HIDDEN}d${DEPTH}.npz" ]; then
        echo "--- skip (exists): $tag"; continue
    fi
    echo "--- pretrain $tag  $(date)"
    PYTHONPATH=$REPO python "$REPO/scripts/pretrain_sign_mlp.py" \
        --target ed --Lx "$LX" --Ly "$LY" --hx "$hx" --hz "$hz" \
        --hidden "$HIDDEN" --depth "$DEPTH" --epochs "$EPOCHS" \
        --fs_target "$FS_TARGET" --seed "$SEED" --out_dir "$OUT" \
        || echo "!!! pretrain failed at $tag"
done
echo "=== pretrain driver done $(date) ==="
