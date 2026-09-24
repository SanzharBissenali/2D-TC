#!/bin/bash
# Learned-vs-gated sign head benchmark (docs/signhead_benchmark_plan.md): three
# arms sharing the DS trunk + QEC head machinery but differing in how the sign
# enters --
#   cnnqM   arm M:     psi = A(sigma) * tanh(m_theta(eps, x)), m_theta random init
#   cnnqMp  arm M-pre: same architecture, m_theta warm-started from a
#           supervised pre-fit to the exact ED signs (scripts/pretrain_sign_mlp.py)
#   cnnqT   arm T:     psi = a A_triv(sigma) + (-1)^s(sigma) A_top(sigma), a a signed
#           real mix scalar (init --mix_init 0.05; a = 0 = head-only arm)
# All three: real trunk(s), complex log psi (non-holomorphic QGT, same path as
# the Phase-4b residual arm cnnqR -- --minsr_mode complex). Phase-3 recipe
# (docs/signhead_benchmark_plan.md Sec 1): minSR lr 0.01, diag_shift 6e-5,
# 350 steps @ dt 0.01, 8192 samples, custom sampler, seed 0.
#
# cnnqMp additionally needs a pretrained npz
# (MLP_DIR/mlp_hc{Lx}x{Ly}_hx{hx:g}_hz{hz:g}_h64d2.npz, written by
# scripts/pretrain_sign_mlp.py --hidden 64 --depth 2) -- FAILS LOUDLY (skips
# just that point, keeps going) if it is missing rather than silently
# training cnnqMp cold like cnnqM.
#
# Env: LX (2), LY (3), ARMS ('+'-separated tokens from {cnnqM, cnnqMp, cnnqT};
#      default all three), POINTS -- hx:hz pairs, ':' inner comma, '+' between
#      pairs (cluster.sh submit flattens ssh args / --export splits on commas),
#      e.g. POINTS=0.4:0+0.8:0.2+1.2:0.4; SIM_TIME (3.5), SEED (0), WANDB (1),
#      MLP_DIR (results/pretrain).
# Submit (shared queue; timing TBD at first run -- budget like cnnqR, ~1.5x
# cnnqB's 3.6 s/step, plus the per-batch host callback for the sign). Plan
# Sec 4 stage d = chunks of 3 runs per 1:45 job, e.g. all three arms at one
# point:
#   ... submit jobs/nersc_signbench.sh -t 1:45:00 \
#       --export=ALL,LX=2,LY=3,ARMS=cnnqM+cnnqMp+cnnqT,POINTS=0.4:0
# jobids use the canonical %g strings (0.4:0 -> hx0.4_hz0, the Phase-4
# convention), so POINTS=0.40:0.0 names the same run as POINTS=0.4:0.
# MLP_DIR may be absolute or repo-relative.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:45:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hc_sgnb
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
mkdir -p "$OUTDIR"

LX="${LX:-2}"
LY="${LY:-3}"
ARMS="${ARMS:-cnnqM+cnnqMp+cnnqT}"; ARMS="${ARMS//+/ }"
POINTS="${POINTS:-0.4:0}"; POINTS="${POINTS//+/ }"
SIM_TIME="${SIM_TIME:-3.5}"
SEED="${SEED:-0}"
MLP_DIR="${MLP_DIR:-results/pretrain}"
case "$MLP_DIR" in /*) ;; *) MLP_DIR="$REPO/$MLP_DIR" ;; esac   # absolute or repo-relative

python -c "import pymatching" 2>/dev/null \
    || { echo "!!! pymatching missing in 2dtc (pip install on a login node)"; exit 1; }

WANDB="${WANDB:-1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-hc-signbench}"
    echo "=== W&B ON (offline, group ${WANDB_GROUP:-hc-signbench}) ==="
fi

declare -A ARM_FLAGS
ARM_FLAGS[cnnqM]="--sign_head qec --sign_impl mlp --mlp_hidden 64 --mlp_depth 2 --minsr_mode complex"
ARM_FLAGS[cnnqMp]="--sign_head qec --sign_impl mlp --mlp_hidden 64 --mlp_depth 2 --minsr_mode complex"  # + --mlp_init below
ARM_FLAGS[cnnqT]="--sign_head qec --sign_impl twobranch --minsr_mode complex"
ARM_FLAGS[cnnqTp]="--sign_head qec --sign_impl twobranch --mix_positive --minsr_mode complex"
for arm in $ARMS; do
    [ -n "${ARM_FLAGS[$arm]+set}" ] || { echo "!!! unknown arm '$arm' (known: ${!ARM_FLAGS[*]})"; exit 1; }
done

# Seed-1 replicas (plan Sec 4 stage f, plan Sec 6 verdict rule -- "wins on
# BOTH seeds"): SEED=0 keeps the established unsuffixed jobid so it never
# collides with/duplicates any prior run; any other SEED appends "_s${SEED}"
# (mirrors nersc_phase4_fields.sh's HY_SEG/DEC_SUFFIX pattern). scripts/
# signbench_summary.py looks for this exact suffix.
SEED_SUFFIX=""; [ "$SEED" != "0" ] && SEED_SUFFIX="_s${SEED}"

echo "=== signbench: ${LX}x${LY} ds points=[$POINTS] arms=[$ARMS] sim_time=$SIM_TIME seed=$SEED ==="
for pt in $POINTS; do
  hx="${pt%%:*}"; hz="${pt##*:}"
  # Python's f'{x:g}' formatting drives the pretrain npz filename; normalize
  # the bash hx/hz the same way (awk %g) so a POINTS token like "0.40" still
  # finds the "hx0.4" cache written by scripts/pretrain_sign_mlp.py, and the
  # jobid is canonical (hx0.4_hz0) whatever the POINTS spelling. LC_ALL=C
  # is load-bearing: under a comma-decimal locale (observed: kk_KZ.UTF-8)
  # awk's %g prints "0,4" instead of "0.4", silently breaking the lookup.
  hx_g=$(LC_ALL=C awk -v v="$hx" 'BEGIN{printf "%g", v}')
  hz_g=$(LC_ALL=C awk -v v="$hz" 'BEGIN{printf "%g", v}')
  for arm in $ARMS; do
    jobid="hc${LX}x${LY}_ds_hx${hx_g}_hz${hz_g}_${arm}${SEED_SUFFIX}"
    base="$OUTDIR/G-equiv_1_${jobid}"
    if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
        echo "=== skip $jobid (complete) ==="
        continue
    fi
    extra=""
    if [ "$arm" = "cnnqMp" ]; then
        mlp_npz="$MLP_DIR/mlp_hc${LX}x${LY}_hx${hx_g}_hz${hz_g}_h64d2.npz"
        if [ ! -f "$mlp_npz" ]; then
            echo "!!! $jobid FAILED: missing pretrained MLP $mlp_npz -- run" \
                 "'python scripts/pretrain_sign_mlp.py --Lx $LX --Ly $LY --hx $hx_g --hz $hz_g' first -- skipping"
            continue
        fi
        extra="--mlp_init $mlp_npz"
    fi
    echo "=== NQS $jobid ==="
    ( cd "$OUTDIR" && PYTHONPATH=$REPO python "$REPO/main.py" \
        --outindex 1 --jobid "$jobid" \
        --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" \
        --hx "$hx" --hy 0.0 --hz "$hz" \
        --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 \
        --sim_time "$SIM_TIME" --seed "$SEED" --kernel_size 2 \
        --n_samples_fin 8192 --use_custom_sampler \
        --channels_noninv 1,16 --channels_inv 16,8,1 \
        ${ARM_FLAGS[$arm]} $extra $WB_FLAGS ) \
        || echo "!!! $jobid FAILED (exit $?) -- continuing"
  done
done
echo "=== signbench driver done ==="
