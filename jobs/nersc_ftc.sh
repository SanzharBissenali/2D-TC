#!/bin/bash
# Fermionic toric code (--ftc): dressed stars A'_v = A_v * B_NE(v) (CKR bosonization
# Gauss law, arXiv:1711.00515). Same unperturbed GS as the plain TC (E0 = -25 at L=4)
# but non-stoquastic H and gap 4 (paired excitations) instead of 2.
#   Phase 3 (default, CURRENT plan): fixed point h=0, arms cnn + plaincnn. Identity-
#     init Combo CNN IS the exact GS (E0=-25 at L=4), so "cnn" is a wiring check;
#     "plaincnn" has NO built-in symmetry and must find that GS from a generic init
#     via VMC alone -- this is the actual test (hypothesis: it fails even at h=0).
#     Block-1 kernel_size stays 2 (LOCAL nearest-neighbor, unlike Block 3's GLOBAL
#     kernel_size_inv=Lx-1, already auto-set per L in config.py -- these are two
#     different kernels, not one): cnn = Combo-small, 1681 params @L4; plaincnn =
#     channels_noninv 1,8,6,2, ~1800 params (matched-generous, no Wilson/invariant).
#   Phase 4 (LATER, not the current plan): hx sweep, e.g. HX_LIST="0.10 0.30 0.50 0.70".
#     Perturbed points force dtype=complex (signful GS): budget ~20 min XLA JIT per
#     point at L=4 (hy-cut lesson) and consider DIAG_SHIFT=1e-3 if SR goes unstable.
#   Env:  LX (4), HX_LIST ("0.00"), SIM_TIME (3.5), ARMS ("cnn"), SEED (0),
#         DIAG_SHIFT (6e-5), ED (1 => run the --ftc ED companion per point first).
#   Array chunking (optional): submit with --array=0-K-1 to split HX_LIST across K
#   tasks (mirrors jobs/nersc_hy_sweep.sh); omit --array to run the whole list serially.
#
# Submit:
#   bash scripts/cluster.sh submit jobs/nersc_ftc.sh -J tc_ftc -t 0:45:00 \
#       --export=ALL,ARMS="cnn plaincnn",WANDB=1                                        # Phase 3 (h=0, current)
#   bash scripts/cluster.sh submit jobs/nersc_ftc.sh -J tc_ftc -t 1:30:00 --array=0-3 \
#       --export=ALL,HX_LIST="0.10 0.30 0.50 0.70",ARMS="cnn plaincnn",WANDB=1          # Phase 4 (later)
#
# Resumable: skip-if-COMPLETE via scripts/is_complete.py; ED skipped if its JSON exists.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_ftc
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUTDIR=$REPO/results/nqs
EDDIR=$REPO/results/ed
mkdir -p "$OUTDIR" "$EDDIR"

HZ=0.0
SEED="${SEED:-0}"
LX="${LX:-4}"
SIM_TIME="${SIM_TIME:-3.5}"
HX_LIST="${HX_LIST:-0.00}"
ARMS="${ARMS:-cnn}"
DIAG_SHIFT="${DIAG_SHIFT:-6e-5}"
ED="${ED:-1}"

# --- optional W&B logging (offline; `wandb sync` from a login node afterward) ---
WANDB="${WANDB:-0}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-ftc_L${LX}}"
    echo "=== W&B ON (mode=$WANDB_MODE dir=$WANDB_DIR group=${WANDB_GROUP:-ftc_L${LX}}) ==="
fi

# Block-1 kernel_size = 2 (LOCAL nearest-neighbor, same for every L -- distinct from
# Block 3's GLOBAL kernel_size_inv=Lx-1, already auto-set in config.py). cnn =
# Combo-small (1681 params @L4); plaincnn = unconstrained baseline, same local-conv
# primitive, no Wilson/invariant block, param-matched-generous (~1800 vs cnn's 1681).
declare -A ARM_FLAGS
ARM_FLAGS[cnn]=""
ARM_FLAGS[plaincnn]="--architecture PlainCNN --channels_noninv 1,8,6,2 --kernel_size 2"

# Fail loudly on an unknown arm (adversarial-audit finding, 2026-08-14): under
# `set -u`, referencing ${ARM_FLAGS[$arm]} for a key not in the map above aborts
# the WHOLE script with a cryptic "unbound variable" -- killing every pending
# hx/arm combination, not just the bad one. Check up front instead.
for arm in $ARMS; do
    [ -n "${ARM_FLAGS[$arm]+set}" ] || { echo "!!! unknown arm '$arm' (known: ${!ARM_FLAGS[*]})"; exit 1; }
done

# Optional SLURM array chunking: --array=0-K-1 splits HX_LIST across K tasks.
HX_ARR=($HX_LIST)
N_HX=${#HX_ARR[@]}
ARR_T=${SLURM_ARRAY_TASK_COUNT:-1}
ARR_I=${SLURM_ARRAY_TASK_ID:-0}
PPT=$(( (N_HX + ARR_T - 1) / ARR_T ))
START=$(( ARR_I * PPT ))
END=$(( START + PPT )); [ "$END" -gt "$N_HX" ] && END=$N_HX
HX_CHUNK="${HX_ARR[@]:$START:$((END-START))}"

echo "=== driver: LX=$LX SIM_TIME=$SIM_TIME ARMS=[$ARMS] HX_CHUNK=[$HX_CHUNK] (task $ARR_I/$ARR_T of [$HX_LIST]) SEED=$SEED DS=$DIAG_SHIFT ED=$ED ==="
for hx in $HX_CHUNK; do
    # --ftc ED companion (L=4 only; the fTC spectrum differs from the plain TC even
    # at h=0 -- gap 4 vs 2 -- so TC ED files are NOT reusable). Real H => float64
    # sparse, ~10 GB, fits the shared node. Also records neg_amp_fraction (sign check).
    edjson=$(printf "%s/ed_ftc_L%d_hx%.2f_hz%.2f.json" "$EDDIR" "$LX" "$hx" "$HZ")
    if [ "$ED" = "1" ] && [ "$LX" -eq 4 ] && [ ! -f "$edjson" ]; then
        echo "=== ED (ftc) L$LX hx=$hx ==="
        ( cd "$REPO" && python -m exact.lanczos_ed --Lx "$LX" --hx "$hx" --hz "$HZ" --k 4 \
            --ftc --out "$edjson" ) || echo "!!! ED hx=$hx FAILED (exit $?) — continuing"
    fi

    for arm in $ARMS; do
        jobid=$(printf "ftc_L%d_hx%.2f_hz%.2f_%s" "$LX" "$hx" "$HZ" "$arm")
        base="$OUTDIR/G-equiv_1_${jobid}"
        if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
            echo "=== skip $jobid (complete) ==="
            continue
        fi
        echo "=== NQS $jobid (sim_time=$SIM_TIME) ==="
        ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
            --outindex 1 --jobid "$jobid" --ftc \
            --Lx "$LX" --hx "$hx" --hy 0.0 --hz "$HZ" \
            --dt 0.01 --diag_shift "$DIAG_SHIFT" --sim_time "$SIM_TIME" --seed "$SEED" \
            --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
            --n_samples_fin 8192 --use_custom_sampler \
            ${ARM_FLAGS[$arm]} $WB_FLAGS ) || echo "!!! $jobid FAILED (exit $?) — continuing"
    done
done
echo "=== driver done ==="
