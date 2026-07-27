#!/bin/bash
# Variant 3 (pure plaquette transformer, gated full attention) on the h_x cut.
# The ansatz is EXACTLY A_v-invariant, so it is only honestly benchmarked at
# hz=hy=0 (the perturbation must commute with every A_v). Default: L=4, hx=0.20.
#   Arms:  v3  (gated content attention, alpha_h trainable from 0)
#          v3f (frozen alpha_h=0 => normalized-factored ablation)
#          v3s (small d=16 n_l=2 fallback if the 35k-param SR solve is too slow)
#   Env:   LX (4), HX_LIST ("0.20"), SIM_TIME (3.5), ARMS ("v3 v3f"), SEED (0),
#          ED (1 => run the missing L=4 ED companion point first).
#
# Submit:
#   bash scripts/cluster.sh submit jobs/nersc_v3.sh -J tc_v3 -t 2:00:00
#   bash scripts/cluster.sh submit jobs/nersc_v3.sh -J tc_v3s -t 1:00:00 --export=ALL,ARMS=v3s,ED=0
#
# Resumable: skip-if-COMPLETE via scripts/is_complete.py; ED skipped if its JSON exists.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_v3
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
HX_LIST="${HX_LIST:-0.20}"
ARMS="${ARMS:-v3 v3f}"
ED="${ED:-1}"

# --- optional W&B logging (offline; `wandb sync` from a login node afterward) ---
WANDB="${WANDB:-0}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="$REPO/wandb"; mkdir -p "$WANDB_DIR"
    WB_FLAGS="--wandb --wandb_project ${WANDB_PROJECT:-2d-tc} --wandb_group ${WANDB_GROUP:-v3_L${LX}}"
    echo "=== W&B ON (mode=$WANDB_MODE dir=$WANDB_DIR group=${WANDB_GROUP:-v3_L${LX}}) ==="
fi

V3_COMMON="--symmetric_block plaquette_transformer --tf_layers 4 --tf_dmodel 32 --tf_heads 4 --tf_ffn_mult 2 --tf_activation gelu"
declare -A ARM_FLAGS
ARM_FLAGS[v3]="$V3_COMMON"
ARM_FLAGS[v3f]="$V3_COMMON --no-tf_content"
ARM_FLAGS[v3s]="--symmetric_block plaquette_transformer --tf_layers 2 --tf_dmodel 16 --tf_heads 4 --tf_ffn_mult 2 --tf_activation gelu"

echo "=== driver: LX=$LX SIM_TIME=$SIM_TIME ARMS=[$ARMS] HX=[$HX_LIST] SEED=$SEED ED=$ED ==="
for hx in $HX_LIST; do
    # ED companion (L=4 only; float64 h_x point, ~10 GB, fits the shared node)
    edjson=$(printf "%s/ed_L%d_hx%.2f_hz%.2f.json" "$EDDIR" "$LX" "$hx" "$HZ")
    if [ "$ED" = "1" ] && [ "$LX" -eq 4 ] && [ ! -f "$edjson" ]; then
        echo "=== ED L$LX hx=$hx ==="
        ( cd "$REPO" && python -m exact.lanczos_ed --Lx "$LX" --hx "$hx" --hz "$HZ" --k 4 \
            --out "$edjson" ) || echo "!!! ED hx=$hx FAILED (exit $?) — continuing"
    fi

    for arm in $ARMS; do
        jobid=$(printf "L%d_hx%.2f_hz%.2f_%s" "$LX" "$hx" "$HZ" "$arm")
        base="$OUTDIR/G-equiv_1_${jobid}"
        if python "$REPO/scripts/is_complete.py" "$base" "$LX"; then
            echo "=== skip $jobid (complete) ==="
            continue
        fi
        echo "=== NQS $jobid (sim_time=$SIM_TIME) ==="
        ( cd "$OUTDIR" && PYTHONPATH="$REPO" python "$REPO/main.py" \
            --outindex 1 --jobid "$jobid" \
            --Lx "$LX" --hx "$hx" --hy 0.0 --hz "$HZ" \
            --dt 0.01 --diag_shift 6e-5 --sim_time "$SIM_TIME" --seed "$SEED" \
            --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
            --n_samples_fin 8192 --use_custom_sampler \
            ${ARM_FLAGS[$arm]} $WB_FLAGS ) || echo "!!! $jobid FAILED (exit $?) — continuing"
    done
done
echo "=== driver done ==="
