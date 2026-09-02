#!/bin/bash
# 30-step smoke of the compiled sign head + per-step t_head logging on a debug
# GPU node (decoder-scaling study, 2026-09-02): 1x2 DS at (0.4,0), minSR, arms
# operator/greedy, operator/mwpm, operator/tie_sum, model/mwpm. Outputs land in
# logs/smoke_head/ (gitignored); the script greps each run JSON for the new
# t_head / n_head_configs fields and prints the last step's values.
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J hd_smoke
#SBATCH -o logs/%x_%j.out
set -uo pipefail
module load conda
conda activate 2dtc
REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/logs/smoke_head
mkdir -p "$OUT"; cd "$OUT"
LX="${LX:-1}"; LY="${LY:-2}"; STEPS="${SIM_TIME:-0.3}"
run() {
  name=$1; shift
  echo "=== $name  $(date) ==="
  PYTHONPATH=$REPO python "$REPO/main.py" --outindex 0 --jobid "$name" \
    --lattice honeycomb --model ds --Lx "$LX" --Ly "$LY" --hx 0.4 --hy 0.0 --hz 0.0 \
    --optimizer minsr --lr 0.01 --dt 0.01 --diag_shift 6e-5 --sim_time "$STEPS" --seed 0 \
    --kernel_size 2 --n_samples_fin 8192 --use_custom_sampler \
    --channels_noninv 1,16 --channels_inv 16,8,1 --sign_head qec "$@" \
    || { echo "!!! $name FAILED (exit $?)"; return; }
  python - "$OUT/G-equiv_0_${name}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("t_head", "n_head_configs", "energy", "Vscore"):
    v = d.get(k); print(f"  {k}: n={len(v) if v else 0} last={v[-1] if v else None}")
assert d.get("t_head") and float(d["t_head"][-1]) > 0, "t_head missing/zero"
assert int(float(d["n_head_configs"][-1])) > 0, "n_head_configs zero"
print("  OK: head accounting present")
PY
}
run smoke_op_greedy  --sign_impl operator --decoder greedy
run smoke_op_mwpm    --sign_impl operator
run smoke_op_tiesum  --sign_impl operator --decoder tie_sum
run smoke_model_mwpm --sign_impl model
echo "=== smoke done $(date) ==="
