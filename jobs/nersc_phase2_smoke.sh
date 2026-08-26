#!/bin/bash
# Phase-2 integrated smoke on a debug GPU node: full main.py stack for the
# honeycomb Combo/PlainCNN NQS (geometry -> H -> model -> hexagon sampler ->
# tdvp -> init gate -> observables), FULL 350-step runs.
# Gates: TC/Combo must converge to E = -(V+F) (rel-err <~1e-6; square-lattice
# precedent 1e-8 tail); DS/Combo(real,positive) must run stably and PLATEAU
# ABOVE -(V+F) (Hastings floor -- this run is the first measured failure gap);
# PlainCNN must train stably. Outputs land in logs/smoke_phase2/ (gitignored).
#SBATCH -A m5340_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J p2smoke
#SBATCH -o logs/%x_%j.out

set -uo pipefail
module load conda
conda activate 2dtc

REPO=$SLURM_SUBMIT_DIR
OUT=$REPO/logs/smoke_phase2
mkdir -p "$OUT"
cd "$OUT"

run() {
  name=$1; shift
  echo "=== $name ==="
  PYTHONPATH=$REPO python "$REPO/main.py" --outindex 0 --jobid "$name" \
    --lattice honeycomb --hx 0.0 --hy 0.0 --hz 0.0 \
    --dt 0.01 --diag_shift 6e-5 --kernel_size 2 \
    --n_samples_fin 8192 --use_custom_sampler --sim_time 3.5 "$@" \
    || echo "!!! $name FAILED (exit $?) -- continuing"
}

run smk_tc12 --model tc --Lx 1 --Ly 2 --channels_noninv 1,16 --channels_inv 16,8,1
run smk_tc22 --model tc --Lx 2 --Ly 2 --channels_noninv 1,16 --channels_inv 16,8,1
run smk_ds12 --model ds --Lx 1 --Ly 2 --channels_noninv 1,16 --channels_inv 16,8,1
run smk_pl12 --model tc --Lx 1 --Ly 2 --architecture PlainCNN --channels_noninv 1,32,24,8,2

python - <<'EOF'
import json
import numpy as np
E0 = {"smk_tc12": -12.0, "smk_tc22": -20.0, "smk_ds12": -12.0, "smk_pl12": -12.0}
for name, e0 in E0.items():
    try:
        d = json.load(open(f"G-equiv_0_{name}.json"))
    except FileNotFoundError:
        print(f"GATE {name}: MISSING"); continue
    E = np.array(d["energy"], dtype=float)
    tail = np.median(E[-20:])
    rel = abs(tail - e0) / abs(e0)
    ops = {k: v[-1] for k, v in d.get("order_params", {}).items()}
    print(f"GATE {name}: steps={len(E)} E_tail={tail:.8f} relerr_vs_{e0}={rel:.3e} "
          f"Vscore={d['Vscore'][-1]:.3e} order={ops}")
EOF
echo "=== smoke done ==="
