#!/bin/bash
# Interactive timing benchmark for the NQS TDVP run.
#
# Runs a handful of TDVP steps at each L (6,8,10,12) with the production
# ansatz (Combo-small, custom sampler), so we can measure the steady-state
# per-step cost and extrapolate the full 350-step (sim_time=3.5) run before
# committing an expensive sweep. Also exercises the end-of-training observable
# block (magnetization + Wilson + Renyi), so the reported wall includes the
# one-time observable cost each production point pays.
#
# Run INSIDE an interactive single-GPU allocation:
#   salloc -A m5340_g -C gpu -q interactive -N 1 -G 1 -t 60:00
#   module load conda && conda activate 2dtc
#   bash scripts/bench.sh                # 5 steps/L (default)
#   STEPS=10 LVALUES="6 8" bash scripts/bench.sh    # override
#
# What to record for each L:
#   (a) steady-state s/step  -> read off the live tqdm bar, IGNORING step 0
#       (step 0 includes the one-time JAX compile; steps 1..N-1 are steady state).
#   (b) total wall           -> the "wall=<sec>s" line printed below per L.
# Then, per L:  full_run_s  ~=  (wall - STEPS * s_step)  +  350 * s_step
#            =  compile+overhead+observables  +  350 * s_step
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STEPS=${STEPS:-5}
LVALUES=${LVALUES:-"6 8 10 12"}
SIM_TIME=$(awk "BEGIN{print $STEPS*0.01}")

WORK=$(mktemp -d)            # throwaway CWD: keeps G-equiv_*.{json,mpack} out of the repo
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

printf '%-4s %-6s %-8s %-s\n' "L" "N" "wall_s" "start"
for L in $LVALUES; do
    N=$((2*L*(L-1)))
    echo "================ L=$L  N=$N  steps=$STEPS  start=$(date +%T) ================"
    SECONDS=0
    PYTHONPATH="$REPO" python "$REPO/main.py" \
        --outindex 0 --jobid "bench_L${L}" \
        --Lx "$L" --hx 0.0 --hy 0.0 --hz 0.25 \
        --dt 0.01 --sim_time "$SIM_TIME" --diag_shift 6e-5 \
        --architecture Combo --channels_noninv 1,16 --channels_inv 16,8,1 \
        --kernel_size 2 --n_samples_fin 8192 --use_custom_sampler
    echo "================ L=$L  DONE  wall=${SECONDS}s ================"
done

echo
echo "Done. For each L note the steady s/step from the bar (skip step 0) and the wall above."
echo "Full 350-step run per L  ~=  (wall - ${STEPS}*s_step) + 350*s_step"
