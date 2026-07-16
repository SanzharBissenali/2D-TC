#!/bin/bash
# NQS training sweep on a NERSC Perlmutter GPU node.
# Submit as a job array over the hz sweep:  sbatch --array=0-6 jobs/nersc_nqs.sh
#
# TODO(nersc): set -A <account>, and the conda/env activation for your NERSC setup.
#SBATCH -A mXXXX
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J tc_nqs
#SBATCH -o logs/nqs_%A_%a.out

set -euo pipefail

HZ_VALUES=(0.0 0.05 0.10 0.15 0.20 0.25 0.30)
HX=0.0
LX=4
hz=${HZ_VALUES[$SLURM_ARRAY_TASK_ID]}
jobid=$(printf "L%d_hx%.2f_hz%.2f" "$LX" "$HX" "$hz")

# TODO(nersc): activate your environment, e.g.
# module load conda && conda activate toric

# Combo-small ansatz with the vertex-update custom sampler (paper L=4 defaults).
python main.py \
    --outindex 1 --jobid "$jobid" \
    --Lx "$LX" --hx "$HX" --hy 0.0 --hz "$hz" \
    --dt 0.01 --diag_shift 6e-5 --sim_time 3.5 \
    --architecture Combo \
    --channels_noninv 1,16 --channels_inv 16,8,1 --kernel_size 2 \
    --n_samples_fin 8192 --use_custom_sampler

# main.py writes G-equiv_1_<jobid>.{json,mpack} to the repo root; collect under results/.
mkdir -p results/nqs
mv "G-equiv_1_${jobid}.json" "G-equiv_1_${jobid}.mpack" results/nqs/ 2>/dev/null || true
