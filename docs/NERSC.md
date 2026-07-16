# Running on NERSC (Perlmutter)

Allocation: **`m5340_g`** (GPU-only). Everything charges to `-C gpu` nodes,
including ED (it runs on the GPU node's CPUs — scipy sparse Lanczos).

## One-time setup

### 1. Conda environment (login node)
```bash
module load conda
conda create -n 2dtc python=3.12 -y
conda activate 2dtc
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.2"     # GPU JAX for the NQS job
```
Env lives at `~/.conda/envs/2dtc`. Do `pip install` on a **login node** (compute
nodes have no outbound internet).

### 2. Git access
SSH key (recommended):
```bash
ssh-keygen -t ed25519 -C "nersc-2dtc" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub          # add at https://github.com/settings/keys
ssh -T git@github.com              # expect: Hi SanzharBissenali!
```
Fallback if port 22 is blocked — HTTPS + fine-grained PAT (Contents: read/write):
```bash
git config --global credential.helper store   # then clone https URL, paste PAT as password
```

### 3. Clone + repo-local identity
```bash
git clone git@github.com:SanzharBissenali/2D-TC.git
cd 2D-TC
git config user.name  "SanzharBissenali"
git config user.email "166195693+SanzharBissenali@users.noreply.github.com"
```

### Verify
```bash
python -c "import netket, jax; print(netket.__version__, jax.devices())"   # login: CPU
# GPU check inside an interactive job:
salloc -A m5340_g -C gpu -N1 -G1 -t 10 -q interactive
python -c "import jax; print(jax.devices())"   # expect a CudaDevice
```

## Running jobs

Job scripts live in `jobs/` and are SLURM arrays over the 7-point `hz` sweep
(`hz ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` at fixed `hx=0`, L=4 OBC):
```bash
mkdir -p logs
sbatch --array=0-6 jobs/nersc_ed.sh     # ED benchmark  -> results/ed/
sbatch --array=0-6 jobs/nersc_nqs.sh    # NQS training  -> results/nqs/
squeue --me
```
Both scripts `module load conda && conda activate 2dtc`; edit that line if the env
name changes. Outputs land under `results/`; logs under `logs/` (gitignored).

Memory note: L=4 ED builds the 2^24 sparse operator (~tens of GB). If a node OOMs,
add `--no-observables` (energies only) or reduce `--k` in `jobs/nersc_ed.sh`.

## Result flow (git as source of truth)
Develop locally → push → **on NERSC**: `git pull` → `sbatch` → commit `results/` →
push → pull locally to analyze. Keep the repo as the durable store; `$SCRATCH` is
purged periodically, so if you clone there, re-clone rather than rely on it.
```bash
# after jobs finish, on NERSC:
git add results/ && git commit -m "L=4 hz-sweep results (ED + NQS)" && git push
```

## Compare NQS vs ED
```bash
python scripts/compare.py \
    --nqs results/nqs/G-equiv_1_L4_hx0.00_hz0.10.json \
    --ed  results/ed/ed_L4_hx0.00_hz0.10.json
```
