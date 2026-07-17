# CLAUDE.md

Approximately-symmetric neural quantum states (NQS) for the 2D mixed-field toric
code. Implements the architecture from *Kufel et al., PRL 135, 056702 (2025)*
([arXiv:2405.17541](https://arxiv.org/pdf/2405.17541)). Core claim: an
architecture split into an *exactly symmetric* block and a *non-symmetric* block
captures the toric-code ground state under small/moderate field perturbations.

## Working style
- Keep infra clean, concise, and minimal. Prefer reusing existing modules over
  duplicating logic (e.g. ED reuses the sim's Hamiltonian builder).
- Do not touch `model/`, `simulation/`, `utils/` (upstream paper code) without a
  clear reason; new tooling goes in `exact/`, `scripts/`, `jobs/`.

## Layout
- `main.py` — NQS entry point (TDVP training). Writes `G-equiv_<outindex>_<jobid>.{json,mpack}` to CWD (jobs run it from `results/nqs/` so it doesn't pollute the repo root).
- `model/` — `geometry.py` (lattice/stabilizers), `hamiltonian.py` (Pauli-string H), `networks.py` (Combo & RPP architectures).
- `simulation/` — `optimizer.py` (TDVP), `observables.py`, `custom_sampler.py` (single-flip + vertex-flip MCMC).
- `utils/` — `config.py` (arg parsing, data dict), `io.py` (save/load).
- `exact/lanczos_ed.py` — sparse-Lanczos ED benchmark (reuses `model/` H).
- `scripts/compare.py` — NQS-vs-ED energy comparison.
- `jobs/nersc_{ed,nqs}.sh` — SLURM jobs: each runs the full hz sweep sequentially in one shared-GPU job (skip-if-exists ⇒ resumable).
- `results/{ed,nqs}/` — committed run outputs (git is the source of truth; see Workflow).

## Physics / conventions
- Qubits live on links. OBC qubit count: `N = 2·Lx·Ly − Lx − Ly`. **L=4 → 24 qubits** (Hilbert dim 2^24 ≈ 1.7e7).
- H = `−J·(Σ vertex XXXX + Σ plaquette ZZZZ) − Σ h·(field) − Jbond·(2-qubit)`.
- `dtype` auto-selects `complex` iff `hy≠0` or any `Jy≠0`, else `float64`.
- `hz` sweeps stay diagonal in the computational basis (cheaper/cleaner); `hx`/`hy` do not.
- Combo architecture: non-invariant CNN (identity-initialized ⇒ starts at the exact
  toric ground state) → Wilson nonlinearity → invariant CNN → mean.
- `n_chains` is **overridden at runtime** by device detection (16 on CPU, 1024 on GPU),
  ignoring the `--n_chains` flag — see `utils/config.setup_environment`.
- Training metrics: energy trajectory + `energy_var`, `Vscore` (=`N·Var/⟨E⟩²`), `tau_corr`,
  `Rsplit`, MCMC accept counts are appended to the run JSON **every step** (live-tailable).
  The tqdm bar prints `E ± err | Vscore | s/step` live (`optimizer.py`).
- The nested `12/12` bar during training is the magnetization callback looping over the
  12 bulk qubits; it (and the JSON `order_params` update) fires every 8 steps, which is
  also why the redirected 350-step bar appears to jump in 8s — display only, training is
  every step.
- Gotcha: `config.py` uses `N = 2·Lx·(Lx−1)` while `geometry.py` uses `2·Lx·Ly−Lx−Ly`;
  these agree only for square `Lx=Ly` (always the case here since `Ly:=Lx`).
- Gotcha: `--use_custom_sampler` divides by the number of *bulk* (4-qubit) vertex stars,
  so it crashes for small L with no bulk stars (e.g. L=2). Use the default local sampler there.

## Current work — first experiment array
Validate the architecture at **L=4, OBC** against ED.
- Sweep: `hz ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` at fixed `hx=0` (`dtype=float64`).
- Ansatz: Combo-small (`channels_noninv 1,16`, `channels_inv 16,8,1`, `kernel_size 2`),
  custom sampler on. Optimizer defaults: `dt=0.01`, `sim_time=3.5` (⇒350 steps), `diag_shift=6e-5`.
- Metric: relative energy error `|E_NQS − E_ED|/|E_ED|`; also V-score and magnetizations.
- ED via `nk.exact.lanczos_ed` on the identical H object.
- **Status (2026-07-16):** validated at `hz=0.1` on Perlmutter — `E0_ED=−25.060413`
  vs NQS `−25.06040 ± 3e-5`, `Vscore≈2e-7` (reproduces the paper). Full sweep running
  as shared-GPU jobs. Timings: NQS ~0.58 s/step (~5 min/run on A100); ED ~8 min/point
  with observables (~4 min energies-only).
- Next: `scripts/plot_sweep.py` (energy error + magnetization vs hz) once the sweep lands.

## Current work — phase-transition sweep (L=6,8,10,12)
Detect the topological→trivial transition beyond ED (L≥6 is **NQS-only**: L=6→60
qubits, L=12→264, so no ED ground truth — use finite-size scaling of order params).
- Sweep: `hz ∈ [0.15, 0.50]` step `0.025` = **15 points**, `hx=0` (float64), per L.
  Brackets the expected `h_c ≈ 0.3–0.34` (3D-Ising* single-field transition).
- Ansatz: **same Combo-small as L=4** (fixed, for a clean scaling baseline).
- Steps: **200** (`sim_time=2.0`). L=4 converges to 0.03% energy by ~step 110
  (`scripts/convergence.py`); 350 was ~3× overkill. Caveat: near `h_c` the gap
  closes (~1/L) so those points converge slower at large L — watch the stragglers.
- Observables: BFFM/Wilson (fixed radius 1), magnetization+susceptibility `χ=d⟨σ⟩/dhz`,
  Rényi-2, energy susceptibility `d²E/dhz²`. Wilson/Rényi now run at **L≥6** (was L>6).
- **Measured timings (A100, 200-step projection):** s/step ≈ 0.58/2.0/6.7/20.4/57 for
  L=4/6/8/10/12 (≈`N^2.7`). Per point: L6 ~7 min, L8 ~22 min, L10 ~68 min, L12 ~3.2 h.
- Jobs: `jobs/nersc_sweep.sh` — one array script, chunking driven by `--array` size:
  L6 `--array=0-1` (8/7 pts, `-t 1:30`), L8 `--array=0-2` (5/5/5, `-t 3:00`),
  L10/L12 `--array=0-14` (1 pt/task, `-t 2:00` / `-t 5:00`). ~72 GPU-h total.
- Analysis TODO: `scripts/analyze_sweep.py` (order params + susceptibilities vs hz
  across L; extract `h_c` from crossings / χ-peaks) once data lands.
- Gotcha: `main.py` writes the run `.json` **every step**, so a walltime kill leaves a
  partial-but-existing file. Sweep jobs skip only if **complete** (`scripts/is_complete.py`:
  `.mpack` present ⇒ training done; L≥6 Wilson+Rényi order-params non-empty ⇒ observables
  done) — never bare file-existence, or half-finished points get skipped forever.
- Open before submit: confirm the `shared_gp` QOS max walltime allows `-t 5:00` (L=12).

## Cluster automation & safeguard (IMPORTANT)
Cluster access is **already configured** — Claude drives NERSC directly via
`scripts/cluster.sh` (SSH over an sshproxy 24h cert; connection settings in the
gitignored `scripts/cluster/config.local.sh`, user `sanzharb`; see docs/NERSC.md).
At session start, just run `bash scripts/cluster.sh status` to connect. If it
fails with a "no NERSC key / cert expired" message, the daily cert lapsed — ask
the user to run `sshproxy -u sanzharb` (only they can; it needs their password +
MFA), then retry. Monitoring/fetching is frictionless; **launching compute is gated**.

**Before ever running `submit` (or any new cluster compute), consult the user and
get explicit approval.** Present a short job spec first:
- **Experiment** — what physics/sweep this run does and why now.
- **Resources** — the `#SBATCH` request (queue, `-N/-G/-c`, node fraction).
- **Walltime** — `-t` and the expected runtime, with headroom rationale.
- **Why** — what question the result answers / what it unblocks.

`submit` and `cancel` are deliberately **not** in the permission allowlist, so they
always prompt too — the consult (chat) and the prompt (Claude Code) are both
required. Read-only ops (`status`/`sync`/`logs`/`fetch`) are allowlisted; `fetch`
brings results back (commit+push on cluster, pull locally) and is safe to run
freely. Never allowlist `submit`/`cancel`.

## Commands
```bash
# ED benchmark (one hz point)
python -m exact.lanczos_ed --Lx 4 --hx 0.0 --hz 0.10 --k 4 --out results/ed/ed_L4_hx0.00_hz0.10.json

# NQS training (see jobs/nersc_nqs.sh for the full flag set)
python main.py --outindex 1 --jobid L4_hx0.00_hz0.10 --Lx 4 --hx 0.0 --hy 0.0 --hz 0.10 \
    --dt 0.01 --diag_shift 6e-5 --channels_noninv 1,16 --channels_inv 16,8,1 \
    --kernel_size 2 --n_samples_fin 8192 --use_custom_sampler

# Compare
python scripts/compare.py --nqs results/nqs/G-equiv_1_L4_hx0.00_hz0.10.json --ed results/ed/ed_L4_hx0.00_hz0.10.json

# Live-monitor a running NQS job (energy in the log; full metrics in the JSON)
tail -f logs/nqs_<jobid>.out
```

Cluster runs (Claude-driven, gated): `bash scripts/cluster.sh submit jobs/nersc_ed.sh`
or `... jobs/nersc_nqs.sh`. Monitor with `... status` / `... logs <pat>`, retrieve
with `... fetch`. See docs/NERSC.md.

## Environment
- Pinned in `requirements.txt`: `netket==3.16.1.post1`, `jax==0.5.2`, `flax==0.10.4`.
- **Not installed locally** on this laptop — runs on NERSC. L≤3 ED is laptop-feasible if netket is installed.
- L=4 ED peaks ~10 GB and runs on the GPU node's CPUs (multithreaded BLAS: `user`≫`real`
  in `time` output is expected). `ed_time_s` in the JSON is the diagonalization only;
  the per-site observables (48 sparse builds on 2^24) add ~half the wall time — use
  `--no-observables` for a fast energies-only check.
- NERSC allocation is `m5340_g` (GPU-only), so everything charges to `-C gpu` nodes.
- Pipeline validated locally at L=2 (ED→NQS→compare) with a throwaway venv:
  `python3 -m venv <scratch>/venv && <venv>/bin/pip install -r requirements.txt`.
  (L=2 must use the default sampler, not `--use_custom_sampler`.)

## Workflow (NERSC)
Develop locally → commit/push to GitHub → Claude drives the cluster over
`scripts/cluster.sh`: `sync` (pull code on NERSC) → `submit` a job (gated —
consult first) → `status`/`logs` to monitor → `fetch` (commit+push `results/` on
NERSC, then pull locally) → analyze. Git stays the source of truth. **Full setup
and run instructions: [`docs/NERSC.md`](docs/NERSC.md)** (allocation `m5340_g`,
conda env `2dtc`, sshproxy, job submission). Env `2dtc` working as of 2026-07-16.

Key NERSC gotchas learned:
- Use `-q shared` (partition `shared_gp`) for these 1-GPU / CPU-bound jobs — charged
  ~¼ node vs a whole exclusive node under `-q regular` (`gpu_ss11`). ~4× cheaper.
- The shared GPU queue requires **exactly `-c 32` per GPU** and grants ~55 GB with it.
  **Do NOT set `--mem`** — it's converted to a core count and blows past 32 → submit error.
- ED still needs `-G 1` even though it never touches the GPU (GPU-only allocation rule).
- Do `pip install` on a login node (compute nodes have no internet). `$SCRATCH` is purged
  periodically → git is the durable store, re-clone if needed.
