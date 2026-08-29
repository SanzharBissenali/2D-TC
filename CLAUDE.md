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
- **Measured timings (A100):** s/step ≈ 0.58/2.0/6.7/20.4/57 for L=4/6/8/10/12 (≈`N^2.7`).
  Per point @200 steps: L6 ~7 min, L8 ~22 min, L10 ~68 min, L12 ~3.2 h; all fit walltime.
- Jobs: `jobs/nersc_sweep.sh` — one array script, `--array` size sets chunking:
  L6 `-t1:30 --array=0-1`, L8 `-t3:00 --array=0-2`, L10 `-t2:00`/L12 `-t5:00` (`--array=0-14`).
  `shared_gp` allows `-t5:00`. Skip-if-**complete** via `scripts/is_complete.py` (`.mpack`
  ⇒ training done; L≥6 Wilson+Rényi order-params non-empty ⇒ observables done) — never bare
  file-existence, since `main.py` writes the `.json` every step (partial file would look done).
- **Status (2026-07-17): sweep COMPLETE**, all 4×15 points in `results/nqs/`. Vscore ~1e-6
  deep in the phase, rising to ~1e-3 near hz=0.5 (criticality). **2 points DIVERGED**
  (L8/hz0.425, L10/hz0.35): NaN at step ~4 but they still wrote `.mpack`+observables, so
  `is_complete` marks them "done" (skipped on resubmit) — excluded in analysis.
- Analysis in `analysis/` (json+numpy+matplotlib, **no netket**; figures→`figures/`, gitignored;
  validate cells headless before committing): `01_L4_vs_ED` (rel-err validates the paper's
  1e-6–1e-7); `02_sweep_observables` (learning curves; ⟨σᶻ⟩/⟨W_X⟩; S2 & BFFM ρ_Z + their
  derivatives, 2×2); `03_fss` (tanh & Richards h_c(L) fits → inverse-power extrapolation).
  Helpers: `scripts/convergence.py` (plateau step), `scripts/sweep_summary.py` (per-L report).
- **FSS first pass:** h_c(L) from sigmoid inflections drift 0.44→0.40 with L; fixed
  x=1/ν≈1.59 (3D-Ising) ⇒ h_c(∞)≈0.375–0.384 (spread over BFFM/Rényi × tanh/Richards).
  Theory is `h_c=0.328473(2)`; the ~0.05 gap is finite-size/estimator bias, **not** fit
  freedom. **Free x is degenerate with 4 sizes** (corr(b,x)≈1, x error >100%); honest fixes:
  crossings of dimensionless ratios (vs inflections), a subleading `L^{-x'}` term, or more L.
- **Physics gotcha (hz sweep):** the responsive Wilson loop is `⟨W_X⟩ = ∏σˣ` (product of
  vertex/star ops, disordered by hz, drops 1→0); `⟨W_Z⟩ = ∏σᶻ = ∏plaquettes ≡ 1` (commutes
  with hz — flat, no signal). The **BFFM string order parameter uses the σᶻ open/closed loop**
  (`WilsonBFFM[6]`), rising 0→~0.7 across the transition. (`WilsonBFFM` = [X_mean, X_std,
  X_BFFM, X_BFFM_std, Z_mean, Z_std, Z_BFFM, Z_BFFM_std].)
- Tooling gotcha: `cluster.sh fetch` now commits→**rebase→push** (was `add && commit && push`,
  which stranded results when `commit` found nothing new, or when origin had diverged from
  code pushed off-cluster).

## Current work — transformer Block-3 experiment (branch `transformer-symmetric-block`)
Controlled swap: **replace ONLY Block 3** (the global-kernel invariant CNN + `Final` mean)
with a **factored-attention transformer**, keeping Block 1 (non-inv CNN) + the Wilson
nonlinearity + sampler/optimizer/seed byte-identical. Kept on its own branch for rollback.
- **New module `model/transformer_block.py`:** `TransformerSymmetric` = embed (C=16→d) →
  n_l pre-LN encoder blocks (factored MHA + 2-layer FFN, residuals) → sum-pool → log-cosh
  readout. Factored attention `A_i=Σ_j α_{Δ(i,j)} V x_j` with REAL, position-only `α`.
- **OBC α indexing (critical):** plaquettes live on an `(L-1)×(L-1)` OPEN grid, so `α` is
  indexed by plain signed displacement via `plaquette_displacement_table` — a dense
  `(2L-3)²`/head table, **no `jnp.roll`/modular wrap** (the NetKet ViT tutorial's roll2d is
  PBC-only). No masking needed (every plaquette pair has an in-range displacement).
- **Exact symmetry is preserved for free:** the stabilizer/gauge invariance comes from the
  *retained* Wilson nonlinearity; any function of its output (CNN or transformer) inherits it.
- **This is a parameterization-efficiency test, not expressivity:** factored attention =
  a full-kernel conv (α) + a 1×1 conv (V) (arXiv:2503.10462), i.e. the same operator class as
  the CNN it replaces. Expect equal ED-accuracy at fewer params; the param gap widens with L
  (TF/CNN block-3 ≈ 1.00 at L=4 → 0.21 at L=10, since CNN grows as `136(L-1)²` but the α table
  only as `(2L-3)²`).
- **Selector:** `--symmetric_block {cnn,transformer}` (default `cnn`, byte-identical to before);
  `--tf_layers/--tf_dmodel/--tf_heads/--tf_ffn_mult/--tf_activation/--tf_readout_K/
  --tf_complex_output`; `--seed` (now threaded into `MCState` for paired runs). `tf_dmodel`
  must be divisible by `tf_heads` (asserted in `create_model`).
- **Arms:** primary `n_l=2,d=8,h=2,FFN 2d,ReLU` (~1236 params, +0.2% vs CNN's 1233 @L4);
  bonus `n_l=2,d=6,h=2,FFN 4d,GELU` (~1108). Kill criteria: L=4 hz∈{0.15,0.30} rel-err ≤1e-5
  at BOTH (ED in `results/ed/`, hz=0.15 E0=−25.13713); L=6 wall-clock-to-1e-5 ratio TF/CNN >3
  ⇒ abandon. Job: `jobs/nersc_transformer.sh` (paired cnn/tf/tfg, seed 0).
- **Instrumentation:** per-step wall-clock split `t_sample`/`t_grad`/`t_sr` (JSON; `optimizer.py`
  uses `jax.block_until_ready` to defeat async dispatch); `⟨B_p⟩` mean/std diagnostic
  (`observables.calculate_plaquette_stabilizer`, must stay ≈1 — contamination check).
- **Optimizer note:** the repo's SR is a hand-rolled **dense P×P** QGT solve (`optimizer.py`),
  so fewer params give a *super-linear* solve speedup ⇒ param-matching both arms also matches
  solve cost. A minSR swap (`VMC_SRt` in the pinned netket 3.16.x; renamed `VMC_SR(use_ntk=True)`
  in LATER netkets — 3.16 does NOT have `VMC_SR`, verified against the wheel) is
  P-independent and would change the CNN baseline too ⇒ deliberately out of scope (both-or-neither).
- **Local verify done:** all changed files `py_compile`-clean; `scratchpad/validate_tf.py`
  (numpy-only) confirms the displacement table (no wrap, correct decode/range) and param counts
  vs closed form. netket/jax NOT installed locally ⇒ end-to-end forward pass runs on NERSC.
- **Validated on GPU (2026-07-17):** transformer trains end-to-end at L=4 (interactive salloc):
  1684 total params (Block1 448 + TF Block3 1236), energy descends cleanly, **~1.3 s/step**
  steady-state (vs CNN ~0.58 @L4; first step ~29 s = JIT). `jobs/nersc_transformer.sh` is now
  **env-parameterized** (`--export=ALL,ARMS=…,LX=…,HZ_LIST=…,SIM_TIME=…`) so one script covers the
  per-arm / per-L jobs; keep them **short** — a 4 h job stalled in PENDING, 1 h jobs backfill fast.
  Live run: L=4 rel-err at hz∈{0.10,0.15,0.20,0.25,0.30} as **three 1 h per-arm jobs** (cnn/tf/tfg)
  + **one 1 h L=6 hz=0.15 all-arms job**. Analyze with `scripts/erel_table.py` after
  `cluster.sh fetch` (kill criterion: L=4 rel-err ≤1e-5). Shrink a PENDING job's walltime in place
  via `scontrol update JobId=<id> TimeLimit=…` (raw ssh; no re-queue, keeps job id).
- **y-field (hy≠0) = future work; validated scope is the sign-free hz cut (all-real).** `config.py`
  auto-switches `dtype='complex'`; as written that makes the WHOLE transformer complex (like the CNN
  arm), and since Block 1 + Wilson upstream also go complex the transformer input is already complex.
  The `--tf_complex_output` flag is built for the Viteritti real-deep + complex-shallow readout
  (`out = LN(Dense_re z) + 1j·LN(Dense_im z)`), but using it needs: decouple encoder-vs-readout dtype
  in `TransformerSymmetric` (currently one `dtype`), a complex-safe norm (LayerNorm on complex is the
  gotcha — prefer the arXiv:2503.10462 fixed-scale norm), and SR with `mode='complex'` / non-holo QGT.

## Current work — hy-field first-order transition sweep (CNN, L=4/6/8)
Pure-Y-field cut (`hx=hz=0`), `hy ∈ {0.80..1.20}` step 0.05 (9 pts), **CNN only** (the
transformer's complex path is still future work — see above). Goal: observe the 1st-order
transition (theory `h_c=1.0` in the thermodynamic limit; shifts at finite L). Tooling (new,
this session): `jobs/nersc_hy_sweep.sh` (arm-dispatch + array-chunk, `ARMS` default `cnn`),
`jobs/nersc_hy_ed.sh` (L=4 ED companion), `scripts/hy_summary.py` (per-(L,arm) E/⟨σʸ⟩/⟨B_p⟩
table, locates the ⟨σʸ⟩ jump, skips no-`.mpack` partials), `analysis/04_hy_transition.ipynb`
(⟨σʸ⟩/⟨B_p⟩ vs hy per L + d⟨σʸ⟩/dhy; **convergence-based** loader — keeps runs whose energy has
plateaued (last-20 rel-std < 5e-3), so walltime-killed-but-converged partials count and NaN/early
runs don't), plus `⟨σʸ⟩` added to `exact/lanczos_ed.py`.
- **RESULT (2026-07-18): 1st-order transition OBSERVED; sweep DONE for L4 (9/9) & L6 (9/9).**
  L6 is the cleanest: ⟨σʸ⟩ **jumps 0.24→0.69→0.89** and ⟨B_p⟩ **collapses 0.82→0.46→0.28** across
  hy=1.00/1.05/1.10 ⇒ `h_c(L6)≈1.02–1.03` (steepest step 1.00→1.05); matching energy slope-kink.
  **L4 is a broad, rounded crossover** (⟨σʸ⟩ 0.15→0.87 over 0.80→1.20, no sharp jump — finite-size
  rounding). Finite-size trend: **L4 rounded → L6 sharp**, marching toward a true discontinuity.
  Transition sits **slightly ABOVE 1.0**, not below — consistent with metastability/hysteresis
  (identity-init starts topological, so the ordered phase persists to higher hy). Window brackets it.
- **L8 = topological side only (5/9).** hy=0.80–1.00 converged (⟨σʸ⟩ 0.14→0.30, ⟨B_p⟩ 0.93→0.75,
  energy plateaued <0.1% — usable via the convergence filter even without `.mpack`). **hy=1.05–1.20
  DIVERGED to NaN at step ~4** (SR instability at large L *past* the transition, where the disordered
  GS is far from the topological init; `diag_shift=6e-5` too small there). Those NaN runs still wrote
  `.mpack` ⇒ `is_complete` falsely skips them (delete the 4 `.mpack`+`.json` to force rerun). **TODO
  to finish L8's high-hy side:** raise `diag_shift` (→~1e-3), maybe halve `dt`, resubmit at `-t 5:00`.
- **Gotcha — complex JIT dominates runtime.** `hy≠0 ⇒ dtype=complex` (`config.py`), non-holomorphic
  QGT. XLA constant-folding compile is huge and scales with L: **~18 min @L4, ~80 min @L6/L8**, and
  it is paid PER hy POINT because each point is a fresh `python main.py` process. Steady stepping is
  cheap (L4 ~1.16 s/step) so JIT is ~80% of a point's wall time. ⇒ at the 1:30 walltime L4 does ~4
  pts/dispatch, L6 ~2, L8 ~1. **Fix (TODO, deferred): loop the hy list INSIDE one process** to
  amortize JIT (turns ~25–30 GPU-h of recompute into ~3–4). Resubmit-to-continue works meanwhile
  (skip-if-complete). Use `-t 3:00`/`-t 5:00` for continuations, 1-pt-per-task arrays for L8.
- **Gotcha — complex L=4 ED OOMs on the shared node.** At 2^24 the complex sparse H alone (dense
  off-diagonal X/Y terms, complex128, netket Pauli→sparse intermediates) exceeds ~55 GB and OOM-kills
  (exit 137) **even with `--no-observables`**. Needs `-q regular` (~256 GB exclusive node). ED is
  confirmatory only — the transition is clear from NQS ⟨σʸ⟩/⟨B_p⟩ alone, and the complex CNN energies
  are internally consistent (monotonic in hy, ⟨B_p⟩≈0.93 in the ordered phase), so the `ComplexWarning`
  (discarded-imag in the Wilson-nonlinearity VJP, `networks.py:261`) is very likely benign.

## Current work — fermionic (dyon) field transition (branch `fermionic-perturbation`, L=4 Colab)
New perturbation `H = H_TC − h_f·Σ S_e`, `S_e = X_a·Z_b` = the L-shaped two-body operator
(figure `fermionic-excitation-geo.png`). `Z` moves an `e` (vertex charge, anticommutes with
`A_v=XXXX`), `X` moves an `m` (plaquette flux, anticommutes with `B_p=ZZZZ`); the **product binds
e+m into the composite fermion ε and hops it** — physically distinct from an equal single-site `X+Z`
sum (that condenses e,m *independently*; the product proliferates the *bound* dyon).
- **Operator list (`geometry._generate_fermion_pairs`):** for each link `j`, partner = `j+[½,½]`
  (same offset as `_generate_bonds`); rule is UNIFORM **Z on source `j`, X on partner**. Reproduces
  both figure orientations (row family Z-horizontal/X-vertical + column family Z-vertical/X-horizontal),
  keyed by the unique Z link ⇒ no double-count. **L=4 ⇒ 18 terms** (9+9; 6 dropped at the OBC boundary).
- **Plumbing (mirrors `hz`/`Jbond`/`⟨B_p⟩`):** `--h_f` (config.py) → term in `hamiltonian.py` →
  `⟨S_e⟩` (`Se_mean/std`) + `⟨A_v⟩` (`Av_mean/std`, XXXX, e-diagnostic to complement `⟨B_p⟩`) in
  `observables.py`, all wired end-of-run in `main.py`. Also **fixed `calculate_renyi_entropy` at L=4**
  (empty-`arange` fallback to a single central placement; was why Rényi was gated to `Lx≥6`).
- **SIGN PROBLEM (key gotcha):** `X·Z` has no σʸ so **H is real, but it is NOT sign-free/stoquastic.**
  Small-L ED (`scratchpad/sign_check.py`, numpy/scipy) shows the exact GS is **signful (~46% negative
  amplitudes for any `h_f>0`)** — that sign structure *is* ε's fermionic statistics. The `float64`
  Combo gives a strictly **positive** amplitude (`Final=mean`, real logψ) and **cannot represent it**.
  ⇒ `config.py` now auto-sets **`dtype='complex'` when `h_f≠0`** (unless the complex-readout transformer
  is used), i.e. the same complex-CNN path as the `hy` cut. The `h_f=0` anchor stays the cheap real CNN
  (E≈−25.0, ⟨A_v⟩≈⟨B_p⟩≈1, ⟨S_e⟩≈0). Chosen arm: **complex CNN** (conservative baseline over the
  faster complex-readout transformer).
- **Run:** `colab/fermionic_transition_L4.ipynb` — subprocess-`main.py` driver, **one cell per
  `h_f`∈{0,…,1.0}**, saves `.mpack`+observables to Drive (skip-if-complete), final collect/plot cell
  (E, ⟨S_e⟩ + `d⟨S_e⟩/dh_f`, ⟨A_v⟩/⟨B_p⟩, Rényi-2). Complex JIT ~15–20 min/point at L4 ⇒ ~2–3 Colab
  sessions. High-`h_f` may hit SR instability (as on `hy`) ⇒ re-run `run_hf(hf, diag_shift=1e-3)`.
- **Status (2026-07-27):** code done + locally verified (`py_compile`; 18-pair + sign checks pass);
  NERSC down (~10-day outage) ⇒ training on Colab. Sweep not yet run.

## Current work — Variant 3: pure plaquette transformer (branch `variant3-plaquette-transformer`)
Standalone exactly-A_v-invariant ansatz (spec "Variant 3"): tokens are the CLASSICAL stabilizer
values `t_p = B_p(s) = ∏₄ sᵢ ∈ {±1}` (change of variables ⇒ invariance by construction, no
architectural constraint downstream), 2-entry lookup embedding (2d params, no positional
encoding), then a **full input-dependent** transformer — unlike v1/v2's factored (content-free)
attention. Attention per head: `A_ij = softmax_j(b_h(Δ_ij) + α_h·q·k/√d_head)` with `b_h` a
T5-style learned relative-position bias over the same `(2L−3)²` OBC displacement table, and
`α_h` a per-head content gate **init 0** (init = pure position attention; trained α_h per
layer/head = the content-routing diagnostic, dumped to `*_attn.json` via `dump_attention`).
- **Physics scope:** ONLY the pure-`h_x` cut (`h_z`/`h_y` anticommute with `A_v = XXXX`, so the
  enforced symmetry would be wrong there — that needs the Variant-1 cleaning block, future work).
  `h_x=0` is trivial (all tokens +1 ⇒ ψ constant); benchmark point is **`h_x=0.2, h_z=0`, L=4**
  vs a new ED point `results/ed/ed_L4_hx0.20_hz0.00.json`. Sign-free ⇒ `float64`, real head.
- **Code:** `model/plaquette_transformer.py` (`GatedMHA_OBC`/`EncoderBlockV3`/`PlaquetteTransformer`;
  bias-free everywhere, pre-**RMSNorm** — no LayerNorm; GELU FFN; sum-pool → RMSNorm → Dense_K →
  Σ log-cosh). Selector `--symmetric_block plaquette_transformer` (early branch in `create_model`,
  no Block-1/Wilson/Final); `--tf_content` (BooleanOptionalAction, default on) — `--no-tf_content`
  freezes `α_h≡0` (stop-gradient) ⇒ normalized-factored ablation arm. `--tf_activation` now also
  accepts `tanh`. `main.py`'s `check_Av_invariance` gate + `dump_attention` extended to this arm
  (for Variant 3 the A_v check must pass at ANY params, not just init — full correctness gate).
- **Default arm:** `d=32, 4 layers, 4 heads, FFN 2d` ⇒ **34,560 params** (vs CNN 1,233!). The
  hand-rolled DENSE P×P SR solve would need ~9.5 GB QGT + O(P³)/step at P≈35k ⇒ **minSR added**:
  `--optimizer minsr` → `simulation/optimizer.run_minsr` = NetKet `VMC_SRt` (the 3.16.x minSR
  driver; falls back to the later-netket `VMC_SR(use_ntk=True)` name if SRt is absent) (kernel
  trick, `N_samples×N_samples` solve — bottleneck is the sample budget, not P). Same per-step
  JSON/tqdm/W&B logging as `run_tdvp` (t_sample/t_grad/t_sr not separable inside the driver ⇒ 0;
  `step_time` carries the total); logs the v3 content gates `alpha/<block>_h<i>` to W&B every 8
  steps. New knobs: `--lr` (minsr lr; 0 ⇒ reuse `--dt`), `--n_steps` (0 ⇒ `sim_time/dt`; overrides
  `sim_time` for BOTH optimizer paths, coupled in config-finalize). Default `tdvp` byte-identical.
- **Job:** `jobs/nersc_v3.sh` (env-param like nersc_transformer.sh; arms `v3`/`v3f`(frozen-α)/`v3s`
  (d=16 2-layer fallback); runs the missing L=4 h_x ED companion first — note it predates the minSR
  flags, add `--optimizer minsr` when NERSC is back). Local verify: `scratchpad/validate_v3.py`
  (numpy-only: geometry, tokenization==B_p, exact A_v invariance of tokens, displacement decode,
  param count 34,560, gated-attention properties) — ALL PASS + `py_compile` clean.
- **Colab driver:** `colab/variant3_L4.ipynb` (NERSC outage workaround, fermionic-notebook pattern):
  dedicated Drive clone of this branch, ONE hyperparameter cell (`HP` dict: steps/lr/diag_shift/
  layers/dmodel/heads/act/tag…), `run_v3(arm, content=, force=, **HP-overrides)` streams the live
  per-step `E ± err | Vscore | s/step` (JSON also appended every step ⇒ tail-able elsewhere),
  arms `v3`+`v3f`, optional ED cell (~10 GB RAM ⇒ high-RAM runtime), collect cell (learning curves,
  rel-err vs ED, trained α_h per layer/head). **W&B ONLINE** into project **`2d-tc-transformer`**
  (group `variant3-L4`; Colab has internet — `WANDB_MODE=online` before `wandb_logger`'s
  offline-setdefault). Completion marker = `.mpack` + `*_attn.json` (attn dump is written last).
- **Gotcha — Colab GPU OOM was the NOTEBOOK KERNEL, not the model.** Importing jax in a driver
  notebook preallocates **75% of the GPU** at backend init, so the training SUBPROCESS got only
  ~5 GB of an L4's 22 GB (d=32 OOM'd; d=16 "barely fit"). Fix (in the notebook): set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` BEFORE `import jax` in any cell that imports it, and give
  the subprocess `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`. Extra memory knobs: HP `CHUNK_SIZE`
  (netket chunking; `--chunk_size` → `MCState`, honored by `VMC_SRt`'s jacobian) and `--tf_remat`
  (`nn.remat` per encoder block — recompute activations in backward; for big d/L).
- **Sweep results (2026-07-28, overnight Colab, L=4 hx=0.2, 350 steps, minSR):** best arm =
  **d=16, 4 layers, 4 heads (d_head 4), GELU, lr 0.01, ds 1e-4** → V-score ~1e-5. d=32: slower
  convergence, similar final E/V-score (more SR directions from the same 8192 samples, no needed
  expressivity — 9 tokens, 2-word vocab). **nl=8 degrades badly** (optimization: 9 tokens mix in
  1–2 global layers; depth may pay only at larger L / near criticality). Factored (frozen-α) ≈
  full at d32/nl4 but much worse at nl8 ⇒ the gate helps optimization, not (yet) routing. lr
  0.0033 = just 3× slower. **Default arm is now d16** (notebook HP updated); escalation knob:
  d_model×2. TODO: rel-err vs ED (run the ED cell — kill criterion is ED-based, not V-score),
  seed replication of the top configs, cosine-schedule run (~500–700 steps) to chase the CNN's
  1e-6–1e-7 V-score, and check the trained α_h gates (≈0 ⇒ no content routing at the symmetric
  point, as theory predicts).

## Current work — Variant 1: gauge-combo transformer (same branch as Variant 3)
The spec's Variant 1, built on the validated Variant-3 blocks (NOT the old v2 `full_transformer`,
which is its factored-attention cousin with LayerNorm/GELU/biases and an unresolved V-score floor):
`σ → odd embed σ_e·w_orient (bias-free) → n1 × χ blocks (gated FULL attention over the
orientation-resolved edge-displacement table, zero-init W_O+ffn1 ⇒ IDENTITY at init, tanh FFN,
pre-RMSNorm — RMSNorm is odd, LayerNorm isn't) → fixed Wilson ∏₄tanh(·) channel-wise →
Dense(C→d) → n2 × EncoderBlockV3 (Variant-3 backbone) → sum-pool → RMSNorm → Dense_K → Σlog-cosh`.
A_v exact AT INIT (= Variant 3 with a linear embedding: every channel `B_p×const`); training
breaks it by the learned χ dressing — that's how `h_z` (anticommutes with A_v) is captured.
- **Code:** `model/gauge_combo_transformer.py` (`GaugeComboTransformer`); `zero_init_out` option
  added to `GatedMHA_OBC`/`EncoderBlockV3`. Selector `--symmetric_block variant1`; χ knobs
  `--tf1_layers/--tf1_dmodel/--tf1_heads` (Ω reuses the `--tf_*` flags). main.py A_v gate (init-
  only for this arm) + `dump_attention` cover it; optimizer `_WRAP` += GaugeComboTransformer.
  Defaults: χ C=8/1 block/2 heads + Ω d16/nl4/h4 ⇒ **9,966 params**.
- **Benchmark:** `(h_x,h_z)=(0.2,0.2)`, L=4, vs new ED `ed_L4_hx0.20_hz0.20.json` (the spec's
  ablation-ladder point). `colab/variant1_L4.ipynb`: **ED runs FIRST**, then arms `v1` (gated) /
  `v1f` (frozen-α) / `cnn` (Combo-small baseline, native tdvp + ds 6e-5 — optimizer differs from
  the minSR transformer arms ⇒ compare converged quality, not s/step), all into W&B project
  `2d-tc-transformer` group `variant1-L4`. Watch χ's α gates (`chi0…`) vs Ω's (`block0…`): with
  h_z≠0, χ is where content routing should first switch on (syndrome-matching claim).
- **Local verify:** `scratchpad/validate_v1gc.py` (numpy-only: init factorization t=B_p×const per
  channel, exact A_v invariance over all 16 stars, edge-table `n_disp=16L²−32L+14=142`, param
  count 9,966) — ALL PASS + `py_compile` clean.
- **First result (2026-07-28): cold-start v1 at (0.2,0.2) is BAD — V-score ~1e-2** (vs Variant 3's
  1e-5 at the symmetric point); suspicion: χ and Ω scramble each other early, and/or the tanh-
  product Wilson adapter. **Curriculum rescue implemented** (the spec's quasi-adiabatic loop):
  `--freeze_chi` (stop-gradient ⇒ χ pinned at identity, zero SR update, param tree unchanged) +
  `--init_params <mpack>` (warm start; A_v gate becomes print-only — a trained χ legitimately
  breaks A_v). Notebook curriculum cell: `v1cA` = 250 steps at (0.2,0) χ-frozen (≡ Variant 3
  through the V1 parameterization) → `v1cB` = 250 steps at (0.2,0.2) warm-started. Compare v1cB
  vs cold v1 vs cnn in W&B. ED at (0.2,0.2) on Colab: use `--no-observables` + High-RAM (a 53 GB
  runtime got OOM-killed WITH observables — netket Pauli→sparse intermediates + the 48-op
  observable pass stack up; energies-only fits).

## Current work — dual-basis (star-Wilson) Combo CNN (branch `variant3-plaquette-transformer`)
e↔m duality experiment on the **CNN** (user verdict: CNN >> every transformer arm — default new
experiments to the Combo CNN). `--dual_basis` Hadamard-conjugates H (σx↔σz alias inside
`create_hamiltonian`: stars→Z-products, plaquettes→X-products, hx→σᶻ field, hz→σˣ; **same
spectrum** ⇒ ED files/CLI unchanged) and swaps the Combo's symmetric machinery: Wilson
nonlinearity takes **masked products over vertex stars** (`vertex_all` has −1 sentinels — 12 of 16
L=4 stars are 2/3-link boundary stars; `_Wilson_4spin_plaq` grows a static-branch mask, primal
graph untouched) and the invariant CNN runs on the **L×L star grid** (`KernelManager(dual=True,
dg_v=…, vertex_all=…)` builds `kernel_shifts_CNN_v`, global kernel = Lx taps — the config's
`kernel_size_inv=Lx−1` stays plaquette-only; `CNN_invariant(grid='vertex')`). The
exactly-embedded-at-init symmetry becomes **B_p** (plaquette flips preserve star products;
Block-1's scaled sigmoid maps ±1→±1 EXACTLY at identity init) — enforced by a new init-only
`check_Bp_invariance` gate in `main.py`. Valid standalone cut = pure-hz (dual of pure-hx).
- **Physics recap (why):** in the z-basis A_v is the architectural symmetry and B_p a learned
  sector constraint; sampling in the x-basis (≡ conjugated H in the computational basis) swaps
  the roles exactly. Params: primal 1,681 vs dual 2,633 (16-tap vs 9-tap global kernel — OBC
  is not self-dual: 16 stars vs 9 plaquettes, boundary types swap).
- **Not self-dual ⇒ asserted off in dual mode:** σy→−σy and h_f: X_a·Z_b→Z_a·X_b (hy/Jy_*/h_f
  guarded in `create_hamiltonian`); Jbond is self-dual. Dual is **Combo-CNN only** (config
  finalize assert) — transformer arms hard-code primal A_v machinery. Wilson-loop observables
  at L≥6 would have X/Z labels swapped (warning printed, not remapped).
- **Observables keep PHYSICAL meaning:** `calculate_{plaquette,vertex}_stabilizer`,
  `calculate_magnetizations` (Y negated), `calculate_Se` take `dual=` (callbacks read
  `config['dual_basis']`) so `Bp_mean`/`Av_mean`/`mag_*` JSON keys are basis-independent.
  Custom sampler flips **plaquette clusters** in dual mode (the orbit moves; all 4-valid).
- **Experiment (`colab/dual_basis_L4.ipynb`, W&B group `dual-basis-L4`):** validated CNN recipe
  (tdvp, dt 0.01, ds 6e-5, 350 steps, custom sampler). Arms: `cnnP`(0.2,0) vs `cnnD`(0,0.2)
  — the dual pair, each vs its own ED (ED-first cell; `ed_L4_hx0.00_hz0.20.json` ships,
  E0=−25.24695976382697; (0.2,0)+(0.1,0.1) computed in-notebook, High-RAM + `--no-observables`)
  — plus `cnnP_m`/`cnnD_m` @ (0.1,0.1) where each basis carries the opposite approximate-symmetry
  burden (compare converged rel-err = whose bias/optimization floor is lower). Expected mirror:
  cnnP ⟨A_v⟩≈1/⟨B_p⟩<1; cnnD ⟨B_p⟩≈1/⟨A_v⟩<1.
- **Local verify:** `scratchpad/validate_dual.py` (numpy-only) ALL PASS — masked star-Wilson ==
  ragged product (spins + float features), star products exactly B_p-invariant, vertex kernel
  table decode, param counts 1,681/2,633, and an L=2 dense-kron proof that the constructor swap
  == W·H·W (Hadamard⊗N) with identical spectra. `py_compile` clean; smoke runs happen on Colab.
- **RESULT (2026-07-29, L=4, all runs done): performance is governed by the perturbation's
  DIAGONALITY in the sampling basis, NOT by matching the embedded symmetry to the conserved
  stabilizer family.** The arm whose Wilson tokens are frozen (conserved family) and whose
  Block-1 does a smooth diagonal dressing wins (V ~1e-5..1e-7); the "tokens carry the physics"
  arm plateaus ~1e-3 (off-diagonal field ⇒ noisy E_loc ratios ⇒ noisy SR). Note V-score
  = ⟨H²⟩−⟨H⟩² is basis-independent for a FIXED state — gaps reflect training dynamics
  (variance budget Σ c_t²ε_t²: magnitude enters squared, count ~linearly, per-flip
  learnedness ε_t dominates). OBC non-self-duality adds a constant dual handicap
  (soft 2/3-body boundary stars) ⇒ at L=4 dual only pays off for hx ≳ 2–3·hz (tie observed
  at (0.3,0.1); primal wins 10× at (0.1,0.3)). Experiment CLOSED — purpose was to learn how
  swapping the Wilson-coarse-graining operator affects performance; findings transfer to the
  3D toric-code repo (dual basis ⇒ Wilson over vertex stars, enforces B_p).
- Pre-existing (flagged, NOT fixed): `MultiRule` off-by-one (`custom_sampler.py`,
  `maxval=n_clusters-1` ⇒ last cluster never proposed; ergodicity safe via single-flip rule).

## Current work — fermionic toric code (branch `fermionic-toric-code`)
`--ftc`: replace EVERY vertex star `A_v=XXXX` by the dressed `A'_v = A_v·B_NE(v)` (Chen–Kapustin–
Radičević exact-bosonization Gauss law, arXiv:1711.00515 Eq. 9; identified via lit search — the
user's stabilizer sketch matches CKR's NE-plaquette convention exactly). `XZ=−iY` on the 2 shared
links (v's up+right) ⇒ string `−Y·Y·X..·Z..` with REAL −1 coefficient; built as plain
`LocalOperator` products (X-links then Z-links) so the sign/Y bookkeeping is automatic and the
operator stays float64. Boundary rule (user decision): vertices without an NE plaquette stay bare
⇒ 9 dressed + 7 bare stars at L=4. Composes exactly with `--dual_basis` aliases but asserted off
(untested); mutually exclusive with `h_f`/`Jy_v`.
- **Physics (validated in `scratchpad/validate_ftc.py`, numpy-only, L=2/3 dense + L≤4 symbolic):**
  same stabilizer group ⇒ unperturbed GS **= TC GS exactly** (positive, E0=−#stab=−25 at L=4;
  identity-init Combo starts AT the GS ⇒ the h=0 run is a wiring test, V≈0 from step 0). BUT the
  OBC relation `Π_v A'_v = Π_p B_p` ties m-parity to e-parity ⇒ single flux forbidden ⇒ **fTC gap
  = 4 (paired excitations) vs TC 2**; spectra differ at h=0 ⇒ perturbed fTC needs its own ED
  (`lanczos_ed --ftc`, files `ed_ftc_L4_*.json`; TC files NOT reusable — except see next).
- **hz cut is EXACTLY the TC hz cut in the GS sector** (theorem, ED-verified to 1e-15 at L=2/3):
  hz preserves flux sectors and `A'_v ≡ A_v` on the zero-flux sector ⇒ same E0(hz), same GS,
  positive ⇒ existing `results/ed/` hz E0s DO apply to fTC E0 (not to gaps). No new physics there.
- **hx is the fermionic cut** (proliferates ε = flux+bound-charge; fermions can't condense ⇒
  expect non-Ising/1st-order — OPEN problem, no NQS/VMC literature): E0 genuinely differs
  (fTC resists polarization, dE0 grows +0.06→+3.0 over hx=0.1→1.5 at L=3). SIGN STRUCTURE
  SURPRISE: the fTC+hx GS is **sign-free at L=2/3 for the whole hx∈[0.1,1.5] scan** (through the
  gap minimum ~2.35 at hx≈0.7, deep into the polarized phase) despite the non-stoquastic H. Not
  yet a theorem at L=4 ⇒ config
  keeps `dtype=complex` for ftc+hx only (hz stays float64 by the sector theorem); L=4
  `lanczos_ed --ftc` records `neg_amp_fraction`/`neg_amp_weight` — if 0 across the window, flip
  ftc+hx to float64 (kills the ~20 min/pt complex JIT).
- **Code:** `geometry._generate_dressed_stars` → per-vertex `(x_links, z_links)`, shared links in
  BOTH lists (one-line lookup: `dg_v.positions[v]` is positionally aligned with `vertex_all[v]`,
  same alignment for `dg_p.positions`/`plaq_all`, so the NE-plaquette lookup is a single
  `_mapping2Dto1D(dg_p.positions, dg_v.positions[v] + [1/2,1/2])`, empty ⇒ bare `A_v`; **written
  and cross-checked bit-for-bit against the independent numpy reference at L=2/3/4**);
  `hamiltonian.py` ftc branch in the vertex loop; `config.py` `--ftc` (+legacy-dict default,
  sim_params record, dtype rule); `observables.calculate_dressed_star` → `Avp_mean/std` (ftc-gated
  end-of-run in main.py; ≈1 at h=0 alongside Av/Bp); `lanczos_ed --ftc` + sign diagnostic;
  `jobs/nersc_ftc.sh` (env-param HX_LIST/ARMS/DIAG_SHIFT/ED, SLURM array chunking over HX_LIST;
  default=h=0 wiring run `-t 0:30`; ED companion inline). Sampler/architecture untouched (H still
  commutes with every A_v; custom sampler is geometry-only — cluster moves stay exact symmetry
  moves). NOTHING COMMITTED YET — all of the above sits uncommitted on the branch.
- **Status (2026-08-14):** implementation complete, all local checks pass (`py_compile` clean;
  `validate_ftc.py` ALL PASS; `_generate_dressed_stars` cross-checked against the reference).
  NERSC cert renewed. Phase 3 (h=0 wiring) and Phase 4 (hx sweep + ED) not yet submitted — always
  consult before `submit`.
- **6-agent adversarial swarm (2026-08-14):** independently re-derived (fresh code, not reusing
  `validate_ftc.py`) the dressed-star algebra against the user's original diagram AND the actual
  CKR paper text (arXiv:1711.00515 Eq. 9/4/12, incl. resolving an operator-order sign subtlety —
  CKR write face-then-star, this repo star-then-face; verified by direct matrix computation that
  the two shared anticommuting pairs give `(-1)²=+1` either way) — **PASS, no discrepancy**.
  End-to-end wiring (geometry→hamiltonian→main→observables) — **PASS** on 6/7 items. PlainCNN
  param counts (1681 vs 1800) — **independently confirmed**. **Two real bugs found and FIXED**:
  (1) `--ftc` exclusivity assert (`config.py` + `hamiltonian.py`) didn't block `hy`/`Jy_p`, even
  though `hy` anticommutes with the dressed star on its 2 shared links exactly like `hx` does —
  a second unvalidated fermionic cut that was silently training with no ED companion; now
  excluded alongside `dual_basis`/`h_f`/`Jy_v`. (2) `hamiltonian.py`'s `ftc` branch had no `-1`
  filter on `z_links` (dormant today — OBC plaquettes never carry `-1` — but structurally
  inconsistent with the `x_links`/plain-plaquette loops elsewhere); added, plus a per-entry
  shape assert on `dressed_stars`. **Noted, not fixed (pre-existing, unrelated to this session):**
  the legacy `sys.argv<=12` positional-args path in `config.py` is missing an `'Lx'` default and
  is dead code at HEAD — no current job script exercises it.

## Current work — architecture comparison: does exact symmetry beat the fTC sign problem?
User's framing: fTC is sign-problem-full by construction (non-stoquastic `A'_v`); user's
hypothesis is that BOTH architectures may fail even at the **fixed point h=0** (no field at all)
— that itself is the current test, not a field sweep (explicitly ruled out: "no need for any hx
sweep"). Plan: run TWO architectures at h=0, L=4, same optimizer/sampler hyperparameters (dt/ds/
steps/tdvp/custom-sampler identical across arms — architecture is the only variable).
- **Arm `cnn`:** Combo-small (`channels_noninv 1,16`, `channels_inv 16,8,1`). Identity-init means
  it starts AT the exact GS (E0=-25) by construction — this run is really a wiring check; a
  deviation from -25 would indicate a bug, not a genuine architecture failure.
- **Arm `plaincnn` (NEW):** `--architecture PlainCNN` (`model/networks.py` `create_model`, new
  branch alongside Combo/RPP) — a genuinely unconstrained baseline: a plain stack of the SAME
  local masked-conv `CNN_noninvariant` blocks Combo's Block 1 uses, but with **no
  WilsonNonlinearity and no invariant block** — no architectural symmetry of any kind. Must find
  the GS from a generic init via VMC alone in the same step budget as `cnn` — this IS the actual
  test of the user's hypothesis. Both arms and RPP previously had **no non-symmetric option in
  this repo** before this session.
- **kernel_size clarified (2026-08-14): TWO DIFFERENT kernels, not one.** Block 1
  (`CNN_noninvariant`, the `--kernel_size` flag) is the LOCAL nearest-neighbor kernel — stays 2
  for every L, both arms. Block 3 (`CNN_invariant`, `kernel_size_inv`) is the GLOBAL kernel and
  was ALREADY auto-set to `Lx-1` per L in `config.py`, unrelated to Block 1 — no change needed
  there, it already matches the paper. (A same-session detour briefly set Block 1's kernel_size
  to `Lx-1` too, on a misreading of "kernel_size = L-1 for all the runs" as referring to Block 1;
  reverted — that phrase meant Block 3's per-L global kernel, which the repo already implements.)
- **Params, confirmed at kernel_size=2 (Block 1) / kernel_size_inv=Lx-1 (Block 3):** **cnn = 1681**
  (Block1 448 + Block3 1233). **plaincnn = `channels_noninv 1,8,6,2` → ~1800** (224+1260+316,
  matched-generous vs cnn, not starved). Note "1233" is Block 3 ALONE, not the network total —
  verified by re-deriving `CNN_noninvariant`/`CNN_invariant`'s exact `self.param` shapes from
  `KernelManager`; masking in `mask_kernel` zeroes boundary taps at call time, it does NOT shrink
  the stored parameter shape, so param count is boundary-independent.
- **wandb:** `utils/wandb_logger._tags` now also tags `architecture`, `hx`, and `ftc` (previously
  only `Lx`/`symmetric_block`/`hz`/`hy` — insufficient to slice this comparison on the dashboard).
- **Analysis:** `scripts/ftc_summary.py` (new, stdlib-only) — per-`(Lx,hx,hz)` table, both arms
  side by side, vs `ed_ftc_L4_*.json` (E0/gap/sign diagnostics), tail energy-std as a convergence
  signal, and a closing section quantifying the rel-err gap between arms. Deliberately **no KILL
  constant** (unlike `erel_table.py`) — this experiment expects and wants to measure failure, not
  gate on it. Verified against a synthetic fixture. At h=0, `E0=-25` is the ED reference (already
  in `results/ed/ed_L4_hx0.00_hz0.00.json` — no new ED point needed for this phase).
- **hx sweep is explicitly DEFERRED**, not part of the current plan — kept in `jobs/nersc_ftc.sh`
  as a documented Phase-4 option only (`{0.10, 0.30, 0.50, 0.70}`, matching the L=3 dense-scan's
  gap minimum window) in case the h=0 fixed-point result motivates it later.
- **RESULT (2026-08-14, L=4, h=0 fixed point, job 56952508): user's hypothesis CONFIRMED.**
  `cnn`: E=-25.000000 (median-tail rel-err **1.47e-08**, essentially machine precision, from step
  1 — the wiring check). `plaincnn`: E≈-23.45 (rel-err **6.22e-02**, V-score ~1.4, i.e. **not
  converged** — a ratio of >4,000,000x worse than `cnn`). Crucially, `plaincnn`'s training was
  **stable, not broken** (no NaN/divergence, small per-step energy std ~0.05-0.07) — the
  unconstrained architecture genuinely cannot find the topologically-ordered ground state via VMC
  from a generic init in 350 steps, even at the trivial h=0 point where the answer is a fixed
  stabilizer state. Diagnostic breakdown for `plaincnn`: `Av_mean≈0.9996` (close to 1 — the plain
  vertex-star structure is partially captured) but `Bp_mean≈0.913` and `Avp_mean≈0.951` (both
  degraded) and `magnetization_Xmean` is O(0.1-0.16) per-site (should be ~0 at h=0) — the network
  has drifted toward some partially-polarized non-topological state rather than the true GS.
  Real `--ftc` ED companion (`ed_ftc_L4_hx0.00_hz0.00.json`, actual pipeline, not a hand
  derivation): `E0=-25.000000000000018, gap=3.9999999999999574, neg_amp_fraction=0.0` — matches
  every adversarial-swarm prediction exactly. Synced: wandb project `2d-tc` (7 runs pushed via
  `wandb-sync`), results committed+pulled via `cluster.sh fetch`.
- **OPEN PHYSICS PUZZLE (2026-08-14, found via cross-session collaboration with the 3D fTC repo
  `toric-code-nqs-bf`): the hx-cut is positive despite being genuinely, un-fixably frustrated —
  not just non-stoquastic-looking.** At L=2, `H` has a real positive off-diagonal element
  (config 0001↔1101, traced to `A'_0`); the 3-cycle {0000,0100,1000} has edges (-1,-1,+1) — an
  ODD count of positive edges, an unfixable frustrated triangle in the signed-graph sense
  (Harary balance). **Exhaustively verified** (brute force over all 2^16 diagonal ±1 gauges at
  L=2, not a heuristic): NO diagonal gauge exists making H entrywise non-positive off-diagonal —
  i.e. this is provably NOT gauge-equivalent to a stoquastic Hamiltonian (unlike, e.g., the
  standard Marshall-sign-rule case). The frustrated configs carry substantial GS weight
  (|ψ[0100]|=|ψ[1000]|=0.186 vs max|ψ|=0.301 at hx=1.0 — not a corner the GS avoids) YET the GS
  is still exactly positive (`neg_amp_fraction=0.0`, confirmed L=2 AND L=3, hx∈{0.1,0.5,1.0}).
  **No explanation yet** — some mechanism other than stoquastic-gauge-equivalence is enforcing
  positivity. **(b) resolved: frustration DOES persist at L=3** (peer session found a concrete
  frustrated cycle there too, same structural signature, different location). **Analytic
  mechanism for which edges can even be positive (derived this session, stronger than pattern-
  matching):** every off-diagonal H entry comes from an X-flip (Z is diagonal). Bare-star and
  hx-field terms are pure X-products ⇒ their matrix element is unconditionally +1 ⇒ contribute
  -J/-hx, ALWAYS negative, no config-dependence. Dressed stars are built X-content-then-Z-content
  (`hamiltonian.py`'s ftc branch order) ⇒ acting on |config⟩ the Z-content evaluates FIRST, on
  the PRE-flip config, giving the NE-plaquette's flux eigenvalue (±1, config-dependent) ⇒ the
  edge sign is exactly `-[pre-flip flux eigenvalue]`. **Conclusion: only dressed-star edges can
  ever be positive — bare/field edges cannot, by construction, ever contribute a positive sign.**
  So any frustrated cycle necessarily involves ≥1 dressed-star edge — **exhaustively confirmed by
  the peer session** (per-term-tagged matrices, zero exceptions: at L=2, 8/8 positive entries
  touch a dressed star; at L=3, 8192/8192 positive entries touch a dressed star; bare/field
  entries are exactly +1 at every connected pair, 100% of the time, both sizes). Still open:
  (a) what non-gauge mechanism forces positivity despite this proven frustration; (c) whether
  this is documented in the bosonization/sign-problem literature for this construction class. **Practical fork (2026-08-14): peer session
  recommends a real-valued (float64) ansatz should suffice for Phase 4 given the L=2/L=3
  evidence — NOT yet acted on.** Our own documented criterion was always "confirm
  `neg_amp_fraction=0` at L=4 itself before trusting float64 there" (L=4 is a different sector-
  counting regime, not just bigger); the cheap fix is running ONE `--ftc` ED point at hx>0, L=4
  (~5-10 min, real dtype) before Phase 4 — proposed as the next step, needs user consult/approval
  before submitting (per the cluster safeguard) like any NERSC job. This reframes the earlier
  "sign structure surprise" note below —
  it is NOT merely non-stoquastic-but-lucky, it is a proven-frustrated Hamiltonian with a
  positive ground state, a stronger and more specific claim. Follow up if/when the peer session
  reports back on the L=3 frustration check.
- **Future work flagged by the user:** once both arms are shown to fall short of ED (including
  possibly at h=0), a planned follow-up experiment adds an **analytic sign-head** component to the
  architecture (design not yet specified — to be communicated in a later session/message) and
  re-tests. Treat that as Phase 5, gated on this comparison's results landing first.
- **Status (2026-08-14):** both arms implemented + compile-clean, kernel_size corrected to `Lx-1`
  for both; job script updated (`plaincnn` arm, dynamic `KSIZE`, SLURM array chunking kept for
  Phase 4 only); analysis script ready. Nothing submitted — job spec pending user consult (per
  the cluster safeguard below).

## Current work — doubled semion vs toric code (branch `doubled-semion`)
North star: validate a sign-aware approximately-symmetric NQS for the 3D fermionic TC
(peer repo `toric-code-nqs-bf`); the 2D doubled semion (DS) is the stepping stone with
real ED reach. Plan: show the Kufel-style positive/approx-symmetric ansatz nails the
honeycomb TC but FAILS the DS — failure is a THEOREM, not an expectation (Hastings
arXiv:1506.08883: no non-negative representation of the DS GS in any local product
basis; cite Shackleton 2509.03708 as the counterpoint — the claim is wavefunction
non-positivity, NOT "no sign-free QMC"). Literature scan found NO existing NQS/VMC
study of DS or any twisted quantum double ⇒ novel. Phases: 1 ground truth (DONE) →
2 NQS arms → 3 fixed-point experiment (+3b field sweeps) → 4 sign-aware architecture
(user designs it; explicitly NOT the peer repo's quadratic head).
- **Paper identity gotcha:** arXiv:1202.3120 = Levin–Gu, "Braiding statistics approach
  to SPT phases" (both models in §IV via gauging the Z2 paramagnet). The 3D
  generalization paper is von Keyserlingk–Burnell–Simon arXiv:1208.5128.
- **Locked conventions (user-approved):** honeycomb, spins on links, Levin–Gu form:
  vertex Q_v = ∏Z (the diagonal family ⇒ Phase-2 Wilson tokens), plaquette = X-flip.
  H_TC = −ΣQ_v − ΣX_hex − hxΣσˣ − hzΣσᶻ (paper's TC plaquette projector OMITTED —
  GS-sector identical since [X_hex,Q_v]=0; only charge-excited sectors differ;
  documented in the builder docstring). H_DS = −ΣQ_v **+** Σ X_hex·D_p·P_p − fields,
  D_p = ∏_{existing legs} i^{(1−σᶻ)/2}, P_p = ∏_{6 verts}(1+Q_v)/2. Ŝ_pP_p is
  Hermitian on the FULL space (P_p forces even down-leg parity ⇒ D real), order-
  insensitive, all terms commute; H has {0,±1} matrix elements ⇒ EXACTLY real ⇒
  float64 everywhere (no complex-JIT tax).
- **Smooth OBC (user decision; PBC rejected — OBC is the repo's NQS playground):**
  patch = Lx×Ly COMPLETE hexagons, staggered brick rows (bottom-left corners
  hx=(hy%2)+2i). Plaquettes never truncated; vertex terms 2-body at the boundary; legs
  truncated. Loops stay closed at degree-2 boundary vertices ⇒ (−1)^{#loops} well-
  defined; ∏_v Q_v = 1 is the single stabilizer relation ⇒ unique GS. Counting:
  F=LxLy, N=3F+2(Lx+Ly)−1, V=2(F+Lx+Ly), E0=−(V+F), gap 2, GS support = 2^F closed-
  loop configs, DS amplitudes = (−1)^{#loops}; DS neg-frac 0.5/0.75/0.75/0.875 at
  1×1/2×1(=1×2)/3×1/2×2. ED ladder: 6/11/16/19 qubits local (throwaway venv), 2×3=27
  on NERSC (deferred to Phase 3).
- **Code (Phase 1):** `model/honeycomb_geometry.py` (brick-wall tables: vertex_all V×3
  slots [left-h, right-h, vertical] −1-padded; plaq_all F×6 always full; legs_all /
  plaq_vertices F×6 slot-aligned CCW from bottom-left; link_endpoints; construction-
  time consistency net incl. coordinate slot-semantics and pairwise shared-edge ≤ 1).
  `create_honeycomb_hamiltonian` (hamiltonian.py) — PauliStrings algebra ONLY: netket
  operator products use `@` (not `*`), and a 12-site LocalOperator product OOMs on
  dense 4096² blocks; ≤2^10 nonzero strings per DS plaquette after netket pruning;
  weight reality comes from the T↔T^c corner-subset pairing (NOT even-Y-count — the
  leg factors (1±i)/2 are complex), asserted then cast float64. `lanczos_ed --lattice
  honeycomb --model {tc,ds} --Ly` (all-up-config sign anchor — the argmax anchor is
  tie-ambiguous at h=0 and can report 1−f; new `neg_amp_fraction_support` key; square
  path byte-identical; ds-on-square guarded). config.py flags + exclusivity asserts
  (honeycomb: OBC-only, no dual_basis/ftc/h_f/hy/Jy/Jbond/custom-sampler; ds ⇒
  honeycomb); main.py hard-guards honeycomb right after parse (ED-only until Phase 2).
  wandb `_tags` += lattice/model.
- **Validation (2026-08-20, all green):** two independent implementation lanes
  (production netket vs from-scratch numpy) reconciled 14/14 at ~1e-13 incl. perturbed
  (hx,hz) points and all sign fractions; 6-agent adversarial swarm found ZERO physics
  discrepancies — entrywise dense equality vs fresh independent derivations (24
  matrices incl. non-unit J and fields), reality mechanism re-proven, loop counts
  independently reproduced; all swarm findings (ds-mislabel guard, main.py guard,
  docstring corrections, _validate hardening, wandb tags) fixed same-day. Physics:
  TC and DS spectra are IDENTICAL under pure hz on simply-connected patches (hz
  preserves vertex sectors) — **hx is the discriminating axis** (1×2, hx=0.1: DS
  −12.0203 vs TC −12.0275; the projectors resist charge creation).
- **Phase 1 CLOSED (2026-08-20):** Term Atlas diagram approved by the user;
  `count_loops` implemented (Claude, per user instruction: union-find over down-link
  vertices, strict even-degree validation) — all harness cases + independent counters
  agree. Session validator artifacts (validate_honeycomb.py,
  honeycomb_reference_values.json, reconcile_phase1.py) live in the session scratchpad.
- **ED phase COMPLETE (2026-08-21): 10/10 points at 2×3 (27 qubits) + 22 local points
  (1×1..2×2), all gates green, all with Tier-2 observables.** E0=−28 exact, gap 2 at
  h=0; hz∈{0.1,0.2}: TC≡DS in the ZERO-DEFECT sector (E0 always; all k=6 levels at 2×3,
  but charge-sector levels DIFFER — DS projectors push them UP — and interleave into
  the low spectrum at small sizes: 1×2 eigs 3-5, 2×2 eig 5), DS neg-frac RIGID at
  0.75 (=48/64, count_loops-predicted); hx∈{0.1,0.2}: models split (DS E0 higher/resists
  polarization; DS neg-frac 0.774→0.803; TC exactly positive, ⟨X̂ₚ⟩≡1 and gap≡2 —
  X̂ₚ conserved under hx ⇒ frozen flux sectors). ED campaign gotchas, all committed:
  (1) netket Pauli→sparse OOMs at 2^27 even for 7-pattern TC h=0 (intermediates ≫
  final 12 GB CSR) ⇒ honeycomb ED = direct scipy CSR builder `_honeycomb_direct_ed`
  (own bit order: site i↔bit i, all-up=0; validated ~1e-13 vs netket on all 22 local
  points); (2) Tier-2 matrix-free observables `_honeycomb_observables` (σᶻ/Q_v from
  |ψ|², plaq/σˣ via single XOR gathers; ~5e-15 vs dense); (3) TC+hx = exactly
  degenerate flux-sector clusters ⇒ ARPACK needs `--ncv`≈48 (TC-hx ~110 min/pt vs
  DS ~55, light pts ~15-20 min at 2^27); (4) Perlmutter `squeue --me` transiently
  returns an EMPTY table ⇒ job watches must poll `cluster.sh sacct` terminal states
  (subcommand added), never squeue presence; (5) `cluster.sh submit` flattens ssh
  args ⇒ `--export` values use `+` separators (job script converts).
- **Phase 2 COMPLETE (2026-08-26): honeycomb Combo/PlainCNN NQS built, smoked on
  debug GPU, Sonnet-swarmed.** `model/honeycomb_networks.py` (`create_honeycomb_model`):
  Block-1 = 3 link-orientation kernels (v + two zigzag h-parities) × 5 slots,
  identity init, scaled sigmoid; Wilson vertex tokens == Q_v at init (masked
  products); Block-3 = full all-to-all displacement kernels PER SUBLATTICE
  (user-locked option (a): 45+45 / 69+69 taps at 2×2 / 2×3 ⇒ Combo 12,489 / 19,017
  params); PlainCNN (1,32,24,8,2) = 15,120. Sampler: MultiRule maxval off-by-one
  FIXED (all callers), hexagon clusters, `SectorInitWeightedRule` (all-up chains).
  main.py wiring + init-only hexflip gate (exact 0 at identity init; invariance is
  init-only — the square reference behaves identically, 0.0 at init / O(1) at random
  params). **Smoke (350 steps, minSR lr 0.01, A100):** TC 1×2/2×2 rel-err 1.4e-9 /
  2.5e-10 (V-score ~1e-13; tdvp cross-check at 1×2: 1.4e-10); **DS 1×2 converged to
  the EXACT positive-state (Hastings) floor −10−√2 = −11.41421356 to 8 decimals,
  ⟨plaq⟩=−1/√2, V-score 8e-11** (the optimum has constant local energy on its
  support — analytic + swarm-verified); PlainCNN fell to the trivial |+⟩ product
  state (E≈−2=−F, ⟨X̂ₚ⟩≡1, ⟨Q_v⟩≈0, 83% rel-err) — cannot find topological order.
  **Phase-3 prediction (exact combinatorial positivity-constrained minimization):
  DS 2×2 positive floor = −19.5699527513 (2.15% gap).** Gotchas: dense-SR tdvp is
  O(P²⁻³)/step ⇒ ~4 s/step at 1×2 with option-(a) P — honeycomb arms run **minSR**
  (all arms, both-or-neither); debug-QOS smoke jobs must fit 30 min; `--channels_inv`
  is argparse-required even for PlainCNN (pass a dummy). Swarm findings fixed:
  docstring init-only invariance wording, honeycomb architecture/symmetric_block
  assert (transformer arms would be silently ignored), cluster.sh `get` temp-file
  write. Known inherited quirks (documented, not bugs): complex-path Wilson tokens
  = rescale·Q_v; `_identity_init_local` is identity only for single-layer Block-1
  (multi-layer channels_noninv would sum channels — caught by the init gate).
- **Phase 3 RESULT (2026-08-27, jobs 57642001/010/033/036, all COMPLETED): the
  three-tier story is quantitative at both sizes.** h=0, minSR lr 0.01, 350 steps,
  seed 0; DS floors pre-registered BEFORE the runs (commit f801a2e; multi-restart
  optimizer matched the exact 1×2/2×2 floors to 1e-12): 1×2 → −10−√2, 2×2 →
  −19.5699527513, 2×3 → −27.2189973793 (gaps 4.89%/2.15%/2.79%, not monotonic).
  (1) **cnn (positive Combo):** TC exact — 2×2 −20.00000000 (5e-9, V 7e-14), 2×3
  −27.99999975 (2.5e-7); **DS pinned AT the Hastings floor** — 2×2 −19.56988161
  (7e-5 above), 2×3 −27.21404541 (5e-3 above; correct side). (2) **cnnc (complex
  Combo):** TC exact (2×2 5e-8, 2×3 1e-10 — complexity costs nothing); **DS: the
  generic phase head does NOT solve the signs** — 2×2 collapsed ONTO the positive
  solution (−19.56960691, ⟨plaq⟩ = positive-optimum value −0.8924); 2×3 dipped
  marginally BELOW the floor (−27.26834504 = 6.3% of the floor→GS gap recovered,
  V-score 1.6e-3 i.e. least-converged run) but stays 0.73 above the GS (2.6%
  rel-err vs TC's 1e-9). (3) **plaincnn:** trivial |+⟩ product state on TC
  (E ≈ −F: −3.98/−5.97, ⟨X̂ₚ⟩≡1, ⟨Q_v⟩≈0); junk on DS (−6.04/−6.00). W&B synced
  (project 2d-tc, groups hc-phase3-2x2/-2x3). Results in results/nqs/
  G-equiv_1_hc{2x2,2x3}_{tc,ds}_h0_{cnn,cnnc,plaincnn}.*  — note cnnc JSONs
  serialize energies as complex STRINGS ('(-3.86+0j)'): parse with complex().
- **Phase 4 design (user's, 2026-08-27): QEC sign head.** log ψ = log A_θ(σ) [positive
  Combo, real float64] + iπ·s(σ), s deterministic & parameter-free: read the Q_v
  syndrome (diagonal — B_p has no value on a config; flux data is the head's OUTPUT),
  MWPM-recover ε (exact minimal-cardinality = leading-order PT; pymatching sparse
  blossom, unit weights), s = [#loops(ε·σ) mod 2] via count_loops (2D closed form of
  the anchor+dressed-flip-covariance recursion — the recursion form is what ports to
  3D). Exact at hx=0 (any hz, sector theorem); at hx≠0 the ONLY error channel is
  tie-degenerate minimal recoveries (= semion braiding), which are interference-
  suppressed. Head keeps QGT real/untouched. Optional learned residual = Phase 4b.
- **Phase 4 gate 0 PASSED (2026-08-27): sign-fidelity diagnostic vs ED**
  (`scripts/sign_fidelity.py`, full 2^N enumeration; F_s = |ψ|²-weighted sign
  agreement = max fidelity of positive-net×head; results/diagnostics/, jobs
  57647055/59 + 57648190/98/200). All 4 sizes incl. 2×3=2^27: **F_s = 1 EXACTLY at
  hx=0 for every hz** (0–0.5); hx>0: wrong_weight = tie_weight/2 at every point/size
  (ties are the SOLE error mode), ~hx^10 scaling (interference kills tied configs at
  leading order), **size-independent** (hx=0.1: 2.3e-10 @1×2 → 1.9e-10 @2×3; even
  hx=0.5: 3.6e-6). F_plus baselines reproduce neg-fracs (0.25/0.25/0.125/0.25).
  E0 9/9 match committed ED. Swarm: independent brute-force reimpl (own dense DS H,
  own MWPM, own loop counter) — all disagreements provably tie-degenerate, zero on
  unique-recovery configs; conventions attacker — numerics all PASS, fixed bare
  `--out` makedirs crash; fusion-blossom agent — 219k configs, cardinality identical
  everywhere, every sign diff tie-verified ⇒ head is solver-agnostic. Caveat:
  `tie_disagree_weight` is a LOWER bound (linear-in-index perturbations miss
  sum-degenerate ties, ~0.003% relative). Gotchas: 2^27 head pass ~18 min (2 MWPM
  decodes × 134M configs) + hx ED ~13 min ⇒ 31 min total JUST misses debug 30-min
  wall — hx points need `-q regular -t 1:30`; light (h0/hz) points fit debug.
  lanczos_ed gained optional `tol` (default 0, unchanged); pymatching installed in
  2dtc env + local venv.
- **Phase 4 RESULT (2026-08-27, jobs 57653805/809): HASTINGS FLOORS SMASHED —
  DS at h=0 reaches TC-grade precision with the deterministic head.** Wiring
  (commit 2f5c199): `model/sign_head.QECSignHead` (frozen decoder-A MWPM + 2^F
  loop-sign table); TWO exactly-equivalent formulations, both implemented per
  user requirement — **B (production)** `model/sign_frame.SignFramedOperator`:
  train the positive real Combo on H̃=SHS (mels × sign(σ)sign(σ'), host numba
  path; sampler/dtype/QGT byte-identical to Phase 3); **A (witness)**
  `honeycomb_networks.SignedModel`: logψ += iπ·s via jax.pure_callback (R→C).
  Flags `--sign_head {none,qec}` / `--sign_impl {operator,model}` /
  `--minsr_mode`; hexflip gate runs on the base net; off-diagonal observables
  re-framed (SOS) under impl operator only. `scripts/ab_equivalence.py` (1×2,
  synced params): Re logψ and per-config Re E_loc BITWISE equal, Im=πs exact,
  imag dust = sin(π)=1.2e-16, 150-step paired trajectories identical to 1e-15,
  E=−11.99 through the −11.414 floor. **Cluster (2×2 both arms + 2×3 cnnqB,
  350 steps, Phase-3 recipe): E_tail −20.000000002 (rel 8e-11, V 1e-14, cnnqA)
  / −19.999999992 (rel 4e-10, cnnqB) / −28.000000016 (rel 6e-10, V 1e-12,
  2×3) — all THROUGH the floors (−19.56995/−27.21900); ⟨plaq_term⟩=−1.000000
  exactly (floor state: −0.8924), ⟨Q_v⟩=1.** In-vivo A/B trajectories differ
  early (max|dE| 0.74) ONLY because flax folds init RNG by module path
  ('base/...' vs bare ⇒ different Block-3 draws — the harness syncs params,
  main.py arms don't), converging to the same state (median |dE| 1e-6). W&B
  group hc-phase4-h0.
- **Phase 4 FIELD RESULT (2026-08-28, jobs 57655134/137/142/144, 13/13
  COMPLETED, W&B hc-phase4-fields): the head is nowhere the bottleneck.**
  cnnqB, Phase-3 recipe, all vs exact E0 refs (signfid JSONs + ED files).
  **hz line (head provably exact): TC-grade** — 2×2 rel-err 2.5e-9/2.3e-10/
  7.7e-8 at hz 0.1/0.2/0.5; 2×3 4.9e-9/8.2e-9 at 0.1/0.2; Qv=1.00000 exactly
  (sector theorem in vivo); ⟨plaq⟩ tracks −0.905→−0.331. **hx line:
  optimization-limited, NOT head-limited** — rel-err 6.4e-6/3.0e-5/1.1e-4/
  3.3e-4 at 2×2 hx 0.1/0.2/0.3/0.5 (2×3: 5.5e-7/5.2e-5/1.4e-4 at 0.1/0.2/0.3;
  mixed (0.1,0.1): 2.7e-6). Gate-0 head ceiling is 3-4 orders BELOW these
  (2e-10 rel at hx=0.1; ≲5% of the error even at hx=0.5) and V-scores
  1.7e-5..3e-3 say the 350-step budget is the limiter — more steps/samples
  would shrink it; vs the positive-ansatz ceiling (F_plus≈0.125 ⇒ %-level)
  the head wins 3-4 orders everywhere. No NaN/divergence anywhere. Phase 4
  CLOSED at 2D scope.
- **Phase 4b (2026-08-28, commits e50c0e3/950b3c6): three sign arms across the
  (hx,hz) plane — campaign LAUNCHED, results pending.** Arms: `cnnqB` (real
  trunk + head, production), `cnnqC` (`--complex_ansatz --sign_head qec
  --sign_impl operator` — complex trunk on the framed H̃), `cnnqR`
  (`--sign_impl residual --res_hidden 16` — real trunk + head + tie-gated
  residual phase MLP: logψ += i(πs + t·MLP(s,d,r)), zero-init ⇒ step 0
  bitwise == head-only; t-gate ⇒ theorem regime untouchable by training).
  sign_head.py gained decoder-B + `features()` = [s, d, r=ε_A⊕ε_B, t].
  **Ceiling map (25-pt 2×2 grid {0,.2,.4,.6,.8}², signfid_hc2x2_ds_grid.json):
  F_s = 1 exactly on the whole hz axis; ≥ 0.9975 EVERYWHERE — the head prior
  survives far outside the topological phase; C/R have room only in the
  hx≳0.6 band; F_plus stays ~0.12 along the hx axis.** 3-agent swarm:
  features() PASS (exhaustive brute force incl. all-2^N-subset recovery
  enumeration; zero false tie flags; r always a valid even-degree cycle);
  wiring PASS (asserts loud; all 3 arms end-to-end at 1×2; gate base-routing;
  cnnqR params = base+369 pre-fix/385 post-fix); expressivity oracle FOUND A
  REAL DESIGN FLAW — r is a deterministic function of d (the decoder sees
  only the syndrome) and tie partners share d, so a (d,r)-MLP provably cannot
  split tie pairs (fixes only 40/14/2.4% of the head gap at 1×2); FIXED by
  feeding s into the MLP (950b3c6). At 2×2 the s-class recovers 86-97% of the
  head gap (7-36× error cut) but genuine (s,d,r) collisions cap it above the
  gate ceiling (weight-suppressed: 4e-8/3e-6 at (0.2,0.2)/(0.5,0.5)) — exact
  saturation would need config-dependent (non-syndrome-derived) features =
  future design. **In queue (Perlmutter partitions DOWN for maintenance at
  submit time):** 16 wave jobs 57670508-527 (2×2: 25 pts × B/R in 4 chunks
  each + C in 5 chunks; 2×3 diagonal mixed pts × R(1)/C(2)); debug smoke
  57670501 (cnnqC+cnnqR@1×2 — its agent verifies + deletes artifacts);
  yesterday's 2×3 mixed ED (57670296-301) + cnnqB NQS (57670302). Grade with
  scripts/phase4b_summary.py (ceiling columns built in).
- **Phase 4b RESULT (2026-08-29, 87/87 runs COMPLETE, W&B synced): all three
  arms are statistically indistinguishable across the entire plane — the
  shared QEC head sets the error; the sign treatment on top of it does not
  matter in this regime.** User's reading (formed independently first, then
  discussed): "all competitive, same ballpark, none loses" — CONFIRMED, with
  the structural why: achieved rel-err sits 1-4 orders ABOVE the head ceiling
  1−F_s at nearly every point (optimization/budget-limited at fixed 350
  steps; V-scores track rel-err), so C's complex freedom and R's residual had
  almost nothing left to win. Only the hx≳0.6, hz∈[0.2,0.6] corner has
  ceiling ≈ achieved error (both ~1e-3) — there C/R show ≲15-20% hints over
  B, within single-seed noise (seed replicas = the open follow-up). Cost at
  equal accuracy: B (real, cheapest) > R (+385 params, host callback) > C
  (complex-QGT/JIT tax, ~2x wall). Ops scars, all healed by resubmits: 2
  R-chunk timeouts at the trimmed 1:45 wall (each ate its chunk's last
  point); 3 jobs fast-"COMPLETED" on a broken GPU slice (nid004069,
  CUDA_ERROR_UNKNOWN at init — sacct says COMPLETED, only the log/elapsed-
  time sanity check tells). Viz: analysis/05_phase4b_plane.ipynb (house
  rcParams per the peer repo's conventions; continuous log-space-cubic error
  fields + 2×3 bars + achieved-vs-ceiling scatter). Gotcha: matplotlib
  mathtext RecursionError on stale font caches — rm ~/.matplotlib/
  fontlist*.json + kernel restart. NEXT: the 3D port (recursion form of the
  head); optional corner seed replicas if the C/R-vs-B hints ever matter.
- Known trap (pre-existing): invoking main.py with ≤12 argv entries hits the legacy
  positional path (`eval(sys.argv[...])`) and dies with a confusing NameError — always
  pass full flag sets.

## Cluster automation & safeguard (IMPORTANT)
Cluster access is **already configured** — Claude drives NERSC directly via
`scripts/cluster.sh` (SSH over an sshproxy 24h cert; connection settings in the
gitignored `scripts/cluster/config.local.sh`, user `sanzharb`; see docs/NERSC.md).
At session start, just run `bash scripts/cluster.sh status` to connect. If it
fails with a "no NERSC key / cert expired" message, the daily cert lapsed — ask
the user to run `sshproxy -u sanzharb` (only they can; it needs their password +
MFA), then retry. Monitoring/fetching is frictionless; **launching compute is gated**.

**Submit policy (updated by the user 2026-08-20):** SHORT jobs — shared-queue or
`debug`-QOS, roughly ≤ ~2 h, single-node-fraction — may be submitted without prior
consultation, **but every submission must be reported to the user** (job id, what it
runs, resources, why). **LONG/HUGE runs (exclusive multi-hour nodes, big arrays, full
sweeps) still require prior consultation and explicit approval.** For those, present
a short job spec first:
- **Experiment** — what physics/sweep this run does and why now.
- **Resources** — the `#SBATCH` request (queue, `-N/-G/-c`, node fraction).
- **Walltime** — `-t` and the expected runtime, with headroom rationale.
- **Why** — what question the result answers / what it unblocks.

`submit` and `cancel` are deliberately **not** in the permission allowlist, so they
always prompt too — the consult (chat) and the prompt (Claude Code) are both
required. Read-only ops (`status`/`sync`/`logs`/`fetch`) are allowlisted; `fetch`
brings results back (commit+push on cluster, pull locally) and is safe to run
freely. Never allowlist `submit`/`cancel`.

## Experiment tracking (Weights & Biases) — opt-in, offline-by-default
Live-ish monitoring of E / V-score / energy-std / grad+dtheta norms / per-step timings
/ QGT cond, plus final observables as run summary. **Opt-in** (`--wandb`; off => runs
byte-identical), **offline by default** (compute nodes have no internet — logs to
`wandb/`, pushed later from a login node). Zero-dep when off: `utils/wandb_logger.py`
only imports `wandb` inside `init_run` and every entry point no-ops unless `wandb.run`
is live.
- **Code:** `utils/wandb_logger.py` (guarded init/log_step/log_summary/finish);
  `optimizer.run_tdvp` logs one row/step; `main.py` inits after config-finalize +
  logs summary/finish at the end. Args in `config.py`: `--wandb --wandb_project
  (2d-tc) --wandb_entity --wandb_group`.
- **Jobs:** `nersc_v2.sh` / `nersc_transformer.sh` take `WANDB=1` (per-submit
  `--export=ALL,WANDB=1[,WANDB_GROUP=…,WANDB_PROJECT=…]`) → sets `WANDB_MODE=offline`,
  `WANDB_DIR=$REPO/wandb`, appends `--wandb …`.
- **One-time setup (login node):** `pip install wandb` (not in `2dtc` env yet) then
  `wandb login` (needs the user's API key from wandb.ai/authorize). Key lives only on
  the login node; never in git or on compute nodes.
- **Push to dashboard:** `bash scripts/cluster.sh wandb-sync` (`wandb sync --sync-all`
  from the login node; safe to re-run mid-training for near-live, dedupes by run id).
- **Online (truly-live) is unverified** — would need compute-node internet (maybe an
  `https_proxy`); test with a tiny job + `WANDB_MODE=online` before relying on it.

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
