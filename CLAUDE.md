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
  solve cost. A `VMC_SR(use_ntk=True)` (SRt/minSR; `VMC_SRt` deprecated in netket 3.16) swap is
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
  `--optimizer minsr` → `simulation/optimizer.run_minsr` = NetKet `VMC_SR(use_ntk=True)` (kernel
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
- **Status (2026-07-27):** code + minSR + notebook complete, locally validated (`py_compile`, cell
  ASTs, validate_v3); NOT yet trained. First run: open the notebook, expect the A_v gate to print
  exactly 0; if minSR diverges, raise diag_shift→1e-3 / halve lr (the hy-run fix).

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
