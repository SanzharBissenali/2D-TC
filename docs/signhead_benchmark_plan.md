# Learned-vs-gated sign head: benchmark plan (2026-09-23)

North star: two 3×3 `(h_x, h_z)` heatmaps per model — relative energy error and
exact fidelity `F = |⟨ψ_ED|ψ_NQS⟩|²` — for three arms, on the 2D doubled semion
(this repo, 2×3 honeycomb, N=27) and the 3D fermionic toric code (peer repo
`toric-code-nqs-fsign`, L=2 OBC / 2×2×3 OBC). Everything else in this plan
serves those 6 heatmaps.

## 0. What the oracle already settled (training-free, exact vs ED)

- Syndrome-only features carry no sign information (ceiling == positive ansatz
  to 5e-13). Any head must see the repaired configuration.
- Given `r = ε·σ`, the on-support sign is closed-form: `(−1)^{poly(x)}` with
  `x` the hexagon-flip (application) variables, `poly` = Levin–Gu Z·CZ·CCZ
  cubic (2D; verified 1×2..4×4) / the cup-product quadratic (3D, peer).
  Exact at `h_x = 0` for all `h_z`; leading-order PT otherwise.
- The head fails only as the state polarises: 2D crossover `h_x* ≈ 1.09`
  (2×2), 3D `h_x* ≈ 0.57` (2×2×3); the best global switch between "head" and
  "no head" leaves a 1–2% hole there, deepening with size. An ideal
  per-configuration keep/drop of the head leaves ≤ 2.6e-4 (3D) ⇒ the true
  crossover sign is "topological sign, kept or dropped per configuration".

## 1. The three arms (identical information, identical everything else)

Shared: positive real Combo trunk `A(σ)` (2D Phase-3 recipe: minSR lr 0.01,
ds 6e-5, 350 steps, 8192 samples, seed 0 + one replica seed); deterministic
feature map `σ → (ε, x)`: `Q_v` syndrome → MWPM (pymatching, production
tie-break) → `ε` → `r = σ⊕ε` → `x = G⁻¹ r` over GF(2) (precomputed
pseudo-inverse of the hexagon mask, unique on the simply-connected patch).

| arm | wavefunction | trainable sign? |
|---|---|---|
| **M** (MLP sign) | `ψ = A(σ)·tanh(m_θ(ε, x))`, `m_θ` = MLP (N+F → 64 → 64 → 1, tanh), random init | yes, from energy alone |
| **M-pre** | same; `m_θ` pre-fit supervised to ED signs (\|ψ_ED\|²-weighted BCE on all 2^N configs at the SAME size), then VMC | yes, warm-started |
| **T** (two-branch) | `ψ = a·A_triv(σ) + s_head(σ)·A_top(σ)`; two Combo trunks + one SIGNED real scalar `a` (`mix`, init +0.05 ⇒ step 0 ≈ production head-only arm; `a = 0` IS the head-only arm; `a < 0` swaps which head-sector the trivial branch can flip); `s_head = (−1)^{poly(x)}` | no |

Notes
- All three have complex `log ψ` with real parameters (same non-holomorphic
  QGT path as the Phase-4b residual arm `cnnqR`, `--minsr_mode complex`).
  `a₁ = log A_triv`, `a₂ = log A_top`, `m = max(a₁, a₂)`, `w = a·e^{a₁−m} + s·e^{a₂−m}`,
  `log ψ_T = m + log|w| + iπ[w<0]` (stable; `a` stays O(1) outside the max/log,
  so `a = 0` and `a < 0` are both valid — no `e^c` reparameterisation).
- Nodes: both M and T have `ψ → 0` on configurations whose sign is flipping;
  `∂ log ψ` is large there but those configurations are sampled ∝ |ψ|². Watch
  `diag_shift`; report divergences rather than tune per point.
- M-pre is pre-trained at the benchmark size (ED ψ exists at 2×3 via the
  sign_fidelity enumeration machinery). It is the "can VMC even find it"
  control, not a size-transfer claim; a size-transferable CNN variant of
  `m_θ` is a follow-up, not part of this benchmark.
- M sees `(ε, x)` exactly as T's head does; T additionally applies the
  closed form. That asymmetry is the experiment.

## 2. Field points

2D DS (2×3): `h_x ∈ {0.4, 0.8, 1.2}` × `h_z ∈ {0, 0.2, 0.4}`. Spans deep
topological (0.4; head exact to 1e-6) → head-still-wins (0.8) → past the
crossover (1.2; positive ansatz ceiling 2e-3 at 2×2, head 5e-2).
Existing 2×3 ED/ceiling points: (0.4,0), (0.4,0.4), (0.8,0), (0.8,0.4).
New ED needed: the `h_z = 0.2` column (3) and the `h_x = 1.2` column (3)
= 6 points, `-q regular -t 1:30` each (hx ED ~13 min + head pass ~18 min).

3D fTC (peer repo, 2×2×3 OBC, hz=0 line has the crossover at 0.57):
`h_x ∈ {0.2, 0.5, 0.8}` × `h_z ∈ {0, 0.2, 0.4}`. Dense ED there is
4096/2^12–2^20-dim, cheap.

Pre-registered predictions (from the ceilings): T ≤ 1e-3 everywhere except
possibly the (0.8, ·) crossover-adjacent column in 2D (ceiling 1e-4..2e-3) and
(0.5, ·) in 3D; M-cold fails at every weak-field point (parity-hardness;
expect it to land near the positive-ansatz F ≈ 0.75 or worse) and may succeed
only at 1.2 where the true state is nearly positive; M-pre tracks T at weak
field if VMC does not destroy the warm start, and is the arm to watch at the
crossover.

## 3. Implementation (2D, this repo, branch `learned-sign-head` off `doubled-semion`)

1. `model/sign_head.py`: `QECSignHead.features_ex(states)` → float64
   `(…, N + F)` = `[ε bits, x bits]` (decoder-A correction + GF(2) solve);
   `x_of_r(r)` helper using a precomputed `G⁻¹`; `poly_sign01(x)` (the
   Levin–Gu cubic, asserted == `loop_parity01` on all 2^F sector configs at
   construction, F ≤ 16). Adds `n_features_ex`.
2. `model/honeycomb_networks.py`: `MLPSignModel(base, feats, K, hidden,
   depth, init_params=None)` and `TwoBranchModel(base_triv, base_top,
   head_s01, mix_init=0.05)`, both wrapping the whole batched net like
   `SignedModel` (one host callback per batch). Trunk factory reused; the two
   trunks of T get distinct flax scopes ('triv'/'top') ⇒ independent init.
3. `utils/config.py`: `--sign_impl {…, mlp, twobranch}`, `--mlp_hidden 64
   --mlp_depth 2 --mlp_init <npz>`, `--mix_init 0.05`; exclusivity asserts
   (real trunk only; `--decoder mwpm` only; ds only); `sim_params` record;
   wandb `_tags` += `sign_impl`. `main.py`: two new branches next to the
   `residual` one; hexflip gate on the base trunk(s); off-diagonal observables
   via the model's own sign (as for residual). `optimizer._WRAP` += new
   classes if needed.
4. `scripts/pretrain_sign_mlp.py` (GPU node): enumerate 2^N configs →
   `features_ex` (host, ~18 min at 2×3) → jax MLP fit, |ψ_ED|²-weighted BCE,
   report achieved sign fidelity vs the ceiling (must reach ≥ 1 − 1e-4 or the
   MLP is under-capacity — that itself is a result) → save npz. Reuses
   `scripts/sign_fidelity.py`'s ED/ψ loader.
5. `scripts/nqs_fidelity.py`: `parse_arm` / `build_vstate` learn the two new
   arm tokens (`cnnqM`, `cnnqMp`, `cnnqT`); sign taken from the model itself
   (residual path). Energy self-check unchanged.
6. Jobs: `jobs/nersc_signbench.sh` (env-param `ARMS`, `POINTS`, `LX/LY`,
   chunked like `nersc_phase4_fields.sh`); ED + ceilings for the 6 new points
   via `jobs/nersc_signfid.sh`; fidelity via `jobs/nersc_fidelity.sh`.
7. Analysis: `scripts/signbench_summary.py` (stdlib; rel-err + F table, both
   vs ED refs) and `analysis/07_signbench_heatmaps.ipynb` (2 metrics × 3 arms,
   3×3 cells, shared log colour scale; 2D and 3D panels side by side once the
   peer data land).
8. Local verification before any submit: `py_compile`; numpy-only
   `scratchpad/validate_signbench.py` (features_ex round-trip `mask·x == r`,
   poly == loop parity on all sector configs 1×2..3×3, stable log-ψ_T
   identity vs direct evaluation, M-pre loader shape checks); debug-QOS smoke
   at 1×2 for all three arms (`-t 0:30`, verifies step-0 ψ_T == head-only arm
   when c → −∞, and that M's gradient flows).

## 4. Runs (2D), in order

| stage | what | queue / walltime | count |
|---|---|---|---|
| a | ED + ceilings, 6 new 2×3 points | regular, 1:30 | 6 |
| b | debug smoke, 3 arms @ 1×2 (0.4,0) | debug, 0:30 | 1 job |
| c | pretrain `m_θ` at 2×3, 9 points | regular (ED memory) + GPU, 1:00 | 9 (one job, sequential) |
| d | VMC, 3 arms × 9 points × seed 0 | shared, 1:45 per chunk of 3 runs (T ≈ 1.5× the 3.6 s/step of cnnqB; sampler callback adds 4N host round-trips per sweep — measure at smoke) | 27 runs / 9 chunks |
| e | exact fidelities, 27 records | regular, 3:00 per 9 | 3 |
| f | seed-1 replica of whichever arm/point pairs sit within 3× of each other | shared | ≤ 27 |

Stage b/c/d/e submissions are short shared/debug jobs (report each); stage a
and c use regular nodes (ED memory) — consult before submitting.

## 5. 3D half (peer repo — same spec, their code)

Hand this file over. Arms map 1:1: trunk = their approx-symmetric trunk;
`s_head` = their `linear`-or-`pt2`-decoded cup-product sign (use pt2, the
production head there, so T's topological branch is the best available);
`(ε, x)` = their recovery + application variables (`notes/fermionic_stencil_
obstruction.md` already defines `x`). Their dense ED covers all 9 points;
pretraining is a 4096–2^20-config fit. Deliverable: the same two heatmaps.

## 6. What decides the benchmark

Per cell: rel-err and `1−F` for M, M-pre, T, each with its ceiling
(`1−F_s` of the head for T; 0 for M/M-pre since `(ε,x)` determines σ).
Verdict criteria, fixed now: an arm "wins" a cell if its `1−F` is ≥ 3× lower
than the other's on both seeds. Expect the story to be: weak field — T exact,
M fails, M-pre ≈ T; crossover — T ≤ 1e-3 by the per-config gate, M-pre is the
open question; polarised — both fine. If M-cold matches T at weak field the
parity-hardness argument is wrong and that is the headline instead.

## 7. Amendments (2026-09-23)

- **T's ceiling is `T_gate`, not `1−F_s`** (peer-repo observation, adopted):
  with positive trunks T can realise either sign where `s_head = −1` but only
  the global sign where `s_head = +1`, so its exact representability ceiling is
  `T_gate = min_g Σ|ψ_ED|²[s_head=+1 ∧ sign ψ_ED ≠ g]` — report it per cell next
  to `1−F_s`. (Oracle: at 3D L=2 OBC `T_gate = 0` at all 9 points while
  `1−F_s(pt2) = 3.6e-2` at h_x=0.8.)
  **Definition as implemented (2026-09-24, `scripts/sign_fidelity.py`):** with
  `s ∈ {±1}` the mwpm head sign over ALL 2^N configs, `ψ_ED` anchored exactly as
  for `F_s` (all-up config positive), `g ∈ {±1}` the global sign of the pinned
  sector, and `sign ψ ≠ g` evaluated on nonzero amplitudes only:
  `T_gate_plus  = min_g Σ|ψ_ED|²[s=+1 ∧ sign ψ_ED ≠ g]`,
  `T_gate_minus = min_g Σ|ψ_ED|²[s=−1 ∧ sign ψ_ED ≠ g]`,
  `T_gate = min(T_gate_plus, T_gate_minus)`, `T_gate_branch ∈ {plus, minus}`.
  `plus` is the `a > 0` realisation (trivial branch flips only `s=−1`), `minus`
  the `a < 0` one (flips only `s=+1`); since `a` is a signed trainable scalar T
  can reach either, so its ceiling is the min. Per-point keys in the signfid
  JSONs; `signbench_summary.py` uses `T_gate` for T's ceiling when present,
  else `1−F_s` flagged `(1-Fs)`.
- **3D deviations (peer, 2×2×3 OBC, N=20):** ε from the `linear` decoder (pt2
  commits to no single recovery); `x` = plaquette pair-move bits + 8 star bits
  (needed for injectivity in 3D); heads in-model as tables (no host callback);
  optimizer = the 3D recipe (dense SR, cosine dt 0.02→0.002, ds 1e-3, 300
  steps), identical across arms. 2D and 3D panels are separate experiments.
- **Adversarial review gate (user rule):** before ANY experiment submission,
  2–3 independent agents attack the stack — (i) physics/feature encoding and
  arm-M blindness to ED signs, (ii) JAX/flax wiring + checkpoint rebuild,
  (iii) grading fairness (identical hyperparameters, labels/anchor conventions,
  summary/notebook on fixtures). Fixes land before the smoke → runs sequence.

- **Per-sector mix (2026-09-24, attacker-1 finding):** the exact ceilings say the
  better sign of a single mix scalar flips cell by cell (2×2: `h_z=0` column wants
  `a<0` — exact at (1.2,0), 8× better at (0.8,0); `h_z>0` columns want `a>0` by
  20–100×), and a scalar can only change sign by passing through the head-only
  point. Arm T therefore carries one signed scalar per head sector,
  `ψ = a_{s(σ)}·A_triv + s·A_top`, `mix = [a₊, a₋]`, both init +0.05: no sector
  is pinned, ceiling = min(T_gate_plus, T_gate_minus) per sector.
- **Arm-M init = positive ansatz (same review):** lecun init gave a random-sign
  state at step 0 (34% minority sign, 16% of configs suppressed >10×) — a handicap
  unrelated to the question. The last MLP layer is zero-kernel / bias +1, so M
  starts at `A·tanh(1)` (no sign information) and M-pre's npz overrides it.
