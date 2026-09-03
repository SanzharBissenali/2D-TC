# Phase 4c — decoder scaling & 3D portability study

**Date:** 2026-08-30 · **Branch:** `doubled-semion` · **Method:** five independent
benchmark agents, one per decoder in the `model/decoders.py` ladder, each measuring
our implementations on honeycomb lattices 2×2 → 12×12 (N = 19 → 479 spins) plus mock
3D cubic-lattice graphs, in the local venv (numpy 2.1.3, pymatching 2.4.0, numba
micro-kernels for compiled projections). Raw artifacts: session scratchpad
`scale_{mwpm,anchor,greedy,uf,tiesum}/` (scripts, JSONs, logs). Exact accuracy
ceilings referenced throughout come from the full-2^N ED gradings in
`results/diagnostics/signfid_hc{2x2,2x3}_ds_*_dl.json`.

**Question answered:** if several decoders tie on accuracy, which wins on speed —
and which port to the 3D fermionic code? **Answer: there is no speed–accuracy
trade-off to navigate. MWPM (pymatching sparse blossom) is simultaneously the most
accurate and, already as shipped, essentially the fastest; its 3D port is trivial
(with one tie-break trap, below). Accuracy beyond MWPM must come from *weights*
(Stace–Barrett / resolvent), never deeper enumeration.**

---

## 1. The frontier at 12×12 (479 spins)

Production sign-head workload: 8192 samples × (1 + N) connected configs ≈ **3.93 M
decodes per VMC step** at hx ≠ 0. The head's *shared*, decoder-independent cost
(loop-parity recounts) is the yardstick: **841 s/step as currently coded**
(`count_loops` ≈ 210 µs/config), dropping to **~20–30 s/step** once the two verified
fixes land (local 12-link hexagon-flip parity rule + compiled union-find counter —
see the 2026-08-30 head-scalability benchmark, `bench_head.py`).

| decoder | µs/decode as coded | optimized (verified equivalent) | s/step optimized | exact ceiling 1−F_s @2×3 (0.4,0) | verdict |
|---|---|---|---|---|---|
| anchor | 120 (int64 matmul) | 3.3 (float32 GEMM) | ~13 | 1.0e-2 | speed floor only; provably unrescuable |
| unionfind | 135–434 effective | 1.3–12 (numba, ×23–44) | 5–48 | 1.5e-5 | dominated by greedy at every 2D physical point |
| greedy | 5–152 | 0.8–27 (numba, ×150–170) | 3–106 | 2.8e-6 | cheap + near-MWPM; 3D accuracy caveat |
| **mwpm** | **0.6–2.9 (C++)** | — | **2.5–12** | **8.5e-7** | **wins both axes; production decoder** |
| tie_sum | 25–285× the shadow | structural (B×K parity tax) | not viable | ≈ mwpm | retire to 2D diagnostic role |

Regimes: "physical" = dilute→moderate defect density (0.5–2% link flips ≈ hx ≲ 0.5).
Dense (8%) and uniform-random columns are in the per-decoder sections.

## 2. Cross-cutting findings (confirmed independently by ≥2 agents)

- **Memo caches are dead at scale.** Per-syndrome caching absorbs everything at
  2×3 (2M possible syndromes) but at 12×12 the production flip-neighborhood
  decodes are **provably 100% unique** (distinct links have distinct endpoint
  pairs ⇒ all N syndrome deltas differ) and correlated-MCMC streams hit only
  1.9–13.6%. Worse: `np.unique`-style dedup costs *more than sparse-blossom
  decoding itself* — MWPM's no-cache design is correct; the Python decoders'
  caches are cosmetic at scale and (UF) would grow 3.2–3.6 GB/step unbounded.
- **`_check_valid` is a hidden tax.** The ladder wrapper's per-call int64 boundary
  matmul costs 60–77 µs/row at 12×12 — up to 22× the MWPM decode it validates,
  and ~⅓ of the parity shadow even on all-hit cached batches. Production rule:
  spot-check validity (or float32-GEMM it), never per-call. The production
  `sign_head.py` mwpm path already bypasses it.
- **numpy integer matmuls are not BLAS.** The float32 GEMM route (exact while
  counts < 2²⁴, then mod 2) is 20–75× faster and bit-equivalent — applies to the
  anchor decode, validity checks, and any future syndrome-table path.

## 3. Per-decoder detail

### mwpm — the production decoder
- 0.17 µs floor + ~0.13–0.15 µs/defect while dilute; **N^1.2 at fixed defect
  density** (matches sparse blossom's published ~O(N^1.32) observed scaling);
  batch-flat from B=256; 12×12 physical band 0.6–2.9 µs/decode ⇒ 2.5–12 s/step.
- Literature anchor: Higgott–Gidney sparse blossom (arXiv:2303.15933, Quantum 9,
  1600) — ~10⁶ errors/core-second; exact MWPM, no approximation.
- **3D mock (cubic OBC, `Matching.from_check_matrix` unchanged):** 512-vertex
  (E=1344): 2.5–16 µs/decode in physical regimes; 1452-vertex (E=3971): 6.7–48 µs.
  Direct literature precedent for 3D point-charge matching (arXiv:2009.11790,
  arXiv:2106.02621).
- **3D tie-degeneracy costs nothing in time — the opposite:** fully-degenerate
  unit weights run 1.15–1.5× *faster* than tie-broken perturbed weights (uniform
  weights batch the region-growth event queue). Ties are ubiquitous in 3D as
  predicted (unit-vs-perturbed corrections differ on 66–97% of rows vs 1.7% in 2D).
- **TRAP (the one real 3D port risk):** `tie_eta = 0.4/N²` falls below pymatching's
  internal weight quantization at 3D sizes — pert vs 2×pert corrections differ on
  1–6% of rows where exact arithmetic mandates 0% (2D 12×12: 0.02%). The frozen
  decoder-A "lexicographic" tie-break silently stops being semantic: still
  deterministic per frozen Matching object, but not reproducible across
  rebuilds/solvers. Port must verify quantization headroom or replace
  η-perturbation with explicit deterministic post-selection among minimal matchings.

### anchor — speed floor + a no-go theorem
- Defect-count-independent (pure (B,V)@(V,N) matmul): 120 µs int64 → **3.3 µs
  float32-GEMM** at 12×12 (GEMM needs B ≳ 8k to amortize).
- Init: the shared `_VertexGraph` all-pairs BFS is O(V^1.9) time / O(V²) memory —
  ~100 s / ~2 GB at V=10⁴. **Fix verified bit-identical:** single-source BFS from
  the anchor only ⇒ 121 ms / 325 MB at V=10,648 (≈800× faster). This fix matters
  for every decoder's 3D port (they share `_VertexGraph`).
- **Theorem (empirically verified at 3 sizes):** any GF(2)-*linear* decoder
  (corr = synd·P) can decode at most V−1 of the N single-link-flip channels to
  themselves; at least N−(V−1) = F channels must receive non-minimal corrections.
  Anchor's wrong set is *exactly* the co-tree links of its BFS tree, corrections =
  link ⊕ fundamental cycle — the ~F·hx² extensive leading-order sign error
  (measured to 0.6% at 1×2, exponent 2.03). Multiple anchors / other trees only
  tune the prefactor; the F·hx² scaling is a cycle-space (topological) invariant.
  **Accuracy therefore fundamentally begins at nonlinear defect pairing.**

### greedy — the cheap near-optimal
- Pairing scan is **d³ confirmed** (fitted exponent 2.87–2.91; ~87 ns/inner
  iteration in Python). Physical-regime 12×12: 5–152 µs/decode uncached ⇒
  0.38–1.0× the as-coded shadow; dense regime blows out 22–26×.
- numba kernel **bit-identical incl. the (dist,u,v) lexicographic tie-break**,
  ×150–170 ⇒ 0.8–27 µs at 12×12 — 8× under the shadow everywhere physical.
- Algorithmic headroom: greedy closest-pair = mutual-nearest-neighbor matching ⇒
  nearest-neighbor-chain gives O(d²) worst case (Eppstein, cs.DS/9912014);
  bucket-queue over integer distances gives O(d² + d·diam). No quantitative
  greedy-matching-decoder accuracy study exists in the QEC literature (nearest:
  arXiv:2602.20238 finite-threshold statement; Astrea-G near-MWPM to d=9).
- **3D caveat:** greedy's mispairing channel (commit the locally closest pair,
  break the global optimum) is **not interference-suppressed** and grows with the
  number of near-equidistant partners in a ball — r³ vs r² — so the measured ~2×
  accuracy penalty vs MWPM should widen in 3D. Re-grade before trusting.

### unionfind — the asymptotic promise, dominated in practice
- Near-linearity visible through Python constants: t ~ N^0.32 at fixed defect
  count, ~d^0.68 in defect count, never super-linear anywhere. Hot spot = the
  per-round growth gather (40→72% with density), not union-find ops (1.0–1.75
  rounds everywhere).
- numba port bit-identical (1000-syndrome check + 3D cross-check), ×23–44 ⇒
  1.3–12 µs at 12×12 (5–48 s/step).
- **On the 2D frontier UF never wins:** in the physical dilute regime greedy is
  simultaneously faster (7 vs 54 µs) *and* more accurate (ceiling 2.8e-6 vs
  1.5e-5); UF beats greedy only at dense/uniform densities where d³ explodes.
  Its 2D error is one mechanism: cluster-merge mispairing of two nearby defect
  pairs (excess-2 corrections on wmin=2 configs ⇒ an hx⁴ channel).
- Literature: Delfosse–Nickerson O(n·α(n)), thresholds 9.9% vs MWPM ~10.3%
  (arXiv:1709.06218); weighted growth narrows but never closes the gap
  (arXiv:2004.04693); no library-grade CPU UF package exists — sparse blossom
  has largely erased UF's CPU speed advantage while being exactly minimal.
- **3D:** ran unmodified on the cubic mock (numba 10–142 µs). Merge-mispairing
  should *worsen* with coordination 6 (swept volume r³; z=3 honeycomb was the
  friendliest case) — 2D's 4–15× ceiling gap is a lower bound for 3D. Role:
  compiled 3D fallback if MWPM ever bottlenecks at high syndrome density.

### tie_sum — retired to diagnostic; v2 = weights, not enumeration
- Two load-bearing code facts (worked around in the harness, must be patched for
  any 6×6+ head): `sign01`'s V ≤ 62 int-packing assert, and `QECSignHead`'s 2^F
  parity table (⇒ per-config `count_loops` at scale).
- Cost is structural: the B×K parity pass (evaluate every class member per row) —
  **25–285× the mwpm head per batch at 12×12 dilute** (one cap-saturated K=4096
  syndrome alone drove a 370 s batch); caches ≤1% relief; 11–43% of dilute rows
  cancel exactly back to the mwpm sign anyway. With defect count extensive at
  fixed field, the constant d_max ⇒ **100% mwpm fallback in every dense batch**
  — silent degradation exactly where the physics is hardest.
- Degeneracy-vs-distance (the 3D-critical curve): 2D honeycomb path counts grow
  ~**1.36^r** (exact brick-lattice binomials, size-independent); 3D cubic
  ~**2.66^r** (multinomial). 3D caps saturate at a single pair separation r=10
  (two pairs at r=6); physical 3D syndromes carry 10⁵–10⁷ minimal recoveries.
  **A capped class sum then returns the sign of a ~10⁻³ sliver of the coherent
  sum — deterministic noise, not a decoder.** Enumeration is DOA in 3D.
- Determinism under caps verified (bitwise across fresh decoders and batch orders).
- **The v2 that survives:** in 2D, exhaustive *resolvent-weighted* enumeration is
  cheap (production dilute chains visit ~280 distinct syndromes/batch, class K
  median 8 / p99 202; all classes enumerate in 0.6 s) — viable once the parity
  kernel is compiled. In 3D: **Stace–Barrett effective-weight MWPM** (fold path
  degeneracy analytically into edge weights, w_eff = w − T·ln #paths, run ONE
  matching — O(1) overhead, enumeration-free) as the default, with weight-ordered
  K-best enumeration (Chegireddy–Hamacher / arXiv:2510.06531) as the small-K
  cross-check. Physics grounding: the S4 swarm proved the true tied signs are
  decided by *unequal resolvent weights* of the minimal recoveries — a
  resolvent-weighted, hz-aware class sign matched ED 440/440 on tied configs and
  graded 0–2e-9 wrong weight (up to 1600× below mwpm) at 1×2.

## 4. 3D port checklist (fermionic code)

1. **Ship MWPM** (pymatching) with an **explicit deterministic tie-break** —
   verify weight-quantization headroom or post-select among minimal matchings
   (the η-perturbation rule silently degrades at 3D sizes).
2. **Single-source BFS init** in `_VertexGraph` (verified bit-identical; 100 s /
   2 GB → 0.12 s / 0.3 GB at V=10⁴) — shared by every decoder.
3. **Patch the small-size shortcuts** now load-bearing: `sign01` V ≤ 62 packing;
   2^F parity table → compiled `count_loops` + the local hexagon-flip parity rule
   (the "dressed-flip covariance" recursion form — already verified in 2D).
4. **Validity checks spot-check only** (never per-call int64 matmuls).
5. **No memo caches** (dedup costs more than decoding; neighborhoods provably
   unique).
6. **Accuracy re-grading required in 3D** for greedy (mispairing) and UF
   (cluster-merge) — both channels are un-suppressed and grow with coordination.
7. **Beyond-MWPM accuracy = weights**: Stace–Barrett effective-weight MWPM
   (+ K-best cross-check). Never deeper enumeration (2.66^r).

## 5. Key references

Sparse blossom: arXiv:2303.15933 (Quantum 9, 1600). Union-find: arXiv:1709.06218
(Quantum 5, 595); weighted UF arXiv:2004.04693; UF-as-approximate-blossom
arXiv:2211.03288. Greedy/mutual-NN: Eppstein cs.DS/9912014. Degeneracy weights:
Stace–Barrett arXiv:0912.1159; multi-path summation arXiv:1709.02154; entropy in
TQEC arXiv:1812.05117; K-best matchings Chegireddy–Hamacher DAM 18, 155 (1987),
arXiv:2510.06531. 3D point-charge matching precedent: arXiv:2009.11790,
arXiv:2106.02621; 3D thresholds quant-ph/0207088, quant-ph/0401101,
arXiv:1808.03092. Geodesic counting: DLMF §26.3.

## 6. Closure — in-vivo timing ladder and the decision to stop in 2D (2026-09-03)

The bench numbers above were followed by the measurement that matters: the head timed *inside* NQS
training (`jobs/nersc_timing_ladder.sh`, `analysis/06_decoder_scaling.ipynb` Figure 1). Same positive
Combo network and DS Hamiltonian at (h_x, h_z) = (0.8, 0.4), one GPU-only arm (no head) plus the five
head arms, 100 minSR steps each on one A100 shared slice, 32 numba threads with the BLAS pool pinned to
one thread (32 + 32 threads starve the kernels 30–50×). Statistic = plateau mean over steps 50–99;
share = Σ t_head / Σ step_wall on the plateau.

| N (size) | GPU-only step | MWPM | greedy | union-find | anchor | tie_sum |
|---|---|---|---|---|---|---|
| 11 (1×2) | 0.35 s | 8.2% | 2.2% | 2.3% | 4.4% | 12% |
| 19 (2×2) | 1.81 s | 3.7% | 1.0% | 1.1% | 2.1% | 7% |
| 27 (2×3) | 4.11 s | 2.9% | 0.9% | 0.9% | 1.3% | 8% |
| 38 (3×3) | 9.02 s | 2.5% | 0.8% | 0.8% | 1.7% | 12% |
| 63 (4×4) | 19.2 s | 2.8% | 1.2% | 1.2% | 2.4% | 30% |
| 94 (5×5) | 33.5 s | 4.0% | 1.9% | 2.0% | 3.2% | 51% |

Verdict: the CPU-resident head costs 1–4% of a VMC step for MWPM, greedy, union-find and anchor at every
size measured — no case for a GPU port or CPU/GPU overlap. tie_sum is the one decoder whose cost grows
into the step at strong field (heavy-tailed per-step cost; mean ≫ median at 4×4), confirming its
retirement. The GPU step itself is launch-latency bound: sampling is 8 · n_sweeps = 4N *sequential*
batch-1024 Metropolis sub-steps, so it grows ~N^1.4–1.5 here and will dominate any production run.

**Not run, by decision (2026-09-03):** 8×8 (N=223, P=159k, Jacobian 10.4 GB) and 10×10 (N=339, P=243k,
15.9 GB). Estimate: GPU-only step 117–190 s / 215–435 s; head 3–20 s for the fast four, 270/550 s for
tie_sum; 100-step runs 3–13 h (fast) and 11–27 h (tie_sum); 12 jobs ≈ 105 GPU-h; shared QOS MaxWall is
48 h so each fits one job. Nothing in it would change the verdict, so the 2D program stops here; the 3D
port checklist in §4 is the deliverable that carries forward.
