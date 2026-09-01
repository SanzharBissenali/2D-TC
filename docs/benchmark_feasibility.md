# Trusted large-scale benchmarks for the perturbed doubled semion (hx ≠ 0) — feasibility memo

*2026-09-02, branch `doubled-semion`. Question: is there any trusted method giving E0 (or the sign structure) of H_DS + hx Σσˣ at 50–500 qubits, so NQS accuracy can be graded vs. system size beyond the 27-qubit ED ceiling? Claims are cited; my own inferences are marked [inference].*

## TL;DR

1. **Nothing published benchmarks DS + hx at any size.** All "sign-free DS" results (Shackleton 2509.03708; Dupont–Gazit–Scaffidi 2008.06509) and all perturbed-string-net numerics live on the **charge-conserving axis** (string tension ≡ our hz, where DS ≡ TC in the GS sector) or on Q_v-preserving TC↔DS interpolations.
2. Both sign-free QMC constructions state explicitly that they need **frozen gauge flux**; that flux is our Q_v, and σˣ on a link is exactly the operator that hops it. **QMC: no-go** without a new idea.
3. **DMRG on our own OBC strips (2×Ly, 3×Ly, 4×Ly) is the only trusted, scalable benchmark**, a 3–5 day TeNPy build. DMRG is sign-agnostic; H is real with {0,±1} entries; Shackleton herself cross-checks her DS QMC with DMRG on an 8×4 open cylinder.
4. High-order series (pCUT/linked-cluster) is possible in principle (integer H0 spectrum) but unpublished for DS, weeks of work, useful only at hx ≲ 0.2 — a later cross-check.
5. "Best arm = reference" is standard NQS practice (Rayleigh–Ritz + V-score, Wu et al. 2302.04919) but relative only; our arms share the QEC head, so a common head bias is invisible.

| Method | Applies to DS+hx? | Trusted? | Effort | Verdict |
|---|---|---|---|---|
| Sign-free QMC (SSE, Shackleton; dual QMC, Dupont et al.) | **No** — both need static Q_v flux; hx hops it | n/a | new-idea research | **NO-GO** |
| Strip DMRG on our 2×Ly…4×Ly patches | Yes (real, local H; signs irrelevant) | Yes, χ-converged, exponential in χ on gapped strips | 3–5 days | **GO — build first** |
| iPEPS / TN literature | no DS+hx energies published; iPEPS ~1e-4 | variational only | — | no (as reference) |
| pCUT / linked-cluster series | in principle; nothing for DS | to truncation error; hx ≲ 0.2 | 2–4 weeks | later cross-check |
| Exact mapping / ungauging | **No** — σˣ_link = flux hopping, non-local in the SPT frame | — | — | NO-GO |
| Best-arm bound + V-score | yes (rigorous upper bound) | relative only | 0 | use, not as absolute reference |

## 1. Quantum Monte Carlo

**Hastings 1506.08883** (J. Math. Phys. 57, 015210) proves less than the folklore: verbatim, "the sign problem is intrinsic for commuting Hamiltonians in the same phase as the double semion model under the technical assumption that TQO-2 holds"; the general local-basis statement is a stated conjecture. Smith–Golan–Ringel 2005.05343 give the S/T-matrix criterion for intrinsic sign problems under local unitaries.

**Shackleton 2509.03708** (PRL 2025) is a genuine counterpoint, but to a different claim: SSE sign-freeness "is fundamentally unrelated to wavefunction positivity". Mechanism (App. B): Levin–Gu SPT = U·(trivial)·U† with U = (−1)^{N_dw} *non-local diagonal*; after gauging, every closed SSE operator cycle has phase 1 — but only because "any non-trivial cycle must live in the flux-free subspace", with vertex operators dressed by projectors so they "only act when the flux on neighboring plaquettes is zero". Sign-free perturbations listed: anything diagonal in the computational basis, the gauge-invariant Ising term J σᶻτᶻσᶻ, and the SPT↔trivial interpolation. Her conclusion: "our results do rely heavily on the assumption that the gauge fluxes are frozen, i.e. no term in the Hamiltonian induces a transition to different flux sectors. It is unclear whether this constraint is fundamental." Dictionary to our model [inference via Levin–Gu 1202.3120 §IV, whose triangle flux μᶻμᶻμᶻ is our Q_v]: her flux = Q_v; her τˣ_link = our σˣ_link (hx); her Ising term, gauge-fixed, = our σᶻ_link (hz). Her numerics (SSE, 16×16 triangular PBC, β = 16–20, J_c ≈ 0.4–0.44, order parameter not E0, DMRG check on 8×4 open cylinder) therefore probe exactly the axis where our sector theorem makes DS ≡ TC. **Our hx is the excluded operator.**

**Dupont–Gazit–Scaffidi 2008.06509** (PRB 103, L140412) simulate the *ungauged* trivial↔Levin–Gu interpolation (10×10 PBC, projective + SSE QMC), dual to TC↔DS. Their SM: sign-free only because "(i) we use periodic boundary conditions, and (ii) π-flux excitations only appear as gapped, static excitations in the gauged models. Relaxing any of these conditions would most likely generate a sign problem." We have OBC *and* dynamical flux. No QMC on twisted doubles / Levin–Wen models with string-breaking fields was found.

**Verdict.** A σᶻ-basis SSE is sign-free iff the configuration graph of H_DS+hx is Harary-balanced (a diagonal ±1 gauge, possibly non-local — Hastings allows that). Our gate-0 data argue against it [inference]: tie configs are interference-suppressed (~hx^10) and F_s < 1 for every decoder — signatures of frustrated cycles, not of a hidden gauge. A one-hour brute-force balance test at 1×1/1×2/2×2 (as done for the fTC at L=2) settles whether any computational-basis QMC is conceivable; a sign-free scheme in another basis would be new physics, not a tool.

## 2. Tensor networks

Literature: Morampudi–von Keyserlingk–Pollmann 1403.0768 (PRB 90, 035117) — 1D exact + **2D ED only**, no link field. Schotte et al. 1909.06284 (PRB 100, 245125), Mariën et al. 1607.05296 (PRB 96, 155127), Schulz et al. 1212.4109 (PRL 110, 147203) and 1401.1033 (PRB 89, 201103), Dusuel–Vidal 1506.03259 (PRB 92, 125150) — all **string tension** (our hz), Z2/Fibonacci/Ising string nets. Zhang–Xu 2508.08376 — iPEPS (D ≤ 7) on a perturbed Z4 quantum double, constraints preserved, **no energies**. Ritz-Zwilling et al. 2108.13425 — unperturbed partition function. **None treats a Q_v-violating field on the DS or reports reference-grade energies.**

**Our own DMRG — GO.** Strips are our native geometry: 2×Ly has N = 8Ly+3 (51/99/243 at Ly = 6/12/30), 3×Ly 11Ly+5, 4×Ly 14Ly+7. Build: snake the `honeycomb_geometry` links along Ly, `SpinHalfSite` chain, H via TeNPy `MPOGraph` / `add_multi_coupling_term` from the real Pauli strings the netket builder already emits (Q_v = ZZZ; each DS plaquette ≤ 2^10 strings on 12 sites — or ≤ 64 if D_p stays as on-site diag(1,i) and only ∏(1+Q_v)/2 is expanded, at the price of a complex MPO). MPO bond dimension ≈ number of distinct partial strings crossing a bond ≲ Σ_{straddling hexagons} 2^{min(sites left,right)} ≈ few × 64 before compression — a few hundred [estimate], harmless (cost linear in it). Entanglement across a width-Lx cut in the gapped phase is ~a·Lx − γ, so χ ≈ 500–2000 (2×Ly) and 2000–6000 (4×Ly) should reach 1e-9 [estimate]; truncation-error/χ extrapolation certifies it. Validate on the 32 committed ED points (1×1…2×3, hx ≤ 0.2) to 1e-10 plus per-config signs vs the head at 1×2. Bonus: perfect sampling from the MPS gives an unbiased F_s of the head at 50–250 qubits — the sign-fidelity diagnostic beyond ED. Caveat: 2×Ly tests N-scaling along one axis; 3×Ly/4×Ly add 2D bulk at higher χ.

## 3. Series expansions

No series exists for the DS in any field. TC precedents: Dusuel–Kamfor–Orús–Schmidt–Vidal 1012.1740 (PRL 106, 107203) — pCUT e0 to **order 10**, gap to order 8, + iPEPS (square); Kott–Mühlhauser–Koziol–Schmidt 2402.15389 (SciPost Phys. 17, 053) — full-graph linked-cluster + SSE for the **honeycomb TC in x/z fields** (our lattice; their z-field is our hx by convention swap, mapping the charge-free sector to the honeycomb TFIM, so hx_c(TC) ≈ 1/2.13 ≈ 0.47 [inference from the standard TFIM value]). For the DS, pCUT applies in principle: integer H0 spectrum (Q_v = ±1, plaquette ∈ {−1,0,+1}); one σˣ flip costs 8 (2 charges + 4 projector-killed plaquettes), so convergence at hx ≤ 0.2 should be good. But the leg-dependent plaquette makes the quasi-particle algebra far richer than the TC's, the finite OBC patch needs boundary-cluster classes, and 1e-8 relative at hx = 0.2 needs order ≳ 14 with unknown radius [estimate]; at hx = 0.4, no. Realistic yield after 2–4 weeks: 1e-6–1e-7 bulk energy density at hx ≤ 0.2 — comparable to the NQS error there (6e-6 / 3e-5 at hx = 0.1 / 0.2). **Later cross-check, not the reference.**

## 4. Other routes

- **Exact mappings.** TC + hx ↔ honeycomb TFIM only because [σˣ, X_hex] = 0. For the DS, σˣ on a hexagon link or leg does not commute with D_p·P_p, so no plaquette sector is conserved (our ED: DS ⟨plaq⟩ moves under hx, TC ⟨X̂ₚ⟩ ≡ 1) — no dual TFIM; the TC↔DS unitary (−1)^{#loops} is non-local and does not commute with σˣ.
- **Ungauging.** Levin–Gu §IV: Q_v ↔ triangle flux; σᶻ_link ↔ Ising τᶻτᶻ (local after ungauging — the hz axis, Shackleton's J term); σˣ_link creates a flux pair ↔ endpoints of a symmetry-twist branch cut, with **no local symmetric counterpart** [inference; Levin–Gu do not discuss link σˣ terms]. Same obstruction as §1.
- **Variational convention.** Each arm's exact expectation is a Rayleigh–Ritz upper bound, so the lowest arm is a legitimate reference; the NQS literature does this and uses the V-score where no exact reference exists (Wu et al. 2302.04919, Science 386, 296 (2024)). Relative only; B/C/R share the head, so a head bias is invisible — DMRG is needed for absolute grading.

## Recommended multi-day path

1. **Day 0 (1 h):** Harary-balance test of H_DS+hx at 1×1/1×2/2×2 — closes the σᶻ-basis QMC question.
2. **Days 1–2:** TeNPy strip DMRG: geometry snake + MPO from existing Pauli strings; validate vs all 32 ED points to 1e-10; per-config signs vs head at 1×2.
3. **Days 3–5:** production 2×Ly (Ly = 6, 12, 24, 30) at hx ∈ {0.1, 0.2, 0.3, 0.5}, hz ∈ {0, 0.2}, χ-extrapolated; then 3×Ly, 4×Ly; MPS-sampled F_s of the head per size. NQS (cnnqB) on the same patches → rel-err vs N: the deliverable.
4. **Later, optional:** DS pCUT/linked-cluster series for the bulk energy density at hx ≤ 0.2 as an independent cross-check of the DMRG bulk value.
