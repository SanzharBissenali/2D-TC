"""Numpy-only local validation for the fermionic toric code (--ftc).

Independent replica of the geometry + a dense-kron Hamiltonian builder (no netket),
mirroring scratchpad/validate_dual.py. Checks:

 1. Geometry: qubit count, star arities (each link in exactly 2 stars), plaquette
    count (L-1)^2, dressed-star count (L-1)^2 with exactly 2 shared links each,
    corner (0,0) fully shared (star = {up, right} both inside the NE plaquette).
 2. Pairwise commutation of ALL stabilizers (dressed/bare stars + plaquettes),
    symbolically via Pauli anticommutation counting (L = 2, 3, 4).
 3. Dense (L = 2, 3): A'_v equals the matrix product A_v @ B_NE(v); equals the
    explicit -1 * (Y on shared, X on star-only, Z on plaq-only) Pauli string
    (sign-convention check: X.Z = -i sigma_y per shared link, (-i)^2 = -1);
    Hermitian, squares to identity. Built EXACTLY as hamiltonian.py consumes
    (x_links products, then z_links products).
 4. Dense spectra (L = 2, 3): E0 = -(#stars + #plaquettes); unperturbed fTC ground
    state IDENTICAL to the TC ground state (overlap 1, positive amplitudes);
    TC gap = 2 but fTC gap = 4 -- the OBC relation Prod_v A'_v = Prod_p B_p ties
    m-parity to e-parity, so a single flux is forbidden and excitations pair up.
    (Consequence: perturbed fTC needs NEW ED reference points; TC files don't apply.)
 5. Field cuts (L = 2, 3; hx or hz = 0.23): E0(fTC) vs E0(TC) and the sign structure
    (fraction + weight of negative GS amplitudes) -> decides the real-vs-complex
    ansatz per cut. h=0 must be exactly sign-free.

Run:  python scratchpad/validate_ftc.py     (~1-2 min; eigh at dim 4096)
"""

import numpy as np

# ---------------------------------------------------------------- geometry replica

def build_geometry(L):
    """Replicates model/geometry.py OBC conventions (order may differ from netket's
    site order; every check below is order-independent)."""
    pts = []
    for i in range(L):
        for j in range(L):
            pts += [[i + 0.5, j], [i, j + 0.5]]
    arr = np.array([p for p in pts if 0 <= p[0] <= L - 1 and 0 <= p[1] <= L - 1])
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]

    def to1d(c):
        hit = np.argwhere((arr[:, 0] == c[0]) & (arr[:, 1] == c[1]))
        return int(hit[0][0]) if len(hit) else -1

    offs = [np.array(o) for o in ([0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5])]
    v_pos = [np.array([i, j]) for i in range(L) for j in range(L)]
    p_pos = [np.array([i + 0.5, j + 0.5]) for i in range(L - 1) for j in range(L - 1)]
    stars = [[to1d(c + o) for o in offs] for c in v_pos]
    plaqs = [[to1d(c + o) for o in offs] for c in p_pos]

    # dressed stars: (x_links, z_links); z_links = NE plaquette of v, [] at boundary
    dressed = []
    for c, star in zip(v_pos, stars):
        x_links = [q for q in star if q != -1]
        ne = c + np.array([0.5, 0.5])
        hit = [k for k, pc in enumerate(p_pos) if pc[0] == ne[0] and pc[1] == ne[1]]
        z_links = plaqs[hit[0]] if hit else []
        dressed.append((x_links, z_links))
    return arr, stars, plaqs, dressed


# ---------------------------------------------------------------- dense Pauli tools

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on(site_mats, N):
    """Dense operator from [(site, 2x2)] factors; same-site factors multiply in the
    given order (exactly how LocalOperator products behave in hamiltonian.py)."""
    loc = [I2.astype(complex) for _ in range(N)]
    for s, m in site_mats:
        loc[s] = loc[s] @ m
    out = loc[0]
    for m in loc[1:]:
        out = np.kron(out, m)
    return out


def dressed_dense(x_links, z_links, N):
    """As consumed by hamiltonian.py: all sigma_x factors first, then all sigma_z."""
    return op_on([(j, SX) for j in x_links] + [(j, SZ) for j in z_links], N)


def build_H(L, hx=0.0, hz=0.0, ftc=False):
    _, stars, plaqs, dressed = build_geometry(L)
    N = 2 * L * L - 2 * L
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    if ftc:
        for x_links, z_links in dressed:
            H -= dressed_dense(x_links, z_links, N)
    else:
        for star in stars:
            H -= op_on([(j, SX) for j in star if j != -1], N)
    for pl in plaqs:
        H -= op_on([(j, SZ) for j in pl], N)
    for j in range(N):
        if hx:
            H -= hx * op_on([(j, SX)], N)
        if hz:
            H -= hz * op_on([(j, SZ)], N)
    assert np.max(np.abs(H.imag)) < 1e-12, "H must be real (even Y count)"
    return H.real


def pauli_dict(x_links, z_links):
    """Site -> 'X'/'Y'/'Z' for a dressed (or bare, z_links=[]) star."""
    d = {}
    for j in x_links:
        d[j] = 'X'
    for j in z_links:
        d[j] = 'Y' if j in d else 'Z'
    return d


def commutes(d1, d2):
    return sum(1 for s in set(d1) & set(d2) if d1[s] != d2[s]) % 2 == 0


def neg_stats(vec):
    v = np.real(vec) * np.sign(np.real(vec[np.argmax(np.abs(vec))]))
    cut = 1e-12 * np.max(np.abs(v))
    return float(np.mean(v < -cut)), float(np.sum(v[v < -cut] ** 2))


# ---------------------------------------------------------------- checks

for L in (2, 3, 4):
    arr, stars, plaqs, dressed = build_geometry(L)
    N = 2 * L * L - 2 * L
    assert len(arr) == N
    arities = [sum(1 for q in s if q != -1) for s in stars]
    assert sum(arities) == 2 * N, "each link must belong to exactly 2 stars"
    assert len(plaqs) == (L - 1) ** 2 and all(-1 not in p for p in plaqs)
    ndress = sum(1 for _, z in dressed if z)
    assert ndress == (L - 1) ** 2, f"expected {(L-1)**2} dressed stars, got {ndress}"
    for x_links, z_links in dressed:
        if z_links:
            shared = set(x_links) & set(z_links)
            assert len(shared) == 2, "every dressed star has exactly 2 shared links"
    x0, z0 = dressed[0]  # vertex (0,0): corner, star = {right, up}, both shared
    assert len(x0) == 2 and z0 and set(x0) <= set(z0), "corner star fully shared"
    print(f"[1] L={L}: geometry OK (N={N}, {ndress} dressed + {len(stars)-ndress} bare stars)")

    ops = [pauli_dict(x, z) for x, z in dressed] + [pauli_dict([], p) for p in plaqs]
    npairs = 0
    for a in range(len(ops)):
        for b in range(a + 1, len(ops)):
            assert commutes(ops[a], ops[b]), f"stabilizers {a},{b} do not commute"
            npairs += 1
    print(f"[2] L={L}: all {len(ops)} stabilizers pairwise commute ({npairs} pairs)")

for L in (2, 3):
    arr, stars, plaqs, dressed = build_geometry(L)
    N = 2 * L * L - 2 * L
    nstab = len(stars) + len(plaqs)

    for v, (x_links, z_links) in enumerate(dressed):
        if not z_links:
            continue
        Ap = dressed_dense(x_links, z_links, N)
        Av = op_on([(j, SX) for j in x_links], N)
        Bp = op_on([(j, SZ) for j in z_links], N)
        assert np.allclose(Ap, Av @ Bp), "A'_v != A_v @ B_NE(v)"
        assert np.allclose(Ap, Ap.conj().T) and np.allclose(Ap @ Ap, np.eye(2 ** N))
        d = pauli_dict(x_links, z_links)
        mats = {'X': SX, 'Y': SY, 'Z': SZ}
        string = -op_on([(s, mats[p]) for s, p in d.items()], N)
        assert np.allclose(Ap, string), "sign convention: A'_v must equal -Y.Y.X..Z.."
    print(f"[3] L={L}: dressed stars == A_v @ B_NE == -1 * (Y,Y,X..,Z..) strings")

    H_tc = build_H(L, ftc=False)
    H_ft = build_H(L, ftc=True)
    w_tc, v_tc = np.linalg.eigh(H_tc)
    w_ft, v_ft = np.linalg.eigh(H_ft)
    assert abs(w_tc[0] + nstab) < 1e-9 and abs(w_ft[0] + nstab) < 1e-9, "E0 = -#stabilizers"
    assert w_tc[1] - w_tc[0] > 1e-9 and w_ft[1] - w_ft[0] > 1e-9, "unique GS"
    assert abs(abs(np.vdot(v_tc[:, 0], v_ft[:, 0])) - 1) < 1e-9, "fTC GS != TC GS"
    gap_tc, gap_ft = w_tc[1] - w_tc[0], w_ft[1] - w_ft[0]
    assert abs(gap_tc - 2) < 1e-9, f"TC gap should be 2, got {gap_tc}"
    assert abs(gap_ft - 4) < 1e-9, f"fTC gap should be 4 (paired excitations), got {gap_ft}"
    frac, wgt = neg_stats(v_ft[:, 0])
    assert frac == 0.0, "unperturbed fTC GS must be positive"
    print(f"[4] L={L}: E0={w_ft[0]:+.6f}=-{nstab}, shared GS (positive), "
          f"gaps TC={gap_tc:.1f} / fTC={gap_ft:.1f}")

    for tag, kw in (("hz", dict(hz=0.23)), ("hx", dict(hx=0.23))):
        wt, _ = np.linalg.eigh(build_H(L, ftc=False, **kw))
        wf, vf = np.linalg.eigh(build_H(L, ftc=True, **kw))
        frac, wgt = neg_stats(vf[:, 0])
        print(f"[5] L={L} {tag}=0.23: E0 fTC={wf[0]:+.6f} vs TC={wt[0]:+.6f} "
              f"(dE0={wf[0]-wt[0]:+.2e}, max spec diff={np.max(np.abs(wf-wt)):.3f}); "
              f"fTC GS neg-amp fraction={frac:.4f}, weight={wgt:.4f}")

print("\nALL FTC CHECKS PASSED")
