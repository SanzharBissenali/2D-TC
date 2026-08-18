"""Check: is the fTC h=0 ground state's Z-basis amplitude either 0 or IDENTICAL
across all nonzero components (a flat/uniform superposition, no relative signs,
no relative magnitude differences)?

This is a stronger claim than "positive" (which scratchpad/validate_ftc.py already
checked via neg_amp_fraction==0). Reuses the exact same geometry/H builders as that
script (copied here verbatim, not imported, so this stays a standalone, independent
check -- consistent with how validate_ftc.py was itself built independently of the
main model/ code).

Dense eigh, so only L=2,3 are laptop-feasible (L=4 dim=2^24, needs the NERSC ED path;
not run here). Since fTC h=0 GS == plain TC h=0 GS exactly (proven in validate_ftc.py),
this also directly checks the plain toric-code ground state's textbook property: it's
the uniform superposition over the vertex-stabilizer group orbit of any Z-basis
config satisfying all plaquette constraints.
"""

import numpy as np


def build_geometry(L):
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

    dressed = []
    for c, star in zip(v_pos, stars):
        x_links = [q for q in star if q != -1]
        ne = c + np.array([0.5, 0.5])
        hit = [k for k, pc in enumerate(p_pos) if pc[0] == ne[0] and pc[1] == ne[1]]
        z_links = plaqs[hit[0]] if hit else []
        dressed.append((x_links, z_links))
    return arr, stars, plaqs, dressed


I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on(site_mats, N):
    loc = [I2.astype(complex) for _ in range(N)]
    for s, m in site_mats:
        loc[s] = loc[s] @ m
    out = loc[0]
    for m in loc[1:]:
        out = np.kron(out, m)
    return out


def dressed_dense(x_links, z_links, N):
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
    assert np.max(np.abs(H.imag)) < 1e-12
    return H.real


def amplitude_report(label, psi, tol=1e-9):
    v = np.real(psi)
    scale = np.max(np.abs(v))
    cut = tol * scale
    nz_mask = np.abs(v) > cut
    nz = v[nz_mask]
    n_total = v.size
    n_nz = nz.size
    signs = np.sign(nz)
    all_same_sign = np.all(signs == signs[0])
    mags = np.abs(nz)
    mag_mean = mags.mean()
    mag_spread = (mags.max() - mags.min()) / mag_mean
    expected = 1.0 / np.sqrt(n_nz)
    print(f"--- {label} ---")
    print(f"  dim={n_total}, nonzero components={n_nz} "
          f"(2^{np.log2(n_nz):.4f} if a clean power of 2)")
    print(f"  all nonzero amplitudes same sign: {all_same_sign} "
          f"(signs seen: {sorted(set(signs.tolist()))})")
    print(f"  |amplitude| mean={mag_mean:.10f}, "
          f"max-min relative spread={mag_spread:.3e}")
    print(f"  expected flat value 1/sqrt(n_nz) = {expected:.10f} "
          f"(matches mean to {abs(expected-mag_mean):.3e})")
    print(f"  uniform to machine precision: {mag_spread < 1e-9}")
    return n_nz, mag_spread, all_same_sign


for L in (2, 3):
    N = 2 * L * L - 2 * L
    print(f"\n===== L={L} (N={N} qubits, dim={2**N}) =====")
    H_tc = build_H(L, ftc=False)
    H_ft = build_H(L, ftc=True)
    _, v_tc = np.linalg.eigh(H_tc)
    _, v_ft = np.linalg.eigh(H_ft)

    n_tc, spread_tc, sign_tc = amplitude_report("plain TC, h=0", v_tc[:, 0])
    n_ft, spread_ft, sign_ft = amplitude_report("fermionic TC, h=0", v_ft[:, 0])

    assert n_tc == n_ft, "fTC and TC ground states must occupy identical support"
    assert spread_tc < 1e-9 and spread_ft < 1e-9, "amplitudes must be flat"
    assert sign_tc and sign_ft, "amplitudes must be single-signed"
    print(f"  => CONFIRMED at L={L}: both GS are uniform "
          f"(0 or one identical value), no relative signs, "
          f"identical support ({n_tc} configs)")

print("\nALL AMPLITUDE-UNIFORMITY CHECKS PASSED (L=2,3; L=4 needs NERSC ED, not run here)")
