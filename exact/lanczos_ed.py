"""
Exact diagonalization (sparse Lanczos) benchmark for the mixed-field toric code.

Reuses the *same* geometry and Hamiltonian builder as the NQS simulation
(model/geometry.py, model/hamiltonian.py) so the ED energy is computed for
exactly the operator the network is trained against.

For L=4 (24 qubits, Hilbert dim 2^24 ~ 1.7e7) this is memory-heavy and is meant
to run on a NERSC CPU node, not locally. L<=3 runs on a laptop.

Example:
    python -m exact.lanczos_ed --Lx 4 --hx 0.0 --hz 0.1 --k 4 \
        --out results/ed/ed_L4_hx0.00_hz0.10.json
"""

import argparse
import json
import os
import time

import numpy as np
import netket as nk

from model.geometry import ToricCodeGeometry
from model.hamiltonian import create_hamiltonian, create_honeycomb_hamiltonian
from model.honeycomb_geometry import HoneycombGeometry


def _expect(psi, sparse_op):
    """<psi| op |psi> for a normalized state vector and a scipy sparse operator."""
    return float(np.real(np.vdot(psi, sparse_op @ psi)))


def _honeycomb_direct_ed(geometry, model, J, hx, hz, k, ncv=None, tol=0, hy=0.0,
                         return_H=False):
    """Direct scipy Lanczos for the honeycomb models -- bypasses netket's
    Pauli->sparse conversion, whose intermediates OOM at 2^27 (observed: job
    57315881 lost even the 7-pattern TC h=0 build to exit 137 on the 55 GB
    shared node, while the final CSR is only ~12 GB).

    Basis convention (used by ALL honeycomb ED downstream code): site i <-> bit
    i of the state index, bit 1 = spin DOWN (sigma^z = -1), all-up config =
    index 0. H row r holds the diagonal plus ONE entry per flip pattern
    (F hexagons; N single flips if hx != 0) at column c = r^mask with value
    w(c), so the CSR arrays are assembled directly by strided writes -- no COO
    intermediates. Memory at 2^27: ~15 GB for h=0/hz (shared-node safe),
    ~60 GB + Lanczos workspace for hx != 0 (regular node).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    N, V, F = geometry.N, geometry.n_vertices, geometry.n_plaqs
    dim = 1 << N
    s = np.arange(dim, dtype=np.int64)

    def zval(i):
        return (1 - 2 * ((s >> int(i)) & 1)).astype(np.int8)

    def qval(v):
        q = np.ones(dim, dtype=np.int8)
        for l in geometry.vertex_all[int(v)]:
            if l != -1:
                q = q * zval(l)
        return q

    Hd = np.zeros(dim)
    for v in range(V):
        Hd -= J * qval(v)
    if hz != 0.0:
        for i in range(N):
            Hd -= hz * zval(i)

    masks, weights = [], []   # weight = scalar, or per-COLUMN int8 vector (x J)
    for p in range(F):
        mask = 0
        for e in geometry.plaq_all[p]:
            mask |= 1 << int(e)
        masks.append(mask)
        if model == "ds":
            w = np.ones(dim, dtype=np.int8)
            for v in geometry.plaq_vertices[p]:
                w = w * ((1 + qval(v)) // 2).astype(np.int8)
            nsum = np.zeros(dim, dtype=np.int8)
            for l in geometry.legs_all[p]:
                if l != -1:
                    nsum += ((1 - zval(l)) // 2).astype(np.int8)
            # D(s)P(s): P in {0,1} forces even leg parity, where D = (-1)^{nsum/2}
            weights.append(w * np.where(nsum % 4 == 0, 1, -1).astype(np.int8))
        else:
            weights.append(None)  # constant -J
    if hx != 0.0 or hy != 0.0:
        # hx and hy both connect r <-> r^bit(i): ONE merged mask per site.
        # <c|sigma^y_i|r> = i*z_r(i)  (sigma^y|up> = i|down>, sigma^y|down> = -i|up>),
        # so the entry is -hx - i*hy*z_r(i): Hermitian since z flips with the bit.
        for i in range(N):
            masks.append(1 << i)
            weights.append(("xy", i))

    P = len(masks)
    step = P + 1
    cdtype = np.complex128 if hy != 0.0 else np.float64
    indices = np.empty(dim * step, dtype=np.int32)
    data = np.empty(dim * step, dtype=cdtype)
    indices[0::step] = s
    data[0::step] = Hd
    for j, (m, w) in enumerate(zip(masks, weights)):
        c = s ^ m
        indices[j + 1::step] = c
        if isinstance(w, np.ndarray):
            data[j + 1::step] = J * w[c].astype(np.float64)   # +J X.D.P (ds)
        elif isinstance(w, tuple):                            # merged field flip
            i = w[1]
            if hy != 0.0:
                # CSR entry is H[r, c] = <r|H|c> with c = r^bit(i):
                # <r|sigma^y_i|c> = i*z_c(i) = -i*z_r(i)  =>  -hy term = +i*hy*z_r(i)
                # (swarm finding 2026-09-01: evaluating the KET's z at the row
                # requires the sign flip; the conjugate matrix is Hermitian too,
                # which is why E0/ceiling gates could not catch it)
                data[j + 1::step] = -hx + 1j * hy * zval(i).astype(np.float64)
            else:
                data[j + 1::step] = -hx
        else:
            data[j + 1::step] = -J                            # -J X_hex (tc)
    del Hd, weights
    indptr = np.arange(dim + 1, dtype=np.int64) * step
    H = sp.csr_matrix((data, indices, indptr), shape=(dim, dim))
    del data, indices

    # ncv: TC+hx conserves every X_hex, so its low spectrum sits in exactly
    # degenerate flux-sector clusters -- ARPACK with the default ncv=20 thrashes
    # (jobs 57338156/62 hit 1:45 walls at 2^27 while the non-degenerate DS
    # points converged in ~50 min). A larger Krylov block (ncv ~ 48) is the fix;
    # memory cost is ncv vectors (1 GB each at 2^27) -- regular-node territory.
    # tol=0 (default) = ARPACK machine precision, as in the ED campaign.
    # A loose tol (~1e-8) cuts iterations for vector-consumers that only
    # need amplitude signs (scripts/sign_fidelity.py) inside debug walltime.
    evals, evecs = eigsh(H, k=k, which="SA", ncv=ncv, tol=tol)
    order = np.argsort(evals)
    if return_H:
        return evals[order], np.asarray(evecs[:, order[0]]), H
    return evals[order], np.asarray(evecs[:, order[0]])


def _honeycomb_observables(psi0, geometry, model):
    """Matrix-free expectation values in the honeycomb ground state (Tier 2).

    Unlike the square path's per-site sparse-operator pass (expensive at 2^24),
    nothing here builds an operator: diagonal observables (sigma^z, Q_v) come
    straight from |psi|^2 via bit arithmetic, and the plaquette term / sigma^x
    have exactly ONE connected element per basis state, so each is a single
    XOR-indexed inner product. Uses the _honeycomb_direct_ed bit convention
    (site i <-> bit i, bit 1 = down). O((N+V+F) passes over 2^N).

    DS plaquette weight D(s)P(s) is computed real: P = prod (1+Q_v)/2 in {0,1},
    and on its image the leg down-parity is even, so D = (-1)^{(sum n_legs)/2}
    (odd-parity configs get an arbitrary sign there -- P already zeroes them).
    """
    N = geometry.N
    dim = 1 << N
    psi = np.asarray(psi0)
    if not np.iscomplexobj(psi):
        psi = np.real(psi)
    prob = np.abs(psi) ** 2
    psic = np.conj(psi)

    s = np.arange(dim, dtype=np.int64)
    _zc = {}

    def zval(i):
        """sigma^z eigenvalue (+1 = up) of site i for every state index (cached)."""
        i = int(i)
        if i not in _zc:
            _zc[i] = (1 - 2 * ((s >> i) & 1)).astype(np.int8)
        return _zc[i]

    def qval(v):
        q = np.ones(dim, dtype=np.int8)
        for l in geometry.vertex_all[int(v)]:
            if l != -1:
                q = q * zval(l)
        return q

    obs = {}
    sz = [float((prob * zval(i)).sum()) for i in range(N)]
    # <sigma^x_i> = sum_r conj(psi[r^m]) psi[r]; real by hermiticity (exactly so
    # for real psi; take .real for the complex hy path)
    sx = [float(np.real((psic[s ^ (1 << i)] * psi).sum())) for i in range(N)]
    obs["magnetization_Z"] = sz
    obs["magnetization_X"] = sx
    if np.iscomplexobj(psi):
        # <sigma^y_i> = sum_r conj(psi[r^m]) * (i z_r(i)) * psi[r]
        sy = [float(np.real((psic[s ^ (1 << i)] * (1j * zval(i)) * psi).sum()))
              for i in range(N)]
        obs["magnetization_Y"] = sy
        obs["magnetization_Y_mean"] = float(np.mean(sy))
    obs["magnetization_Z_mean"] = float(np.mean(sz))
    obs["magnetization_X_mean"] = float(np.mean(sx))

    qv = [float((prob * qval(v)).sum()) for v in range(geometry.n_vertices)]
    obs["Qv_per_vertex"] = qv
    obs["Qv_mean"] = float(np.mean(qv))
    obs["Qv_min"] = float(np.min(qv))

    plaq = []
    for p in range(geometry.n_plaqs):
        mask = 0
        for e in geometry.plaq_all[p]:
            mask |= 1 << int(e)
        if model == "ds":
            w = np.ones(dim, dtype=np.int8)
            for v in geometry.plaq_vertices[p]:
                w = w * ((1 + qval(v)) // 2).astype(np.int8)
            nsum = np.zeros(dim, dtype=np.int8)
            for l in geometry.legs_all[p]:
                if l != -1:
                    nsum += ((1 - zval(l)) // 2).astype(np.int8)
            w = w * np.where(nsum % 4 == 0, 1, -1).astype(np.int8)
            plaq.append(float(np.real((psic[s ^ mask] * w * psi).sum())))
        else:
            plaq.append(float(np.real((psic[s ^ mask] * psi).sum())))
    obs["plaq_term_per_hex"] = plaq
    obs["plaq_term_mean"] = float(np.mean(plaq))
    return obs


def run_ed(Lx, hx, hz, hy=0.0, J=1.0, bc="OBC", k=1, observables=True, ftc=False,
           lattice="square", model="tc", Ly=0, ncv=0):
    """Diagonalize the toric-code Hamiltonian and return a results dict."""
    hinfo = None
    if lattice == "honeycomb":
        # Levin-Gu honeycomb TC / doubled semion on the smooth-OBC brick-wall patch.
        # Both models are exactly real (even Y-count strings) => float64 + the sign
        # diagnostics below, which are the primary deliverable for the DS model
        # ((-1)^{#loops} ground state). Square-lattice observables don't apply.
        assert bc == "OBC" and not ftc, \
            "honeycomb ED: smooth OBC only, no ftc"
        Ly = Ly if Ly else Lx
        dtype = "complex" if hy != 0.0 else "float64"
        geometry = HoneycombGeometry(Lx, Ly)
        H = None  # honeycomb ED bypasses netket entirely -- see _honeycomb_direct_ed
        hinfo = {"builder": "direct-scipy"}
        # observables stay ON: the honeycomb pass is matrix-free (Tier 2, see
        # _honeycomb_observables) -- no sparse operator builds, minutes even at 2^27.
    else:
        assert model == "tc", \
            "--model ds requires --lattice honeycomb (square branch would run the " \
            "plain TC and mislabel the output JSON)"
        # ftc dressed stars are real (even Y count: X.Z = -i*sigma_y on the two shared
        # links, (-i)^2 = -1), so the operator stays float64 like the plain TC.
        dtype = "complex" if hy != 0.0 else "float64"
        Ly = Lx
        geometry = ToricCodeGeometry(Lx, Lx, bc)
        hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)

        H = create_hamiltonian(
            hi=hi,
            vertex_all=geometry.vertex_all,
            plaq_all=geometry.plaq_all,
            bonds=geometry.bonds,
            hx=hx, hy=hy, hz=hz, J=J,
            ftc=ftc,
            dressed_stars=geometry.dressed_stars,
            dtype=dtype,
        )

    t0 = time.time()
    if lattice == "honeycomb":
        # Direct scipy path in OUR bit convention (site i <-> bit i, all-up = 0).
        # netket's Pauli->sparse conversion intermediates OOM at 2^27 even for the
        # 7-pattern TC h=0 matrix (job 57315881, exit 137 on the 55 GB shared node).
        evals, psi0 = _honeycomb_direct_ed(geometry, model, J, hx, hz, k=max(k, 1),
                                           ncv=ncv if ncv else None, hy=hy)
    else:
        # Eigenvectors are needed for observables AND for the real-dtype sign/amplitude
        # diagnostics below, so --no-observables only skips the (expensive) per-site
        # operator pass, not the eigenvector. Complex --no-observables keeps the old
        # memory-lean path (that combination exists because of OOM on shared nodes).
        want_vec = observables or dtype == "float64"
        if want_vec:
            evals, evecs = nk.exact.lanczos_ed(H, k=max(k, 1), compute_eigenvectors=True)
            psi0 = np.asarray(evecs[:, 0])
        else:
            evals = nk.exact.lanczos_ed(H, k=max(k, 1), compute_eigenvectors=False)
            psi0 = None
    ed_time = time.time() - t0

    result = {
        "lattice": lattice, "model": model,
        "Lx": Lx, "Ly": Ly, "N": geometry.N, "bc": bc,
        "hx": hx, "hy": hy, "hz": hz, "J": J, "ftc": ftc, "dtype": dtype,
        "energies": [float(e) for e in np.real(evals)],
        "E0": float(np.real(evals[0])),
        "gap": float(np.real(evals[1] - evals[0])) if len(evals) > 1 else None,
        "ed_time_s": ed_time,
    }
    if lattice == "honeycomb":
        result["n_vertices"] = int(geometry.n_vertices)
        result["n_plaqs"] = int(geometry.n_plaqs)
        result.update(hinfo)  # n_pauli_strings, plaq_string_counts, build_time_s

    # Sign-structure diagnostic for real Hamiltonians: fraction of negative GS
    # amplitudes (global phase fixed by the largest-|amplitude| component) and the
    # weight they carry. 0 => positive/stoquastic-representable state (plain float64
    # Combo suffices); finite => signful GS (needs the complex ansatz). This is the
    # decisive check for the non-stoquastic ftc under hx/hz perturbations.
    if psi0 is not None and dtype == "float64":
        if lattice == "honeycomb":
            # Anchor the global sign to the all-up (zero-loop) configuration: in the
            # (-1)^{#loops} gauge it always carries amplitude +|a|. The argmax anchor
            # below is ARBITRARY for uniform-|amplitude| stabilizer states (it picks
            # the first max index, which may be an odd-loop config and flips every
            # reported sign fraction to 1-x).
            # direct bit convention: the all-up (zero-loop) config is index 0
            anchor = np.real(psi0[0])
            if abs(anchor) < 1e-12 * np.max(np.abs(psi0)):
                anchor = np.real(psi0[np.argmax(np.abs(psi0))])
            v = np.real(psi0) * np.sign(anchor)
        else:
            v = np.real(psi0) * np.sign(np.real(psi0[np.argmax(np.abs(psi0))]))
        scale = np.max(np.abs(v))
        cut = 1e-12 * scale
        result["neg_amp_fraction"] = float(np.mean(v < -cut))
        result["neg_amp_weight"] = float(np.sum(v[v < -cut] ** 2))
        # Fraction of negatives among the NONZERO amplitudes (the whole-vector
        # fraction above dilutes by the 2^N/support ratio -- misleading for
        # stabilizer-like states with tiny support, e.g. the doubled semion where
        # the support-normalized value is the (-1)^{#loops} signal).
        n_nonzero_all = int(np.sum(np.abs(v) > cut))
        result["neg_amp_fraction_support"] = (
            float(np.sum(v < -cut) / n_nonzero_all) if n_nonzero_all else 0.0
        )
        # Amplitude-DISTRIBUTION diagnostic (stronger than the sign check): at a
        # stabilizer fixed point the GS should be 0 or one identical value on every
        # Z-basis string (uniform superposition over the star-group orbit), i.e.
        # n_nonzero a power of 2, amp_rel_spread ~ machine eps, one entry in
        # amp_values_scaled. The signed histogram (normalized by max|amp|) is the
        # compact payload for plotting at L=4 without shipping the 2^24 vector.
        nz = v[np.abs(v) > cut]
        mags = np.abs(nz)
        result["n_nonzero"] = int(nz.size)
        result["amp_rel_spread"] = float((mags.max() - mags.min()) / mags.mean())
        vals, counts = np.unique(np.round(nz / scale, 9), return_counts=True)
        top = np.argsort(-counts)[:8]
        result["amp_values_scaled"] = [[float(vals[i]), int(counts[i])] for i in top]
        result["amp_n_distinct"] = int(len(vals))
        hist, edges = np.histogram(v / scale, bins=400, range=(-1.0, 1.0))
        result["amp_hist_counts"] = hist.tolist()
        result["amp_hist_edges"] = edges.tolist()

    if observables and psi0 is not None and lattice == "honeycomb":
        t1 = time.time()
        result.update(_honeycomb_observables(psi0, geometry, model))
        result["obs_time_s"] = time.time() - t1

    if observables and psi0 is not None and lattice == "square":
        sz = [_expect(psi0, nk.operator.spin.sigmaz(hi, j).to_sparse()) for j in range(geometry.N)]
        sx = [_expect(psi0, nk.operator.spin.sigmax(hi, j).to_sparse()) for j in range(geometry.N)]
        result["magnetization_Z"] = sz
        result["magnetization_X"] = sx
        result["magnetization_Z_mean"] = float(np.mean(sz))
        result["magnetization_X_mean"] = float(np.mean(sx))
        # <sigma^y> is the order parameter for the pure-hy transition. sigmay is
        # complex, so this is only meaningful (and only built) when dtype=complex.
        if dtype == "complex":
            sy = [_expect(psi0, nk.operator.spin.sigmay(hi, j).to_sparse()) for j in range(geometry.N)]
            result["magnetization_Y"] = sy
            result["magnetization_Y_mean"] = float(np.mean(sy))

    return result


def main():
    p = argparse.ArgumentParser(description="Sparse Lanczos ED for the mixed-field toric code")
    p.add_argument("--Lx", type=int, required=True)
    p.add_argument("--Ly", type=int, default=0,
                   help="honeycomb: hexagon rows (0 => Ly = Lx); ignored for square")
    p.add_argument("--lattice", choices=["square", "honeycomb"], default="square",
                   help="honeycomb = Lx x Ly complete hexagons, smooth OBC (Levin-Gu)")
    p.add_argument("--model", choices=["tc", "ds"], default="tc",
                   help="honeycomb model: toric code or doubled semion")
    p.add_argument("--hx", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=0.0)
    p.add_argument("--hy", type=float, default=0.0)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--bc", choices=["OBC", "PBC"], default="OBC")
    p.add_argument("--k", type=int, default=4, help="number of lowest eigenvalues")
    p.add_argument("--ncv", type=int, default=0,
                   help="ARPACK Krylov block size (0 = scipy default; raise to ~48 "
                        "for degenerate low spectra, e.g. honeycomb TC with hx)")
    p.add_argument("--ftc", action="store_true",
                   help="fermionic toric code: dressed stars A'_v = A_v * B_NE(v)")
    p.add_argument("--no-observables", action="store_true",
                   help="skip per-site magnetizations (energies only)")
    p.add_argument("--out", type=str, default=None, help="output JSON path")
    args = p.parse_args()

    result = run_ed(
        Lx=args.Lx, hx=args.hx, hz=args.hz, hy=args.hy, J=args.J,
        bc=args.bc, k=args.k, ncv=args.ncv, observables=not args.no_observables,
        ftc=args.ftc,
        lattice=args.lattice, model=args.model, Ly=args.Ly,
    )

    if args.out is None and args.lattice == "honeycomb":
        hy_seg = f"_hy{args.hy:.2f}" if args.hy != 0.0 else ""
        args.out = (f"results/ed/ed_hc{result['Lx']}x{result['Ly']}_{args.model}"
                    f"_hx{args.hx:.2f}_hz{args.hz:.2f}{hy_seg}.json")

    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, list) or len(v) <= 8}, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved ED results to {args.out}")


if __name__ == "__main__":
    main()
