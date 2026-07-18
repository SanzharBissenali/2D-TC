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
from model.hamiltonian import create_hamiltonian


def _expect(psi, sparse_op):
    """<psi| op |psi> for a normalized state vector and a scipy sparse operator."""
    return float(np.real(np.vdot(psi, sparse_op @ psi)))


def run_ed(Lx, hx, hz, hy=0.0, J=1.0, bc="OBC", k=1, observables=True):
    """Diagonalize the toric-code Hamiltonian and return a results dict."""
    dtype = "complex" if hy != 0.0 else "float64"
    geometry = ToricCodeGeometry(Lx, Lx, bc)
    hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)

    H = create_hamiltonian(
        hi=hi,
        vertex_all=geometry.vertex_all,
        plaq_all=geometry.plaq_all,
        bonds=geometry.bonds,
        hx=hx, hy=hy, hz=hz, J=J,
        dtype=dtype,
    )

    t0 = time.time()
    # k>=2 lets us see the low-lying gap; eigenvectors needed for observables.
    if observables:
        evals, evecs = nk.exact.lanczos_ed(H, k=max(k, 1), compute_eigenvectors=True)
        psi0 = np.asarray(evecs[:, 0])
    else:
        evals = nk.exact.lanczos_ed(H, k=max(k, 1), compute_eigenvectors=False)
        psi0 = None
    ed_time = time.time() - t0

    result = {
        "Lx": Lx, "N": geometry.N, "bc": bc,
        "hx": hx, "hy": hy, "hz": hz, "J": J, "dtype": dtype,
        "energies": [float(e) for e in np.real(evals)],
        "E0": float(np.real(evals[0])),
        "gap": float(np.real(evals[1] - evals[0])) if len(evals) > 1 else None,
        "ed_time_s": ed_time,
    }

    if observables and psi0 is not None:
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
    p.add_argument("--hx", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=0.0)
    p.add_argument("--hy", type=float, default=0.0)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--bc", choices=["OBC", "PBC"], default="OBC")
    p.add_argument("--k", type=int, default=4, help="number of lowest eigenvalues")
    p.add_argument("--no-observables", action="store_true",
                   help="skip per-site magnetizations (energies only)")
    p.add_argument("--out", type=str, default=None, help="output JSON path")
    args = p.parse_args()

    result = run_ed(
        Lx=args.Lx, hx=args.hx, hz=args.hz, hy=args.hy, J=args.J,
        bc=args.bc, k=args.k, observables=not args.no_observables,
    )

    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, list) or len(v) <= 8}, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved ED results to {args.out}")


if __name__ == "__main__":
    main()
