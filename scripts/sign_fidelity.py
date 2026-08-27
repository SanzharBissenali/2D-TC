"""Sign-fidelity diagnostic for the Phase-4 QEC sign head (doubled semion).

Measures, against exact ED ground-state vectors, how much of the DS sign
structure the DETERMINISTIC head captures BEFORE any NQS training:

    head(sigma) = (-1)^{#loops(eps . sigma)},  eps = MWPM recovery of the
    Q_v syndrome of sigma (pymatching, unit link weights).

The headline number per (size, model, hx, hz) is the |psi|^2-weighted sign
agreement F_s = sum_sigma |psi(sigma)|^2 * [head(sigma) == sign psi(sigma)].
F_s is exactly the maximum fidelity ANY positive amplitude network combined
with this head can reach (the optimal amplitude zeroes the wrong-sign
configs), so it is the head's ceiling. Baseline: F_plus = weight of positive
amplitudes = the ceiling of the sign-free (Hastings-floor) ansatz.

Pre-registered predictions (physics discussion, 2026-08-27):
  - hx = 0 (any hz, incl. 0): F_s = 1 exactly -- the GS never leaves the
    zero-charge sector, where the head is the exact (-1)^{#loops} sign.
  - hx != 0: F_s < 1; errors concentrated on defected (off-sector) configs
    where minimal recoveries tie with opposite signs (semion braiding), whose
    true amplitudes are interference-suppressed -- so the lost weight is small.

Conventions (match exact/lanczos_ed._honeycomb_direct_ed): site i <-> bit i,
bit 1 = spin DOWN (sigma^z = -1), all-up config = index 0. The ED global sign
is anchored at the all-up amplitude (positive in the (-1)^{#loops} gauge).

The MWPM tie-breaking is probed by decoding twice with two opposite tiny
deterministic weight perturbations (eta * link_index); minimal-cardinality
recoveries are preserved (eta * N^2 << 1) but ties resolve differently.
The weight where the two tie-breakings give DIFFERENT signs bounds the
tie-sensitive part of the head.

Local sizes (full 2^N enumeration): 1x2 (N=11), 3x1 (N=16), 2x2 (N=19).
2x3 (N=27) needs an exclusive cluster node (jobs/nersc_signfid.sh).

Example:
    PYTHONPATH=. python scripts/sign_fidelity.py --Lx 2 --Ly 2 --model ds \
        --points "0,0;0,0.1;0,0.2;0,0.5;0.05,0;0.1,0;0.2,0;0.3,0;0.5,0;0.1,0.1" \
        --out results/diagnostics/signfid_hc2x2_ds.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact.lanczos_ed import _honeycomb_direct_ed          # noqa: E402
from exact.loops import count_loops                        # noqa: E402
from model.honeycomb_geometry import HoneycombGeometry     # noqa: E402

TIE_ETA = 1e-4  # weight perturbation; minimality preserved while eta*N^2 < 1


def loop_sign_table(g):
    """(-1)^{#loops} for every zero-charge config, keyed by config integer.

    The 2^F hexagon-flip subsets enumerate the sector exactly once
    (simply-connected patch, independent flips) -- asserted.
    """
    masks = [int(np.bitwise_or.reduce(1 << g.plaq_all[p].astype(np.int64)))
             for p in range(g.n_plaqs)]
    table = {}
    for S in range(1 << g.n_plaqs):
        x = 0
        for p in range(g.n_plaqs):
            if (S >> p) & 1:
                x ^= masks[p]
        z = 1 - 2 * ((x >> np.arange(g.N)) & 1)
        table[x] = 1 - 2 * (count_loops(z, g.link_endpoints) % 2)
    assert len(table) == 1 << g.n_plaqs, "hexagon flips are not independent?!"
    return table


def build_matchings(g):
    """Two pymatching decoders on the link graph, opposite tie-breaking."""
    import scipy.sparse as sp
    from pymatching import Matching

    rows = g.link_endpoints.T.ravel()                    # (2N,) vertex ids
    cols = np.tile(np.arange(g.N), 2)
    H = sp.csc_matrix((np.ones(2 * g.N, dtype=np.uint8), (rows, cols)),
                      shape=(g.n_vertices, g.N))
    pert = TIE_ETA * np.arange(g.N)
    assert pert.max() * g.N < 0.5, "perturbation could reorder cardinalities"
    return (Matching.from_check_matrix(H, weights=1.0 + pert),
            Matching.from_check_matrix(H, weights=1.0 + pert[::-1]))


def head_signs(g, configs, table, matchings):
    """(sign_1, sign_2, n_defects) for a chunk of config integers.

    Syndrome -> MWPM recovery -> zero-charge config -> table lookup, once per
    tie-breaking. Recovered configs are asserted to clear the syndrome (they
    must land in the loop-sign table).
    """
    n = configs.shape[0]
    synd = np.zeros((n, g.n_vertices), dtype=np.uint8)
    for v in range(g.n_vertices):
        for l in g.vertex_all[v]:
            if l != -1:
                synd[:, v] ^= ((configs >> int(l)) & 1).astype(np.uint8)
    ndef = synd.sum(axis=1).astype(np.int64)
    assert not (ndef % 2).any(), "odd syndrome weight (violates prod Q_v = 1)"

    bit_values = (1 << np.arange(g.N, dtype=np.int64))
    signs = []
    for m in matchings:
        corr = m.decode_batch(synd).astype(np.int64)
        recovered = configs ^ (corr @ bit_values)
        s = np.empty(n, dtype=np.int8)
        for i, x in enumerate(recovered):
            s[i] = table[int(x)]   # KeyError here == recovery failed
        signs.append(s)
    return signs[0], signs[1], ndef


def run_point(g, model, hx, hz, k, table, matchings, chunk, tol=0):
    t0 = time.time()
    evals, psi = _honeycomb_direct_ed(g, model, 1.0, hx, hz, k=k, tol=tol)
    t_ed = time.time() - t0
    psi = np.real(np.asarray(psi))
    anchor = psi[0]                          # all-up amplitude
    assert abs(anchor) > 1e-10 * np.max(np.abs(psi)), \
        "all-up anchor numerically zero -- sign gauge undefined"
    psi *= np.sign(anchor)
    w = psi ** 2
    w /= w.sum()
    sgn_ed = np.where(psi >= 0, 1, -1).astype(np.int8)

    dim = 1 << g.N
    agree1 = agree2 = disagree12 = pos_w = 0.0
    w_by_d, agree_by_d = {}, {}
    wrong_max = 0.0
    wrong_big = 0                            # wrong configs with weight > 1e-9
    t0 = time.time()
    for lo in range(0, dim, chunk):
        c = np.arange(lo, min(lo + chunk, dim), dtype=np.int64)
        s1, s2, ndef = head_signs(g, c, table, matchings)
        wc, ec = w[lo:lo + len(c)], sgn_ed[lo:lo + len(c)]
        ok1, ok2 = (s1 == ec), (s2 == ec)
        agree1 += float(wc[ok1].sum())
        agree2 += float(wc[ok2].sum())
        disagree12 += float(wc[s1 != s2].sum())
        pos_w += float(wc[ec > 0].sum())
        bad = wc[~ok1]
        if bad.size:
            wrong_max = max(wrong_max, float(bad.max()))
            wrong_big += int((bad > 1e-9).sum())
        for d in np.unique(ndef):
            sel = ndef == d
            w_by_d[int(d)] = w_by_d.get(int(d), 0.0) + float(wc[sel].sum())
            agree_by_d[int(d)] = (agree_by_d.get(int(d), 0.0)
                                  + float(wc[sel & ok1].sum()))
    t_head = time.time() - t0

    fid_by_d = {d: (agree_by_d[d] / w_by_d[d] if w_by_d[d] > 1e-20 else None)
                for d in sorted(w_by_d)}
    return {
        "hx": hx, "hz": hz, "model": model, "k": k,
        "E0": float(evals[0]),
        "evals": [float(e) for e in evals],
        "F_s": agree1,                       # head ceiling fidelity
        "F_s_tie2": agree2,                  # opposite MWPM tie-breaking
        "tie_disagree_weight": disagree12,
        "F_plus": pos_w,                     # positive-head (Hastings) ceiling
        "onsector_weight": w_by_d.get(0, 0.0),
        "onsector_fidelity": fid_by_d.get(0),
        "weight_by_defects": {str(d): w_by_d[d] for d in sorted(w_by_d)},
        "fidelity_by_defects": {str(d): fid_by_d[d] for d in sorted(fid_by_d)},
        "wrong_weight": 1.0 - agree1,
        "wrong_max_single_weight": wrong_max,
        "wrong_configs_above_1e-9": wrong_big,
        "t_ed_s": t_ed, "t_head_s": t_head,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--model", default="ds", choices=["ds", "tc"])
    ap.add_argument("--points", required=True,
                    help="semicolon-separated hx,hz pairs, e.g. '0,0;0.1,0'")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0,
                    help="ARPACK tol (0 = machine precision; ~1e-8 for 2^27)")
    ap.add_argument("--chunk", type=int, default=1 << 22)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = HoneycombGeometry(args.Lx, args.Ly)
    print(f"# {args.Lx}x{args.Ly} {args.model}: N={g.N} V={g.n_vertices} "
          f"F={g.n_plaqs} dim=2^{g.N}", flush=True)
    table = loop_sign_table(g)
    matchings = build_matchings(g)

    points = [tuple(float(x) for x in p.split(","))
              for p in args.points.split(";") if p.strip()]
    hdr = (f"{'hx':>5} {'hz':>5} {'E0':>14} {'F_s':>12} {'F_plus':>12} "
           f"{'on-sect wt':>12} {'tie wt':>10} {'wrong wt':>10}")
    print(hdr, flush=True)
    records = []
    for hx, hz in points:
        r = run_point(g, args.model, hx, hz, args.k, table, matchings,
                      args.chunk, tol=args.tol)
        records.append(r)
        print(f"{hx:5.2f} {hz:5.2f} {r['E0']:14.8f} {r['F_s']:12.9f} "
              f"{r['F_plus']:12.9f} {r['onsector_weight']:12.9f} "
              f"{r['tie_disagree_weight']:10.3e} {r['wrong_weight']:10.3e}",
              flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"Lx": args.Lx, "Ly": args.Ly, "model": args.model,
                       "N": g.N, "head": "MWPM(unit weights) + count_loops",
                       "tie_eta": TIE_ETA, "points": records}, f, indent=1)
        print(f"# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
