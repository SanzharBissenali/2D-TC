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

Pre-registered predictions (physics discussion, 2026-08-27; model=ds -- for
tc the head is deliberately wrong and only the F_plus baseline is meaningful):
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
The weight where the two tie-breakings give DIFFERENT signs LOWER-bounds the
tie-sensitive part of the head: both perturbations are linear in the link
index, so recoveries degenerate in sum-of-link-indices stay tied under both
(adversarial probe, 1x2 exhaustive: 440/540 sign-ties detected; the missed
weight was 1.4e-14 vs 4.7e-10 detected, ~0.003% relative).

Phase 4c (--decoders): grades the decoder LADDER (model/decoders.py) against
the same exact psi in one pass -- per-decoder F_s lands in record['decoders'].
Pre-registered predictions (physics discussion 2026-08-30) AND measured
outcomes (1x2/2x2 grids + swarm S4, same day -- kept side by side for honest
pre-registration):
  anchor    predicted 1 - F_s ~ N*hx^2 -- CONFIRMED with refined prefactor
            ~F*hx^2 (co-tree cycles; 0.6% quantitative match, k=2.03);
  greedy    predicted between anchor and mwpm -- CONFIRMED (~2x mwpm; tie
            losses + closest-pair-first mispairings, both same exponent);
  unionfind predicted ~greedy-class -- CONFIRMED but larger (~10-500x mwpm:
            cluster-merge mispairing of two nearby defect pairs is an
            excess-2 channel on wmin=2 configs, two hx powers cheaper);
  mwpm      predicted tie channel only -- CONFIRMED; NOTE the tie exponent
            is size-dependent (2*wmin of the cheapest tied syndrome: hx^6 at
            1x2, hx^10 at 2x2/2x3), not a universal hx^10;
  tie_sum   predicted ~exact -- REFUTED: ~= mwpm (dominant tied classes sum
            to exactly zero => mwpm fallback; slightly WORSE at hz != 0).
            The tied signs are decided by unequal resolvent weights -- see
            the tie_sum entry in model/decoders.py for the v2 design.
The default --decoders mwpm changes nothing: the top-level keys always come
from the original decoder-A/B path, and the new-path mwpm F_s is asserted to
match it (internal cross-validation).

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


def syndrome_bits(g, configs):
    """(n, V) uint8 Q_v syndrome bits for a chunk of config integers."""
    synd = np.zeros((configs.shape[0], g.n_vertices), dtype=np.uint8)
    for v in range(g.n_vertices):
        for l in g.vertex_all[v]:
            if l != -1:
                synd[:, v] ^= ((configs >> int(l)) & 1).astype(np.uint8)
    return synd


def head_signs(g, configs, table, matchings):
    """(sign_1, sign_2, n_defects) for a chunk of config integers.

    Syndrome -> MWPM recovery -> zero-charge config -> table lookup, once per
    tie-breaking. Recovered configs are asserted to clear the syndrome (they
    must land in the loop-sign table).
    """
    n = configs.shape[0]
    synd = syndrome_bits(g, configs)
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


def build_decoder_tables(g, names, tiesum_dmax, unpack_chunk=1 << 16):
    """Precompute per-decoder corrections over ALL even-weight syndromes.

    There are only 2^(V-1) even syndromes (prod Q_v = 1 forbids odd ones) --
    far fewer than 2^N configs -- so every decoder decodes each syndrome ONCE
    here; the per-point 2^N sweep is then pure array lookups. Corrections are
    packed as int64 link masks. tie_sum stores the degenerate-class masks per
    syndrome instead (None => mwpm fallback beyond d_max). Built once per
    size, reused for every field point."""
    from model.decoders import make_decoder

    V, N = g.n_vertices, g.N
    ints = np.arange(1 << V, dtype=np.int64)
    par = ints.copy()
    for sh in (16, 8, 4, 2, 1):
        par ^= par >> sh
    synd_ints = ints[(par & 1) == 0]                    # (S,)
    S = synd_ints.size
    index = np.full(1 << V, -1, dtype=np.int32)
    index[synd_ints] = np.arange(S, dtype=np.int32)
    synd_bits = np.empty((S, V), dtype=np.uint8)
    for lo in range(0, S, unpack_chunk):
        c = synd_ints[lo:lo + unpack_chunk]
        synd_bits[lo:lo + len(c)] = \
            ((c[:, None] >> np.arange(V, dtype=np.int64)) & 1).astype(np.uint8)

    bitvals = (1 << np.arange(N, dtype=np.int64))
    tables = {"index": index, "names": list(names), "S": S,
              "tiesum_dmax": tiesum_dmax}
    need_mwpm = ("mwpm" in names) or ("tie_sum" in names)
    corr_names = [n for n in names if n != "tie_sum"]
    if need_mwpm and "mwpm" not in corr_names:
        corr_names.append("mwpm")
    for name in corr_names:
        t0 = time.time()
        dec = make_decoder(name, g)
        mask = np.empty(S, dtype=np.int64)
        for lo in range(0, S, unpack_chunk):
            corr = dec.corrections(synd_bits[lo:lo + unpack_chunk])
            mask[lo:lo + corr.shape[0]] = corr.astype(np.int64) @ bitvals
        tables[name] = mask
        print(f"# decoder table {name}: {S} syndromes in "
              f"{time.time() - t0:.1f} s", flush=True)

    if "tie_sum" in names:
        t0 = time.time()
        ts = make_decoder("tie_sum", g, d_max=tiesum_dmax)
        classes = [None] * S                            # None => fallback
        truncated = np.zeros(S, dtype=bool)
        n_fb = 0
        for i in range(S):
            cls, fb, tr = ts.class_for_syndrome(synd_bits[i])
            truncated[i] = tr
            if fb:
                n_fb += 1
            else:
                classes[i] = cls.astype(np.int64) @ bitvals
        tables["tie_sum"] = classes
        tables["tie_sum_truncated"] = truncated
        sizes = [c.size for c in classes if c is not None]
        print(f"# decoder table tie_sum: {S} syndromes in "
              f"{time.time() - t0:.1f} s (fallback {n_fb}, truncated "
              f"{int(truncated.sum())}, max class {max(sizes)}, "
              f"mean class {np.mean(sizes):.2f})", flush=True)
    return tables


def decoder_signs(g, configs, sidx, parity_arr, dec_tables, name):
    """Head signs (+-1 int8) for a chunk under one ladder decoder.

    For tie_sum also returns (fallback, truncated, cancelled) bool masks;
    correction decoders return None there."""
    if name != "tie_sum":
        rec = configs ^ dec_tables[name][sidx]
        sgn = parity_arr[rec]
        assert (sgn != 0).all(), f"{name}: recovery left the sector?!"
        return sgn, None

    classes = dec_tables["tie_sum"]
    truncated_tab = dec_tables["tie_sum_truncated"]
    mwpm_mask = dec_tables["mwpm"]
    sgn = np.zeros(configs.shape[0], dtype=np.int8)
    fallback = np.zeros(configs.shape[0], dtype=bool)
    truncated = np.zeros(configs.shape[0], dtype=bool)
    cancelled = np.zeros(configs.shape[0], dtype=bool)
    order = np.argsort(sidx, kind="stable")
    lo = 0
    while lo < order.size:
        hi = lo
        s = sidx[order[lo]]
        while hi < order.size and sidx[order[hi]] == s:
            hi += 1
        rows = order[lo:hi]
        cfgs = configs[rows]
        cls = classes[s]
        truncated[rows] = truncated_tab[s]
        if cls is None:
            fallback[rows] = True
            sgn[rows] = parity_arr[cfgs ^ mwpm_mask[s]]
        else:
            tot = parity_arr[cfgs[:, None] ^ cls[None, :]] \
                .astype(np.int64).sum(axis=1)
            sg = np.sign(tot).astype(np.int8)
            zero = sg == 0
            if zero.any():
                cancelled[rows[zero]] = True
                sg[zero] = parity_arr[cfgs[zero] ^ mwpm_mask[s]]
            sgn[rows] = sg
        lo = hi
    assert (sgn != 0).all(), "tie_sum: recovery left the sector?!"
    return sgn, (fallback, truncated, cancelled)


def run_point_complex(g, model, hx, hz, hy, k, chunk, dec_tables, tol=0,
                      n_bins=4096):
    """hy != 0 (Phase 4d): the GS is complex, so +-1 'sign agreement' is
    undefined. Graded instead: the PHASE-OPTIMIZED REAL-SIGNED CEILING -- for
    head signs s(sigma) in {+-1} and any positive amplitude A >= 0,

        F_s^C = max_A,theta |<psi| e^{i theta} A s>|^2
              = max_theta sum_sigma max(Re(e^{i theta} s(sigma) psi*(sigma)), 0)^2

    (Cauchy-Schwarz on the positive part). Reduces EXACTLY to the hy=0 F_s for
    real psi. Computed exactly via angular binning: c = s psi* binned by phase
    into n_bins with per-bin moments M0 = sum|c|^2, M2 = sum c^2; for theta on
    the bin grid the window {cos(phi+theta) > 0} is a half circle of bins and
    F(theta) = sum_win [M0 + Re(e^{2 i theta} M2)] / 2 -- the integrand
    vanishes quadratically at the window edges, so bin-boundary error is
    O((2 pi/n_bins)^2 * edge mass), negligible. F_plus^C uses s == +1."""
    t0 = time.time()
    evals, psi = _honeycomb_direct_ed(g, model, 1.0, hx, hz, k=k, tol=tol,
                                      hy=hy)
    t_ed = time.time() - t0
    psi = np.asarray(psi, dtype=np.complex128)
    anchor = psi[0]
    assert abs(anchor) > 1e-12 * np.max(np.abs(psi)), \
        "all-up anchor numerically zero -- phase gauge undefined"
    psi *= np.conj(anchor) / abs(anchor)          # gauge: psi(all-up) real > 0
    psi /= np.sqrt(float((np.abs(psi) ** 2).sum()))

    dnames = dec_tables["names"]
    heads = list(dnames) + ["__plus__"]
    M0 = {h: np.zeros(n_bins) for h in heads}
    M2 = {h: np.zeros(n_bins, dtype=np.complex128) for h in heads}
    w_by_d = {}
    dim = 1 << g.N
    scale = n_bins / (2.0 * np.pi)
    t0 = time.time()
    for lo in range(0, dim, chunk):
        c_idx = np.arange(lo, min(lo + chunk, dim), dtype=np.int64)
        pc = np.conj(psi[lo:lo + len(c_idx)])
        synd = syndrome_bits(g, c_idx)
        ndef = synd.sum(axis=1)
        wc = np.abs(pc) ** 2
        for d in np.unique(ndef):
            w_by_d[int(d)] = w_by_d.get(int(d), 0.0) + float(wc[ndef == d].sum())
        sint = synd.astype(np.int64) @ (1 << np.arange(g.n_vertices,
                                                       dtype=np.int64))
        sidx = dec_tables["index"][sint]
        assert (sidx >= 0).all(), "config produced an odd syndrome?!"
        for h in heads:
            if h == "__plus__":
                c = pc
            else:
                sgn, _ = decoder_signs(g, c_idx, sidx,
                                       dec_tables["parity_arr"], dec_tables, h)
                c = pc * sgn
            b = np.floor((np.angle(c) + np.pi) * scale).astype(np.int64) % n_bins
            M0[h] += np.bincount(b, weights=wc, minlength=n_bins)
            c2 = c * c
            M2[h] += np.bincount(b, weights=c2.real, minlength=n_bins) \
                + 1j * np.bincount(b, weights=c2.imag, minlength=n_bins)

    # window sums: theta_j = j*2pi/n_bins keeps bins with cos(phi_k+theta_j)>0,
    # i.e. phi in (-pi/2 - theta, pi/2 - theta): a half circle sliding with j.
    F = {}
    ks = np.arange(n_bins)
    for h in heads:
        A = 0.5 * M0[h]
        B = 0.5 * M2[h]
        cA = np.concatenate([A, A]).cumsum()
        cB = np.concatenate([B, B]).cumsum()
        half = n_bins // 2
        # bin k covers phi ~ -pi + (k+.5)*2pi/K; window start for theta_j:
        # phi > -pi/2 - theta_j  =>  k >= start_j
        best = -1.0
        thetas = ks * (2.0 * np.pi / n_bins)
        start = np.floor((np.pi / 2.0 - thetas + np.pi) * scale).astype(np.int64) % n_bins
        sumA = cA[start + half - 1] - np.where(start > 0, cA[start - 1], 0.0)
        sumB = cB[start + half - 1] - np.where(start > 0, cB[start - 1], 0.0)
        vals = sumA + np.real(np.exp(2j * thetas) * sumB)
        F[h] = float(vals.max())
    t_head = time.time() - t0

    return {
        "hx": hx, "hz": hz, "hy": hy, "model": model, "k": k,
        "E0": float(evals[0]),
        "evals": [float(e) for e in evals],
        "complex_ceiling": True, "n_bins": n_bins,
        "F_plus": F["__plus__"],
        "onsector_weight": w_by_d.get(0, 0.0),
        "weight_by_defects": {str(d): w_by_d[d] for d in sorted(w_by_d)},
        "decoders": {h: {"F_s": F[h], "wrong_weight": 1.0 - F[h]}
                     for h in dnames},
        "t_ed_s": t_ed, "t_head_s": t_head,
    }


def run_point(g, model, hx, hz, k, table, matchings, chunk, tol=0,
              dec_tables=None):
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
    dnames = dec_tables["names"] if dec_tables is not None else []
    dec_agree = {n: 0.0 for n in dnames}
    ts_w = {"fallback": 0.0, "truncated": 0.0, "cancelled": 0.0}
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
        if dec_tables is not None:
            sint = syndrome_bits(g, c).astype(np.int64) \
                @ (1 << np.arange(g.n_vertices, dtype=np.int64))
            sidx = dec_tables["index"][sint]
            assert (sidx >= 0).all(), "config produced an odd syndrome?!"
            for name in dnames:
                sgn, extra = decoder_signs(g, c, sidx,
                                           dec_tables["parity_arr"],
                                           dec_tables, name)
                dec_agree[name] += float(wc[sgn == ec].sum())
                if extra is not None:
                    fb, tr, cx = extra
                    ts_w["fallback"] += float(wc[fb].sum())
                    ts_w["truncated"] += float(wc[tr].sum())
                    ts_w["cancelled"] += float(wc[cx].sum())
    t_head = time.time() - t0

    fid_by_d = {d: (agree_by_d[d] / w_by_d[d] if w_by_d[d] > 1e-20 else None)
                for d in sorted(w_by_d)}
    dec_block = None
    if dec_tables is not None:
        if "mwpm" in dec_agree:
            # internal cross-validation: the syndrome-table mwpm path must
            # reproduce the original decoder-A path (float sum order only)
            assert abs(dec_agree["mwpm"] - agree1) < 1e-9, \
                (f"mwpm table path F_s={dec_agree['mwpm']!r} != decoder-A "
                 f"path F_s={agree1!r}")
        dec_block = {}
        for name in dnames:
            entry = {"F_s": dec_agree[name],
                     "wrong_weight": 1.0 - dec_agree[name]}
            if name == "tie_sum":
                entry["fallback_weight"] = ts_w["fallback"]
                entry["truncated_weight"] = ts_w["truncated"]
                entry["cancelled_weight"] = ts_w["cancelled"]
            dec_block[name] = entry
    return {
        **({"decoders": dec_block} if dec_block is not None else {}),
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
    ap.add_argument("--decoders", default="mwpm",
                    help="comma list from model/decoders.py "
                         "(mwpm,anchor,greedy,unionfind,tie_sum); the bare "
                         "default 'mwpm' skips the ladder machinery entirely "
                         "(byte-identical legacy output)")
    ap.add_argument("--tiesum_dmax", type=int, default=10,
                    help="tie_sum defect cap (more defects => mwpm fallback)")
    ap.add_argument("--hy", type=float, default=0.0,
                    help="Y field (Phase 4d): complex GS => the graded quantity "
                         "becomes the phase-optimized real-signed ceiling "
                         "(see run_point_complex); hy=0 is byte-identical legacy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = HoneycombGeometry(args.Lx, args.Ly)
    print(f"# {args.Lx}x{args.Ly} {args.model}: N={g.N} V={g.n_vertices} "
          f"F={g.n_plaqs} dim=2^{g.N}", flush=True)
    table = loop_sign_table(g)
    matchings = build_matchings(g)

    dec_names = list(dict.fromkeys(                 # dedupe, keep order (a
        d.strip() for d in args.decoders.split(",") if d.strip()))  # repeated
    # name would silently double-accumulate its F_s -- swarm S2 finding)
    dec_tables = None
    if dec_names != ["mwpm"] or args.hy != 0.0:
        dec_tables = build_decoder_tables(g, dec_names, args.tiesum_dmax)
        parity_arr = np.zeros(1 << g.N, dtype=np.int8)
        for x, s in table.items():
            parity_arr[x] = s
        dec_tables["parity_arr"] = parity_arr

    points = [tuple(float(x) for x in p.split(","))
              for p in args.points.split(";") if p.strip()]
    hdr = (f"{'hx':>5} {'hz':>5} {'E0':>14} {'F_s':>12} {'F_plus':>12} "
           f"{'on-sect wt':>12} {'tie wt':>10} {'wrong wt':>10}")
    if dec_tables is not None:
        hdr += "".join(f" {'1-F_s:' + n:>16}" for n in dec_names)
    print(hdr, flush=True)
    records = []
    for hx, hz in points:
        if args.hy != 0.0:
            r = run_point_complex(g, args.model, hx, hz, args.hy, args.k,
                                  args.chunk, dec_tables, tol=args.tol)
            records.append(r)
            line = (f"{hx:5.2f} {hz:5.2f} {r['E0']:14.8f} {'(complex)':>12} "
                    f"{r['F_plus']:12.9f} {r['onsector_weight']:12.9f} "
                    f"{'-':>10} {'-':>10}")
            line += "".join(f" {r['decoders'][n]['wrong_weight']:16.9e}"
                            for n in dec_names)
            print(line, flush=True)
            continue
        r = run_point(g, args.model, hx, hz, args.k, table, matchings,
                      args.chunk, tol=args.tol, dec_tables=dec_tables)
        records.append(r)
        line = (f"{hx:5.2f} {hz:5.2f} {r['E0']:14.8f} {r['F_s']:12.9f} "
                f"{r['F_plus']:12.9f} {r['onsector_weight']:12.9f} "
                f"{r['tie_disagree_weight']:10.3e} {r['wrong_weight']:10.3e}")
        if dec_tables is not None:
            line += "".join(f" {r['decoders'][n]['wrong_weight']:16.9e}"
                            for n in dec_names)
        print(line, flush=True)

    if args.out:
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"Lx": args.Lx, "Ly": args.Ly, "model": args.model,
                       "N": g.N, "head": "MWPM(unit weights) + count_loops",
                       "tie_eta": TIE_ETA, "hy": args.hy,
                       **({"decoders": dec_names,
                           "tiesum_dmax": args.tiesum_dmax}
                          if dec_tables is not None else {}),
                       "points": records}, f, indent=1)
        print(f"# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
