"""Numpy-only validation of the learned-vs-gated sign-head benchmark pieces
(docs/signhead_benchmark_plan.md Sec. 3 steps 1-3, 8) -- no jax/flax/netket.

Needs pymatching + numba (the head's imports): run under the scratch venv, e.g.
    <venv>/bin/python scratchpad/validate_signbench.py

Checks (PASS/FAIL each):
  a. hexagon-flip variables at 1x2, 2x2, 2x3, 3x3: for ALL 2^F sector configs
     r = G x, x_of_r(r, check=True) == x, and poly_sign01(x) == #loops(r) mod 2
     by the INDEPENDENT python counter exact/loops.count_loops AND == the
     production head s01 on the same configs; an off-sector r fails the check.
  b. TwoBranchModel log-psi formula (numpy replica) vs direct psi = a A1 + s A2
     (a SIGNED real mix) evaluated in 60-digit Decimal, incl. a < 0, a = 0 and
     near-cancellation |psi| ~ 1e-12 A in both sectors; imaginary part in
     {0, pi}; a = 0 == head-only log psi exactly.
  c. MLPSignModel complex log vs direct A * tanh(m).
  d. features_ex at 2x3: shape lead + (N+F,), {0,1} values, eps XOR sigma has
     an EMPTY syndrome, G x == eps XOR sigma, poly(x) == s01(sigma), leading-
     shape passthrough, head accounting, and the non-mwpm assert fires.
  e. model/sign_mlp_io npz spec: key/shape spec, save/load round-trip, and a
     wrong-shape npz is rejected (the M-pre loader contract).
  f. T_gate (plan Sec 7) quadrant formula == definition min_g sum|psi|^2[s=+-1 &
     sign psi != g] on random signed states and on the PRODUCTION head signs at
     1x2; gauge-invariant under psi -> -psi; <= 1 - F_s; 0 when psi = s|psi|;
     the plus/minus branch labelling.
  g. scripts/signbench_summary.build_records on a disk fixture: T's ceiling =
     T_gate with ceiling_kind 'T_gate', falls back to 1-F_s flagged '1-Fs',
     M/M-pre exact 0; n_params from sim_params; prior strings (M-pre warm
     1-Fs read from the pretrain JSON's best_1_minus_Fs); fidelity keyed by hy.
  h. _SignMLP input recentring h = 2u - 1 (numpy replica of the flax module):
     the all-zero feature row no longer sits on m = 0 under zero-bias init, and
     the pretrain script feeds raw {0,1} features (text check).
  i. jobs/nersc_signbench.sh jobid canonicalisation (awk %g): 0.40:0.0 ->
     hx0.4_hz0 == 0.4:0; MLP_DIR absolute/relative resolution.
"""
import os
import sys
import traceback
from decimal import Decimal, getcontext

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exact.loops import count_loops                       # noqa: E402
from model.honeycomb_geometry import HoneycombGeometry    # noqa: E402
from model.sign_head import QECSignHead                   # noqa: E402
from model.sign_mlp_io import (load_mlp_npz, mlp_param_spec,   # noqa: E402
                               nest, save_mlp_npz)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS  {name}")
    except Exception as e:                                # noqa: BLE001
        RESULTS.append((name, False, repr(e)))
        print(f"FAIL  {name}: {e!r}")
        traceback.print_exc()


# ---- a. hexagon-flip variables + Levin-Gu polynomial -------------------------
def _all_sector_configs(head):
    X = ((np.arange(1 << head.F)[:, None] >> np.arange(head.F)) & 1).astype(np.uint8)
    R = ((X.astype(np.int64) @ head._G.T.astype(np.int64)) & 1).astype(np.uint8)
    return X, R


def check_a(Lx, Ly):
    g = HoneycombGeometry(Lx, Ly)
    head = QECSignHead(g)                      # construction gate runs here (F <= 16)
    X, R = _all_sector_configs(head)
    assert len({r.tobytes() for r in R}) == 1 << head.F, "hexagon masks dependent?!"
    # round trip through the GF(2) left-inverse
    assert (head.x_of_r(R, check=True) == X).all()
    # independent reference: python count_loops on +-1 spins
    ref = np.array([count_loops(1 - 2 * r.astype(int), g.link_endpoints) % 2 for r in R],
                   dtype=np.uint8)
    poly = head.poly_sign01(X)
    assert poly.shape == (1 << head.F,) and poly.dtype == np.uint8
    assert (poly == ref).all(), f"poly != count_loops parity on {np.sum(poly != ref)} configs"
    # production head on the sector configs (empty syndrome => eps = 0)
    s01 = head.s01((1 - 2 * R.astype(np.float64)))
    assert (s01.astype(np.uint8) == ref).all()
    # leading-shape passthrough of x_of_r / poly_sign01
    assert head.x_of_r(R.reshape(2, -1, head.N)).shape == (2, R.shape[0] // 2, head.F)
    assert head.poly_sign01(X.reshape(2, -1, head.F)).shape == (2, X.shape[0] // 2)
    # an off-sector config (single down link) must fail the round-trip check
    bad = np.zeros((1, head.N), dtype=np.uint8)
    bad[0, 0] = 1
    try:
        head.x_of_r(bad, check=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("off-sector r passed x_of_r(check=True)")
    print(f"      {Lx}x{Ly}: N={head.N} F={head.F} V={head.V} pairs={len(head._pairs)} "
          f"triples={len(head._triples)} configs={1 << head.F}")


# ---- b. two-branch stable log ------------------------------------------------
def twobranch_log(a1, a2, a, s01):
    """numpy replica of honeycomb_networks.TwoBranchModel.__call__
    (psi = a A1 + s A2, a a SIGNED real scalar kept OUTSIDE the max/log)."""
    s = 1.0 - 2.0 * s01
    m = np.maximum(a1, a2)
    w = a * np.exp(a1 - m) + s * np.exp(a2 - m)
    return m + np.log(np.abs(w)) + 1j * np.pi * (w < 0).astype(np.float64)


def _direct_log_decimal(a1, a2, a, s01):
    getcontext().prec = 60
    psi = Decimal(float(a)) * Decimal(float(a1)).exp() \
        + (1 - 2 * int(s01)) * Decimal(float(a2)).exp()
    re = float(abs(psi).ln())
    im = np.pi if psi < 0 else 0.0
    return re, im


def check_b():
    rng = np.random.default_rng(0)
    B = 3000
    a1 = rng.normal(-5.0, 3.0, B)
    a2 = rng.normal(-5.0, 3.0, B)
    a = rng.normal(0.0, 1.0, B)          # signed mix, both signs, O(1)
    a[:B // 3] = np.abs(a[:B // 3]) * 0.05   # the production-init scale
    a[-B // 3:] = -np.abs(a[-B // 3:])       # a < 0 block
    s01 = rng.integers(0, 2, B).astype(np.float64)
    out = twobranch_log(a1, a2, a, s01)
    for i in range(B):
        re, im = _direct_log_decimal(a1[i], a2[i], a[i], s01[i])
        assert abs(out[i].real - re) < 1e-10 * max(1.0, abs(re)), (i, out[i].real, re)
        assert out[i].imag == im, (i, out[i].imag, im, a[i], s01[i])
    assert set(np.unique(out.imag)) <= {0.0, np.pi}
    # a < 0 flips the OTHER head sector: with s = +1 and |a| A1 > A2 the
    # amplitude is negative; with s = -1 and a < 0 both terms are negative.
    neg = a < 0
    both_neg = neg & (s01 == 1.0)
    assert (out.imag[both_neg] == np.pi).all()
    dom = neg & (s01 == 0.0) & (np.log(np.where(neg, -a, 1.0)) + a1 > a2 + 1e-9)
    assert dom.any() and (out.imag[dom] == np.pi).all()
    # near-cancellation, a > 0: s = -1 and a2 = a1 + log a + log(1 - eps) => psi = a A1 eps
    for eps in (1e-6, 1e-9, 1e-12):
        a1c, ac = 1.3, 0.05
        a2c = a1c + np.log(ac) + np.log1p(-eps)
        got = twobranch_log(np.array([a1c]), np.array([a2c]), np.array([ac]), np.array([1.0]))[0]
        re, im = _direct_log_decimal(a1c, a2c, ac, 1.0)
        # float64 rounding of a2c itself limits agreement to ~1e-16/eps relative in psi
        assert abs(got.real - re) < 1e-3, (eps, got.real, re)
        assert got.imag == im and im == 0.0
        assert abs(got.real - (a1c + np.log(ac) + np.log(eps))) < 1e-3, (eps, got.real)
    # mirror: s = -1, A2 slightly LARGER => psi negative, tiny
    a1c, ac, eps = 0.7, 0.05, 1e-12
    a2c = a1c + np.log(ac) + np.log1p(eps)
    got = twobranch_log(np.array([a1c]), np.array([a2c]), np.array([ac]), np.array([1.0]))[0]
    assert got.imag == np.pi and abs(got.real - (a1c + np.log(ac) + np.log(eps))) < 1e-3
    # near-cancellation, a < 0: s = +1 and |a| A1 = A2 (1 - eps) => psi = +A2 eps
    for eps in (1e-6, 1e-9, 1e-12):
        a1c, ac = -0.4, -0.7
        a2c = a1c + np.log(-ac) - np.log1p(-eps)     # A2 = |a| A1 / (1 - eps)
        got = twobranch_log(np.array([a1c]), np.array([a2c]), np.array([ac]), np.array([0.0]))[0]
        re, im = _direct_log_decimal(a1c, a2c, ac, 0.0)
        assert abs(got.real - re) < 1e-3 and got.imag == im == 0.0, (eps, got, re, im)
    # a = 0: the head-only arm log A_top + i*pi*s01 (no -inf, no nan; real part
    # = m + (a2 - m), i.e. a2 up to one float rounding; imag part EXACT)
    out = twobranch_log(a1, a2, np.zeros(B), s01)
    assert np.isfinite(out.real).all()
    assert np.allclose(out.real, a2, rtol=0, atol=1e-12) and (out.imag == np.pi * s01).all()
    # a = +0.05 at equal trunks: trivial branch is a 5% admixture, sign from head
    out = twobranch_log(np.zeros(1), np.zeros(1), np.array([0.05]), np.array([1.0]))[0]
    assert abs(np.exp(out.real) - 0.95) < 1e-15 and out.imag == np.pi
    out = twobranch_log(np.zeros(1), np.zeros(1), np.array([0.05]), np.array([0.0]))[0]
    assert abs(np.exp(out.real) - 1.05) < 1e-15 and out.imag == 0.0
    # a = -0.05: s = +1 sector is now the one the trivial branch can flip
    out = twobranch_log(np.zeros(1), np.log(np.array([0.01])), np.array([-0.05]), np.array([0.0]))[0]
    assert abs(np.exp(out.real) - 0.04) < 1e-15 and out.imag == np.pi
    # a is never inside the max: huge trunk offsets with tiny a stay finite/exact
    out = twobranch_log(np.array([700.0]), np.array([-700.0]), np.array([1e-3]), np.array([0.0]))[0]
    assert np.isfinite(out.real) and abs(out.real - (700.0 + np.log(1e-3))) < 1e-9


# ---- c. MLP complex log ------------------------------------------------------
def mlp_log(log_a, m):
    """numpy replica of honeycomb_networks.MLPSignModel.__call__ (given m)."""
    t = np.tanh(m)
    return log_a + np.log(np.abs(t)) + 1j * np.pi * (t < 0).astype(np.float64)


def check_c():
    rng = np.random.default_rng(1)
    log_a = rng.normal(-4.0, 2.0, 5000)
    m = rng.normal(0.0, 3.0, 5000)
    m[:10] = np.array([1e-8, -1e-8, 1e-3, -1e-3, 30.0, -30.0, 0.5, -0.5, 2.0, -2.0])
    out = mlp_log(log_a, m)
    psi = np.exp(out)
    direct = np.exp(log_a) * np.tanh(m)
    assert np.allclose(psi, direct, rtol=1e-12, atol=0.0)
    assert set(np.unique(out.imag)) <= {0.0, np.pi}
    assert (out.imag == np.pi * (m < 0)).all()
    assert np.isfinite(out.real).all()


# ---- d. features_ex ----------------------------------------------------------
def check_d():
    g = HoneycombGeometry(2, 3)
    head = QECSignHead(g)
    rng = np.random.default_rng(2)
    B = 500
    states = rng.choice([-1.0, 1.0], size=(B, head.N))
    t0, n0 = head.t_head, head.n_head_configs
    u = head.features_ex(states)
    assert u.shape == (B, head.n_features_ex) == (B, head.N + head.F)
    assert u.dtype == np.float64 and set(np.unique(u)) <= {0.0, 1.0}
    assert head.n_head_configs - n0 == B and head.t_head > t0
    eps = u[:, :head.N].astype(np.uint8)
    x = u[:, head.N:].astype(np.uint8)
    bits = head._bits(states)
    r = bits ^ eps
    assert not head._syndrome(r).any(), "eps did not repair the syndrome"
    assert (head._syndrome(bits).sum(1) % 2 == 0).all()
    assert (((x.astype(np.int64) @ head._G.T.astype(np.int64)) & 1).astype(np.uint8) == r).all(), \
        "G x != sigma XOR eps"
    assert (head.poly_sign01(x) == head.s01(states).astype(np.uint8)).all(), \
        "closed-form sign of the features != production head sign"
    # eps is decoder-A's correction (same Matching object as the production head)
    corr = head._matching.decode_batch(head._syndrome(bits)).astype(np.uint8)
    assert (corr == eps).all()
    # leading-shape passthrough (netket may hand (chains, batch, N))
    u2 = head.features_ex(states.reshape(5, 100, head.N))
    assert u2.shape == (5, 100, head.n_features_ex) and (u2.reshape(B, -1) == u).all()
    # on-sector inputs: eps == 0 and x are the true hexagon-flip variables
    X, R = _all_sector_configs(head)
    u3 = head.features_ex(1 - 2 * R.astype(np.float64))
    assert not u3[:, :head.N].any() and (u3[:, head.N:].astype(np.uint8) == X).all()
    # non-mwpm decoders have no decoder-A features
    alt = QECSignHead(g, decoder='greedy')
    try:
        alt.features_ex(states[:4])
    except AssertionError:
        pass
    else:
        raise AssertionError("features_ex accepted a non-mwpm head")
    print(f"      2x3: K={head.n_features_ex} (N={head.N} + F={head.F}); "
          f"mean syndrome weight {head._syndrome(bits).sum(1).mean():.2f}, "
          f"mean |eps| {eps.sum(1).mean():.2f}")


# ---- e. M-pre npz contract ---------------------------------------------------
def check_e():
    import tempfile
    K, hidden, depth = 27 + 6, 64, 2
    spec = mlp_param_spec(K, hidden, depth)
    assert list(spec) == ["Dense_0/kernel", "Dense_0/bias", "Dense_1/kernel", "Dense_1/bias",
                          "Dense_2/kernel", "Dense_2/bias"]
    assert spec["Dense_0/kernel"] == (K, hidden) and spec["Dense_1/kernel"] == (hidden, hidden)
    assert spec["Dense_2/kernel"] == (hidden, 1) and spec["Dense_2/bias"] == (1,)
    assert list(mlp_param_spec(K, hidden, 0)) == ["Dense_0/kernel", "Dense_0/bias"]
    assert mlp_param_spec(K, hidden, 0)["Dense_0/kernel"] == (K, 1)
    rng = np.random.default_rng(3)
    flat = {k: rng.normal(size=shape) for k, shape in spec.items()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mlp.npz")
        save_mlp_npz(path, flat)
        back = load_mlp_npz(path, spec)
        assert set(back) == set(flat) and all(np.array_equal(back[k], flat[k]) for k in flat)
        tree = nest(back)
        assert set(tree) == {"Dense_0", "Dense_1", "Dense_2"}
        assert set(tree["Dense_0"]) == {"kernel", "bias"}
        # the live tree of a different width must reject it (the loader's contract)
        try:
            load_mlp_npz(path, mlp_param_spec(K, 32, depth))
        except AssertionError:
            pass
        else:
            raise AssertionError("shape mismatch not rejected")
        try:
            load_mlp_npz(path, mlp_param_spec(K, hidden, 3))
        except AssertionError:
            pass
        else:
            raise AssertionError("key mismatch not rejected")
        bad = dict(flat)
        bad["Dense_0/bias"] = np.full(hidden, np.nan)
        save_mlp_npz(path, bad)
        try:
            load_mlp_npz(path, spec)
        except AssertionError:
            pass
        else:
            raise AssertionError("non-finite npz not rejected")


# ---- f. T_gate ceiling ---------------------------------------------------------
def t_gate_quadrants(w, s_head, sgn_ed):
    """numpy replica of scripts/sign_fidelity.run_point's T_gate bookkeeping."""
    quad = {(sh, se): float(w[(s_head == sh) & (sgn_ed == se)].sum())
            for sh in (+1, -1) for se in (+1, -1)}
    tp = min(quad[(+1, -1)], quad[(+1, +1)])
    tm = min(quad[(-1, -1)], quad[(-1, +1)])
    return tp, tm, min(tp, tm), ("plus" if tp <= tm else "minus")


def t_gate_definition(w, s_head, sgn_ed, pinned):
    """min over the global sign g of the weight in the PINNED head-sector whose
    ED sign differs from g (the literal plan-Sec-7 definition)."""
    return min(float(w[(s_head == pinned) & (sgn_ed != g)].sum()) for g in (+1, -1))


def check_f():
    rng = np.random.default_rng(5)
    for trial in range(200):
        n = int(rng.integers(8, 400))
        psi = rng.normal(size=n) * rng.choice([1.0, 0.0], size=n, p=[0.9, 0.1])
        w = psi ** 2
        w /= w.sum()
        sgn_ed = np.where(psi >= 0, 1, -1)
        s_head = rng.choice([1, -1], size=n)
        tp, tm, tg, br = t_gate_quadrants(w, s_head, sgn_ed)
        assert abs(tp - t_gate_definition(w, s_head, sgn_ed, +1)) < 1e-15
        assert abs(tm - t_gate_definition(w, s_head, sgn_ed, -1)) < 1e-15
        assert tg == min(tp, tm) and br == ("plus" if tp <= tm else "minus")
        # <= 1 - F_s: the head itself is one admissible T-state (g = +1 with s pinned)
        one_minus_fs = float(w[s_head != sgn_ed].sum())
        assert tg <= one_minus_fs + 1e-15 and tp <= one_minus_fs + 1e-15
        # global gauge psi -> -psi leaves T_gate invariant (min over g)
        sgn_flip = np.where(-psi >= 0, 1, -1)
        tp2, tm2, tg2, _ = t_gate_quadrants(w, s_head, sgn_flip)
        # zero-amplitude configs carry sign +1 in both gauges but weight 0
        assert abs(tp2 - tp) < 1e-15 and abs(tm2 - tm) < 1e-15 and abs(tg2 - tg) < 1e-15
        # psi = s |psi| exactly => the head is exact and T_gate = 0 in BOTH branches
        tp3, tm3, tg3, _ = t_gate_quadrants(w, s_head, s_head)
        assert tp3 == tm3 == tg3 == 0.0
        # brute force over all 2^n sign assignments is too big; instead check the
        # optimum of the two-branch family directly on a small n: any psi_T with
        # sign = g on the pinned sector and free elsewhere has wrong weight >= T_gate
        if n <= 12:
            pinned = +1
            for g in (+1, -1):
                for free in range(1 << n):
                    sg = np.where(s_head == pinned, g,
                                  np.where(((free >> np.arange(n)) & 1) == 1, 1, -1))
                    wrong = float(w[sg != sgn_ed].sum())
                    assert wrong >= tp - 1e-15
    # production head at 1x2 with a synthetic signed state over ALL 2^N configs
    from model.honeycomb_geometry import HoneycombGeometry
    from model.sign_head import QECSignHead
    g = HoneycombGeometry(1, 2)
    head = QECSignHead(g)
    idx = np.arange(1 << g.N, dtype=np.int64)
    spins = (1 - 2 * ((idx[:, None] >> np.arange(g.N)) & 1)).astype(np.float64)
    s_head = (1 - 2 * np.asarray(head.s01(spins))).astype(np.int64)
    psi = rng.normal(size=1 << g.N)
    w = psi ** 2
    w /= w.sum()
    sgn_ed = np.where(psi >= 0, 1, -1)
    tp, tm, tg, br = t_gate_quadrants(w, s_head, sgn_ed)
    assert abs(tp - t_gate_definition(w, s_head, sgn_ed, +1)) < 1e-15
    assert abs(tm - t_gate_definition(w, s_head, sgn_ed, -1)) < 1e-15
    # the head's own state (psi = s_head |psi|) => 0; its negation => 0 too
    assert t_gate_quadrants(w, s_head, s_head)[2] == 0.0
    assert t_gate_quadrants(w, s_head, -s_head)[2] == 0.0
    # a pure Hastings-positive state: plus-branch wrong weight = weight of s=+1
    # configs vs g=-1 ... i.e. min(W[s=+1], 0) = 0 -- both branches 0, so a
    # positive state is representable by T (a > 0 with A_top -> 0)
    assert t_gate_quadrants(w, s_head, np.ones_like(s_head))[2] == 0.0
    print(f"      1x2 production head: T_gate_plus={tp:.4f} T_gate_minus={tm:.4f} "
          f"branch={br} (random signed psi)")


# ---- g. signbench_summary fixture --------------------------------------------
def check_g():
    import json
    import tempfile
    from scripts.signbench_summary import build_records, load_fidelity, load_runs
    size, hx, hz = "2x3", 0.4, 0.0
    with tempfile.TemporaryDirectory() as d:
        run_dir, diag, pre = [os.path.join(d, x) for x in ("nqs", "diag", "pretrain")]
        for x in (run_dir, diag, pre):
            os.makedirs(x)
        arms = {"cnnqM": (12170, {}), "cnnqMp": (12170, {"mlp_init": ["x.npz"]}),
                "cnnqT": (20000, {"mix_init": [0.05]})}
        for arm, (npar, extra) in arms.items():
            base = os.path.join(run_dir, f"G-equiv_1_hc{size}_ds_hx{hx:g}_hz{hz:g}_{arm}")
            json.dump({"energy": [-27.0 + 1e-4 * i for i in range(30)],
                       "Vscore": [1e-4] * 30,
                       "sim_params": {"n_params": [npar], "mlp_hidden": [64],
                                      "mlp_depth": [2], **extra}},
                      open(base + ".json", "w"))
            open(base + ".mpack", "wb").close()
        json.dump({"Lx": 2, "Ly": 3, "records": [
            {"hx": hx, "hz": hz, "hy": 0.0, "arm": "cnnqM", "F": 0.9},
            {"hx": hx, "hz": hz, "hy": 0.0, "arm": "cnnqMp", "F": 0.99},
            {"hx": hx, "hz": hz, "hy": 0.0, "arm": "cnnqT", "F": 0.999, "mix": 0.03},
            {"hx": hx, "hz": hz, "hy": 0.4, "arm": "cnnqT", "F": 0.5, "mix": -1.0},  # must NOT shadow
        ]}, open(os.path.join(diag, "fidelity_x.json"), "w"))
        json.dump({"best_1_minus_Fs": 3e-5, "final_1_minus_Fs": 5e-5,
                   "min_1_minus_Fs": 3e-5, "best_epoch": 17},
                  open(os.path.join(pre, f"mlp_hc{size}_hx{hx:g}_hz{hz:g}_h64d2.json"), "w"))
        runs, fid = load_runs(run_dir), load_fidelity(diag)
        assert (size, hx, hz, 0.0, "cnnqT", 0) in fid and (size, hx, hz, 0.4, "cnnqT", 0) in fid
        refs_new = {(size, hx, hz): {"E0": -27.1, "F_s": 0.999, "F_plus": 0.2,
                                     "T_gate": 1e-6, "T_gate_branch": "plus"}}
        refs_old = {(size, hx, hz): {"E0": -27.1, "F_s": 0.999, "F_plus": 0.2,
                                     "T_gate": None, "T_gate_branch": None}}
        for refs, kind, ceil in ((refs_new, "T_gate", 1e-6), (refs_old, "1-Fs", 1 - 0.999)):
            recs = {r["arm"]: r for r in build_records(size, [(hx, hz)], list(arms), runs,
                                                       fid, refs, 20, pretrain_dir=pre)}
            t = recs["cnnqT"]
            assert t["ceiling_kind"] == kind and abs(t["ceiling"] - ceil) < 1e-15, t
            assert t["mix"] == 0.03 and abs(t["one_minus_F"] - 0.001) < 1e-12
            assert t["n_params"] == 20000 and t["prior"] == "head-only init (mix=0.05)"
            assert recs["cnnqM"]["ceiling"] == 0.0 and recs["cnnqM"]["ceiling_kind"] == "exact-0"
            assert recs["cnnqM"]["prior"] == "random" and recs["cnnqM"]["n_params"] == 12170
            mp = recs["cnnqMp"]
            assert mp["prior"] == "ED-pretrained@point (warm 1-Fs=3.00e-05)" and mp["warm_1mFs"] == 3e-5
            assert mp["mix"] is None and recs["cnnqM"]["mix"] is None
        # no pretrain JSON => '?' not a crash
        os.remove(os.path.join(pre, f"mlp_hc{size}_hx{hx:g}_hz{hz:g}_h64d2.json"))
        mp = [r for r in build_records(size, [(hx, hz)], ["cnnqMp"], runs, fid, refs_new, 20,
                                       pretrain_dir=pre)][0]
        assert mp["prior"] == "ED-pretrained@point (warm 1-Fs=?)" and mp["warm_1mFs"] is None
        # missing run keeps the fixed schema incl. the new keys
        miss = build_records(size, [(0.8, 0.2)], ["cnnqT"], runs, fid, refs_new, 20, pretrain_dir=pre)[0]
        assert miss["missing"] and all(k in miss for k in ("ceiling_kind", "mix", "n_params", "prior", "warm_1mFs"))


# ---- h. _SignMLP recentring ------------------------------------------------------
def check_h():
    src = open(os.path.join(ROOT, "model", "honeycomb_networks.py")).read()
    body = src[src.index("class _SignMLP"):src.index("class MLPSignModel")]
    assert "h = 2.0 * u - 1.0" in body, "recentring line missing from _SignMLP"
    pre = open(os.path.join(ROOT, "scripts", "pretrain_sign_mlp.py")).read()
    assert "mlp.apply({\"params\": p}, xb.astype(jnp.float64))" in pre, \
        "pretrain script no longer feeds the raw {0,1} features straight into _SignMLP"
    assert "2.0 * " not in pre and "2 * X" not in pre, "pretrain must not pre-recentre"
    # numpy replica: zero biases, lecun-normal kernels; all-zero features
    rng = np.random.default_rng(7)
    K, hidden, depth = 27 + 6, 64, 2
    kernels = [rng.normal(scale=1.0 / np.sqrt(K), size=(K, hidden))]
    kernels += [rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden, hidden)) for _ in range(depth - 1)]
    kernels += [rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden, 1))]

    def mlp(u, recentre):
        h = 2.0 * u - 1.0 if recentre else u
        for k in kernels[:-1]:
            h = np.tanh(h @ k)
        return (h @ kernels[-1])[..., 0]
    u0 = np.zeros((1, K))
    assert mlp(u0, recentre=False)[0] == 0.0          # the bug: all-up config on the node
    assert abs(mlp(u0, recentre=True)[0]) > 1e-3       # fixed: generic nonzero m
    # recentring is a fixed affine map: any function of {0,1}^K is still representable
    # (bias absorbs the shift), and outputs on {0,1} inputs stay bounded like before
    u = rng.integers(0, 2, size=(500, K)).astype(np.float64)
    m = mlp(u, recentre=True)
    assert np.isfinite(m).all() and np.abs(m).max() < 10 * np.sqrt(hidden)


# ---- i. job-script jobid canonicalisation ----------------------------------------
def check_i():
    import subprocess
    job = open(os.path.join(ROOT, "jobs", "nersc_signbench.sh")).read()
    assert 'jobid="hc${LX}x${LY}_ds_hx${hx_g}_hz${hz_g}_${arm}${SEED_SUFFIX}"' in job
    assert 'case "$MLP_DIR" in /*) ;; *) MLP_DIR="$REPO/$MLP_DIR" ;; esac' in job
    assert 'mlp_npz="$MLP_DIR/mlp_hc' in job and '"$REPO/$MLP_DIR/' not in job
    for pt, want in (("0.40:0.0", "hx0.4_hz0"), ("0.4:0", "hx0.4_hz0"),
                     ("1.2:0.20", "hx1.2_hz0.2"), ("0.8:0.4", "hx0.8_hz0.4")):
        out = subprocess.check_output(["bash", "-c", (
            'pt="$1"; hx="${pt%%:*}"; hz="${pt##*:}"; '
            "hx_g=$(LC_ALL=C awk -v v=\"$hx\" 'BEGIN{printf \"%g\", v}'); "
            "hz_g=$(LC_ALL=C awk -v v=\"$hz\" 'BEGIN{printf \"%g\", v}'); "
            'echo "hx${hx_g}_hz${hz_g}"'), "_", pt]).decode().strip()
        assert out == want, (pt, out, want)
    for mdir, want in (("results/pretrain", "/REPO/results/pretrain"), ("/abs/x", "/abs/x")):
        out = subprocess.check_output(["bash", "-c", (
            'REPO=/REPO; MLP_DIR="$1"; case "$MLP_DIR" in /*) ;; *) MLP_DIR="$REPO/$MLP_DIR" ;; esac; '
            'echo "$MLP_DIR"'), "_", mdir]).decode().strip()
        assert out == want, (mdir, out, want)


if __name__ == "__main__":
    for L in ((1, 2), (2, 2), (2, 3), (3, 3)):
        check(f"a. x_of_r round-trip + poly == loop parity, all sector configs @ {L[0]}x{L[1]}",
              lambda L=L: check_a(*L))
    check("b. TwoBranchModel stable log-psi == direct a A1 + s A2 (signed a, a=0, 1e-12 cancellation)",
          check_b)
    check("c. MLPSignModel complex log == direct A * tanh(m)", check_c)
    check("d. features_ex @ 2x3: shapes, empty syndrome of eps XOR sigma, G x == r, poly == s01",
          check_d)
    check("e. sign_mlp_io npz spec / round-trip / rejection (M-pre loader contract)", check_e)
    check("f. T_gate quadrant formula == plan-Sec-7 definition (random + production head @1x2)",
          check_f)
    check("g. signbench_summary fixture: T_gate ceiling / 1-Fs fallback / n_params / prior / hy key",
          check_g)
    check("h. _SignMLP recentring h = 2u - 1 (all-up config off the m = 0 node; pretrain feeds raw)",
          check_h)
    check("i. nersc_signbench.sh canonical %g jobid + MLP_DIR resolution", check_i)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
