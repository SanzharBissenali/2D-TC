"""Numpy-only validation of the learned-vs-gated sign-head benchmark pieces
(docs/signhead_benchmark_plan.md Sec. 3 steps 1-3, 8) -- no jax/flax/netket.

Needs pymatching + numba (the head's imports): run under the scratch venv, e.g.
    <venv>/bin/python scratchpad/validate_signbench.py

Checks (PASS/FAIL each):
  a. hexagon-flip variables at 1x2, 2x2, 2x3, 3x3: for ALL 2^F sector configs
     r = G x, x_of_r(r, check=True) == x, and poly_sign01(x) == #loops(r) mod 2
     by the INDEPENDENT python counter exact/loops.count_loops AND == the
     production head s01 on the same configs; an off-sector r fails the check.
  b. TwoBranchModel log-psi formula (numpy replica) vs direct psi = e^c A1 + s A2
     evaluated in 60-digit Decimal, incl. near-cancellation |psi| ~ 1e-12 A;
     imaginary part in {0, pi}; c -> -inf limit == head-only log psi.
  c. MLPSignModel complex log vs direct A * tanh(m).
  d. features_ex at 2x3: shape lead + (N+F,), {0,1} values, eps XOR sigma has
     an EMPTY syndrome, G x == eps XOR sigma, poly(x) == s01(sigma), leading-
     shape passthrough, head accounting, and the non-mwpm assert fires.
  e. model/sign_mlp_io npz spec: key/shape spec, save/load round-trip, and a
     wrong-shape npz is rejected (the M-pre loader contract).
"""
import os
import sys
import traceback
from decimal import Decimal, getcontext

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
def twobranch_log(a1, a2, c, s01):
    """numpy replica of honeycomb_networks.TwoBranchModel.__call__."""
    a1 = a1 + c
    s = 1.0 - 2.0 * s01
    m = np.maximum(a1, a2)
    w = np.exp(a1 - m) + s * np.exp(a2 - m)
    return m + np.log(np.abs(w)) + 1j * np.pi * (w < 0).astype(np.float64)


def _direct_log_decimal(a1, a2, c, s01):
    getcontext().prec = 60
    psi = (Decimal(float(c)) + Decimal(float(a1))).exp() \
        + (1 - 2 * int(s01)) * Decimal(float(a2)).exp()
    re = float(abs(psi).ln())
    im = np.pi if psi < 0 else 0.0
    return re, im


def check_b():
    rng = np.random.default_rng(0)
    B = 2000
    a1 = rng.normal(-5.0, 3.0, B)
    a2 = rng.normal(-5.0, 3.0, B)
    c = rng.normal(-3.0, 1.0, B)
    s01 = rng.integers(0, 2, B).astype(np.float64)
    out = twobranch_log(a1, a2, c, s01)
    for i in range(B):
        re, im = _direct_log_decimal(a1[i], a2[i], c[i], s01[i])
        assert abs(out[i].real - re) < 1e-10 * max(1.0, abs(re)), (i, out[i].real, re)
        assert out[i].imag == im, (i, out[i].imag, im)
    assert set(np.unique(out.imag)) <= {0.0, np.pi}
    # near-cancellation: s = -1 and a2 = a1 + c + log(1 - eps) => psi = e^{c} A1 * eps
    for eps in (1e-6, 1e-9, 1e-12):
        a1c, cc = 1.3, -2.0
        a2c = a1c + cc + np.log1p(-eps)
        got = twobranch_log(np.array([a1c]), np.array([a2c]), np.array([cc]), np.array([1.0]))[0]
        re, im = _direct_log_decimal(a1c, a2c, cc, 1.0)
        # float64 rounding of a2c itself limits agreement to ~1e-16/eps relative in psi
        assert abs(got.real - re) < 1e-3, (eps, got.real, re)
        assert got.imag == im and im == 0.0
        assert abs(got.real - (a1c + cc + np.log(eps))) < 1e-3, (eps, got.real)
    # mirror: s = -1, A2 slightly LARGER => psi negative, tiny
    a1c, cc, eps = 0.7, -1.5, 1e-12
    a2c = a1c + cc + np.log1p(eps)
    got = twobranch_log(np.array([a1c]), np.array([a2c]), np.array([cc]), np.array([1.0]))[0]
    assert got.imag == np.pi and abs(got.real - (a1c + cc + np.log(eps))) < 1e-3
    # c -> -inf: reduces to the head-only arm log A_top + i*pi*s01
    out = twobranch_log(a1, a2, np.full(B, -1e6), s01)
    assert np.allclose(out.real, a2) and (out.imag == np.pi * s01).all()
    # c = -3 at equal trunks: trivial branch is a ~5% admixture, sign from head
    out = twobranch_log(np.zeros(1), np.zeros(1), np.array([-3.0]), np.array([1.0]))[0]
    assert abs(np.exp(out.real) - (1 - np.exp(-3.0))) < 1e-15 and out.imag == np.pi


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


if __name__ == "__main__":
    for L in ((1, 2), (2, 2), (2, 3), (3, 3)):
        check(f"a. x_of_r round-trip + poly == loop parity, all sector configs @ {L[0]}x{L[1]}",
              lambda L=L: check_a(*L))
    check("b. TwoBranchModel stable log-psi == direct e^c A1 + s A2 (incl. 1e-12 cancellation)",
          check_b)
    check("c. MLPSignModel complex log == direct A * tanh(m)", check_c)
    check("d. features_ex @ 2x3: shapes, empty syndrome of eps XOR sigma, G x == r, poly == s01",
          check_d)
    check("e. sign_mlp_io npz spec / round-trip / rejection (M-pre loader contract)", check_e)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
