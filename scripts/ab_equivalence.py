"""Phase-4 A/B equivalence witness: sign-in-state vs sign-framed Hamiltonian.

Formulation A:  log psi = log A_theta + 1j*pi*s(sigma)   (SignedModel), on H.
Formulation B:  positive A_theta                          on H~ = SHS.

Pre-registered claims this script tests, at shared parameters:
  1. BITWISE: Re log psi_A == log A_B on every config; Im log psi_A == pi*s
     exactly (s in {0,1}).
  2. Per-config local energies: E_loc^B (real) vs Re E_loc^A -- equal to
     ~1e-15 (cos(pi) = -1.0 exactly in float64; the exp(i*pi) dust sits in
     the IMAGINARY part, ~1e-16, reported separately).
  3. <H~>_B == Re <H>_A on the same Monte-Carlo samples (same seed => same
     chains, since sampling sees only Re log psi).
  4. Short paired minSR training: per-step energies side by side. NOT expected
     bit-for-bit (A auto-selects jacobian_mode 'complex', B 'real' => float
     summation order differs); expected to agree within MC error bars.

Run in the CPU venv:
    PYTHONPATH=. python scripts/ab_equivalence.py --Lx 1 --Ly 2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                  # noqa: E402
import netket as nk                                         # noqa: E402
import optax                                                # noqa: E402

from model.honeycomb_geometry import HoneycombGeometry      # noqa: E402
from model.hamiltonian import create_honeycomb_hamiltonian  # noqa: E402
from model.honeycomb_networks import (                      # noqa: E402
    create_honeycomb_model, SignedModel)
from model.sign_head import QECSignHead                     # noqa: E402
from model.sign_frame import SignFramedOperator             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, default=1)
    ap.add_argument("--Ly", type=int, default=2)
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--diag_shift", type=float, default=1e-4)
    args = ap.parse_args()

    g = HoneycombGeometry(args.Lx, args.Ly)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.N)
    H = create_honeycomb_hamiltonian(hi, g, "ds", 1.0, args.hx, args.hz)
    head = QECSignHead(g)
    H_framed = SignFramedOperator(H, head)

    cfg = {"architecture": "Combo", "dtype": "float64",
           "channels_noninv": [1, 16], "channels_inv": [16, 8, 1]}
    base = create_honeycomb_model(cfg, g)
    signed = SignedModel(base, head.s01)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=8)
    vsB = nk.vqs.MCState(sa, base, n_samples=args.n_samples, seed=0)
    vsA = nk.vqs.MCState(sa, signed, n_samples=args.n_samples, seed=0)
    # flax RNG folding depends on the module path ('base/...' vs '...'), so
    # the two inits differ -- sync A's base subtree to B's parameters.
    vsA.parameters = {"base": vsB.parameters}

    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # ---- 1. log psi, bitwise -------------------------------------------
    configs = np.asarray(hi.random_state(jax.random.PRNGKey(7), 512))
    logB = np.asarray(vsB.log_value(configs))
    logA = np.asarray(vsA.log_value(configs))
    s = head.s01(configs)
    check("Re log psi_A == log A_B (bitwise)",
          np.array_equal(np.real(logA), logB))
    check("Im log psi_A == pi*s (bitwise)",
          np.array_equal(np.imag(logA), np.pi * s),
          f"(sign-flipped configs in batch: {int(s.sum())}/512)")

    # ---- 2. per-config local energies ----------------------------------
    def eloc(vstate, op, x, logx):
        xp, mels = op.get_conn_padded(x)
        lead = xp.shape[:-1]
        logp = np.asarray(vstate.log_value(np.asarray(xp).reshape(-1, g.N)))
        logp = logp.reshape(lead)
        return (np.asarray(mels) * np.exp(logp - logx[:, None])).sum(axis=1)

    eB = eloc(vsB, H_framed, configs, logB)
    eA = eloc(vsA, H, configs, logA)
    dre = np.abs(np.real(eA) - np.real(eB)).max()
    dim_ = np.abs(np.imag(eA)).max()
    scale = np.abs(eB).max()
    check("Re E_loc^A == E_loc^B", dre <= 1e-12 * scale,
          f"(max |dRe| = {dre:.3e}, scale {scale:.2e})")
    print(f"[info] E_loc^B exactly real: {np.abs(np.imag(eB)).max():.1e}; "
          f"A's exp(i*pi) imaginary dust: {dim_:.3e}")

    # ---- 3. expectations on the same chains ----------------------------
    EB = vsB.expect(H_framed)
    EA = vsA.expect(H)
    check("<H~>_B == Re <H>_A (same seed/chains)",
          abs(complex(EB.mean).real - complex(EA.mean).real) < 1e-12,
          f"(B {complex(EB.mean).real:.12f}, A {complex(EA.mean).real:.12f})")

    # one off-diagonal observable, framed-vs-bare cross-check
    p0 = None
    for j in g.plaq_all[0]:
        o = nk.operator.spin.sigmax(hi, int(j), dtype="complex").to_pauli_strings()
        p0 = o if p0 is None else p0 @ o
    w = np.asarray(p0.weights)
    p0 = nk.operator.PauliStrings(hi, [str(t) for t in p0.operators],
                                  np.real(w).astype(np.float64))
    oB = complex(vsB.expect(SignFramedOperator(p0, head)).mean).real
    oA = complex(vsA.expect(p0).mean).real
    check("<X_hex0>: framed@B == bare@A", abs(oB - oA) < 1e-12,
          f"(B {oB:.12f}, A {oA:.12f})")

    # ---- 4. paired short training --------------------------------------
    from netket.experimental.driver import VMC_SRt
    drB = VMC_SRt(H_framed, optax.sgd(args.lr), diag_shift=args.diag_shift,
                  variational_state=vsB)
    drA = VMC_SRt(H, optax.sgd(args.lr), diag_shift=args.diag_shift,
                  variational_state=vsA)
    print(f"\nstep {'E_B (framed)':>16} {'Re E_A (signed)':>16} {'|diff|':>10}")
    worst = 0.0
    for it in range(args.steps):
        drB.advance(1)
        drA.advance(1)
        eB_ = complex(drB._loss_stats.mean).real
        eA_ = complex(drA._loss_stats.mean).real
        err = max(drB._loss_stats.error_of_mean, drA._loss_stats.error_of_mean)
        worst = max(worst, abs(eB_ - eA_) / max(err, 1e-300))
        print(f"{it:4d} {eB_:16.10f} {eA_:16.10f} {abs(eB_ - eA_):10.2e}")
    check("paired training: |E_B - E_A| within MC error throughout",
          worst < 5.0, f"(worst |diff|/err = {worst:.2f})")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
