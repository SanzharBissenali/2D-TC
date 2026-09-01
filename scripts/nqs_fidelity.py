"""Exact trained-NQS fidelity vs ED for the honeycomb DS campaigns.

For every requested (point, arm) with a saved checkpoint, rebuilds the network
from the run's sim_params, loads the .mpack into a structurally identical
MCState (main.py's warm-start pattern), evaluates log psi on ALL 2^N
configurations (GPU batches, ED bit convention: site i <-> bit i, bit 1 =
spin DOWN, all-up = index 0), applies the run's decoder sign where the sign
lives in the OPERATOR frame (cnnqB/cnnqC arms; cnnqR's model already carries
its sign; cnn/cnnc/plaincnn have none), and reports

    F        = |<psi_ED | s * psi_trunk>|^2 / norms   (the physical fidelity)
    F_trunk  = |<psi_ED | psi_trunk>|^2 / norms       (head contribution probe)

SELF-VALIDATING: <H> is recomputed from the enumerated psi via the same CSR
the ED used and compared against the run JSON's tail energy -- a wrong model
rebuild or parameter load fails this check loudly instead of producing a
plausible-looking fidelity. Runs on a GPU node (jax forward passes) with the
ED memory profile of jobs/nersc_signfid.sh.

Example:
    PYTHONPATH=. python scripts/nqs_fidelity.py --Lx 2 --Ly 3 --hy 0.4 \
        --points "0,0;0.4,0" --arms cnnqC,cnnqC_anchor \
        --out results/diagnostics/fidelity_hc2x3_hy0.4_c1.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact.lanczos_ed import _honeycomb_direct_ed                # noqa: E402
from model.honeycomb_geometry import HoneycombGeometry           # noqa: E402
from scripts.sign_fidelity import (loop_sign_table,              # noqa: E402
                                   build_decoder_tables, syndrome_bits,
                                   decoder_signs)

# arm token -> (base trunk, complex?, sign source). decoder comes from the
# token suffix (or 'mwpm'); 'operator' arms get the sign applied here.
ARM_BASE = {
    'cnnqB': ('combo', False, 'operator'),
    'cnnqC': ('combo', True, 'operator'),
    'cnnqR': ('combo', False, 'model'),   # ResidualSignedModel carries the sign
    'cnn': ('combo', False, 'none'),
    'cnnc': ('combo', True, 'none'),
    'plaincnn': ('plain', False, 'none'),
}
DEC_NAMES = ('anchor', 'greedy', 'unionfind', 'tie_sum', 'mwpm')


def parse_arm(token):
    for base in sorted(ARM_BASE, key=len, reverse=True):
        if token == base:
            return base, 'mwpm'
        if token.startswith(base + '_'):
            dec = token[len(base) + 1:]
            assert dec in DEC_NAMES, f"unknown decoder suffix in {token!r}"
            return base, dec
    raise ValueError(f"unknown arm token {token!r}")


def run_config(base, is_complex, hx, hz, hy, sp):
    """Campaign recipe + sim_params overrides -> the config dict the model
    factory and sampler need (recipe fixed across every notebook campaign)."""
    return {
        'Lx': int(sp.get('Lx', [2])[0]) if isinstance(sp.get('Lx'), list) else 2,
        'architecture': 'PlainCNN' if base == 'plain' else 'Combo',
        'symmetric_block': 'cnn',
        'channels_noninv': [1, 32, 24, 8, 2] if base == 'plain' else [1, 16],
        'channels_inv': [16, 8, 1],
        'kernel_size': 2,
        'dtype': 'complex' if (is_complex or hy != 0.0) else 'float64',
        'complex_ansatz': is_complex,
        'hx': hx, 'hy': hy, 'hz': hz,
        'use_custom_sampler': True,
        'n_samples': 8192, 'n_chains': 1024, 'n_discard': 8,
        'chunk_size': 2048, 'n_sweeps': 512, 'seed': 0,
        'res_hidden': int((sp.get('res_hidden') or [16])[0]),
    }


def build_vstate(cfg, geometry, head):
    import flax
    import netket as nk
    from model.honeycomb_networks import create_honeycomb_model
    from simulation.custom_sampler import create_custom_sampler

    hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)
    model = create_honeycomb_model(cfg, geometry)
    if cfg.get('_sign_source') == 'model':
        from model.honeycomb_networks import ResidualSignedModel
        model = ResidualSignedModel(model, head.features, head.n_features,
                                    hidden=cfg['res_hidden'])
    sa = create_custom_sampler(geometry, hi, cfg)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg['n_samples'],
                        n_discard_per_chain=cfg['n_discard'],
                        chunk_size=cfg['chunk_size'], seed=cfg['seed'])
    return vs, model


def enumerate_logs(model, variables, N, batch=1 << 17):
    """log psi over all 2^N configs (ED bit order), complex128 on host."""
    import jax
    import jax.numpy as jnp

    dim = 1 << N
    site = np.arange(N, dtype=np.int64)
    out = np.empty(dim, dtype=np.complex128)

    @jax.jit
    def fwd(x):
        return model.apply(variables, x)

    for lo in range(0, dim, batch):
        idx = np.arange(lo, min(lo + batch, dim), dtype=np.int64)
        spins = (1 - 2 * ((idx[:, None] >> site) & 1)).astype(np.float64)
        out[lo:lo + len(idx)] = np.asarray(fwd(jnp.asarray(spins)),
                                           dtype=np.complex128)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Lx', type=int, required=True)
    ap.add_argument('--Ly', type=int, required=True)
    ap.add_argument('--hy', type=float, default=0.0)
    ap.add_argument('--points', required=True,
                    help="semicolon-separated hx,hz pairs")
    ap.add_argument('--arms', required=True, help="comma list of arm tokens")
    ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    g = HoneycombGeometry(args.Lx, args.Ly)
    N = g.N
    print(f"# fidelity {args.Lx}x{args.Ly} hy={args.hy} N={N} dim=2^{N}",
          flush=True)

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    decs = sorted({parse_arm(a)[1] for a in arms
                   if ARM_BASE[parse_arm(a)[0]][2] == 'operator'})
    dec_tables = None
    if decs:
        dec_tables = build_decoder_tables(g, decs, 10)
        parr = np.zeros(1 << N, dtype=np.int8)
        for x, s in loop_sign_table(g).items():
            parr[x] = s
        dec_tables['parity_arr'] = parr

    from model.sign_head import QECSignHead
    head = QECSignHead(g)          # features() provider for the cnnqR arm

    hyseg = f"_hy{args.hy:g}" if args.hy != 0.0 else ""
    records = []
    for pt in args.points.split(';'):
        hx, hz = (float(x) for x in pt.split(','))
        t0 = time.time()
        evals, psi_ed, H = _honeycomb_direct_ed(
            g, 'ds', 1.0, hx, hz, k=1, tol=args.tol, hy=args.hy,
            return_H=True)
        psi_ed = np.asarray(psi_ed, dtype=np.complex128)
        psi_ed /= np.linalg.norm(psi_ed)
        print(f"# ED ({hx},{hz}) E0={evals[0]:.8f} in {time.time()-t0:.0f}s",
              flush=True)

        # decoder signs per config are arm-dependent; syndrome index shared
        sidx = None
        if dec_tables is not None:
            s_all = np.arange(1 << N, dtype=np.int64)
            sint = np.zeros(1 << N, dtype=np.int64)
            for lo in range(0, 1 << N, 1 << 22):
                c = s_all[lo:lo + (1 << 22)]
                sint[lo:lo + len(c)] = syndrome_bits(g, c).astype(np.int64) \
                    @ (1 << np.arange(g.n_vertices, dtype=np.int64))
            sidx = dec_tables['index'][sint]
            del sint

        for token in arms:
            base, dec = parse_arm(token)
            trunk, is_c, sign_src = ARM_BASE[base]
            name = f"hc{args.Lx}x{args.Ly}_ds_hx{hx:g}_hz{hz:g}{hyseg}_{token}"
            path = f"results/nqs/G-equiv_1_{name}"
            if not os.path.exists(path + ".mpack") and hx == hz == args.hy == 0:
                path = f"results/nqs/G-equiv_1_hc{args.Lx}x{args.Ly}_ds_h0_{token}"
            if not os.path.exists(path + ".mpack"):
                print(f"  skip {token} at ({hx},{hz}): no checkpoint", flush=True)
                continue

            sp = json.load(open(path + ".json")).get('sim_params', {})
            cfg = run_config(trunk, is_c, hx, hz, args.hy, sp)
            cfg['_sign_source'] = sign_src
            import flax
            vs, model = build_vstate(cfg, g, head)
            with open(path + ".mpack", 'rb') as f:
                vs = flax.serialization.from_bytes(vs, f.read())

            t1 = time.time()
            logs = enumerate_logs(model, vs.variables, N)
            m = float(np.max(logs.real))
            psi_t = np.exp(logs - m)
            nrm_t = np.linalg.norm(psi_t)
            psi_t /= nrm_t

            if sign_src == 'operator':
                sgn = np.empty(1 << N, dtype=np.int8)
                for lo in range(0, 1 << N, 1 << 22):
                    c = np.arange(lo, min(lo + (1 << 22), 1 << N),
                                  dtype=np.int64)
                    sgn[lo:lo + len(c)], _ = decoder_signs(
                        g, c, sidx[lo:lo + len(c)], dec_tables['parity_arr'],
                        dec_tables, dec)
                psi = psi_t * sgn
            else:
                psi = psi_t          # sign already in the model (or no sign)

            F = float(abs(np.vdot(psi_ed, psi)) ** 2)
            F_trunk = float(abs(np.vdot(psi_ed, psi_t)) ** 2) \
                if sign_src == 'operator' else None

            # SELF-CHECK: <H> from the enumerated state vs the run's tail
            e_chk = float(np.real(np.vdot(psi, H @ psi)))
            e_run = statistics.median(
                [complex(e).real for e in
                 json.load(open(path + ".json"))['energy'][-20:]])
            de = abs(e_chk - e_run)
            ok = de < max(0.02 * abs(evals[0]), 0.05)
            records.append({
                'hx': hx, 'hz': hz, 'hy': args.hy, 'arm': token,
                'F': F, 'F_trunk': F_trunk, 'E0': float(evals[0]),
                'E_check': e_chk, 'E_run_tail': e_run, 'dE': de,
                'energy_check_ok': bool(ok),
                't_eval_s': time.time() - t1,
            })
            print(f"  {token:22s} F={F:.6f}"
                  + (f" F_trunk={F_trunk:.6f}" if F_trunk is not None else "")
                  + f" E_chk={e_chk:.5f} E_run={e_run:.5f} dE={de:.2e} "
                  + ("OK" if ok else "ENERGY-CHECK-FAIL"), flush=True)
        del H, psi_ed

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'Lx': args.Lx, 'Ly': args.Ly, 'hy': args.hy,
                   'records': records}, f, indent=1)
    print(f"# wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
