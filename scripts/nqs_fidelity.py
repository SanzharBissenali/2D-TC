"""Exact trained-NQS fidelity vs ED for the honeycomb DS campaigns.

For every requested (point, arm) with a saved checkpoint, rebuilds the network
from the run's sim_params, loads the .mpack into a structurally identical
MCState (main.py's warm-start pattern), evaluates log psi on ALL 2^N
configurations (GPU batches, ED bit convention: site i <-> bit i, bit 1 =
spin DOWN, all-up = index 0), applies the run's decoder sign where the sign
lives in the OPERATOR frame (cnnqB/cnnqC arms; cnnqR/cnnqM/cnnqMp/cnnqT's
model already carries its own sign; cnn/cnnc/plaincnn have none), and reports

    F        = |<psi_ED | s * psi_trunk>|^2 / norms   (the physical fidelity)
    F_trunk  = |<psi_ED | psi_trunk>|^2 / norms       (head contribution probe)

SELF-VALIDATING: <H> is recomputed from the enumerated psi via the same CSR
the ED used and compared against the run JSON's tail energy -- a wrong model
rebuild or parameter load fails this check loudly instead of producing a
plausible-looking fidelity. Runs on a GPU node (jax forward passes) with the
ED memory profile of jobs/nersc_signfid.sh.

Learned-sign-benchmark arms (docs/signhead_benchmark_plan.md): cnnqM (arm M,
random-init sign MLP), cnnqMp (arm M-pre; STRUCTURALLY identical to cnnqM --
the .mpack already carries the supervised-pretrained-then-VMC-trained 'mlp'
subtree, so --mlp_init must NOT be re-applied here, only the checkpoint's own
values matter), cnnqT (arm T, two-branch mix). All three carry their own sign
in the model's complex log psi (like cnnqR), so the generic F/F_trunk=None
path applies; cnnqT additionally reports a HEAD-STRIPPED F_trunk (s forced to
+1, i.e. fidelity of e^{a1-m}+e^{a2-m} alone against psi_ED -- the two-branch
analogue of "fidelity without the head's sign") and the trained log-mix c.

Example:
    PYTHONPATH=. python scripts/nqs_fidelity.py --Lx 2 --Ly 3 --hy 0.4 \
        --points "0,0;0.4,0" --arms cnnqC,cnnqC_anchor \
        --out results/diagnostics/fidelity_hc2x3_hy0.4_c1.json
"""

import argparse
import json
import os
import re
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
# token suffix (or 'mwpm'); 'operator' arms get the sign applied here; every
# other non-'none' source already carries its sign in the model's own
# complex log psi (decoder suffixes are meaningless there -- see the
# --decoder guard in main()).
ARM_BASE = {
    'cnnqB': ('combo', False, 'operator'),
    'cnnqC': ('combo', True, 'operator'),
    'cnnqR': ('combo', False, 'model'),   # ResidualSignedModel carries the sign
    'cnnqM': ('combo', False, 'mlp'),     # arm M: MLPSignModel, random init
    'cnnqMp': ('combo', False, 'mlp'),    # arm M-pre: same structure; the
                                          # checkpoint already carries the
                                          # supervised pre-fit (never re-apply
                                          # --mlp_init here)
    'cnnqT': ('combo', False, 'twobranch'),  # arm T: TwoBranchModel
    'cnnqTp': ('combo', False, 'twobranch'), # arm T+: positive mix a = exp(c) (--mix_positive)
    'cnn': ('combo', False, 'none'),
    'cnnc': ('combo', True, 'none'),
    'plaincnn': ('plain', False, 'none'),
}
DEC_NAMES = ('anchor', 'greedy', 'unionfind', 'tie_sum', 'mwpm')


_SEED_SUFFIX_RE = re.compile(r'_s\d+$')


def parse_arm(token):
    """(base, decoder). A trailing "_s<seed>" (jobs/nersc_signbench.sh's
    seed-1-replica jobid convention -- seed 0 is never suffixed, so this is
    a no-op for every token used before that script existed) is stripped
    ONLY for this base/decoder match; ``token`` itself (used verbatim for
    the checkpoint filename and recorded as-is in the output) is untouched,
    so e.g. 'cnnqM_s1' -> ('cnnqM', 'mwpm') and 'cnnqB_greedy_s1' ->
    ('cnnqB', 'greedy')."""
    stem = _SEED_SUFFIX_RE.sub('', token, count=1)
    for base in sorted(ARM_BASE, key=len, reverse=True):
        if stem == base:
            return base, 'mwpm'
        if stem.startswith(base + '_'):
            dec = stem[len(base) + 1:]
            assert dec in DEC_NAMES, f"unknown decoder suffix in {token!r}"
            return base, dec
    raise ValueError(f"unknown arm token {token!r}")


def _sp(sp, key, default):
    """sim_params values are 1-element lists; unwrap with a fallback."""
    v = sp.get(key)
    if v is None:
        return default
    if isinstance(v, list) and len(v) == 1 and not isinstance(default, list):
        return type(default)(v[0])
    if isinstance(v, list) and v and isinstance(v[0], list):
        return list(v[0])
    return v


def run_config(base, is_complex, hx, hz, hy, sp):
    """Campaign recipe with sim_params overrides -> the config dict the model
    factory and sampler need. CRITICAL: forward-pass constants (rescale,
    channels, kernel_size) must come from the RUN's own record -- the factory
    default for rescale (10**1.5) differs from the trained value (1.0), which
    silently changes the network function (caught by the validation job's
    energy self-check, 2026-09-01)."""
    return {
        'architecture': 'PlainCNN' if base == 'plain' else 'Combo',
        'symmetric_block': 'cnn',
        'channels_noninv': _sp(sp, 'n_chann_noninv',
                               [1, 32, 24, 8, 2] if base == 'plain' else [1, 16]),
        'channels_inv': _sp(sp, 'n_chann_inv', [16, 8, 1]),
        'kernel_size': _sp(sp, 'kernel_size_noninv', 2),
        'rescale': _sp(sp, 'rescale', 1.0),
        'dtype': 'complex' if (is_complex or hy != 0.0) else 'float64',
        'complex_ansatz': is_complex,
        'hx': hx, 'hy': hy, 'hz': hz,
        'use_custom_sampler': True,
        'n_samples': 8192, 'n_chains': 1024, 'n_discard': 8,
        'chunk_size': 2048, 'n_sweeps': 512, 'seed': 0,
        'res_hidden': _sp(sp, 'res_hidden', 16),
        'mlp_hidden': _sp(sp, 'mlp_hidden', 64),
        'mlp_depth': _sp(sp, 'mlp_depth', 2),
        'mix_init': _sp(sp, 'mix_init', 0.05),
        'mix_positive': bool(_sp(sp, 'mix_positive', False)),
    }


def build_vstate(cfg, geometry, head):
    """(vstate, model) for one arm. ``model`` is the network actually wired
    into ``vs`` (used by enumerate_logs); the base Combo trunk lives at
    vs.parameters['base'] (mlp/model/residual arms) or
    vs.parameters['base_triv']/['base_top'] (twobranch)."""
    import flax
    import netket as nk
    from model.honeycomb_networks import create_honeycomb_model
    from simulation.custom_sampler import create_custom_sampler

    hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)
    model = create_honeycomb_model(cfg, geometry)
    src = cfg.get('_sign_source')
    if src == 'model':
        from model.honeycomb_networks import ResidualSignedModel
        model = ResidualSignedModel(model, head.features, head.n_features,
                                    hidden=cfg['res_hidden'])
    elif src == 'mlp':
        # arms M / M-pre: STRUCTURALLY identical (same hidden/depth); the
        # checkpoint's own 'mlp' subtree already reflects whichever init +
        # training the run used -- no --mlp_init reapplication here.
        from model.honeycomb_networks import MLPSignModel
        model = MLPSignModel(model, head.features_ex, head.n_features_ex,
                             hidden=cfg['mlp_hidden'], depth=cfg['mlp_depth'])
    elif src == 'twobranch':
        from model.honeycomb_networks import create_two_branch_model
        model, _ = create_two_branch_model(cfg, geometry, head)
    sa = create_custom_sampler(geometry, hi, cfg)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg['n_samples'],
                        n_discard_per_chain=cfg['n_discard'],
                        chunk_size=cfg['chunk_size'], seed=cfg['seed'])
    return vs, model


def head_sign_pm1_all(head, N, batch=1 << 17):
    """Decoded head sign s in {+-1} over all 2^N configs (ED bit order)."""
    dim = 1 << N
    site = np.arange(N, dtype=np.int64)
    out = np.empty(dim, dtype=np.float64)
    for lo in range(0, dim, batch):
        idx = np.arange(lo, min(lo + batch, dim), dtype=np.int64)
        spins = (1 - 2 * ((idx[:, None] >> site) & 1)).astype(np.float64)
        out[lo:lo + len(idx)] = 1.0 - 2.0 * np.asarray(head.s01(spins))
    return out


def two_branch_head_stripped_fidelity(cfg, geometry, psi_ed, params, N, s_pm):
    """arm T's F_trunk analogue: force the head sign s -> +1 AND drop the sign
    of the mix (keep both branch amplitudes), i.e. fidelity of the positive
    state |a_s| e^{a1-m} + e^{a2-m} (a1 = log A_triv, a2 = log A_top, a_s the
    per-sector mix selected by the head sign s_pm,
    m = max(a1,a2)) against psi_ED. Evaluates the two bare Combo trunks separately (same
    module definition, different param subtrees -- flax modules are
    stateless) since TwoBranchModel's own forward always applies the true
    decoded sign."""
    from model.honeycomb_networks import create_honeycomb_model

    bare = create_honeycomb_model(cfg, geometry)
    a1 = np.real(enumerate_logs(bare, {'params': params['base_triv']}, N))
    a2 = np.real(enumerate_logs(bare, {'params': params['base_top']}, N))
    m = np.maximum(a1, a2)
    mix = (np.exp(np.asarray(params['log_mix'])).reshape(-1) if 'log_mix' in params
           else np.asarray(params['mix']).reshape(-1))
    a_abs = np.abs(np.where(s_pm > 0, mix[0], mix[-1]))
    psi_hs = (a_abs * np.exp(a1 - m)
              + np.exp(a2 - m)).astype(np.complex128)
    psi_hs /= np.linalg.norm(psi_hs)
    return float(abs(np.vdot(psi_ed, psi_hs)) ** 2)


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
            assert sign_src == 'operator' or dec == 'mwpm', (
                f"{token}: decoder suffixes only apply to the 'operator' "
                f"arms (cnnqB/cnnqC) -- {sign_src} arms carry a fixed mwpm "
                f"head baked into the checkpoint"
            )
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
            mix = None
            if sign_src == 'twobranch':
                # arm T's own F_trunk analogue (head stripped: s -> +1, |a|)
                # and the trained signed mix a, both read straight from the
                # loaded checkpoint's params.
                pm = vs.parameters
                mix = [float(v) for v in (np.exp(np.asarray(pm['log_mix'])).reshape(-1)
                                          if 'log_mix' in pm else np.asarray(pm['mix']).reshape(-1))]
                F_trunk = two_branch_head_stripped_fidelity(
                    cfg, g, psi_ed, vs.parameters, N, head_sign_pm1_all(head, N))

            # SELF-CHECK: <H> from the enumerated state vs the run's tail
            e_chk = float(np.real(np.vdot(psi, H @ psi)))
            e_run = statistics.median(
                [complex(e).real for e in
                 json.load(open(path + ".json"))['energy'][-20:]])
            de = abs(e_chk - e_run)
            ok = de < max(0.02 * abs(evals[0]), 0.05)
            records.append({
                'hx': hx, 'hz': hz, 'hy': args.hy, 'arm': token,
                'F': F, 'F_trunk': F_trunk, 'mix': mix,
                'E0': float(evals[0]),
                'E_check': e_chk, 'E_run_tail': e_run, 'dE': de,
                'energy_check_ok': bool(ok),
                't_eval_s': time.time() - t1,
            })
            print(f"  {token:22s} F={F:.6f}"
                  + (f" F_trunk={F_trunk:.6f}" if F_trunk is not None else "")
                  + (f" mix={mix}" if mix is not None else "")
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
