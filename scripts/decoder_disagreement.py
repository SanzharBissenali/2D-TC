"""Sampled decoder-disagreement proxy for the QEC sign-head ceilings (Phase 4c+).

WHY. The exact head ceiling 1 - F_s(dec) = sum_sigma |psi(sigma)|^2 [sign_dec(sigma)
!= sign_true(sigma)] needs the ED ground state, so it stops at 2x3 (2^27). Beyond
ED we cannot know sign_true, but we CAN draw sigma ~ |psi|^2 from a TRAINED run and
evaluate all five deterministic heads on the same configurations. With mwpm as the
reference (its own ceiling is the tiny tie channel, ~hx^10),

    D(dec, mwpm) = fraction of samples with s01_dec(sigma) != s01_mwpm(sigma)

approximates 1 - F_s(dec) up to the tie floor, at ANY size the network trains at.
The full 5x5 disagreement matrix, per-decoder in-vivo timings (a second,
independent measurement of the scaling study's numbers on the real sampled
distribution), syndrome statistics and recovery-weight statistics come for free.

READING THE NUMBERS. (i) Resolution: with n samples the proxy cannot see rates
below ~3/n (reported as rule_of_three_upper; 65536 samples => 4.6e-5), so the
mwpm-class decoders' ~1e-6 ceilings read as D = 0 while anchor's ~1e-2..1e-3 is
resolved with se_chain error bars (per-chain blocked SE; the binomial SE ignores
MCMC autocorrelation). (ii) The proxy weights configs by the TRAINED |psi_NQS|^2,
not the exact one: the positive trunk cannot zero the wrong-sign configs, so it
over-weights them relative to the GS (2x2 hx=0.4 validation: NQS-weighted
greedy/unionfind rates were ~5-10x the ED-weighted 1 - F_s; anchor agreed to
~3%) -- the proxy is a slight over-estimate, the safe direction. (iii) Timings:
the cold pass includes one-off costs (numba JIT / decoder-table warm-up on the
first call); the warm pass is the steady per-config cost.

VALIDATION (--exact, N <= --exact_max_N, default 24): the same trained |psi|^2 is
enumerated over all 2^N configurations, giving the EXACT |psi_NQS|^2-weighted
disagreement matrix (the sampled proxy's target), and -- when hy == 0 -- the ED
ground state is recomputed (exact/lanczos_ed._honeycomb_direct_ed, the same
routine scripts/sign_fidelity.py grades with) so the TRUE 1 - F_s per decoder and
the trained trunk's fidelity under each head are reported next to the proxy.

HOW THE STATE IS REBUILT (mirrors main.py's honeycomb branch, keyed off the run
JSON's sim_params -- never the factory defaults; nqs_fidelity's 2026-09-01 lesson:
rescale/channels/kernel MUST come from the run's own record):
  sim_params keys read: lattice, model, BC, Lx, Ly, hx, hy, hz, architecture_type,
  n_chann_noninv, n_chann_inv, kernel_size_noninv, kernel_size_inv, rescale,
  param_dtype, sign_head, sign_impl, decoder, res_hidden, n_chains, n_sweeps,
  n_discard_per_chain=, chunk_size=, seed, n_params, n_samples.
  geometry = HoneycombGeometry(Lx, Ly, BC); model = create_honeycomb_model(cfg, g);
  sign_impl 'operator' -> bare positive trunk + SignFramedOperator(H, head) for the
  energy check; 'residual' -> ResidualSignedModel(base, head.features, ...);
  'model' -> SignedModel(base, head.s01). Sampler = create_custom_sampler (hexagon
  flips + single flips, SectorInitWeightedRule) unless --sampler local. Parameters
  are restored from the .mpack 'variables/params' subtree via
  flax.serialization.from_state_dict onto a structurally identical MCState (loud
  failure on any mismatch; n_params cross-checked against sim_params), the sampler
  chains are re-seeded from --seed and burned in for --n_discard samples/chain.
  SANITY: <H> on the drawn samples is compared with the run JSON's tail energy
  (complex-string energies '(-3.86+0j)' parsed with complex()).

OUTPUT JSON SCHEMA (schema "decoder_disagreement/v1"):
  run        {prefix, jobid, json, mpack, load_mode}
  meta       {git_hash, hostname, timestamp, args}
  geometry   {Lx, Ly, N, F, V}
  field      {hx, hy, hz, model}
  arm        {architecture, dtype, sign_head, sign_impl, decoder, channels_noninv,
              channels_inv, kernel_size, rescale, n_params_run, n_params_rebuilt}
  sampling   {n_samples, n_chains, chain_length, n_discard_per_chain, n_sweeps,
              seed, sampler, t_sample_s}
  energy     {sampled_mean, sampled_err, sampled_var, tau_corr, R_hat,
              run_tail_median, run_tail_std, run_tail_n, abs_diff, z, ok}
  syndrome   {mean_defects, frac_zero_syndrome, frac_offsector, max_defects,
              defect_histogram {"d": frac}, n_unique_syndromes}
  decoders   [names]; reference "mwpm"; run_decoder
  disagreement {names, matrix (k x k, D[a][b] = frac s01_a != s01_b),
              vs_ref {dec: {D, n_disagree, se_binomial, se_chain,
                            rule_of_three_upper, D_offsector}}}
  per_decoder {dec: {t_build_s, t_cold_s, us_per_config_cold, t_warm_s,
              us_per_config_warm, s01_mean, corr_weight_mean, corr_weight_max,
              corr_weight_mean_offsector, parity_crosscheck_mismatches,
              [tie_sum only] fallback_frac, truncated_frac, cancelled_frac}}
  exact      null | {skipped: reason} | {N, t_enumerate_s, matrix_nqs_weighted,
              vs_ref_nqs_weighted {dec: D},
              sampled_minus_exact {dec: {diff, z, expected_n_disagree}},
              table_crosscheck_mismatches {dec: n},
              ed: null | {E0, t_ed_s, F_plus, onsector_weight_ed,
                          onsector_weight_nqs, wrong_weight_true {dec},
                          wrong_weight_nqs_weighted {dec}, fidelity_trunk_x_head
                          {dec}, fidelity_trunk_unsigned}}

Runs on the cluster only (netket/jax/pymatching). ``--selftest`` exercises the
numpy aggregation/loader logic with no netket. Example (Perlmutter):
    PYTHONPATH=. python scripts/decoder_disagreement.py \
        --run results/nqs/G-equiv_1_hc2x2_ds_hx0.4_hz0_cnnqB --exact \
        --out results/diagnostics/disagree_hc2x2_ds_hx0.4_hz0_cnnqB.json
"""

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCHEMA = "decoder_disagreement/v1"
DEFAULT_DECODERS = "mwpm,anchor,greedy,unionfind,tie_sum"


# --------------------------------------------------------------------------
# pure-numpy helpers (selftested)
# --------------------------------------------------------------------------

def parse_energy(e):
    """Run-JSON energy entry -> float. Entries are strings ('-20.51', or the
    complex-trunk form '(-3.86+0j)'), plain numbers, or {'real','imag'}."""
    if isinstance(e, dict):
        return float(e["real"])
    if isinstance(e, (int, float)):
        return float(e)
    return complex(str(e).strip()).real


def tail_energy(energies, n=20):
    """(median, std, count) of the last-n parsed energies (NaNs dropped)."""
    vals = [parse_energy(e) for e in energies[-n:]]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return statistics.median(vals), sd, len(vals)


def _sp(sp, key, default):
    """sim_params values are 1-element lists (channel lists are flat lists);
    unwrap with a typed fallback. Also accepts the legacy trailing-'=' keys."""
    v = sp.get(key)
    if v is None and not key.endswith("="):
        v = sp.get(key + "=")
    if v is None:
        return default
    if isinstance(v, list):
        if isinstance(default, list):
            return list(v[0]) if (v and isinstance(v[0], list)) else list(v)
        if len(v) == 1:
            return type(default)(v[0]) if default is not None else v[0]
        return v
    return type(default)(v) if default is not None else v


def run_config_from_sim_params(sp):
    """The config dict main.py's honeycomb branch needs, rebuilt from a run's
    sim_params record (see module docstring for the key list)."""
    lattice = _sp(sp, "lattice", "square")
    assert lattice == "honeycomb", \
        f"decoder_disagreement grades honeycomb DS/TC runs only (lattice={lattice!r})"
    dtype = _sp(sp, "param_dtype", "float64")
    cfg = {
        "lattice": lattice,
        "model": _sp(sp, "model", "tc"),
        "bc": _sp(sp, "BC", "OBC"),
        "Lx": _sp(sp, "Lx", 0), "Ly": _sp(sp, "Ly", 0),
        "hx": _sp(sp, "hx", 0.0), "hy": _sp(sp, "hy", 0.0), "hz": _sp(sp, "hz", 0.0),
        "J": 1.0,
        "architecture": _sp(sp, "architecture_type", "Combo"),
        "channels_noninv": [int(c) for c in _sp(sp, "n_chann_noninv", [1, 16])],
        "channels_inv": [int(c) for c in _sp(sp, "n_chann_inv", [16, 8, 1])],
        "kernel_size": _sp(sp, "kernel_size_noninv", 2),
        "kernel_size_inv": _sp(sp, "kernel_size_inv", 1),
        "rescale": _sp(sp, "rescale", 1.0),
        "dtype": dtype,
        "complex_ansatz": dtype == "complex",
        "sign_head": _sp(sp, "sign_head", "none"),
        "sign_impl": _sp(sp, "sign_impl", "operator"),
        "decoder": _sp(sp, "decoder", "mwpm"),
        "res_hidden": _sp(sp, "res_hidden", 16),
        "n_chains": _sp(sp, "n_chains", 1024),
        "n_sweeps": _sp(sp, "n_sweeps", 0),
        "n_discard": _sp(sp, "n_discard_per_chain", 8),
        "chunk_size": _sp(sp, "chunk_size", 2048),
        "seed": _sp(sp, "seed", 0),
        "n_params": _sp(sp, "n_params", 0),
        "n_samples": _sp(sp, "n_samples", 8192),
        "symmetric_block": "cnn",
        "use_custom_sampler": True,
    }
    assert cfg["Lx"] >= 1 and cfg["Ly"] >= 1, "sim_params missing Lx/Ly"
    if cfg["n_sweeps"] <= 0:                      # config.py default: N // 2
        N = 3 * cfg["Lx"] * cfg["Ly"] + 2 * (cfg["Lx"] + cfg["Ly"]) - 1
        cfg["n_sweeps"] = N // 2
    return cfg


def syndrome_from_spins(spins, vertex_all):
    """(B, N) +-1 spins -> (B, V) uint8 Q_v syndrome bits (1 = Q_v = -1).
    Independent of QECSignHead's implementation (used for the statistics and
    as the selftest oracle). -1 slots in vertex_all are padding."""
    spins = np.asarray(spins)
    bits = (spins.reshape(-1, spins.shape[-1]) < 0).astype(np.uint8)
    va = np.asarray(vertex_all)
    synd = np.zeros((bits.shape[0], va.shape[0]), dtype=np.uint8)
    for v in range(va.shape[0]):
        for l in va[v]:
            if l != -1:
                synd[:, v] ^= bits[:, int(l)]
    return synd


def spins_to_ints(spins):
    """(B, N) +-1 spins -> int64 config index (site i <-> bit i, bit 1 = down),
    the ED / sign_fidelity bit convention. N <= 62."""
    spins = np.asarray(spins)
    N = spins.shape[-1]
    assert N <= 62, "int64 packing needs N <= 62"
    bits = (spins.reshape(-1, N) < 0).astype(np.int64)
    return bits @ (1 << np.arange(N, dtype=np.int64))


def ints_to_spins(ints, N):
    ints = np.asarray(ints, dtype=np.int64)
    return (1 - 2 * ((ints[:, None] >> np.arange(N, dtype=np.int64)) & 1)).astype(np.int8)


def disagreement_matrix(s01_by_dec, names, weights=None):
    """D[a][b] = (weighted) fraction of configs where s01_a != s01_b."""
    k = len(names)
    D = np.zeros((k, k), dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    for i in range(k):
        si = np.asarray(s01_by_dec[names[i]])
        for j in range(i + 1, k):
            neq = si != np.asarray(s01_by_dec[names[j]])
            D[i, j] = D[j, i] = float(neq.mean()) if weights is None \
                else float(w[neq].sum())
    return D


def binomial_se(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else float("nan")


def chain_blocked_se(mask_chains):
    """Standard error of a fraction from MCMC samples shaped (n_chains, L):
    per-chain means are (nearly) independent, so SE = std(chain means)/sqrt(C).
    None when there is a single chain."""
    m = np.asarray(mask_chains, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 2:
        return None
    means = m.mean(axis=1)
    return float(np.std(means, ddof=1) / math.sqrt(m.shape[0]))


def defect_histogram(ndef):
    ndef = np.asarray(ndef)
    vals, cnt = np.unique(ndef, return_counts=True)
    return {str(int(v)): float(c) / ndef.size for v, c in zip(vals, cnt)}


def _py(o):
    """numpy -> JSON-native."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# state rebuild (netket / jax -- cluster only)
# --------------------------------------------------------------------------

def make_parity_fn(geometry):
    """(K, N) uint8 zero-charge bits -> {0.,1.} loop parities. Prefers the
    compiled model.decoders.loop_parity01 (flags off-sector rows with -1);
    falls back to exact.loops.count_loops row by row."""
    try:
        from model.decoders import loop_parity01

        def fn(bits):
            p = loop_parity01(np.asarray(bits, dtype=np.uint8),
                              geometry.link_endpoints, geometry.n_vertices)
            assert (p >= 0).all(), "recovery left the zero-charge sector?!"
            return p.astype(np.float64)
        return fn
    except ImportError:
        from exact.loops import count_loops

        def fn(bits):
            b = np.asarray(bits, dtype=np.uint8).reshape(-1, geometry.N)
            z = 1 - 2 * b.astype(np.int8)
            return np.array([count_loops(z[i], geometry.link_endpoints) % 2
                             for i in range(b.shape[0])], dtype=np.float64)
        return fn


def build_state(cfg, geometry, sampler_kind, n_chains, seed):
    """MCState exactly as main.py builds it for this run's arm (model wrapping
    per sign_impl, custom hexagon sampler), plus the head/H needed for the
    energy sanity check. Returns (vs, model, H_for_expect, head_run)."""
    import netket as nk
    import jax.numpy as jnp
    from model.hamiltonian import create_honeycomb_hamiltonian
    from model.honeycomb_networks import create_honeycomb_model

    hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)
    H = create_honeycomb_hamiltonian(hi, geometry, cfg["model"], J=cfg["J"],
                                     hx=cfg["hx"], hz=cfg["hz"], hy=cfg["hy"])
    model = create_honeycomb_model(cfg, geometry)
    head_run = None
    if cfg["sign_head"] == "qec":
        from model.sign_head import QECSignHead
        head_run = QECSignHead(geometry, decoder=cfg["decoder"])
        if cfg["sign_impl"] == "operator":
            from model.sign_frame import SignFramedOperator
            H = SignFramedOperator(H, head_run)
        elif cfg["sign_impl"] == "residual":
            from model.honeycomb_networks import ResidualSignedModel
            model = ResidualSignedModel(model, head_run.features,
                                        head_run.n_features, hidden=cfg["res_hidden"])
        elif cfg["sign_impl"] == "model":
            from model.honeycomb_networks import SignedModel
            model = SignedModel(model, head_run.s01)
        else:
            raise ValueError(f"unknown sign_impl {cfg['sign_impl']!r}")

    scfg = dict(cfg, n_chains=int(n_chains))
    if sampler_kind == "custom":
        from simulation.custom_sampler import create_custom_sampler
        sa = create_custom_sampler(geometry, hi, scfg)
    else:
        sa = nk.sampler.MetropolisSampler(hi, rule=nk.sampler.rules.LocalRule(),
                                          n_chains=int(n_chains),
                                          n_sweeps=cfg["n_sweeps"], dtype=jnp.int8)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg["n_samples"],
                        n_discard_per_chain=cfg["n_discard"],
                        chunk_size=cfg["chunk_size"], seed=int(seed))
    return vs, model, H, head_run


def load_params(vs, mpack_path):
    """Restore the trained parameters from a whole-MCState .mpack
    (utils/io.save_model = flax to_bytes(vstate)). Only the 'params' subtree
    is used, so the sampler state / chain count of the diagnostic need not
    match the run's; from_state_dict fails loudly on any structural mismatch."""
    import flax

    with open(mpack_path, "rb") as f:
        tree = flax.serialization.msgpack_restore(f.read())
    params = None
    if isinstance(tree, dict) and isinstance(tree.get("variables"), dict):
        params = tree["variables"].get("params")
    if params is None:                       # fall back to a tree walk
        stack = [tree]
        while stack and params is None:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("params"), dict):
                    params = node["params"]
                else:
                    stack.extend(node.values())
    assert params is not None, f"no 'params' subtree in {mpack_path}"
    vs.parameters = flax.serialization.from_state_dict(vs.parameters, params)
    return "variables/params via from_state_dict"


def draw_samples(vs, n_samples, n_discard):
    """sigma ~ |psi|^2: (n_chains, chain_length, N) int8 array of +-1."""
    n_chains = int(vs.sampler.n_chains)
    chain_length = max(1, math.ceil(n_samples / n_chains))
    n_eff = chain_length * n_chains
    if vs.chunk_size is not None and n_eff % vs.chunk_size:
        g = math.gcd(n_eff, int(vs.chunk_size))
        vs.chunk_size = g if g > 1 else None
    vs.n_discard_per_chain = int(n_discard)
    vs.n_samples = n_eff
    vs.reset()
    t0 = time.perf_counter()
    samples = np.asarray(vs.sample())
    t = time.perf_counter() - t0
    assert samples.ndim == 3 and samples.shape[-1] == vs.hilbert.size, \
        f"unexpected sample shape {samples.shape}"
    return samples.astype(np.int8), t, chain_length


def enumerate_logs(model, variables, N, batch=1 << 17):
    """log psi over all 2^N configs (ED bit order), complex128 on host."""
    import jax
    import jax.numpy as jnp

    dim = 1 << N
    site = np.arange(N, dtype=np.int64)
    out = np.empty(dim, dtype=np.complex128)
    fwd = jax.jit(lambda x: model.apply(variables, x))
    for lo in range(0, dim, batch):
        idx = np.arange(lo, min(lo + batch, dim), dtype=np.int64)
        spins = (1 - 2 * ((idx[:, None] >> site) & 1)).astype(np.float64)
        out[lo:lo + len(idx)] = np.asarray(fwd(jnp.asarray(spins)), dtype=np.complex128)
    return out


# --------------------------------------------------------------------------
# head evaluation on the sampled configurations
# --------------------------------------------------------------------------

def evaluate_heads(geometry, samples2d, decoders, chunk):
    """Timed QECSignHead.s01 per decoder (cold pass, then a warm repeat that
    exposes memoization), returning (s01_by_dec, timing_by_dec)."""
    from model.sign_head import QECSignHead

    n = samples2d.shape[0]
    s01, timing = {}, {}
    for d in decoders:
        t0 = time.perf_counter()
        head = QECSignHead(geometry, decoder=d)
        t_build = time.perf_counter() - t0
        out = np.empty(n, dtype=np.float64)
        t_cold = 0.0
        for lo in range(0, n, chunk):
            t0 = time.perf_counter()
            out[lo:lo + chunk] = head.s01(samples2d[lo:lo + chunk])
            t_cold += time.perf_counter() - t0
        t_warm = 0.0
        for lo in range(0, n, chunk):
            t0 = time.perf_counter()
            again = head.s01(samples2d[lo:lo + chunk])
            t_warm += time.perf_counter() - t0
            assert np.array_equal(again, out[lo:lo + chunk]), \
                f"{d}: head is not a deterministic function of sigma?!"
        assert np.isin(out, (0.0, 1.0)).all(), f"{d}: s01 not in {{0,1}}"
        s01[d] = out
        timing[d] = {"t_build_s": t_build, "t_cold_s": t_cold,
                     "us_per_config_cold": 1e6 * t_cold / n,
                     "t_warm_s": t_warm, "us_per_config_warm": 1e6 * t_warm / n}
        print(f"  head {d:10s} build {t_build:6.2f}s | cold {t_cold:7.2f}s "
              f"({1e6 * t_cold / n:8.2f} us/cfg) | warm {t_warm:7.2f}s "
              f"({1e6 * t_warm / n:8.2f} us/cfg) | <s>={out.mean():.4f}", flush=True)
    return s01, timing


def correction_stats(geometry, samples2d, synd, decoders, s01_by_dec, chunk):
    """Recovery Hamming weights per decoder (model.decoders public API) with a
    parity cross-check against the head's s01; tie_sum reports its
    fallback/truncated/cancelled fractions (its class weight == the minimal
    == mwpm weight, so mwpm's weights are reported for it)."""
    from model.decoders import make_decoder

    parity = make_parity_fn(geometry)
    bits = (samples2d < 0).astype(np.uint8)
    n = bits.shape[0]
    off = synd.any(axis=1)
    out = {}
    mwpm_w = None
    corr_names = [d for d in decoders if d != "tie_sum"]
    if "tie_sum" in decoders and "mwpm" not in corr_names:
        corr_names.append("mwpm")
    for d in corr_names:
        dec = make_decoder(d, geometry)
        w = np.empty(n, dtype=np.int64)
        mism = 0
        for lo in range(0, n, chunk):
            corr = np.asarray(dec.corrections(synd[lo:lo + chunk]), dtype=np.uint8)
            assert corr.shape == bits[lo:lo + chunk].shape
            w[lo:lo + chunk] = corr.sum(axis=1)
            if d in s01_by_dec:
                p = parity(bits[lo:lo + chunk] ^ corr)
                mism += int((p != s01_by_dec[d][lo:lo + chunk]).sum())
        if d == "mwpm":
            mwpm_w = w
        if d in s01_by_dec:
            assert mism == 0, (f"{d}: model.decoders recovery parity disagrees with "
                               f"QECSignHead.s01 on {mism} configs")
            out[d] = {"corr_weight_mean": float(w.mean()), "corr_weight_max": int(w.max()),
                      "corr_weight_mean_offsector": float(w[off].mean()) if off.any() else 0.0,
                      "parity_crosscheck_mismatches": mism}
    if "tie_sum" in decoders:
        ts = make_decoder("tie_sum", geometry)
        s = np.empty(n, dtype=np.float64)
        fb = np.zeros(n, dtype=bool)
        tr = np.zeros(n, dtype=bool)
        cx = np.zeros(n, dtype=bool)
        for lo in range(0, n, chunk):
            r = ts.sign01(bits[lo:lo + chunk], synd[lo:lo + chunk], parity)
            s[lo:lo + chunk], fb[lo:lo + chunk], tr[lo:lo + chunk], cx[lo:lo + chunk] = r
        mism = int((s != s01_by_dec["tie_sum"]).sum())
        if mism:
            print(f"  WARNING tie_sum: standalone TieSumDecoder disagrees with the head "
                  f"on {mism} configs (cap/cache differences?)", flush=True)
        out["tie_sum"] = {
            "corr_weight_mean": float(mwpm_w.mean()), "corr_weight_max": int(mwpm_w.max()),
            "corr_weight_mean_offsector": float(mwpm_w[off].mean()) if off.any() else 0.0,
            "parity_crosscheck_mismatches": mism,
            "fallback_frac": float(fb.mean()), "truncated_frac": float(tr.mean()),
            "cancelled_frac": float(cx.mean())}
    return out


# --------------------------------------------------------------------------
# exact validation route (N <= exact_max_N)
# --------------------------------------------------------------------------

def exact_route(geometry, cfg, model, variables, decoders, ref, samples2d,
                s01_sampled, D_sampled_vs_ref, se_vs_ref, ed_tol, tiesum_dmax):
    from scripts.sign_fidelity import (loop_sign_table, build_decoder_tables,
                                       syndrome_bits, decoder_signs)

    g, N = geometry, geometry.N
    dim = 1 << N
    t0 = time.perf_counter()
    logs = enumerate_logs(model, variables, N)
    t_enum = time.perf_counter() - t0
    m = float(np.max(logs.real))
    psi_t = np.exp(logs - m)
    psi_t /= np.linalg.norm(psi_t)
    w_nqs = np.abs(psi_t) ** 2

    tables = build_decoder_tables(g, list(decoders), tiesum_dmax)
    parr = np.zeros(dim, dtype=np.int8)
    for x, s in loop_sign_table(g).items():
        parr[x] = s
    tables["parity_arr"] = parr
    vbits = 1 << np.arange(g.n_vertices, dtype=np.int64)

    sgn = {d: np.empty(dim, dtype=np.int8) for d in decoders}
    for lo in range(0, dim, 1 << 22):
        c = np.arange(lo, min(lo + (1 << 22), dim), dtype=np.int64)
        sidx = tables["index"][syndrome_bits(g, c).astype(np.int64) @ vbits]
        assert (sidx >= 0).all(), "odd syndrome?!"
        for d in decoders:
            sgn[d][lo:lo + len(c)], _ = decoder_signs(g, c, sidx, parr, tables, d)
    s01_all = {d: (sgn[d] < 0).astype(np.float64) for d in decoders}
    D_exact = disagreement_matrix(s01_all, decoders, weights=w_nqs)
    iref = decoders.index(ref)

    # the sampled configs re-signed through the table route == head route?
    ints = spins_to_ints(samples2d)
    cross = {d: int((s01_all[d][ints] != s01_sampled[d]).sum()) for d in decoders}
    for d, nm in cross.items():
        if nm:
            print(f"  WARNING {d}: table-route signs differ from QECSignHead on "
                  f"{nm} sampled configs", flush=True)
    for d in decoders:
        if d != "tie_sum":
            assert cross[d] == 0, f"{d}: sign_fidelity table route != head route"

    out = {"N": N, "t_enumerate_s": t_enum, "names": list(decoders),
           "matrix_nqs_weighted": D_exact,
           "vs_ref_nqs_weighted": {d: float(D_exact[i, iref])
                                   for i, d in enumerate(decoders)},
           "sampled_minus_exact": {}, "table_crosscheck_mismatches": cross,
           "onsector_weight_nqs": None, "ed": None}
    n_samp = samples2d.shape[0]
    for i, d in enumerate(decoders):
        diff = D_sampled_vs_ref[d] - float(D_exact[i, iref])
        se = se_vs_ref[d]
        out["sampled_minus_exact"][d] = {
            "diff": diff, "z": (diff / se) if (se and se > 0) else None,
            # disagreements the sample size could have resolved at the exact
            # NQS-weighted rate (< ~3 => the proxy is resolution-limited here)
            "expected_n_disagree": float(D_exact[i, iref]) * n_samp}

    synd_int = np.zeros(dim, dtype=np.int64)
    for lo in range(0, dim, 1 << 22):
        c = np.arange(lo, min(lo + (1 << 22), dim), dtype=np.int64)
        synd_int[lo:lo + len(c)] = syndrome_bits(g, c).astype(np.int64) @ vbits
    onsec = synd_int == 0
    out["onsector_weight_nqs"] = float(w_nqs[onsec].sum())

    if cfg["hy"] != 0.0:
        out["ed"] = {"skipped": "hy != 0: complex GS, +-1 sign agreement undefined"}
        return out
    from exact.lanczos_ed import _honeycomb_direct_ed
    t0 = time.perf_counter()
    evals, psi = _honeycomb_direct_ed(g, cfg["model"], cfg["J"], cfg["hx"], cfg["hz"],
                                      k=1, tol=ed_tol)
    t_ed = time.perf_counter() - t0
    psi = np.real(np.asarray(psi))
    assert abs(psi[0]) > 1e-10 * np.max(np.abs(psi)), "all-up anchor ~0: gauge undefined"
    psi *= np.sign(psi[0])
    psi /= np.linalg.norm(psi)
    w_ed = psi ** 2
    sgn_ed = np.where(psi >= 0, 1, -1).astype(np.int8)
    ed = {"E0": float(evals[0]), "t_ed_s": t_ed,
          "F_plus": float(w_ed[sgn_ed > 0].sum()),
          "onsector_weight_ed": float(w_ed[onsec].sum()),
          "wrong_weight_true": {}, "wrong_weight_nqs_weighted": {},
          "fidelity_trunk_x_head": {},
          "fidelity_trunk_unsigned": float(abs(np.vdot(psi, psi_t)) ** 2)}
    for d in decoders:
        wrong = sgn[d] != sgn_ed
        ed["wrong_weight_true"][d] = float(w_ed[wrong].sum())
        ed["wrong_weight_nqs_weighted"][d] = float(w_nqs[wrong].sum())
        ed["fidelity_trunk_x_head"][d] = float(abs(np.vdot(psi, sgn[d] * psi_t)) ** 2)
    out["ed"] = ed
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(args):
    from model.honeycomb_geometry import HoneycombGeometry

    prefix = args.run[:-5] if args.run.endswith(".json") else args.run
    prefix = prefix[:-6] if prefix.endswith(".mpack") else prefix
    jpath, mpath = prefix + ".json", prefix + ".mpack"
    assert os.path.exists(jpath), f"missing {jpath}"
    assert os.path.exists(mpath), f"missing {mpath} (run not complete?)"
    with open(jpath) as f:
        data = json.load(f)
    cfg = run_config_from_sim_params(data.get("sim_params", {}))
    jobid = os.path.basename(prefix)
    jobid = jobid[len("G-equiv_1_"):] if jobid.startswith("G-equiv_1_") else jobid

    decoders = list(dict.fromkeys(d.strip() for d in args.decoders.split(",") if d.strip()))
    ref = args.ref
    assert ref in decoders, f"--ref {ref} must be in --decoders {decoders}"

    g = HoneycombGeometry(cfg["Lx"], cfg["Ly"], cfg["bc"])
    n_chains = args.n_chains or cfg["n_chains"]
    print(f"# run {jobid}: {cfg['Lx']}x{cfg['Ly']} {cfg['model']} N={g.N} F={g.n_plaqs} "
          f"V={g.n_vertices} h=({cfg['hx']},{cfg['hy']},{cfg['hz']}) arch={cfg['architecture']}"
          f" dtype={cfg['dtype']} head={cfg['sign_head']}/{cfg['sign_impl']}/{cfg['decoder']}"
          f" rescale={cfg['rescale']} ch={cfg['channels_noninv']}+{cfg['channels_inv']}",
          flush=True)

    t0 = time.perf_counter()
    vs, model, H, head_run = build_state(cfg, g, args.sampler, n_chains, args.seed)
    load_mode = load_params(vs, mpath)
    n_par = int(vs.n_parameters)
    if cfg["n_params"] and n_par != cfg["n_params"]:
        print(f"  WARNING: rebuilt n_params {n_par} != run's {cfg['n_params']}", flush=True)
    print(f"# state rebuilt + loaded ({load_mode}) in {time.perf_counter() - t0:.1f}s, "
          f"n_params={n_par}", flush=True)

    samples, t_sample, chain_length = draw_samples(vs, args.n_samples, args.n_discard)
    C, L, N = samples.shape
    samples2d = samples.reshape(-1, N)
    n = samples2d.shape[0]
    print(f"# sampled {n} configs ({C} chains x {L}) in {t_sample:.1f}s", flush=True)

    st = vs.expect(H)
    e_mean = complex(np.asarray(st.mean)).real
    e_err = float(np.real(st.error_of_mean))
    run_med, run_sd, run_n = tail_energy(data.get("energy", []), args.tail)
    de = abs(e_mean - run_med)
    tol = max(5.0 * math.hypot(e_err, run_sd if math.isfinite(run_sd) else 0.0),
              0.02 * abs(run_med) if math.isfinite(run_med) else 0.0)
    z = de / e_err if e_err > 0 else None
    ok = bool(de < tol) if math.isfinite(run_med) else None
    print(f"# energy: sampled {e_mean:.6f} +- {e_err:.1e} | run tail median {run_med:.6f} "
          f"(std {run_sd:.1e}, n={run_n}) | dE={de:.2e} "
          + ("OK" if ok else ("ENERGY-CHECK-FAIL" if ok is False else "n/a")), flush=True)

    synd = syndrome_from_spins(samples2d, g.vertex_all)
    ndef = synd.sum(axis=1).astype(np.int64)
    assert not (ndef % 2).any(), "odd syndrome weight (violates prod Q_v = 1)"
    off = ndef > 0
    frac_off = float(off.mean())
    syn_stats = {"mean_defects": float(ndef.mean()),
                 "frac_zero_syndrome": 1.0 - frac_off, "frac_offsector": frac_off,
                 "max_defects": int(ndef.max()), "defect_histogram": defect_histogram(ndef),
                 "n_unique_syndromes": int(np.unique(synd, axis=0).shape[0])}
    print(f"# syndrome: <d>={ndef.mean():.3f} zero-frac={1 - frac_off:.4f} "
          f"unique={syn_stats['n_unique_syndromes']}", flush=True)

    s01, timing = evaluate_heads(g, samples2d, decoders, args.chunk)
    D = disagreement_matrix(s01, decoders)
    iref = decoders.index(ref)
    vs_ref, se_ref = {}, {}
    for i, d in enumerate(decoders):
        neq = s01[d] != s01[ref]
        Dd = float(D[i, iref])
        se_c = chain_blocked_se(neq.reshape(C, L))
        se_b = binomial_se(Dd, n)
        se_ref[d] = se_c if se_c else se_b
        vs_ref[d] = {"D": Dd, "n_disagree": int(neq.sum()), "se_binomial": se_b,
                     "se_chain": se_c, "rule_of_three_upper": 3.0 / n,
                     "D_offsector": float(neq[off].mean()) if off.any() else 0.0}
        print(f"  D({d:10s}, {ref}) = {Dd:.3e}  (n={int(neq.sum())}, se_bin {se_b:.1e}, "
              f"se_chain {se_c if se_c is not None else float('nan'):.1e})", flush=True)

    cstats = correction_stats(g, samples2d, synd, decoders, s01, args.chunk)
    per_dec = {d: {**timing[d], "s01_mean": float(s01[d].mean()), **cstats.get(d, {})}
               for d in decoders}

    exact = None
    if args.exact:
        if N > args.exact_max_N:
            exact = {"skipped": f"N={N} > --exact_max_N {args.exact_max_N}"}
            print(f"# exact route skipped: {exact['skipped']}", flush=True)
        else:
            print("# exact route: enumerating 2^N ...", flush=True)
            exact = exact_route(g, cfg, model, vs.variables, decoders, ref, samples2d,
                                s01, {d: vs_ref[d]["D"] for d in decoders}, se_ref,
                                args.ed_tol, args.tiesum_dmax)
            for d in decoders:
                line = (f"  {d:10s} D_exact(nqs-weighted)={exact['vs_ref_nqs_weighted'][d]:.3e} "
                        f"sampled-exact={exact['sampled_minus_exact'][d]['diff']:+.2e}")
                if exact["ed"] and "wrong_weight_true" in exact["ed"]:
                    line += (f" | true 1-F_s={exact['ed']['wrong_weight_true'][d]:.3e}"
                             f" F(trunk x head)={exact['ed']['fidelity_trunk_x_head'][d]:.6f}")
                print(line, flush=True)

    result = {
        "schema": SCHEMA,
        "run": {"prefix": prefix, "jobid": jobid, "json": jpath, "mpack": mpath,
                "load_mode": load_mode},
        "meta": {"git_hash": git_hash(), "hostname": socket.gethostname(),
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)},
        "geometry": {"Lx": cfg["Lx"], "Ly": cfg["Ly"], "N": N, "F": g.n_plaqs,
                     "V": g.n_vertices},
        "field": {"hx": cfg["hx"], "hy": cfg["hy"], "hz": cfg["hz"], "model": cfg["model"]},
        "arm": {"architecture": cfg["architecture"], "dtype": cfg["dtype"],
                "sign_head": cfg["sign_head"], "sign_impl": cfg["sign_impl"],
                "decoder": cfg["decoder"], "channels_noninv": cfg["channels_noninv"],
                "channels_inv": cfg["channels_inv"], "kernel_size": cfg["kernel_size"],
                "rescale": cfg["rescale"], "n_params_run": cfg["n_params"],
                "n_params_rebuilt": n_par},
        "sampling": {"n_samples": n, "n_chains": C, "chain_length": L,
                     "n_discard_per_chain": args.n_discard, "n_sweeps": cfg["n_sweeps"],
                     "seed": args.seed, "sampler": args.sampler, "t_sample_s": t_sample},
        "energy": {"sampled_mean": e_mean, "sampled_err": e_err,
                   "sampled_var": float(np.real(st.variance)),
                   "tau_corr": float(np.real(st.tau_corr)), "R_hat": float(np.real(st.R_hat)),
                   "run_tail_median": run_med, "run_tail_std": run_sd, "run_tail_n": run_n,
                   "abs_diff": de, "z": z, "ok": ok},
        "syndrome": syn_stats,
        "decoders": decoders, "reference": ref, "run_decoder": cfg["decoder"],
        "disagreement": {"names": decoders, "matrix": D, "vs_ref": vs_ref},
        "per_decoder": per_dec,
        "exact": exact,
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_py(result), f, indent=1)
    os.replace(tmp, args.out)
    print(f"# wrote {args.out}", flush=True)


# --------------------------------------------------------------------------
# selftest (numpy only)
# --------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(0)
    # energy parsing incl. complex strings and dicts
    assert parse_energy("-20.5") == -20.5
    assert parse_energy("(-3.86+0j)") == -3.86
    assert parse_energy({"real": -1.5, "imag": 0.2}) == -1.5
    assert parse_energy(-2) == -2.0
    med, sd, cnt = tail_energy(["(-1+0j)", "-3", "-2", "nan"], n=4)
    assert (med, cnt) == (-2.0, 3) and abs(sd - math.sqrt(2 / 3)) < 1e-12

    # sim_params unwrapping
    sp = {"Lx": [2], "rescale": [1.0], "n_chann_inv": [16, 8, 1], "chunk_size=": [2048],
          "hx": [0.4], "decoder": ["greedy"], "n_discard_per_chain=": [8]}
    assert _sp(sp, "Lx", 0) == 2 and isinstance(_sp(sp, "Lx", 0), int)
    assert _sp(sp, "n_chann_inv", [1]) == [16, 8, 1]
    assert _sp(sp, "chunk_size", 0) == 2048 and _sp(sp, "n_discard_per_chain", 0) == 8
    assert _sp(sp, "decoder", "mwpm") == "greedy" and _sp(sp, "missing", 7) == 7

    # loader on a real run record if present
    real = os.path.join(REPO, "results/nqs/G-equiv_1_hc2x2_ds_hx0.4_hz0_cnnqB.json")
    if os.path.exists(real):
        with open(real) as f:
            cfg = run_config_from_sim_params(json.load(f)["sim_params"])
        assert (cfg["Lx"], cfg["Ly"], cfg["model"]) == (2, 2, "ds")
        assert cfg["hx"] == 0.4 and cfg["hz"] == 0.0 and cfg["hy"] == 0.0
        assert cfg["sign_head"] == "qec" and cfg["sign_impl"] == "operator"
        assert cfg["decoder"] == "mwpm" and cfg["rescale"] == 1.0
        assert cfg["channels_noninv"] == [1, 16] and cfg["channels_inv"] == [16, 8, 1]
        assert cfg["kernel_size"] == 2 and cfg["dtype"] == "float64"
        assert cfg["n_chains"] == 1024 and cfg["chunk_size"] == 2048
        assert cfg["n_discard"] == 8 and cfg["n_sweeps"] == 9 and cfg["n_params"] == 12489
        with open(real) as f:
            med, sd, cnt = tail_energy(json.load(f)["energy"], 20)
        assert cnt == 20 and -21 < med < -20, (med, cnt)
        print("  loader on real 2x2 run record: OK")
    else:
        print("  loader on real run record: SKIPPED (file absent)")
    try:
        run_config_from_sim_params({"lattice": ["square"], "Lx": [4]})
        raise RuntimeError("square lattice must be rejected")
    except AssertionError:
        pass

    # disagreement matrix vs brute force, symmetry, zero diagonal, weights
    names = ["a", "b", "c"]
    s = {k: rng.integers(0, 2, 1000).astype(np.float64) for k in names}
    D = disagreement_matrix(s, names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            assert abs(D[i, j] - np.mean(s[a] != s[b])) < 1e-15
            assert D[i, j] == D[j, i]
        assert D[i, i] == 0.0
    Dw = disagreement_matrix(s, names, weights=np.ones(1000))
    assert np.allclose(D, Dw)
    w = np.zeros(1000)
    w[:10] = 1.0
    Dw = disagreement_matrix(s, names, weights=w)
    assert abs(Dw[0, 1] - np.mean(s["a"][:10] != s["b"][:10])) < 1e-15

    # error bars
    assert abs(binomial_se(0.5, 100) - 0.05) < 1e-15 and binomial_se(0.0, 10) == 0.0
    m = rng.integers(0, 2, (8, 50)).astype(float)
    se = chain_blocked_se(m)
    assert abs(se - np.std(m.mean(1), ddof=1) / math.sqrt(8)) < 1e-15
    assert chain_blocked_se(m[:1]) is None
    h = defect_histogram(np.array([0, 0, 2, 4, 2]))
    assert h == {"0": 0.4, "2": 0.4, "4": 0.2}

    # syndromes + bit conventions on a real honeycomb patch (numpy-only module)
    from model.honeycomb_geometry import HoneycombGeometry
    g = HoneycombGeometry(1, 2)
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(500, g.N))
    synd = syndrome_from_spins(spins, g.vertex_all)
    for v in range(g.n_vertices):
        links = [int(l) for l in g.vertex_all[v] if l != -1]
        q = np.prod(spins[:, links], axis=1)
        assert np.array_equal(synd[:, v], (q < 0).astype(np.uint8))
    assert not (synd.sum(axis=1) % 2).any(), "prod Q_v = 1 must force even defect counts"
    ints = spins_to_ints(spins)
    assert np.array_equal(ints_to_spins(ints, g.N), spins)
    assert spins_to_ints(np.ones((1, g.N)))[0] == 0            # all-up = index 0
    assert spins_to_ints(-np.ones((1, g.N)))[0] == (1 << g.N) - 1

    # parity fallback vs exact.loops on zero-charge configs built by hexagon flips
    from exact.loops import count_loops
    parity = make_parity_fn(g)
    masks = [np.bitwise_or.reduce(1 << g.plaq_all[p].astype(np.int64)) for p in range(g.n_plaqs)]
    xs = []
    for S in range(1 << g.n_plaqs):
        x = 0
        for p in range(g.n_plaqs):
            if (S >> p) & 1:
                x ^= int(masks[p])
        xs.append(x)
    zs = ints_to_spins(np.array(xs), g.N)
    p = parity((zs < 0).astype(np.uint8))
    ref = np.array([count_loops(z, g.link_endpoints) % 2 for z in zs], dtype=float)
    assert np.array_equal(p, ref)
    assert (syndrome_from_spins(zs, g.vertex_all) == 0).all()

    # correction plumbing with the pure-numpy ladder decoders (no pymatching):
    # boundary == syndrome, and recovered configs are on-sector
    try:
        from model.decoders import make_decoder
        inc = np.zeros((g.n_vertices, g.N), dtype=np.int64)
        for l, (u, v) in enumerate(g.link_endpoints):
            inc[u, l] = inc[v, l] = 1
        bits = (spins < 0).astype(np.uint8)
        for name in ("anchor", "greedy", "unionfind"):
            corr = np.asarray(make_decoder(name, g).corrections(synd), dtype=np.uint8)
            assert corr.shape == bits.shape
            assert np.array_equal((corr.astype(np.int64) @ inc.T) % 2, synd), name
            rec = bits ^ corr
            assert (syndrome_from_spins(1 - 2 * rec.astype(np.int8), g.vertex_all) == 0).all()
            parity(rec)                      # must not raise (on-sector)
        print("  ladder decoders anchor/greedy/unionfind plumbing: OK")
    except ImportError as e:                 # numba-less fallback is built in; only
        print(f"  ladder decoder plumbing: SKIPPED ({e})")   # a missing module skips
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", help="run prefix, e.g. results/nqs/G-equiv_1_<jobid>")
    ap.add_argument("--n_samples", type=int, default=65536)
    ap.add_argument("--n_discard", type=int, default=64,
                    help="burn-in samples per chain (chains restart in the loop sector)")
    ap.add_argument("--n_chains", type=int, default=0,
                    help="0 => the run's recorded n_chains (1024 on GPU)")
    ap.add_argument("--sampler", choices=["custom", "local"], default="custom")
    ap.add_argument("--decoders", default=DEFAULT_DECODERS)
    ap.add_argument("--ref", default="mwpm", help="reference decoder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8192, help="head evaluation batch")
    ap.add_argument("--tail", type=int, default=20, help="run-JSON tail length for <E>")
    ap.add_argument("--exact", action="store_true",
                    help="also enumerate 2^N (N <= --exact_max_N) + ED reference")
    ap.add_argument("--exact_max_N", type=int, default=24)
    ap.add_argument("--ed_tol", type=float, default=0.0)
    ap.add_argument("--tiesum_dmax", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true", help="numpy-only unit tests")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.run, "--run is required (or --selftest)"
    if not args.out:
        jobid = os.path.basename(args.run)
        jobid = jobid[len("G-equiv_1_"):] if jobid.startswith("G-equiv_1_") else jobid
        args.out = os.path.join(REPO, "results/diagnostics", f"disagree_{jobid}.json")
    run(args)


if __name__ == "__main__":
    main()
