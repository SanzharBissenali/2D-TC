#!/usr/bin/env python
"""Measure A_v (vertex/star) symmetry breaking of trained NQS as a function of hz.

At hz=0 the toric-code ground state obeys the exact vertex symmetry A_v|psi>=|psi>,
i.e. in the sigma^z basis  psi(A_v . sigma) == psi(sigma), where A_v flips the spins on
the qubits touching vertex v. At hz != 0 the field does not commute with A_v, so the
symmetry is *physically* broken; the paper's non-symmetric Block 1 (identity-init CNN)
is what reproduces that breaking.

For each trained run we form the per-(config, vertex) log-amplitude difference
    delta_v(sigma) = log psi(A_v sigma) - log psi(sigma)          (real on the hz cut)
    r_v(sigma)     = psi(A_v sigma) / psi(sigma) = exp(delta_v)
which is normalization-free (a difference of log-amplitudes on the same network).

We average delta over sigma drawn two ways:
  * BORN  sigma ~ |psi|^2 from the model's own MCMC sampler (physical primary). For real
    psi, E_born[r_v] = <A_v> (the vertex-stabilizer expectation): 1 at hz=0, dropping with hz.
  * UNIFORM random +-1 configs (tail stress-test; probes psi as a bare function).

Reuses model/geometry.py + model/networks.py + utils/io.load_model, and mirrors the
flip+evaluate primitive in simulation/observables.check_Av_invariance.

Writes one compact JSON per run to results/symmetry/ (summary stats + delta histograms;
no full raw arrays). netket/jax are NERSC-only, so this runs on the cluster.

Usage:
    python scripts/symmetry_check.py --Lx 4 --n_samples 8192 --out-dir results/symmetry
"""

import argparse
import glob
import json
import os
import re

import numpy as np


# --------------------------------------------------------------------------------------
# Metric aggregation --- the physics definition of "how strongly the symmetry is broken".
# --------------------------------------------------------------------------------------
def symmetry_breaking_metric(delta: np.ndarray, ratio: np.ndarray) -> dict:
    """Aggregate per-(config, vertex) symmetry-breaking into summary scalars.

    Args:
        delta: real array, shape (n_configs, n_vertices), delta_v(sigma) = log psi(A_v s) - log psi(s).
        ratio: real array, same shape, r_v(sigma) = exp(delta) = psi(A_v s)/psi(s).

    Returns:
        dict of named scalar summaries (JSON-serializable floats).

    NOTE (learning-mode decision point): this function *defines* what "how strongly broken"
    means. Reasonable, physically-motivated choices you may keep / adjust / combine:
      * mean_abs_delta = mean |delta|        -- mean absolute log-amplitude deviation (nats)
      * rms_delta      = sqrt(mean delta^2)  -- penalizes large deviations more
      * mean_abs_ratdev= mean |r - 1|        -- deviation of the amplitude ratio from 1
      * Av             = mean r              -- physical <A_v> anchor (=1 at hz=0). Its
                                                std is the fluctuational breaking around <A_v>.
    All are 0 (Av -> 1) at hz=0 and grow with hz. Pick a primary; keep the rest as context.
    """
    n = delta.size
    return {
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "rms_delta": float(np.sqrt(np.mean(delta ** 2))),
        "max_abs_delta": float(np.max(np.abs(delta))) if n else 0.0,
        "mean_abs_ratio_dev": float(np.mean(np.abs(ratio - 1.0))),
        "Av_mean": float(np.mean(ratio)),                       # = <A_v> for Born sigma
        "Av_sem": float(np.std(ratio) / np.sqrt(max(n, 1))),    # standard error of the mean
    }


# --------------------------------------------------------------------------------------
# Config reconstruction from a run's JSON (config lives under sim_params, 1-elem lists).
# --------------------------------------------------------------------------------------
def config_from_sim_params(js: dict) -> dict:
    sp = js["sim_params"]

    def one(key, default=None):
        v = sp.get(key, default)
        return v[0] if isinstance(v, list) and len(v) else v

    cfg = {
        "architecture": one("architecture_type", "Combo"),
        "channels_noninv": list(sp["n_chann_noninv"]),
        "channels_inv": list(sp["n_chann_inv"]),
        "kernel_size": one("kernel_size_noninv"),
        "kernel_size_inv": one("kernel_size_inv"),
        "Lx": one("Lx"),
        "Ly": one("Lx"),                       # Ly := Lx throughout this repo
        "bc": one("BC", "OBC"),
        "dtype": one("param_dtype", "float64"),
        "rescale": one("rescale", 1.0),
        "symmetric_block": one("symmetric_block", "cnn"),
        "dual_basis": bool(one("dual_basis", False)),
        "seed": one("seed", 0),
        # sampler bookkeeping (keys literally carry a trailing '=' in the JSON)
        "n_chains": one("n_chains", 16),
        "n_samples_saved": one("n_samples"),   # rebuild target with saved shape => clean from_bytes
        "n_sweeps": one("n_sweeps"),
        "n_discard": one("n_discard_per_chain=", 8),
        "chunk_size": one("chunk_size=", 2048),
    }
    # transformer arms carry extra tf_* keys; pull them through if present
    for k in ("tf_layers", "tf1_layers", "tf_gamma_init", "tf_dmodel", "tf_heads",
              "tf_ffn_mult", "tf_activation", "tf_readout_K", "tf_complex_output"):
        if k in sp:
            cfg[k] = one(k)
    return cfg


# --------------------------------------------------------------------------------------
# Build model + restore params (netket/jax imported lazily so pure-numpy parts run locally).
# --------------------------------------------------------------------------------------
def load_run(base: str, cfg: dict):
    """Return (geometry, model, params, vstate). vstate carries the model's sampler."""
    import jax.numpy as jnp
    import netket as nk
    from model.geometry import ToricCodeGeometry
    from model.networks import KernelManager, create_model
    from simulation.custom_sampler import create_custom_sampler
    from utils.io import load_model

    geometry = ToricCodeGeometry(cfg["Lx"], cfg["Ly"], cfg["bc"])
    hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)
    kernel_manager = KernelManager(
        Lx=cfg["Lx"], Ly=cfg["Ly"], bc=cfg["bc"],
        kernel_size=cfg["kernel_size"], kernel_size_inv=cfg["kernel_size_inv"],
        arr_coord=geometry.arr_coord, dg_p=geometry.dg_p, N=geometry.N,
        dg_v=geometry.dg_v, vertex_all=geometry.vertex_all,
        dual=cfg.get("dual_basis", False),
    )
    model = create_model(cfg, geometry.plaq_all, kernel_manager)

    # Rebuild the training sampler so from_bytes matches the saved MCState structure.
    # Custom (vertex-flip) sampler needs bulk stars; fall back to LocalRule if none.
    n_chains = int(cfg["n_chains"])
    n_sweeps = int(cfg["n_sweeps"]) if cfg["n_sweeps"] is not None else geometry.N // 2
    has_bulk = len(geometry.vertex_bulk_hetero) > 0
    if has_bulk:
        sa = create_custom_sampler(
            geometry, hi,
            {"n_chains": n_chains, "n_sweeps": n_sweeps,
             "dual_basis": cfg.get("dual_basis", False)},
        )
    else:
        sa = nk.sampler.MetropolisSampler(
            hi, rule=nk.sampler.rules.LocalRule(),
            n_chains=n_chains, n_sweeps=n_sweeps, dtype=jnp.int8,
        )

    # Build the restore target with the SAME n_samples/n_chains as saved so flax.from_bytes
    # matches the serialized MCState structure exactly (avoids _samples-cache shape mismatch);
    # we override n_samples afterwards when drawing our own Born configs.
    n_samp_saved = int(cfg["n_samples_saved"]) if cfg.get("n_samples_saved") else n_chains * 8
    vs = load_model(
        base, sa, model,
        n_samples=n_samp_saved,
        n_discard_per_chain=int(cfg["n_discard"]),
        chunk_size=int(cfg["chunk_size"]),
    )
    return geometry, model, vs.parameters, vs


# --------------------------------------------------------------------------------------
# Core: draw configs, apply every vertex flip, form delta arrays.
# --------------------------------------------------------------------------------------
def deltas_for_configs(model, params, X, stars):
    """delta[i, k] = log psi(flip star_k of config i) - log psi(config i), real.

    X: (n_configs, N) in {-1,+1}. stars: list of index-lists (one per vertex).
    Returns (n_configs, n_stars) real array.
    """
    X = np.asarray(X, dtype=np.float64)
    base = np.asarray(model.apply({"params": params}, X)).real
    out = np.empty((X.shape[0], len(stars)), dtype=np.float64)
    for k, star in enumerate(stars):
        Xf = X.copy()
        Xf[:, np.asarray(star)] *= -1.0
        flipped = np.asarray(model.apply({"params": params}, Xf)).real
        out[:, k] = flipped - base
    return out


def summarize(delta, hist_range, n_bins):
    """Metric dict + histogram (edges, counts) for a delta array; None-safe if empty."""
    if delta.size == 0:
        return {}, {"edges": [], "counts": []}
    ratio = np.exp(delta)
    metric = symmetry_breaking_metric(delta, ratio)
    counts, edges = np.histogram(delta.ravel(), bins=n_bins, range=hist_range)
    return metric, {"edges": edges.tolist(), "counts": counts.tolist()}


def process_run(base, n_samples, n_discard_sample, seed, n_bins, hist_halfwidth):
    with open(base + ".json") as f:
        js = json.load(f)
    cfg = config_from_sim_params(js)

    L = int(cfg["Lx"])
    hz = float(js["sim_params"]["hz"][0])
    dtype = cfg["dtype"]
    if dtype != "float64":
        print(f"  SKIP {os.path.basename(base)}: dtype={dtype!r} (complex path out of scope)")
        return None

    geometry, model, params, vs = load_run(base, cfg)
    N = geometry.N
    bulk = [list(map(int, s)) for s in geometry.vertex_bulk_hetero]
    edge = [list(map(int, s)) for s in geometry.vertex_edge_hetero]

    # --- Born configs from the model's own sampler (sigma ~ |psi|^2) ---
    vs.n_discard_per_chain = int(n_discard_sample)
    vs.n_samples = int(n_samples)
    vs.reset()
    X_born = np.asarray(vs.samples).reshape(-1, N)[:n_samples]

    # --- Uniform +-1 configs (same count, fixed seed) ---
    rng = np.random.default_rng(seed)
    X_unif = rng.choice([-1.0, 1.0], size=(n_samples, N))

    hist_range = (-hist_halfwidth, hist_halfwidth)
    result = {
        "L": L, "hz": hz, "N": N, "dtype": dtype,
        "n_samples": int(n_samples),
        "n_vertices_bulk": len(bulk), "n_vertices_edge": len(edge),
        "file": os.path.basename(base),
        "energy_last": float(js["energy"][-1]) if js.get("energy") else None,
        "n_iters": len(js.get("energy", [])),
    }
    for tag, X in (("born", X_born), ("unif", X_unif)):
        for vset, stars in (("bulk", bulk), ("edge", edge)):
            if not stars:
                continue
            delta = deltas_for_configs(model, params, X, stars)
            metric, hist = summarize(delta, hist_range, n_bins)
            result[f"{tag}_{vset}"] = {"metric": metric, "hist": hist}
    return result


def discover_runs(nqs_dir, Lx, suffix=""):
    """One canonical (base) path per hz, for a CONSISTENT architecture arm.

    suffix="" selects the plain baseline (name ends right after the hz value, i.e. no
    arm suffix -- the Combo-small CNN "first experiment array"). suffix="tf"/"tfg"/"v2c8"
    etc. selects that arm instead (name must end with _hz<val>_<suffix>). Requires a paired
    .mpack; ties broken by most training iters.
    """
    end = re.escape("_" + suffix) if suffix else ""
    pat = re.compile(rf"_L{Lx}_hx[0-9.]+_hz([0-9]+\.[0-9]+){end}$")
    by_hz = {}
    for jp in glob.glob(os.path.join(nqs_dir, "*.json")):
        base_name = os.path.basename(jp)[:-len(".json")]
        m = pat.search(base_name)
        if not m:
            continue
        base = jp[:-len(".json")]
        if not os.path.exists(base + ".mpack"):
            continue
        try:
            js = json.load(open(jp))
        except Exception:
            continue
        niter = len(js.get("energy", []) or [])
        hz = round(float(m.group(1)), 5)
        if hz not in by_hz or niter > by_hz[hz][0]:
            by_hz[hz] = (niter, base)
    return [by_hz[hz][1] for hz in sorted(by_hz)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Lx", type=int, default=4, help="lattice size to sweep (default 4)")
    ap.add_argument("--nqs-dir", default="results/nqs", help="dir with G-equiv_*.{json,mpack}")
    ap.add_argument("--out-dir", default="results/symmetry", help="output dir")
    ap.add_argument("--suffix", default="", help="arm suffix ('' = plain CNN baseline; "
                    "e.g. 'tf','tfg','v2c8') -- keeps the architecture consistent across hz")
    ap.add_argument("--n_samples", type=int, default=8192, help="configs per run (Born & uniform)")
    ap.add_argument("--n_discard_sample", type=int, default=64, help="MCMC burn-in per chain for Born")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for uniform configs")
    ap.add_argument("--n_bins", type=int, default=81, help="histogram bins for delta")
    ap.add_argument("--hist_halfwidth", type=float, default=4.0, help="delta histogram range [-w, w]")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bases = discover_runs(args.nqs_dir, args.Lx, args.suffix)
    if not bases:
        raise SystemExit(f"No L={args.Lx} suffix={args.suffix!r} runs with paired .mpack "
                         f"found in {args.nqs_dir}")
    arm = args.suffix or "cnn-baseline"
    print(f"Found {len(bases)} L={args.Lx} [{arm}] runs:")
    for b in bases:
        print("  ", os.path.basename(b))

    for base in bases:
        print(f"\n=== {os.path.basename(base)} ===")
        res = process_run(base, args.n_samples, args.n_discard_sample,
                          args.seed, args.n_bins, args.hist_halfwidth)
        if res is None:
            continue
        res["arm"] = arm
        tag = "" if not args.suffix else f"_{args.suffix}"
        out = os.path.join(args.out_dir, f"sym_L{res['L']}_hz{res['hz']:.3f}{tag}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        bb = res.get("born_bulk", {}).get("metric", {})
        print(f"  hz={res['hz']:.3f}  born_bulk: <A_v>={bb.get('Av_mean', float('nan')):.4f}"
              f"  mean|delta|={bb.get('mean_abs_delta', float('nan')):.3e}  -> {out}")


if __name__ == "__main__":
    main()
