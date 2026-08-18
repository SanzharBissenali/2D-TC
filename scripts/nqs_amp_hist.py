"""Amplitude distribution of a TRAINED NQS over the full Z-basis (L<=4).

Answers "does the trained network's wavefunction show the 0-or-one-identical-value
spike structure of the exact fTC/TC h=0 ground state?" directly: rebuild the model
from a run JSON's sim_params, load the trained .mpack, evaluate log psi on ALL 2^N
basis strings (batched; 2^24 at L=4 is a ~2 min GPU pass for these tiny CNNs), and
record the amplitude distribution split by the EXACT ground-state support (a config
is in-support iff every plaquette product is +1 -- at h=0 the GS is the uniform
superposition over that 2^(N - #plaquettes)-dim zero-flux sector under OBC).

A real-log-psi network can never emit an exact 0 (that would be log psi = -inf), so
the interesting numbers are: how uniform the support amplitudes are (rel spread),
how far DOWN the off-support amplitudes sit (leakage weight), and the overlap with
the exact uniform state, |<psi_exact|psi_NQS>| = sum_support(a) / sqrt(|support|).

Output: <run_base>_amp.json next to the input, with the same histogram schema as
exact/lanczos_ed.py's diagnostics (amp_hist_* over a/max(a) in [-1,1], 400 bins)
plus a log10 histogram (amp_log10_hist_* over log10(a/max) in [-16, 0], 400 bins)
which is the right axis for NQS spikes separated by orders of magnitude.

Run (NERSC, needs netket/jax):
    python scripts/nqs_amp_hist.py --json results/nqs/G-equiv_1_ftc_L4_hx0.00_hz0.00_cnn.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def config_from_run_json(path):
    """Rebuild the config dict create_model/KernelManager need from sim_params."""
    with open(path) as f:
        sp = json.load(f)["sim_params"]

    def one(key, default=None):
        v = sp.get(key, [default])
        return v[0] if isinstance(v, list) else v

    cfg = {
        "Lx": int(one("Lx")), "Ly": int(one("Lx")), "bc": one("BC", "OBC"),
        "architecture": one("architecture_type", "Combo"),
        "channels_noninv": [int(c) for c in sp["n_chann_noninv"]],
        "channels_inv": [int(c) for c in sp["n_chann_inv"]],
        "kernel_size": int(one("kernel_size_noninv", 2)),
        "kernel_size_inv": int(one("kernel_size_inv")),
        "rescale": float(one("rescale", 1.0)),
        "dtype": one("param_dtype", "float64"),
        "symmetric_block": one("symmetric_block", "cnn"),
        "dual_basis": bool(one("dual_basis", False)),
        "ftc": bool(one("ftc", False)),
        "hx": float(one("hx", 0.0)), "hz": float(one("hz", 0.0)),
    }
    # transformer knobs pass through untouched if a non-cnn block ever uses this
    for k in ("tf_layers", "tf_dmodel", "tf_heads", "tf_ffn_mult", "tf_activation",
              "tf_readout_K", "tf_content", "tf_remat", "tf1_layers", "tf1_dmodel",
              "tf1_heads", "freeze_chi"):
        if k in sp:
            cfg[k] = one(k)
    return cfg


def load_params_from_mpack(path):
    """Extract the 'params' subtree from a serialized MCState .mpack (whole-state
    to_bytes), without reconstructing the sampler the run used."""
    import flax

    with open(path, "rb") as f:
        tree = flax.serialization.msgpack_restore(f.read())

    found = []

    def walk(node):
        if isinstance(node, dict):
            if "params" in node and isinstance(node["params"], dict):
                found.append(node["params"])
            for v in node.values():
                walk(v)

    walk(tree)
    assert found, f"no 'params' subtree in {path} (top-level keys: {list(tree)})"
    return found[0]


def main():
    p = argparse.ArgumentParser(description="Full-basis amplitude histogram of a trained NQS")
    p.add_argument("--json", required=True, nargs="+", help="run JSON(s) with sim_params")
    p.add_argument("--batch", type=int, default=1 << 18)
    args = p.parse_args()

    import jax
    import netket as nk
    from model.geometry import ToricCodeGeometry
    from model.networks import KernelManager, create_model

    for run_json in args.json:
        cfg = config_from_run_json(run_json)
        assert cfg["Lx"] <= 4, "full-basis evaluation is 2^N -- L<=4 only"
        base = run_json[:-5] if run_json.endswith(".json") else run_json
        mpack = base + ".mpack"
        out = base + "_amp.json"

        geometry = ToricCodeGeometry(cfg["Lx"], cfg["Ly"], cfg["bc"])
        hi = nk.hilbert.Spin(s=1 / 2, N=geometry.N)
        km = KernelManager(
            Lx=cfg["Lx"], Ly=cfg["Ly"], bc=cfg["bc"],
            kernel_size=cfg["kernel_size"], kernel_size_inv=cfg["kernel_size_inv"],
            arr_coord=geometry.arr_coord, dg_p=geometry.dg_p, N=geometry.N,
            dg_v=geometry.dg_v, vertex_all=geometry.vertex_all,
            dual=cfg["dual_basis"],
        )
        model = create_model(cfg, geometry.plaq_all, km)
        params = load_params_from_mpack(mpack)

        n_states = hi.n_states
        plaq = np.asarray(geometry.plaq_all)  # OBC plaquettes carry no -1
        fwd = jax.jit(lambda s: model.apply({"params": params}, s))

        logpsi = np.empty(n_states, dtype=np.float64)
        support = np.empty(n_states, dtype=bool)
        t0 = time.time()
        for a in range(0, n_states, args.batch):
            b = min(a + args.batch, n_states)
            s = np.asarray(hi.numbers_to_states(np.arange(a, b)))
            lp = np.asarray(fwd(s))
            assert np.max(np.abs(np.imag(lp))) < 1e-10 if np.iscomplexobj(lp) else True
            logpsi[a:b] = np.real(lp)
            support[a:b] = np.all(s[:, plaq].prod(axis=2) > 0, axis=1)
        eval_time = time.time() - t0

        # normalized positive amplitudes (real log psi => psi > 0 everywhere)
        w = np.exp(logpsi - logpsi.max())
        amp = w / np.sqrt(np.sum(w ** 2))
        scale = amp.max()
        a_sup, a_off = amp[support], amp[~support]

        n_sup = int(support.sum())
        result = {
            "run_json": os.path.basename(run_json),
            "Lx": cfg["Lx"], "N": geometry.N, "hx": cfg["hx"], "hz": cfg["hz"],
            "ftc": cfg["ftc"], "architecture": cfg["architecture"],
            "n_states": int(n_states),
            "support_size": n_sup,
            "support_rel_spread": float((a_sup.max() - a_sup.min()) / a_sup.mean()),
            "support_weight": float(np.sum(a_sup ** 2)),
            "offsupport_weight": float(np.sum(a_off ** 2)),
            "offsupport_max_over_scale": float(a_off.max() / scale),
            "overlap_exact_uniform": float(np.sum(a_sup) / np.sqrt(n_sup)),
            "eval_time_s": eval_time,
        }
        hist, edges = np.histogram(amp / scale, bins=400, range=(-1.0, 1.0))
        result["amp_hist_counts"] = hist.tolist()
        result["amp_hist_edges"] = edges.tolist()
        lg = np.log10(np.maximum(amp / scale, 1e-300))
        lhist, ledges = np.histogram(np.clip(lg, -16.0, 0.0), bins=400, range=(-16.0, 0.0))
        result["amp_log10_hist_counts"] = lhist.tolist()
        result["amp_log10_hist_edges"] = ledges.tolist()

        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{os.path.basename(base)}] support {n_sup}/{n_states} "
              f"(2^{np.log2(max(n_sup, 1)):.2f}), support rel-spread "
              f"{result['support_rel_spread']:.3e}, off-support weight "
              f"{result['offsupport_weight']:.3e}, overlap with exact uniform "
              f"{result['overlap_exact_uniform']:.10f}  ({eval_time:.1f}s) -> {out}")


if __name__ == "__main__":
    main()
