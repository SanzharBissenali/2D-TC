"""Supervised pretraining of the learned sign MLP m_theta(eps, x) (arm M-pre,
docs/signhead_benchmark_plan.md Sec. 3 step 4).

Fits model.honeycomb_networks._SignMLP to the EXACT ED sign structure of the
doubled-semion ground state at one (Lx, Ly, hx, hz) point:

  1. enumerate psi_ED over all 2^N configs the same way scripts/sign_fidelity.py
     does (exact/lanczos_ed._honeycomb_direct_ed; bit convention site i <-> bit
     i, bit 1 = spin DOWN, all-up = index 0; sign gauge fixed by the all-up
     anchor, matching sign_fidelity.run_point);
  2. compute model.sign_head.QECSignHead.features_ex(sigma) = [eps, x] on every
     config, in host chunks (~18 min at 2x3, N=27 -- printed progress);
  3. fit the MLP (same architecture as honeycomb_networks._SignMLP: `depth`
     tanh Dense(hidden) layers then Dense(1), float64) to the |psi_ED|^2-
     weighted sign, via full-batch Adam (optax; the "batch" is deterministic
     gradient ACCUMULATION over device chunks each epoch -- jax.lax.scan --
     so a single 2^27-row epoch never materializes all activations at once).

Loss: weighted logistic regression of the raw logit m against the target sign
y in {-1,+1} (targets = sign psi_ED relative to the all-up anchor):
    L = sum_sigma w(sigma) * softplus(-y(sigma) * m(sigma)),   w = |psi_ED|^2.
Reported every --report_every epochs (and always at the end): the weighted
sign fidelity F_s^MLP = sum_sigma w(sigma) * [sign(m(sigma)) == y(sigma)] --
this is what MLPSignModel actually represents (sign(tanh(m)) == sign(m), tanh
being sign-preserving). Training stops early once 1 - F_s^MLP <= --target.
The final 1 - F_s^MLP is ALWAYS printed and recorded (the plan's ceiling-check
line: reaching >= 1 - target says the MLP captured the whole head-exact tail;
falling short at hx=0, where the closed form is exact, means the MLP is
under-capacity -- itself a result, not a failure of this script).

Two-stage so the slow ED+features pass can be reused across hidden/depth
sweeps: --save_features writes the {features, weights, targets, meta} cache;
--fit_only reads it back and skips ED entirely (needs only jax/flax/optax/
numpy -- no netket, no pymatching).

Outputs (arm M-pre input, model/sign_mlp_io.py spec):
    <out_dir>/mlp_hc{Lx}x{Ly}_hx{hx:g}_hz{hz:g}_h{hidden}d{depth}.npz
    <out_dir>/mlp_hc{Lx}x{Ly}_hx{hx:g}_hz{hz:g}_h{hidden}d{depth}.json

Runs on a GPU node (NERSC 2dtc env: jax 0.5.2, optax 0.2.5, netket
3.16.1.post1, pymatching 2.4.0). Not importable/runnable on this laptop
(netket is required transitively via exact.lanczos_ed, and jax/optax are not
installed) -- py_compile clean, numpy-only sub-logic validated separately.

Example:
    PYTHONPATH=. python scripts/pretrain_sign_mlp.py --Lx 2 --Ly 3 \
        --hx 0.4 --hz 0 --hidden 64 --depth 2 --out_dir results/pretrain

    # reuse the feature cache for a second (hidden, depth):
    PYTHONPATH=. python scripts/pretrain_sign_mlp.py --Lx 2 --Ly 3 \
        --hx 0.4 --hz 0 --hidden 32 --depth 1 \
        --fit_only results/pretrain/features_hc2x3_hx0.4_hz0.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_features_and_targets(geometry, hx, hz, tol=0.0, chunk=1 << 20,
                                progress_every=8):
    """ED (lazy netket import) -> (X uint8 (dim, K), w float64 (dim,) summing
    to 1, y int8 (dim,) in {-1,+1}, E0). Mirrors scripts/sign_fidelity.py's
    run_point anchor-sign gauge exactly (all-up config = index 0)."""
    from exact.lanczos_ed import _honeycomb_direct_ed
    from model.sign_head import QECSignHead

    head = QECSignHead(geometry)          # decoder='mwpm' (default; features_ex requires it)
    t0 = time.time()
    evals, psi = _honeycomb_direct_ed(geometry, 'ds', 1.0, hx, hz, k=1, tol=tol)
    psi = np.real(np.asarray(psi))
    anchor = psi[0]                       # all-up amplitude
    assert abs(anchor) > 1e-10 * np.max(np.abs(psi)), \
        "all-up anchor numerically zero -- sign gauge undefined"
    psi = psi * np.sign(anchor)
    w = psi ** 2
    w = w / w.sum()
    y = np.where(psi >= 0, 1, -1).astype(np.int8)
    print(f"# ED ({hx},{hz}): E0={evals[0]:.8f} in {time.time() - t0:.0f}s",
          flush=True)

    N, F = geometry.N, geometry.n_plaqs
    dim = 1 << N
    K = head.n_features_ex
    assert K == N + F
    X = np.empty((dim, K), dtype=np.uint8)
    site = np.arange(N, dtype=np.int64)
    t0 = time.time()
    n_chunks = (dim + chunk - 1) // chunk
    for ci, lo in enumerate(range(0, dim, chunk)):
        hi = min(lo + chunk, dim)
        idx = np.arange(lo, hi, dtype=np.int64)
        spins = (1 - 2 * ((idx[:, None] >> site) & 1)).astype(np.float64)
        X[lo:hi] = head.features_ex(spins).astype(np.uint8)
        if ci % progress_every == 0 or hi == dim:
            print(f"# features {hi}/{dim} ({100.0 * hi / dim:5.1f}%) "
                  f"chunk {ci + 1}/{n_chunks} elapsed {time.time() - t0:.0f}s",
                  flush=True)
    return X, w.astype(np.float64), y, float(evals[0])


def save_features_npz(path, X, w, y, meta):
    """Cache {features (dim,K) uint8, weights (dim,) float64, targets (dim,)
    int8} + scalar meta (Lx, Ly, hx, hz, N, F, E0), reused by --fit_only."""
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(path, features=X, weights=w, targets=y,
              **{f"meta_{k}": v for k, v in meta.items()})


def load_features_npz(path):
    with np.load(path) as z:
        X = np.asarray(z["features"])
        w = np.asarray(z["weights"], dtype=np.float64)
        y = np.asarray(z["targets"], dtype=np.int8)
        meta = {k[len("meta_"):]: (z[k].item() if z[k].shape == () else z[k])
                for k in z.files if k.startswith("meta_")}
    assert X.shape[0] == w.shape[0] == y.shape[0], \
        f"{path}: features/weights/targets length mismatch"
    return X, w, y, meta


def fit_mlp(X, w, y, hidden, depth, lr, epochs, target, seed,
            report_every=25, chunk=1 << 20):
    """Full-batch (chunk-accumulated) Adam fit of _SignMLP to the weighted
    sign targets. Returns (flax params tree of the BEST epoch by weighted
    1 - F_s, summary dict incl. best_1_minus_Fs / best_epoch / final_1_minus_Fs)."""
    import jax
    jax.config.update("jax_enable_x64", True)   # standalone run: netket may
    import jax.numpy as jnp                      # not have been imported yet
    import optax

    from model.honeycomb_networks import _SignMLP

    dim, K = X.shape
    chunk = min(chunk, dim)
    n_chunks = (dim + chunk - 1) // chunk
    pad = n_chunks * chunk - dim
    if pad:
        X = np.concatenate([X, np.zeros((pad, K), dtype=X.dtype)], axis=0)
        w = np.concatenate([w, np.zeros(pad, dtype=w.dtype)])
        y = np.concatenate([y, np.ones(pad, dtype=y.dtype)])   # weight 0 => inert

    Xc = jnp.asarray(X.reshape(n_chunks, chunk, K))
    Wc = jnp.asarray(w.reshape(n_chunks, chunk).astype(np.float64))
    Yc = jnp.asarray(y.reshape(n_chunks, chunk).astype(np.float64))

    mlp = _SignMLP(hidden=hidden, depth=depth)
    key = jax.random.PRNGKey(seed)
    params = mlp.init(key, jnp.zeros((1, K), dtype=jnp.float64))["params"]
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def epoch_step(params, Xc, Wc, Yc):
        zero_grad = jax.tree_util.tree_map(jnp.zeros_like, params)

        def body(carry, chunk_data):
            loss_acc, correct_acc, grad_acc = carry
            xb, wb, yb = chunk_data

            def loss_fn(p):
                m = mlp.apply({"params": p}, xb.astype(jnp.float64))
                loss = jnp.sum(wb * jax.nn.softplus(-yb * m))
                return loss, m

            (l, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
            correct = jnp.sum(wb * (jnp.sign(m) == yb))
            grad_acc = jax.tree_util.tree_map(lambda a, b: a + b, grad_acc, g)
            return (loss_acc + l, correct_acc + correct, grad_acc), None

        init = (jnp.asarray(0.0, jnp.float64), jnp.asarray(0.0, jnp.float64),
                zero_grad)
        (loss_sum, correct_sum, grads), _ = jax.lax.scan(
            body, init, (Xc, Wc, Yc))
        return loss_sum, correct_sum, grads

    loss_curve, one_minus_fs_curve = [], []
    t0 = time.time()
    achieved = False
    ep = 0
    best_params, best_1mfs, best_ep = params, float("inf"), -1
    for ep in range(epochs):
        loss_sum, correct_sum, grads = epoch_step(params, Xc, Wc, Yc)
        fs = float(correct_sum)          # total weight sums to 1 (padding => 0)
        # correct_sum was evaluated on the PRE-update params of this epoch:
        # keep those as the best-epoch snapshot (the saved tree is then the
        # one whose 1-F_s is reported, not an unevaluated post-update step).
        if 1.0 - fs < best_1mfs:
            best_params, best_1mfs, best_ep = params, 1.0 - fs, ep
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_curve.append(float(loss_sum))
        one_minus_fs_curve.append(1.0 - fs)
        reached = (1.0 - fs) <= target
        if ep % report_every == 0 or ep == epochs - 1 or reached:
            print(f"epoch {ep:5d} loss={float(loss_sum):.6e} "
                  f"1-F_s={1.0 - fs:.3e} ({time.time() - t0:.0f}s)", flush=True)
        if reached:
            achieved = True
            print(f"# target reached at epoch {ep}: 1-F_s={1.0 - fs:.3e} "
                  f"<= {target:.1e}", flush=True)
            break

    assert best_1mfs == min(one_minus_fs_curve)
    return best_params, {
        "epochs_run": ep + 1, "achieved_target": achieved,
        "final_1_minus_Fs": one_minus_fs_curve[-1],
        "final_loss": loss_curve[-1],
        "best_1_minus_Fs": best_1mfs, "best_epoch": best_ep,
        "min_1_minus_Fs": min(one_minus_fs_curve),
        "loss_curve": loss_curve, "one_minus_Fs_curve": one_minus_fs_curve,
        "wall_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--hx", type=float, required=True)
    ap.add_argument("--hz", type=float, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--target", type=float, default=1e-5,
                    help="stop once 1 - F_s^MLP <= target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/pretrain")
    ap.add_argument("--fit_only", default="",
                    help="reuse a cached features/weights/targets npz "
                         "(--save_features of a previous run); skips ED "
                         "and the features_ex pass entirely")
    ap.add_argument("--save_features", default="",
                    help="write the {features, weights, targets, meta} cache "
                         "to this path (skipped if empty)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="ARPACK tol for the ED call (0 = machine precision)")
    ap.add_argument("--feature_chunk", type=int, default=1 << 20,
                    help="host chunk size for the features_ex pass")
    ap.add_argument("--batch_chunk", type=int, default=1 << 20,
                    help="device chunk size for the scan-accumulated fit "
                         "(bounds per-chunk activation memory, not accuracy: "
                         "the gradient is summed over ALL chunks every epoch)")
    ap.add_argument("--report_every", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from model.honeycomb_geometry import HoneycombGeometry
    g = HoneycombGeometry(args.Lx, args.Ly)
    print(f"# pretrain_sign_mlp {args.Lx}x{args.Ly} hx={args.hx} hz={args.hz} "
          f"N={g.N} F={g.n_plaqs} K={g.N + g.n_plaqs} dim=2^{g.N} "
          f"hidden={args.hidden} depth={args.depth}", flush=True)

    if args.fit_only:
        print(f"# loading cached features from {args.fit_only}", flush=True)
        X, w, y, meta = load_features_npz(args.fit_only)
        for k, want in (("Lx", args.Lx), ("Ly", args.Ly)):
            assert int(meta[k]) == want, \
                f"cached {k}={meta[k]!r} != --{k} {want} (wrong cache file?)"
        for k, want in (("hx", args.hx), ("hz", args.hz)):
            assert abs(float(meta[k]) - want) < 1e-9, \
                f"cached {k}={meta[k]!r} != --{k} {want} (wrong cache file?)"
        E0 = float(meta.get("E0", float("nan")))
    else:
        X, w, y, E0 = build_features_and_targets(
            g, args.hx, args.hz, tol=args.tol, chunk=args.feature_chunk)
        if args.save_features:
            meta = {"Lx": args.Lx, "Ly": args.Ly, "hx": args.hx, "hz": args.hz,
                    "N": g.N, "F": g.n_plaqs, "E0": E0}
            save_features_npz(args.save_features, X, w, y, meta)
            print(f"# wrote features cache {args.save_features}", flush=True)

    params, summary = fit_mlp(
        X, w, y, args.hidden, args.depth, args.lr, args.epochs, args.target,
        args.seed, report_every=args.report_every, chunk=args.batch_chunk)

    import flax
    from flax import traverse_util
    from model.sign_mlp_io import save_mlp_npz

    flat = traverse_util.flatten_dict(params, sep="/")
    tag = f"mlp_hc{args.Lx}x{args.Ly}_hx{args.hx:g}_hz{args.hz:g}_h{args.hidden}d{args.depth}"
    npz_path = os.path.join(args.out_dir, tag + ".npz")
    save_mlp_npz(npz_path, flat)
    print(f"# wrote {npz_path}", flush=True)

    json_path = os.path.join(args.out_dir, tag + ".json")
    payload = {
        "Lx": args.Lx, "Ly": args.Ly, "hx": args.hx, "hz": args.hz,
        "hidden": args.hidden, "depth": args.depth, "lr": args.lr,
        "seed": args.seed, "target": args.target, "E0": E0,
        **summary,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"# wrote {json_path}", flush=True)
    print(f"# CEILING CHECK: best 1-F_s = {summary['best_1_minus_Fs']:.3e} "
          f"(epoch {summary['best_epoch']}, SAVED) final 1-F_s = "
          f"{summary['final_1_minus_Fs']:.3e} vs target {args.target:.1e} -- "
          + ("REACHED (MLP captured the head-exact tail)" if summary["achieved_target"]
             else "NOT REACHED (MLP is under-capacity at this hidden/depth -- a result)"),
          flush=True)


if __name__ == "__main__":
    main()
