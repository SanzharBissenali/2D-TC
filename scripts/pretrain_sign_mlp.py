"""Supervised pretraining of the learned sign MLP m_theta(eps, x) (arm M-pre,
docs/signhead_benchmark_plan.md Sec. 3 step 4).

Two modes, selected by --target (default 'ed', byte-identical to the original
single-mode script):

  --target ed (original path, UNCHANGED): fit to the EXACT ED sign structure
    of the doubled-semion ground state at one (Lx, Ly, hx, hz) point -- see
    below. Needs --hx/--hz; the float early-stop threshold that used to be
    --target is now --fs_target (renamed 2026-09-24 so --target could become
    the ed/head mode selector; default unchanged at 1e-5, and the JSON
    payload still records it under the key "target" for downstream readers).

  --target head (NEW, docs/signhead_benchmark_plan.md addendum 2026-09-24): a
    field-independent, ED-free warm start. Fits the SAME _SignMLP to the
    CLOSED-FORM head sign itself, sign(sigma) = (-1)^{poly_sign01(x_of_r(sigma
    XOR eps))} = 1 - 2*head.s01(sigma) (model/sign_head.py), on synthetic data:
    a random closed-loop config r = G x (mod 2, x uniform over {0,1}^F, G the
    head's own hexagon-link incidence matrix -- the matrix x_of_r inverts) XOR
    k random single-link flips, k ~ Uniform{0..--k_max} per config (recorded).
    No ED anywhere in this path -- --synthetic/--n_val govern the streamed
    minibatch fit (optax.adam, --batch, --steps), reporting held-out sign
    error (overall + per-k) into a JSON `curve` every --report_every steps and
    snapshotting params at the same cadence. The snapshots can then be graded
    against exact ED at specific (hx, hz) points -- either inline via
    --ed_points (pulls in the ED memory profile into the same job) or, for
    the recommended split (fast debug-queue fit now, heavier regular-queue ED
    grading later), via a SEPARATE invocation: `--eval_only
    <tag>_snapshots.npz --ed_points "hx:hz,..."` -- which needs no jax/flax
    (mlp_forward_numpy is a plain-numpy replica of _SignMLP) beyond netket
    (ED) + pymatching/numba (the head).

Fits model.honeycomb_networks._SignMLP to the EXACT ED sign structure of the
doubled-semion ground state at one (Lx, Ly, hx, hz) point (--target ed):

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


# =====================================================================
# --target head: field-independent, ED-free warm start (2026-09-24 addendum)
# =====================================================================

def _pack_bits(bits):
    """(n, N) uint8 {0,1} rows -> (n,) opaque row keys for np.unique / np.isin
    (bytes-packed rows viewed as one fixed-width void per row; no width limit,
    unlike the int64 packing in model/decoders.py)."""
    packed = np.ascontiguousarray(np.packbits(bits.astype(np.uint8), axis=1))
    return packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)


def _draw_closed_loop_plus_k(geometry, head, n, k_max, rng):
    """n synthetic zero/near-zero-syndrome configs: x ~ Uniform({0,1}^F) ->
    r = G x (mod 2) [head's own G, the matrix x_of_r inverts -- see the
    QECSignHead.__init__ comment introducing self._G] XOR k random
    single-link flips, k ~ Uniform{0..k_max} PER ROW (vectorized via a
    per-row random-rank trick, no python loop over rows). Returns
    (bits (n, N) uint8, ks (n,) int64); each row's syndrome weight is
    <= 2*k (a single-link flip touches at most 2 vertices)."""
    N, F = geometry.N, geometry.n_plaqs
    x = rng.integers(0, 2, size=(n, F)).astype(np.uint8)
    r = ((x.astype(np.int64) @ head._G.T.astype(np.int64)) % 2).astype(np.uint8)
    ks = rng.integers(0, k_max + 1, size=n).astype(np.int64)
    ranks = np.argsort(np.argsort(rng.random((n, N)), axis=1), axis=1)
    flips = (ranks < ks[:, None]).astype(np.uint8)
    bits = np.bitwise_xor(r, flips)
    return bits, ks


def synthetic_dataset(geometry, head, n_train, n_val, k_max, seed,
                       oversample=1.5, max_rounds=200):
    """Deduplicated synthetic train/val split: both sets are internally
    unique, and val is disjoint from train (checked on the packed config
    codes -- two configs never share features_ex since (eps, x) determines
    sigma exactly, so config-level dedup is also feature-level dedup).
    Returns (train_bits, train_ks, val_bits, val_ks), each (*, N) uint8 /
    (*,) int64. Raises RuntimeError if max_rounds of oversampled draws still
    can't fill the request (k_max/N too small for n_train + n_val)."""
    rng = np.random.default_rng(seed)

    def fill(target_n, exclude_codes):
        have_bits, have_ks, have_codes = [], [], []
        total = 0
        for _ in range(max_rounds):
            if total >= target_n:
                break
            need = target_n - total
            batch = max(int(need * oversample), 64)
            bits, ks = _draw_closed_loop_plus_k(geometry, head, batch, k_max, rng)
            codes = _pack_bits(bits)
            codes_u, first = np.unique(codes, return_index=True)
            seen = np.concatenate([exclude_codes] + have_codes) if have_codes \
                else exclude_codes
            keep = ~np.isin(codes_u, seen, assume_unique=True)
            codes_u, first = codes_u[keep], first[keep]
            take = min(codes_u.size, need)
            if take:
                have_bits.append(bits[first[:take]])
                have_ks.append(ks[first[:take]])
                have_codes.append(codes_u[:take])
                total += take
        if total < target_n:
            raise RuntimeError(
                f"synthetic_dataset: only drew {total}/{target_n} unique "
                f"configs disjoint from the exclusion set after {max_rounds} "
                f"rounds -- k_max={k_max} too small (or N too small) for the "
                f"requested sample count?")
        N = geometry.N
        bits_all = (np.concatenate(have_bits, axis=0) if have_bits
                    else np.zeros((0, N), dtype=np.uint8))
        ks_all = (np.concatenate(have_ks, axis=0) if have_ks
                  else np.zeros(0, dtype=np.int64))
        codes_all = (np.concatenate(have_codes, axis=0) if have_codes
                     else exclude_codes[:0])
        return bits_all, ks_all, codes_all

    empty = _pack_bits(np.zeros((0, geometry.N), dtype=np.uint8))
    train_bits, train_ks, train_codes = fill(n_train, empty)
    val_bits, val_ks, val_codes = fill(n_val, train_codes)
    assert not np.isin(val_codes, train_codes, assume_unique=True).any(), \
        "val overlaps train after dedup (bug)"
    return train_bits, train_ks, val_bits, val_ks


def fit_mlp_head(Xtr, ytr, wtr, ktr, Xval, yval, kval, hidden, depth, lr,
                  steps, batch, seed, report_every=25):
    """Minibatch Adam fit of _SignMLP to the synthetic (features, head-sign)
    pairs. Unlike fit_mlp (full-batch scan over a 2^N enumeration), the data
    here are streamed arrays -- a fresh random minibatch of `batch` train
    rows every step (with replacement, host RNG, independent of the init
    key), `steps` steps total. Every `report_every` steps (and the last
    step), BEFORE that step's update: evaluates held-out UNWEIGHTED sign
    error on Xval overall and per-k into a `curve` entry alongside that
    step's train loss, snapshots the flattened params (for --eval_only /
    --ed_points to grade later without retraining), and keeps the
    best-by-val-error params. Returns (best_params, curve, summary,
    snapshots) where snapshots = {step: flat _SignMLP param dict}."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from flax import traverse_util

    from model.honeycomb_networks import _SignMLP

    n_train, K = Xtr.shape
    n_val = Xval.shape[0]
    batch = min(batch, n_train)
    k_max = int(max(ktr.max() if n_train else 0, kval.max() if n_val else 0))

    Xtr_d = jnp.asarray(Xtr, dtype=jnp.float64)
    ytr_d = jnp.asarray(ytr, dtype=jnp.float64)
    wtr_d = jnp.asarray(wtr, dtype=jnp.float64)
    Xval_d = jnp.asarray(Xval, dtype=jnp.float64)
    yval_d = jnp.asarray(yval, dtype=jnp.float64)

    mlp = _SignMLP(hidden=hidden, depth=depth)
    key = jax.random.PRNGKey(seed)
    params = mlp.init(key, jnp.zeros((1, K), dtype=jnp.float64))["params"]
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, idx):
        xb, yb, wb = Xtr_d[idx], ytr_d[idx], wtr_d[idx]

        def loss_fn(p):
            m = mlp.apply({"params": p}, xb)
            wsum = jnp.sum(wb)
            return jnp.sum(wb * jax.nn.softplus(-yb * m)) / jnp.maximum(wsum, 1e-300)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def val_wrong(params):
        m = mlp.apply({"params": params}, Xval_d)
        return jnp.sign(m) != yval_d

    # train-side sign error on a FIXED subset of the training set (same size
    # as val), so train vs held-out curves are directly comparable.
    n_tr_sub = min(n_train, max(n_val, 1))
    tr_idx = np.random.default_rng(seed + 2).choice(n_train, size=n_tr_sub, replace=False)
    Xtr_sub = jnp.asarray(Xtr[tr_idx], dtype=jnp.float64)
    ytr_sub = jnp.asarray(ytr[tr_idx], dtype=jnp.float64)

    def train_wrong(params):
        return jnp.sign(mlp.apply({"params": params}, Xtr_sub)) != ytr_sub

    rng = np.random.default_rng(seed + 1)   # independent of the init key
    curve, snapshots = [], {}
    best_params, best_err, best_step = params, float("inf"), -1
    t0 = time.time()
    for step in range(steps):
        report = (step % report_every == 0) or (step == steps - 1)
        pre_params = params
        if report:
            wrong = np.asarray(val_wrong(pre_params)) if n_val else np.zeros(0, dtype=bool)
            overall = float(wrong.mean()) if n_val else float("nan")
            per_k = {str(kk): float(wrong[kval == kk].mean())
                     for kk in range(k_max + 1) if (kval == kk).any()}
            train_err = float(np.asarray(train_wrong(pre_params)).mean())
        idx = jnp.asarray(rng.integers(0, n_train, size=batch))
        params, opt_state, loss = train_step(params, opt_state, idx)
        if not report:
            continue
        if overall < best_err:
            best_params, best_err, best_step = pre_params, overall, step
        curve.append({"step": step, "train_loss": float(loss), "train_err": train_err,
                      "val_err": overall, "val_err_per_k": per_k})
        snapshots[step] = traverse_util.flatten_dict(pre_params, sep="/")
        print(f"step {step:6d} loss={float(loss):.6e} train_err={train_err:.4e} val_err={overall:.4e} "
              f"({time.time() - t0:.0f}s)", flush=True)

    summary = {"steps_run": steps, "best_val_err": best_err,
              "best_step": best_step,
              "final_val_err": curve[-1]["val_err"] if curve else float("nan"),
              "final_loss": curve[-1]["train_loss"] if curve else float("nan"),
              "wall_s": time.time() - t0}
    return best_params, curve, summary, snapshots


def mlp_forward_numpy(flat_params, u):
    """Pure-numpy forward pass of model.honeycomb_networks._SignMLP (same
    {0,1} -> {-1,+1} feature recentring, `depth` tanh Dense(hidden) layers,
    final linear Dense(1)) -- lets --eval_only grade saved snapshots WITHOUT
    jax/flax/optax (only netket-for-ED + pymatching/numba-for-the-head +
    numpy are then needed). `depth` is inferred from the flat param dict's
    highest 'Dense_i/kernel' index (model/sign_mlp_io.mlp_param_spec).
    Returns the raw logit m, shape u.shape[:-1]."""
    depth = max(int(k.split("/")[0].split("_")[1]) for k in flat_params
                if k.startswith("Dense_") and k.endswith("/kernel"))
    h = np.asarray(u, dtype=np.float64) * 2.0 - 1.0
    for i in range(depth):
        h = np.tanh(h @ flat_params[f"Dense_{i}/kernel"] + flat_params[f"Dense_{i}/bias"])
    out = h @ flat_params[f"Dense_{depth}/kernel"] + flat_params[f"Dense_{depth}/bias"]
    return out[..., 0]


def save_snapshot_npz(path, snapshots, meta):
    """snapshots: {step:int -> flat _SignMLP param dict}. One npz with keys
    'step<N>/Dense_i/kernel' etc + a '_steps' index array + scalar meta
    (Lx, Ly, hidden, depth, k_max, seed) for --eval_only's consistency
    check -- mirrors save_features_npz's meta_ prefix convention."""
    payload = {}
    steps_sorted = sorted(snapshots)
    for step in steps_sorted:
        for k, v in snapshots[step].items():
            payload[f"step{step}/{k}"] = np.asarray(v, dtype=np.float64)
    payload["_steps"] = np.asarray(steps_sorted, dtype=np.int64)
    for k, v in meta.items():
        payload[f"meta_{k}"] = v
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(path, **payload)


def load_snapshot_npz(path):
    """Inverse of save_snapshot_npz: (steps list[int], {step: flat dict},
    meta dict)."""
    with np.load(path) as z:
        steps = [int(s) for s in z["_steps"]]
        snapshots = {}
        for step in steps:
            prefix = f"step{step}/"
            snapshots[step] = {k[len(prefix):]: np.asarray(z[k], dtype=np.float64)
                               for k in z.files if k.startswith(prefix)}
        meta = {k[len("meta_"):]: (z[k].item() if z[k].shape == () else z[k])
                for k in z.files if k.startswith("meta_")}
    return steps, snapshots, meta


def build_features_all(geometry, chunk=1 << 20, progress_every=8):
    """features_ex over ALL 2^N configs (host chunked pass), independent of
    any field point -- the expensive part of an ED-weighted grading
    (~18 min@2x3). Returns (X uint8 (dim, K), head). Mirrors the features
    loop inside build_features_and_targets, split out so it can be paid ONCE
    and reused across every --ed_points entry (ed_targets() below is the
    only per-point cost)."""
    from model.sign_head import QECSignHead

    head = QECSignHead(geometry)
    N = geometry.N
    dim = 1 << N
    K = head.n_features_ex
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
    return X, head


def ed_targets(geometry, hx, hz, tol=0.0):
    """(w, y, E0) via _honeycomb_direct_ed -- the SAME anchor-sign gauge as
    build_features_and_targets / scripts/sign_fidelity.py (all-up config
    positive). No features here: build_features_all covers that, once, for
    every --ed_points entry."""
    from exact.lanczos_ed import _honeycomb_direct_ed

    t0 = time.time()
    evals, psi = _honeycomb_direct_ed(geometry, 'ds', 1.0, hx, hz, k=1, tol=tol)
    psi = np.real(np.asarray(psi))
    anchor = psi[0]
    assert abs(anchor) > 1e-10 * np.max(np.abs(psi)), \
        "all-up anchor numerically zero -- sign gauge undefined"
    psi = psi * np.sign(anchor)
    w = psi ** 2
    w = w / w.sum()
    y = np.where(psi >= 0, 1, -1).astype(np.int8)
    print(f"# ED ({hx},{hz}): E0={evals[0]:.8f} in {time.time() - t0:.0f}s",
          flush=True)
    return w.astype(np.float64), y, float(evals[0])


def grade_snapshots(g, steps, snapshots, ed_points_str, tol, feature_chunk, max_snapshots=25):
    """{'hx:hz' -> {hx, hz, E0, head_ceiling, curve:[{step, ed_err}]}} for
    every comma-separated 'hx:hz' token in ed_points_str, against the
    snapshot set {step: flat _SignMLP param dict}. Also prints, per point,
    the deterministic head's OWN |psi_ED|^2-weighted ceiling computed
    directly from the closed form on the same feature pass (should equal
    1-F_s in results/diagnostics/signfid_hc*.json -- printed side by side
    with the MLP's achieved error for a direct sanity cross-check)."""
    N = g.N
    X_all, head = build_features_all(g, chunk=feature_chunk)          # uint8, kept as such
    head_sign_all = head.poly_sign01(X_all[:, N:]).astype(np.float64) * -2.0 + 1.0
    # Grade a log-spaced subset of the snapshots (801 x 2^27 rows is hours of
    # numpy), and stream the forward pass in row chunks: a 2^27 x hidden
    # float64 activation is ~70 GB, which OOM-killed the first attempt.
    avail = np.array(sorted(int(k) for k in snapshots))
    if len(avail) > max_snapshots:
        targets = np.concatenate([[avail[0]], np.geomspace(max(avail[1], 1), avail[-1],
                                                             num=max_snapshots - 1)])
        steps = sorted(set(int(avail[np.abs(avail - t).argmin()]) for t in targets))
    else:
        steps = [int(v) for v in avail]
    print(f"# grading {len(steps)} snapshots: {steps[:5]} ... {steps[-3:]}", flush=True)
    rows = X_all.shape[0]
    chunk_rows = 1 << 20
    mlp_signs = np.empty((len(steps), rows), dtype=np.int8)
    try:                                   # GPU forward when jax is around (cluster):
        import jax, jax.numpy as jnp       # np.tanh over 2^27 x hidden x 23 snapshots is
        jax.config.update("jax_enable_x64", True)   # >1 h single-threaded on the host
        stacked = {k: jnp.asarray(np.stack([snapshots[st][k] for st in steps]))
                   for k in snapshots[steps[0]]}
        depth = max(int(k.split("/")[0].split("_")[1]) for k in stacked
                    if k.startswith("Dense_") and k.endswith("/kernel"))

        @jax.jit
        def fwd_all(Xc):                    # (S, rows_c) signs for all snapshots at once
            h = jnp.broadcast_to(Xc * 2.0 - 1.0, (len(steps),) + Xc.shape)
            for i in range(depth):
                h = jnp.tanh(jnp.einsum("srk,skh->srh", h, stacked[f"Dense_{i}/kernel"])
                             + stacked[f"Dense_{i}/bias"][:, None, :])
            out = jnp.einsum("srk,skh->srh", h, stacked[f"Dense_{depth}/kernel"]) \
                + stacked[f"Dense_{depth}/bias"][:, None, :]
            return jnp.sign(out[..., 0]).astype(jnp.int8)
        chunk_rows = 1 << 18
        for lo in range(0, rows, chunk_rows):
            Xc = jnp.asarray(X_all[lo:lo + chunk_rows], dtype=jnp.float64)
            mlp_signs[:, lo:lo + chunk_rows] = np.asarray(fwd_all(Xc))
    except ImportError:
        for lo in range(0, rows, chunk_rows):
            Xc = X_all[lo:lo + chunk_rows].astype(np.float64)
            for si, step in enumerate(steps):
                mlp_signs[si, lo:lo + chunk_rows] = np.sign(mlp_forward_numpy(snapshots[step], Xc))
    del X_all

    points = []
    for tok in ed_points_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        hx_s, hz_s = tok.split(":")
        points.append((float(hx_s), float(hz_s)))
    assert points, f"--ed_points parsed to nothing: {ed_points_str!r}"

    results = {}
    for hx, hz in points:
        w, y, E0 = ed_targets(g, hx, hz, tol=tol)
        head_ceiling = float((w * (head_sign_all != y)).sum())
        print(f"# head ceiling (1-F_s) at hx={hx} hz={hz}: {head_ceiling:.6e} "
              f"(cf. results/diagnostics/signfid_hc*.json)", flush=True)
        curve = [{"step": int(step), "ed_err": float((w * (mlp_signs[si] != y)).sum())}
                 for si, step in enumerate(steps)]
        print(f"#   step {steps[-1]}: ed_err={curve[-1]['ed_err']:.6e} "
              f"(head_ceiling {head_ceiling:.6e})", flush=True)
        results[f"{hx:g}:{hz:g}"] = {"hx": hx, "hz": hz, "E0": E0,
                                     "head_ceiling": head_ceiling, "curve": curve}
    return results


def run_head_target(args, g):
    """--target head: fit _SignMLP to the CLOSED-FORM head sign itself on
    synthetic (closed-loop base XOR k flips) configs -- a field-independent,
    ED-free warm start. No ED call anywhere in this function unless
    --ed_points is also given, so the default invocation is a fast
    debug-queue job at any (Lx, Ly); the recommended decoupled ED grading is
    a separate --eval_only invocation instead (module docstring)."""
    from model.sign_head import QECSignHead

    N, F = g.N, g.n_plaqs
    assert args.k_max >= 0 and args.synthetic > 0 and args.n_val > 0
    head = QECSignHead(g)
    print(f"# pretrain_sign_mlp --target head {args.Lx}x{args.Ly} N={N} F={F} "
          f"K={head.n_features_ex} n_train={args.synthetic} n_val={args.n_val} "
          f"k_max={args.k_max} hidden={args.hidden} depth={args.depth} "
          f"seed={args.seed}", flush=True)

    t0 = time.time()
    train_bits, train_ks, val_bits, val_ks = synthetic_dataset(
        g, head, args.synthetic, args.n_val, args.k_max, args.seed)
    print(f"# synthetic data: {train_bits.shape[0]} train / {val_bits.shape[0]} "
          f"val, unique & train/val-disjoint, drawn in {time.time() - t0:.1f}s",
          flush=True)

    def labels_and_features(bits):
        states = bits.astype(np.float64) * -2.0 + 1.0   # bit=1 (down) -> -1
        y = (head.s01(states) * -2.0 + 1.0).astype(np.int8)
        X = head.features_ex(states).astype(np.uint8)
        return X, y

    Xtr, ytr = labels_and_features(train_bits)
    Xval, yval = labels_and_features(val_bits)

    if args.weight_by_k:
        # mild geometric decay favouring low-k (high-ED-weight-in-the-weak-
        # field-regime) configs; a heuristic, not fit to any field point --
        # the alternative to the (default) uniform weighting.
        wtr = args.k_decay ** train_ks.astype(np.float64)
        wtr = wtr / wtr.sum()
    else:
        wtr = np.full(train_bits.shape[0], 1.0 / train_bits.shape[0])

    tr_hist = np.bincount(train_ks, minlength=args.k_max + 1).tolist()
    val_hist = np.bincount(val_ks, minlength=args.k_max + 1).tolist()
    print(f"# k-histogram train={tr_hist} val={val_hist}", flush=True)

    best_params, curve, summary, snapshots = fit_mlp_head(
        Xtr, ytr, wtr, train_ks, Xval, yval, val_ks,
        args.hidden, args.depth, args.lr, args.steps, args.batch, args.seed,
        report_every=args.report_every)

    from flax import traverse_util
    from model.sign_mlp_io import save_mlp_npz

    tag = f"mlp_hc{args.Lx}x{args.Ly}_head_h{args.hidden}d{args.depth}_k{args.k_max}"
    npz_path = os.path.join(args.out_dir, tag + ".npz")
    save_mlp_npz(npz_path, traverse_util.flatten_dict(best_params, sep="/"))
    print(f"# wrote {npz_path}", flush=True)

    snap_path = os.path.join(args.out_dir, tag + "_snapshots.npz")
    meta = {"Lx": args.Lx, "Ly": args.Ly, "hidden": args.hidden,
            "depth": args.depth, "k_max": args.k_max, "seed": args.seed}
    save_snapshot_npz(snap_path, snapshots, meta)
    print(f"# wrote {snap_path} ({len(snapshots)} snapshots)", flush=True)

    val_path = os.path.join(args.out_dir, tag + "_val.npz")
    np.savez(val_path, bits=val_bits, k=val_ks, y=yval, features=Xval,
             meta_Lx=args.Lx, meta_Ly=args.Ly, meta_N=N, meta_F=F,
             meta_k_max=args.k_max, meta_seed=args.seed,
             meta_n_val=val_bits.shape[0])
    print(f"# wrote {val_path}", flush=True)

    payload = {
        "target": "head", "Lx": args.Lx, "Ly": args.Ly, "N": N, "F": F,
        "hidden": args.hidden, "depth": args.depth, "lr": args.lr,
        "steps": args.steps, "batch": args.batch, "seed": args.seed,
        "k_max": args.k_max, "n_train": int(train_bits.shape[0]),
        "n_val": int(val_bits.shape[0]), "weight_by_k": bool(args.weight_by_k),
        "k_decay": args.k_decay if args.weight_by_k else None,
        "train_k_hist": tr_hist, "val_k_hist": val_hist,
        "npz_path": npz_path, "snapshots_path": snap_path, "val_path": val_path,
        "curve": curve, **summary,
    }
    if args.ed_points:
        payload["ed_points"] = grade_snapshots(
            g, sorted(snapshots.keys()), snapshots, args.ed_points,
            args.tol, args.feature_chunk)

    json_path = os.path.join(args.out_dir, tag + ".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"# wrote {json_path}", flush=True)
    print(f"# BEST head-fit val_err={summary['best_val_err']:.3e} at step "
          f"{summary['best_step']} (final {summary['final_val_err']:.3e})",
          flush=True)


def run_eval_only(args):
    """--eval_only <snapshots.npz> --ed_points 'hx:hz,...': grade a
    previously-fit --target head snapshot set against exact ED, decoupled
    from the fitting run (module docstring). No jax/flax/optax needed here
    (mlp_forward_numpy is a plain-numpy MLP forward); needs netket (ED) +
    pymatching/numba (the head)."""
    assert args.ed_points, "--eval_only needs --ed_points"
    from model.honeycomb_geometry import HoneycombGeometry

    steps, snapshots, meta = load_snapshot_npz(args.eval_only)
    if "Lx" in meta:
        assert int(meta["Lx"]) == args.Lx and int(meta["Ly"]) == args.Ly, \
            (f"{args.eval_only}: meta Lx/Ly {meta['Lx']}/{meta['Ly']} != "
             f"--Lx {args.Lx} --Ly {args.Ly} (wrong snapshots file?)")
    g = HoneycombGeometry(args.Lx, args.Ly)
    print(f"# eval_only {args.Lx}x{args.Ly} N={g.N} F={g.n_plaqs} "
          f"{len(steps)} snapshots from {args.eval_only}", flush=True)

    results = grade_snapshots(g, steps, snapshots, args.ed_points, args.tol,
                              args.feature_chunk)

    base = args.eval_only[:-4] if args.eval_only.endswith(".npz") else args.eval_only
    out_path = base + "_edeval.json"
    payload = {"Lx": args.Lx, "Ly": args.Ly, "N": g.N, "F": g.n_plaqs,
               "snapshots_path": args.eval_only, "steps": [int(s) for s in steps],
               "points": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"# wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ed", "head"], default="ed",
                    help="'ed' (default): supervised fit to ED signs at one "
                         "(hx,hz) point, byte-identical to the pre-existing "
                         "path (needs --hx/--hz). 'head': field-independent "
                         "fit to the closed-form head sign on synthetic "
                         "data, no ED anywhere -- see --synthetic/--n_val/"
                         "--k_max and --eval_only/--ed_points for the "
                         "optional ED-weighted grading (module docstring).")
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--hx", type=float, default=None,
                    help="required for --target ed")
    ap.add_argument("--hz", type=float, default=None,
                    help="required for --target ed")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3000,
                    help="--target ed only")
    ap.add_argument("--fs_target", type=float, default=1e-5,
                    help="--target ed only: stop once 1 - F_s^MLP <= this "
                         "(this flag was named --target before 2026-09-24; "
                         "renamed so --target could become the ed/head mode "
                         "selector -- default unchanged, and the JSON output "
                         "still records it under the key \"target\")")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/pretrain")
    ap.add_argument("--fit_only", default="",
                    help="--target ed only: reuse a cached "
                         "features/weights/targets npz (--save_features of "
                         "a previous run); skips ED and the features_ex "
                         "pass entirely")
    ap.add_argument("--save_features", default="",
                    help="--target ed only: write the {features, weights, "
                         "targets, meta} cache to this path (skipped if "
                         "empty)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="ARPACK tol for any ED call (0 = machine precision)")
    ap.add_argument("--feature_chunk", type=int, default=1 << 20,
                    help="host chunk size for a full-enumeration features_ex "
                         "pass (--target ed, or --ed_points/--eval_only)")
    ap.add_argument("--batch_chunk", type=int, default=1 << 20,
                    help="--target ed only: device chunk size for the "
                         "scan-accumulated fit (bounds per-chunk activation "
                         "memory, not accuracy: the gradient is summed over "
                         "ALL chunks every epoch)")
    ap.add_argument("--report_every", type=int, default=25)
    # --target head
    ap.add_argument("--synthetic", type=int, default=200000,
                    help="--target head: N_TRAIN, synthetic training configs")
    ap.add_argument("--n_val", type=int, default=20000,
                    help="--target head: N_VAL, held-out synthetic configs")
    ap.add_argument("--k_max", type=int, default=12,
                    help="--target head: K, max single-link flips added to "
                         "a closed-loop base config (k ~ Uniform{0..K})")
    ap.add_argument("--weight_by_k", action="store_true",
                    help="--target head: geometric-decay-in-k sample "
                         "weights (see --k_decay) instead of the default "
                         "uniform per-sample weight")
    ap.add_argument("--k_decay", type=float, default=0.5,
                    help="--target head, only with --weight_by_k: per-k "
                         "weight decay base (weight ~ k_decay**k)")
    ap.add_argument("--batch", type=int, default=4096,
                    help="--target head: minibatch size")
    ap.add_argument("--steps", type=int, default=20000,
                    help="--target head: optimizer steps (minibatch SGD, "
                         "not epochs over the synthetic set)")
    ap.add_argument("--ed_points", default="",
                    help="comma-separated 'hx:hz' pairs; with --target head, "
                         "grades the fitted snapshots against exact ED "
                         "INLINE (pulls the ED memory profile into this same "
                         "job -- prefer --eval_only for the decoupled "
                         "invocation); required (and the only thing used) "
                         "with --eval_only")
    ap.add_argument("--eval_only", default="",
                    help="path to a *_snapshots.npz written by a previous "
                         "--target head run; grades it against --ed_points "
                         "and exits WITHOUT any training (needs --Lx/--Ly to "
                         "match the run that produced it)")
    args = ap.parse_args()

    if args.eval_only:
        run_eval_only(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    from model.honeycomb_geometry import HoneycombGeometry
    g = HoneycombGeometry(args.Lx, args.Ly)

    if args.target == "head":
        run_head_target(args, g)
        return

    assert args.hx is not None and args.hz is not None, \
        "--target ed needs --hx and --hz (or use --target head / --eval_only)"
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
        X, w, y, args.hidden, args.depth, args.lr, args.epochs, args.fs_target,
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
        "seed": args.seed, "target": args.fs_target, "E0": E0,
        **summary,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"# wrote {json_path}", flush=True)
    print(f"# CEILING CHECK: best 1-F_s = {summary['best_1_minus_Fs']:.3e} "
          f"(epoch {summary['best_epoch']}, SAVED) final 1-F_s = "
          f"{summary['final_1_minus_Fs']:.3e} vs target {args.fs_target:.1e} -- "
          + ("REACHED (MLP captured the head-exact tail)" if summary["achieved_target"]
             else "NOT REACHED (MLP is under-capacity at this hidden/depth -- a result)"),
          flush=True)


if __name__ == "__main__":
    main()
