"""Can an MLP LEARN the doubled-semion sign, or only memorise it? Pure supervised
test (no VMC, no energy): fit the arm-M sign MLP m_theta(eps, x) to sign labels
with an 80/20 train/validation split and log both learning curves.

Two data sources
  --source ed    (needs --hx --hz; ED-reachable sizes): the --n_data configs of
                 largest |psi_ED|^2, labelled with the EXACT ground-state sign
                 (all-up anchor gauge), weighted by |psi_ED|^2 (renormalised
                 inside each split).
  --source head  (any size, no ED): synthetic configs (random closed-loop set
                 XOR k <= --k_max random link flips, deduplicated), labelled
                 with the closed-form head sign, uniform weights. Exact ground
                 truth on the h_x = 0 line; the head's sign elsewhere.

  --source xsyn  (any size, no decoder, no eps): the MLP sees ONLY the F hexagon-
                 flip bits, x ~ uniform over distinct patterns, label = the
                 Levin-Gu polynomial (--degree 3, default), its quadratic part
                 (--degree 2) or full parity sum(x) (--degree 1). Isolates
                 lattice size / polynomial degree from the eps nuisance inputs.

Two splits of the SAME dataset, each run as --folds-fold cross-validation (every
configuration / every x pattern is held out exactly once; fold f also seeds the
MLP init):
  random   configurations held out at random. Every hexagon-flip pattern x can
           appear on both sides => tests memorisation.
  pattern  whole x patterns held out: validation only contains x patterns never
           seen in training => tests generalisation.

Loss = weighted softplus(-y m) (binary cross-entropy on the sign), full-batch
Adam. Curves (every --log_every steps): train/val loss, train/val weighted sign
error, and the PATTERN-BALANCED validation error (mean over validation x
patterns of the within-pattern weighted error -- immune to one heavy pattern,
e.g. x = 0 at h_z > 0, dominating the weight). Constant references on the
validation set, in both weightings: the computed head sign, the best constant
sign (hindsight) and the training-majority sign. NOTE the label is (nearly) a
function of x alone and 48/64 patterns are negative at 2x3, so the honest
baseline for a held-out pattern is the constant guess, not 50%.

Output: <out_dir>/signlearn_hc<Lx>x<Ly>_<source-tag>_h<hidden>d<depth>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ed_dataset(g, head, hx, hz, n_data):
    """Top-n_data configs by |psi|^2 -> (bits (n,N) uint8, y int8 +-1, w, meta)."""
    from exact.lanczos_ed import _honeycomb_direct_ed

    t0 = time.time()
    evals, psi = _honeycomb_direct_ed(g, 'ds', 1.0, hx, hz, k=1, tol=0.0)
    psi = np.real(np.asarray(psi))
    assert abs(psi[0]) > 1e-10 * np.abs(psi).max(), "all-up anchor ~ 0"
    psi = psi * np.sign(psi[0])
    w_all = psi ** 2
    w_all /= w_all.sum()
    n = min(n_data, int((w_all > 1e-20 * w_all.max()).sum()))   # no zero-weight noise rows
    idx = np.argpartition(w_all, -n)[-n:]
    idx = idx[np.argsort(-w_all[idx], kind='stable')]
    bits = ((idx[:, None] >> np.arange(g.N, dtype=np.int64)) & 1).astype(np.uint8)
    y = np.where(psi[idx] >= 0, 1, -1).astype(np.int8)
    meta = dict(E0=float(evals[0]), covered_weight=float(w_all[idx].sum()),
                ed_time_s=time.time() - t0)
    print(f"# ED ({hx},{hz}) E0={evals[0]:.8f}  top-{n} configs cover "
          f"{meta['covered_weight']:.6f} of the weight  ({meta['ed_time_s']:.0f}s)",
          flush=True)
    return bits, y, w_all[idx].astype(np.float64), meta


def head_dataset(g, head, n_data, k_max, seed):
    """n_data distinct synthetic configs (closed-loop set XOR k <= k_max random
    link flips) with the head's sign, uniform weights. Duplicates are dropped
    keeping DRAW order (pretrain_sign_mlp.synthetic_dataset keeps the smallest
    packed codes of each round, which biases the leading links -- review
    2026-09-29), so the kept set is an unbiased sample of the distinct draws."""
    from scripts.pretrain_sign_mlp import _draw_closed_loop_plus_k, _pack_bits

    assert n_data <= 2 ** (g.N - 2), f"n_data={n_data} too large for N={g.N}"
    rng = np.random.default_rng(seed)
    bits = np.zeros((0, g.N), dtype=np.uint8)
    for _ in range(50):
        new, _ = _draw_closed_loop_plus_k(g, head, int(1.3 * n_data), k_max, rng)
        bits = np.concatenate([bits, new], axis=0)
        _, first = np.unique(_pack_bits(bits), return_index=True)
        bits = bits[np.sort(first)]
        if bits.shape[0] >= n_data:
            break
    assert bits.shape[0] >= n_data, "could not draw n_data distinct configs"
    bits = bits[:n_data]
    y = (1 - 2 * head.s01(1.0 - 2.0 * bits.astype(np.float64))).astype(np.int8)
    return bits, y, np.full(bits.shape[0], 1.0), dict(k_max=k_max)


def pattern_ids(X, F):
    """Row -> integer id of its x pattern (last F feature columns)."""
    packed = np.ascontiguousarray(np.packbits(X[:, -F:], axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
    _, inv = np.unique(keys, return_inverse=True)
    return inv.reshape(-1)


def xsyn_dataset(head, F, n_data, degree, seed):
    """n distinct uniform x patterns and y = (-1)^poly_degree(x) (degree 3 = the
    head polynomial; 2 drops the vertex triples; 1 = parity of sum x)."""
    rng = np.random.default_rng(seed)
    n = min(n_data, 2 ** F)
    if F <= 22:
        codes = rng.choice(2 ** F, size=n, replace=False)
        x = ((codes[:, None] >> np.arange(F)) & 1).astype(np.uint8)
    else:
        x = np.zeros((0, F), dtype=np.uint8)
        while x.shape[0] < n:
            new = rng.integers(0, 2, size=(int(1.2 * (n - x.shape[0])) + 16, F), dtype=np.uint8)
            x = np.concatenate([x, new], axis=0)
            _, first = np.unique(np.packbits(x, axis=1).view(np.dtype((np.void, (F + 7) // 8))).reshape(-1),
                                 return_index=True)
            x = x[np.sort(first)]
        x = x[:n]
    xi = x.astype(np.int64)
    tot = xi.sum(axis=1)
    if degree >= 2 and head._pairs.size:
        tot += (xi[:, head._pairs[:, 0]] * xi[:, head._pairs[:, 1]]).sum(axis=1)
    if degree >= 3 and head._triples.size:
        tot += (xi[:, head._triples[:, 0]] * xi[:, head._triples[:, 1]]
                * xi[:, head._triples[:, 2]]).sum(axis=1)
    y = (1 - 2 * (tot & 1)).astype(np.int8)
    if degree == 3:
        assert (y == 1 - 2 * head.poly_sign01(x).astype(np.int8)).all()
    return x, y, np.ones(n), dict(degree=degree, n_pairs=int(head._pairs.shape[0]),
                                 n_triples=int(head._triples.shape[0]))


def hamming1_vote(xbits, y, w, val):
    """Weighted validation error of a Hamming-1 neighbour majority vote over the
    training patterns (F <= 62), plus the share of validation patterns with >= 1
    training neighbour. A baseline that interpolates without learning the function."""
    F = xbits.shape[1]
    if F > 62:
        return dict(hamming1_vote_err=None, hamming1_coverage=None)
    code = (xbits.astype(np.int64) << np.arange(F, dtype=np.int64)).sum(axis=1)
    tc, ty = code[~val], y[~val].astype(np.int64)
    tc_u, first = np.unique(tc, return_index=True)
    lab = ty[first]
    vc, vy, vw = code[val], y[val], w[val] / w[val].sum()
    votes = np.zeros(vc.size)
    hit = np.zeros(vc.size, dtype=bool)
    for j in range(F):
        nb = vc ^ (np.int64(1) << j)
        pos = np.clip(np.searchsorted(tc_u, nb), 0, tc_u.size - 1)
        ok = tc_u[pos] == nb
        votes += np.where(ok, lab[pos], 0)
        hit |= ok
    const = 1 if ty.sum() >= 0 else -1
    pred = np.where(votes > 0, 1, np.where(votes < 0, -1, const))
    return dict(hamming1_vote_err=float(vw[pred != vy].sum()),
                hamming1_coverage=float(hit.mean()))


def make_split(kind, pid, fold, n_folds, rng):
    """Boolean validation mask of fold ``fold``: 'random' partitions the
    configurations, 'pattern' partitions the distinct x patterns."""
    n, n_pat = pid.size, int(pid.max()) + 1
    if kind == 'random':
        val = (rng.permutation(n) % n_folds) == fold
    else:
        val = ((rng.permutation(n_pat) % n_folds) == fold)[pid]
    assert val.any() and (~val).any(), "degenerate split"
    seen = np.zeros(n_pat, dtype=bool)
    seen[pid[~val]] = True
    info = dict(n_patterns=n_pat, n_val_patterns=int(np.unique(pid[val]).size),
                val_rows_pattern_seen=float(seen[pid[val]].mean()))
    assert kind == 'random' or info['val_rows_pattern_seen'] == 0.0, "pattern leak"
    return val, info


def balanced_weights(w, pid):
    """Weights summing to 1 with every x pattern carrying equal total weight."""
    tot = np.bincount(pid, weights=w)
    wb = w / tot[pid]
    return wb / wb.sum()


def fit(X, y, w, wbal, val, hidden, depth, lr, steps, log_every, seed):
    """Returns (curve dict, final per-row validation 'wrong' mask)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from model.honeycomb_networks import _SignMLP

    def part(mask, ww):
        return (jnp.asarray(X[mask], dtype=jnp.float64),
                jnp.asarray(y[mask], dtype=jnp.float64),
                jnp.asarray(ww[mask] / ww[mask].sum()))

    tr, va = part(~val, w), part(val, w)
    vbal = jnp.asarray(wbal)
    mlp = _SignMLP(hidden=hidden, depth=depth)
    params = mlp.init(jax.random.PRNGKey(seed), tr[0][:1])["params"]
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_err(p, d):
        m = mlp.apply({"params": p}, d[0])
        wrong = (jnp.sign(m) != d[1])
        return jnp.sum(d[2] * jax.nn.softplus(-d[1] * m)), jnp.sum(d[2] * wrong), wrong

    @jax.jit
    def step(p, s, d):          # data as arguments, not closure constants
        g = jax.grad(lambda q: loss_err(q, d)[0])(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s

    @jax.jit
    def ev(p, a, b, bw):
        la, ea, _ = loss_err(p, a)
        lb, eb, wrong = loss_err(p, b)
        return dict(train_loss=la, train_err=ea, val_loss=lb, val_err=eb,
                    val_err_bal=jnp.sum(bw * wrong)), wrong

    curve = {k: [] for k in ("step", "train_loss", "train_err", "val_loss",
                             "val_err", "val_err_bal")}
    t0 = time.time()
    for it in range(steps + 1):
        if it % log_every == 0 or it == steps:
            vals, wrong = ev(params, tr, va, vbal)
            curve["step"].append(it)
            for k, v in vals.items():
                curve[k].append(float(v))
            if it % (20 * log_every) == 0 or it == steps:
                print(f"  step {it:6d}  loss {curve['train_loss'][-1]:.3e}/"
                      f"{curve['val_loss'][-1]:.3e}  err {curve['train_err'][-1]:.3e}/"
                      f"{curve['val_err'][-1]:.3e}  (train/val)  val_bal "
                      f"{curve['val_err_bal'][-1]:.3e}  {time.time() - t0:.0f}s",
                      flush=True)
        if it < steps:
            params, state = step(params, state, tr)
    return curve, np.asarray(wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--source", choices=["ed", "head", "xsyn"], required=True)
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--n_data", type=int, default=200000)
    ap.add_argument("--k_max", type=int, default=-1, help="head source; -1 => 2F")
    ap.add_argument("--folds", type=int, default=5, help="k-fold CV => 1/k held out")
    ap.add_argument("--max_folds", type=int, default=0,
                    help="run only the first this-many folds of the partition (0 => all)")
    ap.add_argument("--degree", type=int, default=3, choices=[1, 2, 3], help="xsyn only")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--splits", default="random,pattern")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out_dir", default="results/signlearn")
    args = ap.parse_args()

    from model.honeycomb_geometry import HoneycombGeometry
    from model.sign_head import QECSignHead

    g = HoneycombGeometry(args.Lx, args.Ly)
    head = QECSignHead(g)
    N, F = g.N, g.n_plaqs
    if args.source == "ed":
        bits, y, w, meta = ed_dataset(g, head, args.hx, args.hz, args.n_data)
        src = f"ed_hx{args.hx:g}_hz{args.hz:g}"
    elif args.source == "head":
        k_max = 2 * F if args.k_max < 0 else args.k_max
        bits, y, w, meta = head_dataset(g, head, args.n_data, k_max, seed=0)
        src = f"head_k{k_max}"
    else:
        bits, y, w, meta = xsyn_dataset(head, F, args.n_data, args.degree, seed=0)
        src = f"xsyn_d{args.degree}"
    if args.source == "xsyn":
        X, y_head = bits, y          # MLP input = x only; label is the polynomial itself
    else:
        states = 1.0 - 2.0 * bits.astype(np.float64)
        X = head.features_ex(states).astype(np.uint8)
        y_head = (1 - 2 * head.s01(states)).astype(np.int8)
        # (eps, x) must encode sigma losslessly: sigma = G x XOR eps
        back = ((X[:, N:].astype(np.int64) @ head._G.T.astype(np.int64)) & 1) ^ X[:, :N]
        assert (back == bits).all(), "features_ex is not a lossless encoding"
    print(f"# {args.Lx}x{args.Ly} N={N} F={F} source={src} n={bits.shape[0]} "
          f"neg weight={float(w[y < 0].sum() / w.sum()):.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"signlearn_hc{args.Lx}x{args.Ly}_{src}"
                                      f"_h{args.hidden}d{args.depth}{args.tag}.json")
    out = dict(Lx=args.Lx, Ly=args.Ly, N=N, F=F, source=args.source,
               hx=args.hx, hz=args.hz, n_data=int(bits.shape[0]),
               hidden=args.hidden, depth=args.depth, lr=args.lr,
               steps=args.steps, folds=args.folds, max_folds=args.max_folds, meta=meta, runs=[],
               complete=False)

    def dump():                 # atomic, after EVERY fit (walltime-safe)
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)

    pid = pattern_ids(X, F)

    def refs(ww, v, tag):
        """Weighted validation error of three sign rules under weights ww[v]."""
        wv = ww[v] / ww[v].sum()
        yv = y[v]
        tr_major = 1 if w[~v][y[~v] > 0].sum() >= w[~v][y[~v] < 0].sum() else -1
        return {f"head_val_err{tag}": float(wv[y_head[v] != yv].sum()),
                f"bestconst_val_err{tag}": float(min(wv[yv > 0].sum(), wv[yv < 0].sum())),
                f"trainmajority_val_err{tag}": float(wv[yv != tr_major].sum())}

    for kind in args.splits.split(","):
        for fold in range(min(args.max_folds or args.folds, args.folds)):
            val, info = make_split(kind, pid, fold, args.folds,
                                   np.random.default_rng(1000))
            wbal = balanced_weights(w[val], pid[val])
            h1 = hamming1_vote(X[:, -F:], y, w, val)
            wfull = np.zeros_like(w)
            wfull[val] = wbal
            ref = dict(n_train=int((~val).sum()), n_val=int(val.sum()),
                       val_weight_share=float(w[val].sum() / w.sum()),
                       n_train_patterns=int(np.unique(pid[~val]).size), input_dim=int(X.shape[1]),
                       **h1, **refs(w, val, ""), **refs(wfull, val, "_bal"), **info)
            print(f"== split={kind} fold={fold} {ref}", flush=True)
            curve, wrong = fit(X, y, w, wbal, val, args.hidden, args.depth,
                               args.lr, args.steps, args.log_every, fold)
            run = dict(split=kind, fold=fold, **ref, curve=curve)
            if info["n_val_patterns"] <= 256:    # per-pattern final verdicts
                pv, wv = pid[val], w[val]
                tot = np.bincount(pv, weights=wv, minlength=info["n_patterns"])
                err = np.bincount(pv, weights=wv * wrong, minlength=info["n_patterns"])
                neg = np.bincount(pv, weights=wv * (y[val] < 0), minlength=info["n_patterns"])
                ids = np.flatnonzero(tot > 0)
                run["val_patterns"] = dict(
                    weight=(tot[ids] / w.sum()).tolist(),
                    err=(err[ids] / tot[ids]).tolist(),
                    neg_share=(neg[ids] / tot[ids]).tolist())
            out["runs"].append(run)
            dump()

    out["complete"] = True
    dump()
    print(f"# wrote {path}", flush=True)


if __name__ == "__main__":
    main()
