"""Online (streaming) sign-learnability test: can the arm-M MLP m_theta(eps, x)
learn the doubled-semion sign from a stream of fresh samples, never a fixed pool?

No VMC, no energy. Every step draws a FRESH i.i.d. minibatch (with replacement:
repeats occur wherever the configuration/pattern space is small -- 2x3 ED, small F);
the validation set is a fixed draw that training can never produce (a stateless hash partition of the
stream, see below). Reported: training loss/error (window means over fresh data,
so themselves a generalisation estimate), validation loss/error, against the
number of samples seen.

Sources
  head  (any size, no ED): x uniform over {0,1}^F, r = G x, XOR k ~ U{0..k_max}
        random link flips (k_max = 2F); (eps, x) = head.features_ex(sigma);
        label = head sign = (-1)^poly(x).
  ed    (2x3): configs drawn i.i.d. from |psi_ED|^2, exact ground-state sign.
        ED is solved in a subprocess (no jax/netket in this process, so forked
        workers are safe) and cached as cumulative-probability + sign arrays.

Splits (20% held out, fixed for the whole run, both defined by a hash)
  random   a configuration is held out iff hash(config) % 5 == 0
  pattern  an x pattern is held out iff hash(x) % 5 == 0 (exact-count table for
           F <= 24); validation only contains x patterns that training can
           never produce.

The three seeds differ only in MLP initialisation; they are trained together on
the SAME streamed batches (one vmapped pytree), so data generation, the CPU
bottleneck, is paid once.

Workers are SPAWNED (clean interpreters; nothing forks after jax starts).

Output: <out_dir>/signonline_hc<Lx>x<Ly>_<src>_<split>_h<H>d<D>.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker process

import argparse
import json
import multiprocessing as mp
import queue as queue_mod
import subprocess
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELD_FRAC_DEN = 5                            # held out <=> hash % 5 == 0  (20%)


# ----------------------------------------------------------------- hashing ---
def mix64(z):
    """splitmix64 finaliser on a uint64 array (wraps silently)."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def config_hash(bits, salt=0):
    """(n, N) {0,1} rows -> (n,) uint64 hash of the whole row."""
    n, N = bits.shape
    nb = (N + 7) // 8
    packed = np.zeros((n, ((nb + 7) // 8) * 8), dtype=np.uint8)
    packed[:, :nb] = np.packbits(bits.astype(np.uint8), axis=1)
    words = packed.view(np.uint64)
    h = np.full(n, np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64)
    for i in range(words.shape[1]):
        h = mix64(h ^ (words[:, i] + np.uint64(i + 1)))
    return h


def pattern_key(x):
    """(n, F) {0,1} rows -> (n,) uint64 key, F <= 62 (exact, injective)."""
    return (x.astype(np.uint64) << np.arange(x.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64)


class Partition:
    """Stateless 80/20 partition. ``held(bits, x)`` -> bool mask of held-out rows."""

    def __init__(self, kind, F, seed=12345):
        self.kind, self.F, self.seed = kind, F, seed
        self.table = None                     # pattern split, F <= 24: exact-count table, built lazily

    def __getstate__(self):                 # do not ship the 2^F table to every worker
        d = dict(self.__dict__)
        d["table"] = None
        return d

    def _build_table(self):
        rng = np.random.default_rng(self.seed)
        n_held = max(1, int(round(2 ** self.F / HELD_FRAC_DEN)))
        table = np.zeros(2 ** self.F, dtype=bool)
        table[rng.choice(2 ** self.F, size=n_held, replace=False)] = True
        return table

    def held(self, bits, x):
        if self.kind == "pattern" and self.F <= 24 and self.table is None:
            self.table = self._build_table()
        if self.kind == "random":
            return (config_hash(bits, self.seed) % np.uint64(HELD_FRAC_DEN)) == 0
        key = pattern_key(x)
        if self.table is not None:
            return self.table[key.astype(np.int64)]
        return (mix64(key ^ np.uint64(self.seed)) % np.uint64(HELD_FRAC_DEN)) == 0


# ------------------------------------------------------------------ stream ---
class Stream:
    """Draws fresh (X uint8 (B, N+F), y int8) batches from one side of the partition."""

    def __init__(self, cfg, part):
        from model.honeycomb_geometry import HoneycombGeometry
        from model.sign_head import QECSignHead
        self.cfg, self.part = cfg, part
        g = HoneycombGeometry(cfg["Lx"], cfg["Ly"])
        self.g, self.N, self.F = g, g.N, g.n_plaqs
        self.head = QECSignHead(g, n_threads=1)
        self.k_max = 2 * self.F if cfg["k_max"] < 0 else cfg["k_max"]
        if cfg["source"] == "ed":
            self.cum = np.load(cfg["cum_path"], mmap_mode="r")
            self.sign = np.load(cfg["sign_path"], mmap_mode="r")

    def raw(self, rng, n):
        if self.cfg["source"] == "head":
            from scripts.pretrain_sign_mlp import _draw_closed_loop_plus_k
            bits, _ = _draw_closed_loop_plus_k(self.g, self.head, n, self.k_max, rng)
            return bits, None
        u = rng.random(n) * float(self.cum[-1])
        idx = np.minimum(np.searchsorted(self.cum, u, side="right"), self.cum.shape[0] - 1)
        bits = ((idx[:, None] >> np.arange(self.N, dtype=np.int64)) & 1).astype(np.uint8)
        return bits, np.asarray(self.sign[idx])

    def batch(self, rng, B, side):
        Xs, ys, have, drawn_tot = [], [], 0, 0
        want_held = side == "val"
        for _round in range(4000):
            if have >= B:
                break
            acc_rate = max(have / drawn_tot, 1e-5) if drawn_tot else 0.7     # adaptive draw size
            n = int(min(5_000_000, 1.4 * (B - have) / acc_rate)) + 64
            drawn_tot += n
            bits, y = self.raw(rng, n)
            if self.part.kind == "random":
                m = self.part.held(bits, None) == want_held
                bits, y = bits[m], (None if y is None else y[m])
            X = self.head.features_ex(1.0 - 2.0 * bits.astype(np.float64)).astype(np.uint8)
            if y is None:
                y = (1 - 2 * self.head.poly_sign01(X[:, self.N:]).astype(np.int64)).astype(np.int8)
            if self.part.kind == "pattern":
                m = self.part.held(None, X[:, self.N:]) == want_held
                X, y = X[m], y[m]
            Xs.append(X), ys.append(y)
            have += X.shape[0]
            drawn = getattr(self, "_drawn", 0) + n
            self._drawn = drawn
        if have < B:
            raise RuntimeError(f"{side} side of the {self.part.kind} split is (nearly) empty: "
                               f"{have}/{B} rows after 4000 rounds ({drawn_tot} draws)")
        return np.concatenate(Xs)[:B], np.concatenate(ys)[:B].astype(np.int8)

    def side_weight(self, rng, n):
        """Probability mass of the held-out side under the stream's distribution (Monte Carlo)."""
        bits, _ = self.raw(rng, n)
        X = self.head.features_ex(1.0 - 2.0 * bits.astype(np.float64)).astype(np.uint8)
        m = self.part.held(bits, X[:, self.N:]) if self.part.kind == "random" else self.part.held(None, X[:, self.N:])
        return float(m.mean())


def _worker(cfg, part, wid, q, stop, chunk_rows):
    try:
        st = Stream(cfg, part)
        rng = np.random.default_rng([cfg["seed"], 1, wid])
        while not stop.is_set():
            q.put(st.batch(rng, chunk_rows, "train"))
    except BaseException:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------- ED ----
def make_psi(Lx, Ly, hx, hz, cum_path, sign_path):
    """Run in a subprocess: ED, then save cumulative |psi|^2 and the sign (anchor gauge)."""
    from exact.lanczos_ed import _honeycomb_direct_ed
    from model.honeycomb_geometry import HoneycombGeometry

    g = HoneycombGeometry(Lx, Ly)
    evals, psi = _honeycomb_direct_ed(g, "ds", 1.0, hx, hz, k=1, tol=0.0)
    psi = np.real(np.asarray(psi))
    assert abs(psi[0]) > 1e-10 * np.abs(psi).max(), "all-up anchor ~ 0"
    psi = psi * np.sign(psi[0])
    sign = np.where(psi >= 0, 1, -1).astype(np.int8)
    cum = np.cumsum(psi ** 2)
    for arr, path in ((cum, cum_path), (sign, sign_path)):
        tmp = f"{path}.{os.getpid()}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
    print(f"# ED ({hx},{hz}) E0={evals[0]:.8f}; cached {cum_path}", flush=True)


def ed_cache_paths(a):
    d = a.cache_dir or os.path.join(os.environ.get("SCRATCH", "/tmp"), "signonline_cache")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"ed_hc{a.Lx}x{a.Ly}_hx{a.hx:g}_hz{a.hz:g}")
    return base + "_cum.npy", base + "_sign.npy"


# ------------------------------------------------------------------- train ---
def train_split(a, cfg, split):
    t_all = time.time()
    part = Partition(split, cfg["F"])
    st = Stream(cfg, part)
    N, F = st.N, st.F
    K = N + F
    rng = np.random.default_rng([cfg["seed"], 0])
    Xv, yv = st.batch(rng, a.n_val, "val")
    Xb, yb = st.batch(rng, 100000, "train")
    const = 1 if (yb > 0).mean() >= 0.5 else -1
    refs = dict(trainmajority_val_err=float((yv != const).mean()),
                bestconst_val_err=float(min((yv > 0).mean(), (yv < 0).mean())),
                val_neg_frac=float((yv < 0).mean()), train_neg_frac=float((yb < 0).mean()),
                val_distinct_patterns=int(np.unique(pattern_key(Xv[:, N:])).size),
                val_side_weight=st.side_weight(np.random.default_rng([cfg["seed"], 2]), 200000),
                train_batch_distinct_configs=int(np.unique(config_hash(
                    ((Xb[:, N:].astype(np.int64) @ st.head._G.T.astype(np.int64)) & 1) ^ Xb[:, :N])).size))
    if cfg["source"] == "ed":
        # head sign vs exact sign on the validation set (sigma = G x XOR eps)
        sig = ((Xv[:, N:].astype(np.int64) @ st.head._G.T.astype(np.int64)) & 1) ^ Xv[:, :N]
        refs["head_val_err"] = float((1 - 2 * st.head.s01(1.0 - 2.0 * sig.astype(np.float64)) != yv).mean())
    print(f"# split={split}: val {Xv.shape[0]} rows (held-out side carries {100 * refs['val_side_weight']:.1f}% of the mass), "
          f"{refs['val_distinct_patterns']} distinct x patterns, "
          f"{refs['train_batch_distinct_configs']}/{Xb.shape[0]} distinct configs in a 100k train batch, "
          f"const-guess val err {refs['trainmajority_val_err']:.4f}", flush=True)

    ctx = mp.get_context("spawn")      # clean interpreters: no fork after jax/CUDA init (2nd split)
    q = ctx.Queue(maxsize=2 * a.workers)
    stop = ctx.Event()
    chunk = a.batch * a.chunk_batches
    procs = [ctx.Process(target=_worker, args=(cfg, part, w, q, stop, chunk), daemon=True)
             for w in range(a.workers)]
    for p in procs:
        p.start()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from model.honeycomb_networks import _SignMLP

    S = a.seeds
    mlp = _SignMLP(hidden=a.hidden, depth=a.depth)
    keys = jax.random.split(jax.random.PRNGKey(a.seed), S)
    params = jax.vmap(lambda k: mlp.init(k, jnp.zeros((1, K), jnp.float64))["params"])(keys)
    opt = optax.adam(a.lr)
    ostate = opt.init(params)
    apply = jax.vmap(lambda p, X: mlp.apply({"params": p}, X), in_axes=(0, None))   # -> (S, B)

    def stats(p, X, y):
        m = apply(p, X.astype(jnp.float64))
        yf = y.astype(jnp.float64)
        loss = jax.nn.softplus(-yf * m)
        return loss, (jnp.sign(m) != yf)

    @jax.jit
    def step(p, s, X, y, acc):
        def lf(q_):
            loss, err = stats(q_, X, y)
            lm = loss.mean(axis=1)
            return lm.sum(), (lm, err.mean(axis=1))
        (_, (lm, em)), g = jax.value_and_grad(lf, has_aux=True)(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s, acc + jnp.stack([lm, em])

    @jax.jit
    def evaluate(p, X, y):
        loss, err = stats(p, X, y)
        return loss.mean(axis=1), err.mean(axis=1)

    Xv_d, yv_d = jnp.asarray(Xv), jnp.asarray(yv)
    n_steps = a.samples // a.batch
    assert n_steps >= 1, "--samples smaller than --batch"
    eval_at = set(np.unique(np.round(np.geomspace(1, n_steps, a.evals)).astype(int)).tolist())
    curve = {k: [] for k in ("step", "samples", "train_loss", "train_err", "val_loss", "val_err")}
    out = dict(cfg={k: v for k, v in cfg.items() if not k.endswith("_path")}, split=split,
               N=N, F=F, input_dim=K, batch=a.batch, lr=a.lr, hidden=a.hidden, depth=a.depth,
               seeds=S, n_val=int(Xv.shape[0]), budget_samples=int(n_steps * a.batch), refs=refs,
               curve=curve, complete=False)
    path = os.path.join(a.out_dir, f"signonline_hc{a.Lx}x{a.Ly}_{cfg['src']}_{split}_h{a.hidden}d{a.depth}{a.tag}.json")
    os.makedirs(a.out_dir, exist_ok=True)

    def dump():
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)

    def log_point(it, acc, n_acc):
        vl, ve = evaluate(params, Xv_d, yv_d)
        curve["step"].append(it); curve["samples"].append(it * a.batch)
        tl = (np.asarray(acc) / max(n_acc, 1)) if n_acc else np.full((2, S), np.nan)
        curve["train_loss"].append(tl[0].tolist()); curve["train_err"].append(tl[1].tolist())
        curve["val_loss"].append(np.asarray(vl).tolist()); curve["val_err"].append(np.asarray(ve).tolist())

    acc = jnp.zeros((2, S)); n_acc = 0
    log_point(0, acc, 0)
    buf_X = buf_y = None
    t0 = time.time(); starved = 0.0
    for it in range(1, n_steps + 1):
        if buf_X is None or buf_X.shape[0] < a.batch:
            t1 = time.time()
            while True:
                try:
                    item = q.get(timeout=60)
                    break
                except queue_mod.Empty:
                    alive = sum(p.is_alive() for p in procs)
                    if alive == 0 or time.time() - t1 > 600:
                        raise RuntimeError(f"no data for {time.time() - t1:.0f}s, {alive}/{len(procs)} workers alive")
            starved += time.time() - t1
            if isinstance(item[0], str):
                raise RuntimeError("worker failed:\n" + item[1])
            buf_X = item[0] if buf_X is None or buf_X.shape[0] == 0 else np.concatenate([buf_X, item[0]])
            buf_y = item[1] if buf_y is None or buf_y.shape[0] == 0 else np.concatenate([buf_y, item[1]])
        Xb_, yb_ = buf_X[:a.batch], buf_y[:a.batch]
        buf_X, buf_y = buf_X[a.batch:], buf_y[a.batch:]
        params, ostate, acc = step(params, ostate, jnp.asarray(Xb_), jnp.asarray(yb_), acc)
        n_acc += 1
        if it in eval_at:
            log_point(it, acc, n_acc)
            acc = jnp.zeros((2, S)); n_acc = 0
            if len(curve["step"]) % 10 == 0 or it == n_steps:
                out["throughput_samples_per_s"] = it * a.batch / (time.time() - t0)
                out["starved_frac"] = starved / (time.time() - t0)
                dump()
            print(f"  [{split}] step {it:8d} samples {it * a.batch:.3e}  train {np.mean(curve['train_err'][-1]):.3e} "
                  f"val {np.mean(curve['val_err'][-1]):.3e} (loss {np.mean(curve['val_loss'][-1]):.3e})  "
                  f"{it * a.batch / (time.time() - t0):.0f} samples/s  starved {100 * starved / (time.time() - t0):.0f}%  "
                  f"workers alive {sum(p.is_alive() for p in procs)}/{len(procs)}",
                  flush=True)
    stop.set()
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(10)
    q.cancel_join_thread()
    q.close()
    out["complete"] = True
    out["wall_s"] = time.time() - t_all
    dump()
    print(f"# wrote {path}  ({out['wall_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["head", "ed"], required=True)
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--splits", default="random,pattern")
    ap.add_argument("--samples", type=int, default=400_000_000, help="fresh samples per split")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=200000)
    ap.add_argument("--k_max", type=int, default=-1)
    ap.add_argument("--evals", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 4))
    ap.add_argument("--chunk_batches", type=int, default=8)
    ap.add_argument("--out_dir", default="results/signonline")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--make_psi", action="store_true", help="internal: ED subprocess")
    a = ap.parse_args()

    if a.make_psi:
        cum_path, sign_path = ed_cache_paths(a)
        return make_psi(a.Lx, a.Ly, a.hx, a.hz, cum_path, sign_path)

    cfg = dict(source=a.source, Lx=a.Lx, Ly=a.Ly, hx=a.hx, hz=a.hz, k_max=a.k_max, seed=a.seed)
    from model.honeycomb_geometry import HoneycombGeometry
    cfg["F"] = HoneycombGeometry(a.Lx, a.Ly).n_plaqs
    if a.source == "head":
        cfg["src"] = f"head_k{2 * cfg['F'] if a.k_max < 0 else a.k_max}"
    else:
        cfg["src"] = f"ed_hx{a.hx:g}_hz{a.hz:g}"
        cum_path, sign_path = ed_cache_paths(a)
        if not (os.path.exists(cum_path) and os.path.exists(sign_path)):
            cmd = [sys.executable, os.path.abspath(__file__), "--make_psi", "--source", "ed",
                   "--Lx", str(a.Lx), "--Ly", str(a.Ly), "--hx", str(a.hx), "--hz", str(a.hz)]
            if a.cache_dir:
                cmd += ["--cache_dir", a.cache_dir]
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
            subprocess.run(cmd, check=True, env=env)        # ED keeps its multithreaded BLAS
        cfg["cum_path"], cfg["sign_path"] = cum_path, sign_path
    print(f"# online sign learning: {cfg['src']} {a.Lx}x{a.Ly} splits={a.splits} samples/split={a.samples:.3e} "
          f"batch={a.batch} lr={a.lr} h={a.hidden}d{a.depth} seeds={a.seeds} workers={a.workers}", flush=True)
    for split in a.splits.split(","):
        train_split(a, cfg, split)


if __name__ == "__main__":
    main()
