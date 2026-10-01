"""High-capacity architectures on the x-only sign problem: can a much bigger / structured
network learn the doubled-semion head sign from a stream of fresh samples?

Task (identical to scripts/sign_learn_online.py, x-only): input = the F hexagon-flip bits x,
label = (-1)^poly(x), poly = Levin-Gu cubic (sum x + sum_{edge pairs} x x + sum_{vertex triples}
x x x), exact on the closed-loop sector. x is uniform over {0,1}^F, so samples are generated ON
THE GPU (no decoder, no CPU bottleneck) and any lattice size is reachable.

Held-out validation: a pattern is held out iff hash(x) % 5 == 0 (hash fixed for the run). The
training stream rejects held-out patterns, so validation patterns are never trained on.

Architectures (float32, TF32 matmuls):
  mlp  residual MLP:   Dense -> n x [h + gelu(Dense(LN(h)))] -> LN -> Dense(1)
  cnn  hexagonal CNN:  7-point stencil conv with DIRECTION-SPECIFIC weights (self + E,W,NE,NW,SE,SW
                       neighbours from the geometry, zero padding at the open boundary), residual,
                       mean-pool over hexagons, small MLP head
  tf   transformer:    one token per hexagon (bit embedding + learned position embedding),
                       pre-LN encoder layers, mean-pool, MLP head

Output: <out_dir>/signarch_hc<Lx>x<Ly>_<arch>.json (curve of window-mean train and validation
loss/error against fresh samples; dumped every few eval points; `complete` flag at the end).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = [(2, 0), (-2, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]      # E W NE NW SE SW in (hx, hy)


def lattice(Lx, Ly):
    """pairs (P,2), triples (T,3), nbr (F,6) with -1 for missing neighbours."""
    from model.honeycomb_geometry import HoneycombGeometry
    from model.sign_head import QECSignHead

    g = HoneycombGeometry(Lx, Ly)
    head = QECSignHead(g)
    F = g.n_plaqs
    pos = {tuple(c): i for i, c in enumerate(g.hex_coords)}
    nbr = -np.ones((F, 6), dtype=np.int64)
    for p, (hx, hy) in enumerate(g.hex_coords):
        for d, (dx, dy) in enumerate(DIRS):
            nbr[p, d] = pos.get((hx + dx, hy + dy), -1)
    adj = {tuple(sorted((p, int(q)))) for p in range(F) for q in nbr[p] if q >= 0}
    assert adj == {tuple(sorted(map(int, r))) for r in head._pairs}, "coordinate adjacency != polynomial pairs"
    return F, head, head._pairs.astype(np.int32), head._triples.astype(np.int32), nbr.astype(np.int32)


def build_model(arch, F, nbr, a):
    import flax.linen as nn
    import jax.numpy as jnp

    class ResMLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            h = nn.Dense(a.width)(2.0 * x - 1.0)
            for _ in range(a.layers):
                h = h + nn.gelu(nn.Dense(a.width)(nn.LayerNorm()(h)))
            return nn.Dense(1)(nn.LayerNorm()(h))[..., 0]

    class HexCNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            B = x.shape[0]
            idx = jnp.where(jnp.asarray(nbr) < 0, F, jnp.asarray(nbr))              # (F,6), F -> zero row
            h = nn.Dense(a.width)((2.0 * x - 1.0)[..., None])                        # (B,F,C)
            for _ in range(a.layers):
                u = nn.LayerNorm()(h)
                up = jnp.concatenate([u, jnp.zeros((B, 1, a.width), u.dtype)], axis=1)
                z = jnp.concatenate([u[:, :, None, :], up[:, idx, :]], axis=2).reshape(B, F, 7 * a.width)
                h = h + nn.gelu(nn.Dense(a.width)(z))
            h = nn.LayerNorm()(h).mean(axis=1)
            return nn.Dense(1)(nn.gelu(nn.Dense(a.width)(h)))[..., 0]

    class Transformer(nn.Module):
        @nn.compact
        def __call__(self, x):
            pos = self.param("pos", nn.initializers.normal(0.02), (F, a.width))
            h = nn.Dense(a.width)((2.0 * x - 1.0)[..., None]) + pos
            for _ in range(a.layers):
                u = nn.LayerNorm()(h)
                h = h + nn.MultiHeadDotProductAttention(num_heads=a.heads, qkv_features=a.width)(u, u)
                u = nn.LayerNorm()(h)
                h = h + nn.Dense(a.width)(nn.gelu(nn.Dense(4 * a.width)(u)))
            h = nn.LayerNorm()(h).mean(axis=1)
            return nn.Dense(1)(nn.gelu(nn.Dense(a.width)(h)))[..., 0]

    return {"mlp": ResMLP, "cnn": HexCNN, "tf": Transformer}[arch]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "cnn", "tf"], required=True)
    ap.add_argument("--Lx", type=int, required=True)
    ap.add_argument("--Ly", type=int, required=True)
    ap.add_argument("--width", type=int, default=0, help="0 => arch default (mlp 512, cnn 128, tf 192)")
    ap.add_argument("--layers", type=int, default=0, help="0 => arch default (mlp 12, cnn 12, tf 10)")
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=0, help="0 => by arch and F (memory-safe table)")
    ap.add_argument("--lr", type=float, default=0.0, help="0 => arch default (mlp/cnn 5e-4, tf 3e-4)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1_000_000_000)
    ap.add_argument("--max_minutes", type=float, default=0.0, help="stop gracefully after this wall time")
    ap.add_argument("--n_val", type=int, default=200000)
    ap.add_argument("--evals", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--converged_evals", type=int, default=6,
                    help="stop early once val err < 1e-6 and train err < 1e-5 at this many consecutive evals (0 = never)")
    ap.add_argument("--out_dir", default="results/signarch")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.width = a.width or {"mlp": 512, "cnn": 128, "tf": 192}[a.arch]
    a.layers = a.layers or {"mlp": 12, "cnn": 12, "tf": 10}[a.arch]
    assert a.width % a.heads == 0 or a.arch != "tf", "--width must be divisible by --heads"
    a.lr = a.lr or {"mlp": 5e-4, "cnn": 5e-4, "tf": 3e-4}[a.arch]

    F, head, pairs, triples, nbr = lattice(a.Lx, a.Ly)
    if a.batch == 0:                      # A100-40GB-safe defaults (reviewed activation estimates)
        a.batch = {"mlp": 4096, "cnn": 4096 if F <= 64 else 2048,
                   "tf": 4096 if F <= 36 else (2048 if F <= 64 else 1024)}[a.arch]

    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    p0, p1 = jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    t0, t1, t2 = jnp.asarray(triples[:, 0]), jnp.asarray(triples[:, 1]), jnp.asarray(triples[:, 2])
    rng_np = np.random.default_rng(12345)
    R = jnp.asarray(rng_np.integers(1, 2 ** 32, size=F, dtype=np.uint64).astype(np.uint32) | np.uint32(1))

    def label(x):                                    # x (n,F) int32 {0,1} -> y (n,) float32 +-1
        tot = x.sum(axis=1)
        tot = tot + (x[:, p0] * x[:, p1]).sum(axis=1) + (x[:, t0] * x[:, t1] * x[:, t2]).sum(axis=1)
        return (1 - 2 * (tot & 1)).astype(jnp.float32)

    def held(x):                                     # pattern hash -> held-out iff % 5 == 0
        h = (x.astype(jnp.uint32) * R[None, :]).sum(axis=1, dtype=jnp.uint32)
        h = h ^ (h >> 16); h = h * jnp.uint32(0x85EBCA6B)
        h = h ^ (h >> 13); h = h * jnp.uint32(0xC2B2AE35)
        h = h ^ (h >> 16)
        return (h % jnp.uint32(5)) == 0

    B = a.batch
    DRAW = int(1.5 * B) + 64

    @jax.jit
    def train_batch(key):
        x = jax.random.bernoulli(key, 0.5, (DRAW, F)).astype(jnp.int32)
        idx = jnp.nonzero(~held(x), size=B, fill_value=0)[0]
        x = x[idx]
        return x.astype(jnp.float32), label(x)

    @jax.jit
    def val_draw(key):
        x = jax.random.bernoulli(key, 0.5, (65536, F)).astype(jnp.int32)
        return x, held(x)

    # ---- label correctness against the CPU head polynomial; held fraction; validation set
    xs = np.random.default_rng(0).integers(0, 2, size=(20000, F)).astype(np.int32)
    assert (np.asarray(label(jnp.asarray(xs))) == 1 - 2 * head.poly_sign01(xs.astype(np.uint8)).astype(np.int64)).all(), \
        "GPU label != head.poly_sign01"
    hf = float(np.asarray(held(jnp.asarray(xs))).mean())
    assert (1 - hf) * DRAW > B + 8 * np.sqrt(DRAW * hf * (1 - hf)), \
        f"held fraction {hf:.3f} too large for the fixed-size rejection draw (F={F} too small?)"
    vx, vk = [], jax.random.PRNGKey(7)
    while sum(v.shape[0] for v in vx) < a.n_val:
        vk, k = jax.random.split(vk)
        x, m = val_draw(k)
        vx.append(np.asarray(x)[np.asarray(m)])
    Xv = jnp.asarray(np.concatenate(vx)[: a.n_val].astype(np.float32))
    yv = label(Xv.astype(jnp.int32))
    tb_x, tb_y = train_batch(jax.random.PRNGKey(99))
    const = 1.0 if float((tb_y > 0).mean()) >= 0.5 else -1.0
    refs = dict(trainmajority_val_err=float((yv != const).mean()), val_neg_frac=float((yv < 0).mean()),
                train_neg_frac=float((tb_y < 0).mean()), held_fraction=hf)
    print(f"# {a.arch} {a.Lx}x{a.Ly} F={F} width={a.width} layers={a.layers} batch={B} lr={a.lr} | held fraction {hf:.3f}, "
          f"val {Xv.shape[0]} rows, const-guess val err {refs['trainmajority_val_err']:.4f}", flush=True)

    model = build_model(a.arch, F, nbr, a)
    params = model.init(jax.random.PRNGKey(a.seed), jnp.zeros((2, F), jnp.float32))["params"]
    n_params = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))
    print(f"# parameters: {n_params:,}", flush=True)
    sched = optax.linear_schedule(0.0, a.lr, a.warmup)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched if a.warmup > 0 else a.lr))
    ostate = opt.init(params)

    def stats(p, x, y):
        m = model.apply({"params": p}, x)
        return optax.sigmoid_binary_cross_entropy(m, (y > 0).astype(jnp.float32)), (jnp.sign(m) != y)

    @jax.jit
    def step(p, s, key, acc):
        x, y = train_batch(key)

        def lf(q):
            l, e = stats(q, x, y)
            return l.mean(), (l.mean(), e.mean())

        (_, (lm, em)), g = jax.value_and_grad(lf, has_aux=True)(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s, acc + jnp.stack([lm, em])

    @jax.jit
    def evaluate(p, x, y):
        l, e = stats(p, x, y)
        return l.mean(), e.mean()

    CH = 8192

    def val_metrics(p):
        ls, es, n = 0.0, 0.0, Xv.shape[0]
        for lo in range(0, n, CH):
            l, e = evaluate(p, Xv[lo:lo + CH], yv[lo:lo + CH])
            w = min(CH, n - lo) / n
            ls += float(l) * w; es += float(e) * w
        return ls, es

    n_steps = a.samples // B
    eval_at = set(np.unique(np.round(np.geomspace(1, n_steps, a.evals)).astype(int)).tolist())
    curve = {k: [] for k in ("step", "samples", "train_loss", "train_err", "val_loss", "val_err")}
    out = dict(cfg=dict(source="xsyn", Lx=a.Lx, Ly=a.Ly, arch=a.arch, F=F, seed=a.seed), arch=a.arch, F=F,
               width=a.width, layers=a.layers, heads=a.heads, batch=B, lr=a.lr, n_params=n_params, refs=refs,
               curve=curve, budget_samples=int(n_steps * B), complete=False, stopped="")
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"signarch_hc{a.Lx}x{a.Ly}_{a.arch}{a.tag}.json")

    def dump():
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)

    def log_point(it, acc, n_acc):
        vl, ve = val_metrics(params)
        tl = (np.asarray(acc) / n_acc) if n_acc else np.array([np.nan, np.nan])
        curve["step"].append(it); curve["samples"].append(it * B)
        curve["train_loss"].append([float(tl[0])]); curve["train_err"].append([float(tl[1])])
        curve["val_loss"].append([vl]); curve["val_err"].append([ve])

    key = jax.random.PRNGKey(1000 + a.seed)
    acc = jnp.zeros(2); n_acc = 0; conv = 0
    log_point(0, acc, 0)
    t0w = time.time(); t_last = t0w; last_print = (t0w, 0)
    EVAL_SEC = 240
    for it in range(1, n_steps + 1):
        key, k = jax.random.split(key)
        params, ostate, acc = step(params, ostate, k, acc)
        n_acc += 1
        due = it in eval_at
        if not due and it % 64 == 0:                 # time-based trigger (also enforces --max_minutes)
            acc.block_until_ready()
            now = time.time()
            due = (now - t_last > EVAL_SEC) or bool(a.max_minutes and now - t0w > 60 * a.max_minutes)
        if due:
            log_point(it, acc, n_acc); acc = jnp.zeros(2); n_acc = 0; t_last = time.time()
            ve, te = curve["val_err"][-1][0], curve["train_err"][-1][0]
            if not np.isfinite(curve["val_loss"][-1][0]):
                out["stopped"] = "nan"; break
            conv = conv + 1 if (ve < 1e-6 and te < 1e-5) else 0
            el = time.time() - t0w
            dump()
            rate = (it - last_print[1]) * B / max(t_last - last_print[0], 1e-9); last_print = (t_last, it)
            print(f"  step {it:9d} samples {it * B:.3e}  train {te:.3e} val {ve:.3e} (loss {curve['val_loss'][-1][0]:.3e})  "
                  f"{rate:.0f} samples/s  {el / 60:.1f} min", flush=True)
            if a.max_minutes and el > 60 * a.max_minutes:
                out["stopped"] = "max_minutes"; break
            if a.converged_evals and conv >= a.converged_evals:
                out["stopped"] = "converged"; break
    out["complete"] = True
    out["wall_s"] = time.time() - t0w
    dump()
    print(f"# wrote {path} ({out['wall_s']:.0f}s, stopped: {out['stopped'] or 'budget'})", flush=True)


if __name__ == "__main__":
    main()
