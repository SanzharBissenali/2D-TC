"""Numpy-only validation of the Variant-3 plaquette transformer (no jax/flax/netket).

Checks, mirroring model/plaquette_transformer.py + model/geometry.py at L=4 OBC:
  1. geometry: N=24 edges, 9 plaquettes, every plaquette has 4 valid edges;
  2. tokenization t_p = prod of the 4 plaquette edge spins == brute-force B_p;
  3. EXACT A_v invariance of the token vector under every vertex-star flip
     (bulk 4-edge and boundary 2/3-edge stars) => log psi invariance downstream;
  4. displacement table: values in [0, (2L-3)^2), exact signed decode (no wrap),
     zero-displacement diagonal, antisymmetry D[i,j] <-> D[j,i];
  5. closed-form parameter count for the default arm (d=32, h=4, 4 layers,
     ffn_mult=2, K=d, L=4) == shape-enumerated count;
  6. gated attention reference: alpha=0 => attention weights are input-INDEPENDENT
     and row-stochastic; alpha != 0 => input-dependent.

Run: python scratchpad/validate_v3.py
"""
import numpy as np

Lx = Ly = 4


# ---- geometry (faithful numpy replica of model/geometry.py, OBC) ----------------
def build_geometry(Lx, Ly):
    lattice = np.array([[n // Ly, n % Ly] for n in range(Lx * Ly)], dtype=float)
    atoms = np.concatenate([lattice + np.array([0.5, 0.0]),
                            lattice + np.array([0.0, 0.5])])
    atoms = atoms[np.lexsort((atoms[:, 1], atoms[:, 0]))]
    keep = ((atoms[:, 0] >= 0) & (atoms[:, 0] <= Lx - 1) &
            (atoms[:, 1] >= 0) & (atoms[:, 1] <= Ly - 1))
    arr_coord = atoms[keep]

    def to1d(coord):
        hits = np.where((arr_coord[:, 0] == coord[0]) & (arr_coord[:, 1] == coord[1]))[0]
        return int(hits[0]) if len(hits) else -1

    def stars(centers):
        offs = [np.array([0.5, 0]), np.array([0, 0.5]),
                np.array([-0.5, 0]), np.array([0, -0.5])]
        return [[to1d(c + o) for o in offs] for c in centers]

    plaq_centers = [np.array([r + 0.5, c + 0.5]) for r in range(Lx - 1) for c in range(Ly - 1)]
    vert_centers = [np.array([float(i), float(j)]) for i in range(Lx) for j in range(Ly)]
    return arr_coord, stars(plaq_centers), stars(vert_centers), np.array(plaq_centers)


arr_coord, plaq_all, vertex_all, plaq_pos = build_geometry(Lx, Ly)
N = len(arr_coord)
assert N == 2 * Lx * Ly - Lx - Ly == 24, N
assert len(plaq_all) == (Lx - 1) * (Ly - 1) == 9
assert all(all(e >= 0 for e in p) for p in plaq_all), "plaquette with missing edge"
sizes = sorted(sum(e >= 0 for e in v) for v in vertex_all)
assert sizes[0] >= 2 and sizes[-1] == 4, sizes
print(f"[1] geometry OK: N={N}, n_plaq={len(plaq_all)}, all plaquettes 4-edged; "
      f"vertex stars {sizes.count(4)}x4 + {len(sizes)-sizes.count(4)}x2/3 edges")

# ---- tokenization + exact A_v invariance ----------------------------------------
rng = np.random.default_rng(0)
P = np.array(plaq_all)
for _ in range(50):
    s = rng.choice([-1.0, 1.0], size=N)
    t = np.prod(s[P], axis=-1)                       # module's fancy-index tokenization
    t_brute = np.array([np.prod([s[e] for e in p]) for p in plaq_all])
    assert np.array_equal(t, t_brute)
    assert set(np.unique(t)) <= {-1.0, 1.0}
    for star in vertex_all:                          # every A_v, incl. boundary stars
        sf = s.copy()
        sf[[e for e in star if e >= 0]] *= -1
        assert np.array_equal(np.prod(sf[P], axis=-1), t), "A_v changed a token!"
print("[2/3] tokenization == brute-force B_p; tokens EXACTLY A_v-invariant "
      f"(50 configs x {len(vertex_all)} stars)")

# ---- displacement table (replica of plaquette_displacement_table) ---------------
coords = np.floor(plaq_pos).astype(np.int64)
span_x, span_y = 2 * Lx - 3, 2 * Ly - 3
dx = coords[:, 0][None, :] - coords[:, 0][:, None] + (Lx - 2)
dy = coords[:, 1][None, :] - coords[:, 1][:, None] + (Ly - 2)
D = dx * span_y + dy
n_disp = span_x * span_y
assert D.min() >= 0 and D.max() < n_disp == (2 * Lx - 3) ** 2 == 25
dec_dx, dec_dy = D // span_y - (Lx - 2), D % span_y - (Ly - 2)   # exact decode, no wrap
assert np.array_equal(dec_dx, coords[:, 0][None, :] - coords[:, 0][:, None])
assert np.array_equal(dec_dy, coords[:, 1][None, :] - coords[:, 1][:, None])
assert np.all(np.diag(dec_dx) == 0) and np.all(np.diag(dec_dy) == 0)
assert np.array_equal(dec_dx, -dec_dx.T) and np.array_equal(dec_dy, -dec_dy.T)
print(f"[4] displacement table OK: range [0,{n_disp}), signed decode exact (no wrap), "
      "antisymmetric")

# ---- parameter count (default arm: d=32, h=4, 4 layers, m=2, K=d) ---------------
d, h, n_layers, m = 32, 4, 4, 2
K = d
shapes = [("embed", (2, d))]
for l in range(n_layers):
    shapes += [(f"block{l}.norm1", (d,)), (f"block{l}.norm2", (d,)),
               (f"block{l}.Wq", (d, d)), (f"block{l}.Wk", (d, d)),
               (f"block{l}.Wv", (d, d)), (f"block{l}.Wo", (d, d)),
               (f"block{l}.b_pos", (h, n_disp)), (f"block{l}.alpha", (h,)),
               (f"block{l}.ffn0", (d, m * d)), (f"block{l}.ffn1", (m * d, d))]
shapes += [("out_norm", (d,)), ("head", (d, K))]
n_params = sum(int(np.prod(s)) for _, s in shapes)
closed = 2 * d + n_layers * (2 * d + 4 * d * d + h * n_disp + h + 2 * m * d * d) + d + d * K
assert n_params == closed == 34560, (n_params, closed)
print(f"[5] param count OK: {n_params} (embed 64 + 4x8360 + head 1056)")

# ---- gated attention reference ---------------------------------------------------
def gated_attention(x, Wq, Wk, b_pos, alpha, D):
    hh, d_head = b_pos.shape[0], x.shape[1] // b_pos.shape[0]
    n = x.shape[0]
    q = (x @ Wq).reshape(n, hh, d_head).transpose(1, 0, 2)
    k = (x @ Wk).reshape(n, hh, d_head).transpose(1, 0, 2)
    logits = b_pos[:, D] + alpha[:, None, None] * np.einsum("hid,hjd->hij", q, k) / np.sqrt(d_head)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

n = len(plaq_all)
Wq, Wk = rng.normal(size=(d, d)), rng.normal(size=(d, d))
b_pos = rng.normal(size=(h, n_disp))
x1, x2 = rng.normal(size=(n, d)), rng.normal(size=(n, d))
A1 = gated_attention(x1, Wq, Wk, b_pos, np.zeros(h), D)
A2 = gated_attention(x2, Wq, Wk, b_pos, np.zeros(h), D)
assert np.allclose(A1, A2), "alpha=0 must make attention input-independent"
assert np.allclose(A1.sum(axis=-1), 1.0)
A1g = gated_attention(x1, Wq, Wk, b_pos, 0.5 * np.ones(h), D)
A2g = gated_attention(x2, Wq, Wk, b_pos, 0.5 * np.ones(h), D)
assert not np.allclose(A1g, A2g), "alpha!=0 must make attention input-dependent"
assert A1.shape == (h, n, n)
print("[6] gated attention OK: alpha=0 => input-independent row-stochastic; "
      "alpha!=0 => input-dependent")

print("\nALL VARIANT-3 CHECKS PASSED")
