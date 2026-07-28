"""Numpy-only validation of Variant 1 (gauge-combo transformer) init math. No jax/flax.

Checks at L=4 OBC, mirroring model/gauge_combo_transformer.py:
  1. AT INIT (chi blocks = identity by zero-init), the Wilson tokens factorize per
     channel as  t_{p,c} = B_p(s) * const_{p,c}  with const = prod tanh(w[orient]);
  2. the tokens (hence log psi) are EXACTLY A_v-invariant under every vertex star;
  3. orientation-resolved edge displacement table: n_disp == 16 L^2 - 32 L + 14;
  4. closed-form parameter count for the default arm (chi C=8/h=2/nl1=1;
     Omega d=16/h=4/nl=4; L=4).

Run: python scratchpad/validate_v1gc.py
"""
import numpy as np

Lx = Ly = 4

# ---- geometry (same replica as validate_v3.py) -----------------------------------
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
    return arr_coord, stars(plaq_centers), stars(vert_centers)


arr_coord, plaq_all, vertex_all = build_geometry(Lx, Ly)
N, P = len(arr_coord), np.array(plaq_all)
assert N == 24 and all(all(e >= 0 for e in p) for p in plaq_all)

# ---- [1]+[2] init factorization + exact A_v invariance ---------------------------
# orient: y half-integer => vertical (same rule as edge_displacement_table)
c2 = np.round(arr_coord * 2).astype(np.int64)
orient = (c2[:, 1] % 2).astype(np.int64)
rng = np.random.default_rng(0)
C = 8
w = rng.normal(size=(2, C))

def tokens(s):
    x = s[:, None] * w[orient]              # odd embedding; chi blocks == identity at init
    return np.prod(np.tanh(x)[P], axis=1)   # fixed Wilson step -> (n_plaq, C)

for _ in range(50):
    s = rng.choice([-1.0, 1.0], size=N)
    t = tokens(s)
    Bp = np.prod(s[P], axis=-1)                                  # (n_plaq,)
    const = np.prod(np.tanh(w[orient])[P], axis=1)               # (n_plaq, C)
    assert np.allclose(t, Bp[:, None] * const), "init factorization broken"
    for star in vertex_all:                                      # every A_v
        sf = s.copy()
        sf[[e for e in star if e >= 0]] *= -1
        assert np.allclose(tokens(sf), t), "A_v changed a Wilson token at init!"
print(f"[1/2] init tokens == B_p x const per channel; EXACTLY A_v-invariant "
      f"(50 configs x {len(vertex_all)} stars)")

# ---- [3] edge displacement table (replica of edge_displacement_table) ------------
lut = {}
for i in range(N):
    for j in range(N):
        k = (int(orient[i]), int(orient[j]),
             int(c2[i, 0] - c2[j, 0]), int(c2[i, 1] - c2[j, 1]))
        lut.setdefault(k, len(lut))
n_disp_e = len(lut)
assert n_disp_e == 16 * Lx**2 - 32 * Lx + 14 == 142, n_disp_e
print(f"[3] edge displacement table OK: n_disp_e = {n_disp_e} = 16L^2-32L+14")

# ---- [4] parameter count (default arm) --------------------------------------------
def block_params(d, h, m, n_disp):
    return 2 * d + 4 * d * d + h * n_disp + h + 2 * m * d * d   # norms+QKVO+b_pos+alpha+FFN

Cc, hc, nl1 = 8, 2, 1          # chi
d, h, nl, m = 16, 4, 4, 2      # Omega (the Variant-3 sweep winner)
n_disp_p = (2 * Lx - 3) ** 2
K = d
total = (2 * Cc                                   # embed_w
         + nl1 * block_params(Cc, hc, m, n_disp_e)
         + Cc * d                                 # proj C->d
         + nl * block_params(d, h, m, n_disp_p)
         + d + d * K)                             # out_norm + head
assert total == 9966, total
print(f"[4] param count OK: {total} (chi {nl1 * block_params(Cc, hc, m, n_disp_e)} "
      f"+ Omega {nl * block_params(d, h, m, n_disp_p)} + embed/proj/head)")

print("\nALL VARIANT-1 CHECKS PASSED")
