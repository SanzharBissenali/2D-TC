"""Numpy-only validation of the dual-basis star-Wilson Combo CNN (no jax/flax/netket).

Checks, mirroring model/networks.py + model/hamiltonian.py + model/geometry.py:
  1. geometry at L=4 OBC: N=24 edges, 16 vertex stars = 4 bulk (4-link) + 8 edge
     (3-link) + 4 corner (2-link), -1 sentinels, total star-link incidences = 2N;
  2. masked star-Wilson product (the _Wilson_4spin_plaq -1 branch) == brute-force
     ragged product, on +-1 spin configs AND on float feature arrays (the Wilson
     input is Block-1 features, not raw spins), multi-channel;
  3. EXACT B_p invariance: all 16 star products unchanged under each of the 9
     plaquette flips => log psi invariance of the dual net at identity init;
  4. vertex-grid invariant-CNN kernel table (KernelManager dual build): shape
     (Lx*Ly, Lx*Ly), correct -1 pad counts per shift, exact position decode;
  5. parameter counts: primal Combo-small total 1,681 (Block-1 448 + Block-3
     1,233) and dual total 2,633 (Block-3 star grid 2,185), by shape enumeration
     against closed forms;
  6. L=2 dense-matrix check: the dual Hamiltonian built by the sigma_x <-> sigma_z
     swap equals W H W with W = Hadamard^(x)N, and has an identical spectrum, at
     (hx,hz) = (0.2, 0.0) and (0.1, 0.1).

Run: python scratchpad/validate_dual.py
"""
import numpy as np


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
    return arr_coord, stars(plaq_centers), stars(vert_centers), np.array(vert_centers)


Lx = Ly = 4
arr_coord, plaq_all, vertex_all, vert_pos = build_geometry(Lx, Ly)
N = len(arr_coord)
assert N == 2 * Lx * Ly - Lx - Ly == 24, N
arities = [sum(e >= 0 for e in v) for v in vertex_all]
assert sorted(arities).count(4) == 4 and sorted(arities).count(3) == 8 \
    and sorted(arities).count(2) == 4, arities
assert sum(arities) == 2 * N, "each link must touch exactly 2 stars"
assert all(all(e >= 0 for e in p) for p in plaq_all)
print(f"[1] geometry OK: N={N}, 16 stars = 4x4-link + 8x3-link + 4x2-link, "
      f"incidences {sum(arities)} = 2N")

# ---- masked star-Wilson == brute-force ragged product ----------------------------
VA = np.array(vertex_all)                    # (16, 4) with -1 sentinels
idx = np.where(VA == -1, 0, VA)
mask = (VA != -1)
rng = np.random.default_rng(0)


def masked_wilson(x):                        # x: (C, N) -- replica of the -1 branch
    return np.prod(np.where(mask, x[:, idx], 1.0), axis=-1)


for trial in range(50):
    s = rng.choice([-1.0, 1.0], size=(1, N))          # raw spins (identity-init case)
    f = rng.normal(size=(16, N))                      # Block-1 features, 16 channels
    for x in (s, f):
        got = masked_wilson(x)
        brute = np.array([[np.prod([xc[e] for e in star if e >= 0])
                           for star in vertex_all] for xc in x])
        assert np.allclose(got, brute), "masked star-Wilson != ragged product"
    assert set(np.unique(masked_wilson(s))) <= {-1.0, 1.0}
print("[2] masked star-Wilson == brute-force ragged product "
      "(50 trials, spins + 16-channel float features)")

# ---- exact B_p invariance of star products ---------------------------------------
for _ in range(50):
    s = rng.choice([-1.0, 1.0], size=(1, N))
    t = masked_wilson(s)
    for p in plaq_all:                                # every B_p (all 4-valid)
        sf = s.copy()
        sf[:, p] *= -1
        assert np.array_equal(masked_wilson(sf), t), "B_p flip changed a star product!"
print(f"[3] star products EXACTLY B_p-invariant (50 configs x {len(plaq_all)} plaquettes)")

# ---- vertex-grid invariant-CNN kernel table (KernelManager dual build) -----------
def to1d_vert(coord):
    hits = np.where((vert_pos[:, 0] == coord[0]) & (vert_pos[:, 1] == coord[1]))[0]
    return int(hits[0]) if len(hits) else -1


table = []
for a in range(Lx):
    for b in range(Lx):
        table.append([to1d_vert(vert_pos[j] + np.array([a, b])) for j in range(Lx * Lx)])
table = np.array(table)
assert table.shape == (Lx * Ly, Lx * Ly)
for r, (a, b) in enumerate((a, b) for a in range(Lx) for b in range(Lx)):
    n_valid = (Lx - a) * (Ly - b)                     # shifted positions still on-grid
    assert (table[r] != -1).sum() == n_valid, (r, a, b)
    for j in range(Lx * Lx):                          # exact decode of every entry
        tgt = vert_pos[j] + np.array([a, b])
        expect = to1d_vert(tgt)
        assert table[r, j] == expect
assert (table[0] == np.arange(Lx * Lx)).all()         # zero shift = identity row
print(f"[4] vertex kernel table OK: shape {table.shape}, per-shift -1 pads exact, "
      "entry decode exact, zero-shift row = identity")

# ---- parameter counts -------------------------------------------------------------
def recursive_func(n):
    return 0 if n == 0 else 4 * n + recursive_func(n - 1)


k_edge = 1 + recursive_func(2)                        # non-inv kernel width (kernel_size=2)
block1 = 2 * (16 * 1 * k_edge) + 2 * 16               # Wconv_hor/vert + biases
taps_p, taps_v = (Lx - 1) ** 2, Lx ** 2               # global kernels: 9 vs 16 taps
ch = [16, 8, 1]


def block3(taps):
    return sum(co * ci * taps + co for ci, co in zip(ch, ch[1:]))


primal, dual = block1 + block3(taps_p), block1 + block3(taps_v)
assert block1 == 448 and block3(taps_p) == 1233 and primal == 1681, (block1, primal)
assert block3(taps_v) == 2185 and dual == 2633, dual
print(f"[5] param counts OK: primal {primal} (448 + 1233), dual {dual} (448 + 2185); "
      f"delta = {dual - primal} from 16-tap vs 9-tap global kernels")

# ---- L=2 dense kron: H_dual == W H W, identical spectra ---------------------------
I2 = np.eye(2)
SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]])
HAD = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)

arr2, plaq2, vert2, _ = build_geometry(2, 2)
N2 = len(arr2)
assert N2 == 4 and len(plaq2) == 1 and len(vert2) == 4


def op_on(sites_ops, n):
    mats = [I2] * n
    for site, op in sites_ops:
        mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_H(hx, hz, swap):                            # swap=True mirrors dual_basis
    sx, sz = (SZ, SX) if swap else (SX, SZ)
    H = np.zeros((2 ** N2, 2 ** N2))
    for v in vert2:
        H -= op_on([(e, sx) for e in v if e >= 0], N2)
    for p in plaq2:
        H -= op_on([(e, sz) for e in p], N2)
    for j in range(N2):
        H -= hz * op_on([(j, sz)], N2) + hx * op_on([(j, sx)], N2)
    return H


W = op_on([(j, HAD) for j in range(N2)], N2)
for hx, hz in [(0.2, 0.0), (0.1, 0.1)]:
    H = build_H(hx, hz, swap=False)
    Hd = build_H(hx, hz, swap=True)
    assert np.allclose(Hd, W @ H @ W), f"H_dual != W H W at (hx,hz)=({hx},{hz})"
    assert np.allclose(np.linalg.eigvalsh(Hd), np.linalg.eigvalsh(H))
print("[6] L=2 dense check OK: swapped-constructor H == Hadamard-conjugated H, "
      "identical spectra at (0.2,0.0) and (0.1,0.1)")

print("\nALL DUAL-BASIS CHECKS PASSED")
