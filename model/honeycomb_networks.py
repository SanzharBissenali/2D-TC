"""Honeycomb (brick-wall) networks for the Levin-Gu toric-code / doubled-semion
study: the approximately-symmetric Combo and the unconstrained PlainCNN baseline,
both built from geometry tables (model/honeycomb_geometry.HoneycombGeometry).

Combo = Block-1 local link CNN (non-invariant, identity-init) -> Wilson vertex
tokens (masked products over vertex_all; exactly hexagon-flip invariant AT
IDENTITY-INIT Block-1, where tokens == Q_v -- generic Block-1 parameters
legitimately break it, that is the "approximately symmetric" design; the
invariance is Block-3-parameter-independent though, since at fixed tokens
every hexagon shares 0 or 2 links with every vertex) ->
Block-3 invariant CNN over the vertex-token lattice -> mean = log psi.
PlainCNN = a stack of the same Block-1 primitive, mean over links; no Wilson,
no invariant block, no symmetry (the baseline that must find the GS unaided).

Conventions fixed here (documentation-grade):

Block-1 orientation classes (one kernel each):
  class 0: horizontal link whose LEFT endpoint has (x + y) even
  class 1: horizontal link whose LEFT endpoint has (x + y) odd
  class 2: vertical link
(the honeycomb's three bond orientations in brick coordinates).

Block-1 slot order (5 canonical taps per link, -1 = absent at the boundary):
  slot 0: the link itself
  endpoint-1 := the endpoint with lexicographically smaller (x, y)
                (left endpoint for h-links, bottom endpoint for v-links)
  slots 1-2: the OTHER links at endpoint-1, in vertex_all's fixed
             [left-h, right-h, vertical] slot order, skipping the link itself
  slots 3-4: the same for endpoint-2 (the larger-coordinate endpoint)
No biases in the local convs: Block-1 params = 3 * 5 * C_in * C_out exactly.

Block-3 (option (a), user-locked): FULL all-to-all displacement kernel per
center sublattice (A: vertex (x+y) even, B: odd; honeycomb translations
preserve sublattice). Tap index = rank of (dx, dy) = pos[target] - pos[center]
in the sorted union of displacements realized from that sublattice's centers;
the union covers every (center, target) pair by construction, so the gather
needs no masking. Kernels carry a bias per output channel (square-code parity).
"""

from typing import Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from model.networks import Final, Sequential, WilsonNonlinearity


# --------------------------------------------------------------------------
# static tables (numpy, trace-time constants)
# --------------------------------------------------------------------------

def block1_tables(geometry) -> Tuple[np.ndarray, np.ndarray]:
    """Per-link orientation class (N,) and 5-slot neighbor table (N, 5)."""
    pos = geometry.vertex_positions.astype(int)
    cls = np.empty(geometry.N, dtype=int)
    slots = np.full((geometry.N, 5), -1, dtype=int)
    for l in range(geometry.N):
        a, b = (int(v) for v in geometry.link_endpoints[l])
        if tuple(pos[a]) > tuple(pos[b]):
            a, b = b, a  # endpoint-1 = lexicographically smaller (x, y)
        if geometry.link_type[l] == 1:
            cls[l] = 2
        else:
            cls[l] = (pos[a, 0] + pos[a, 1]) % 2
        slots[l, 0] = l
        for k, v in enumerate((a, b)):
            others = [int(e) for e in geometry.vertex_all[v] if e != -1 and e != l]
            assert len(others) <= 2
            for j, e in enumerate(others):
                slots[l, 1 + 2 * k + j] = e
    return cls, slots


def block3_tables(geometry) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
    """Sublattice (V,), all-to-all tap table (V, V), padded tap count T,
    and the per-sublattice tap counts (T_A, T_B)."""
    pos = geometry.vertex_positions.astype(int)
    V = geometry.n_vertices
    sub = (pos[:, 0] + pos[:, 1]) % 2
    disp_index = []
    for s in (0, 1):
        disps = sorted({(int(pos[u, 0] - pos[v, 0]), int(pos[u, 1] - pos[v, 1]))
                        for v in range(V) if sub[v] == s for u in range(V)})
        disp_index.append({d: i for i, d in enumerate(disps)})
    tap = np.empty((V, V), dtype=int)
    for v in range(V):
        table = disp_index[sub[v]]
        for u in range(V):
            tap[v, u] = table[(int(pos[u, 0] - pos[v, 0]), int(pos[u, 1] - pos[v, 1]))]
    counts = (len(disp_index[0]), len(disp_index[1]))
    return sub, tap, max(counts), counts


def _identity_init_local(key, shape, dtype):
    """Self-tap (slot 0) = 1 for every (class, in, out); all other taps 0.
    With C_in = 1 this makes the layer an exact passthrough of the spins."""
    W = jnp.zeros(shape, dtype)
    return W.at[:, 0, :, :].set(1)


def _scaled_sigmoid(x, dtype):
    """The repo's activation: maps +-1 -> +-1 exactly (real and imag parts)."""
    if dtype == "complex":
        return ((nn.sigmoid(jnp.real(x)) - 1 / 2)
                + 1j * (nn.sigmoid(jnp.imag(x)) - 1 / 2)) * (2 + 2 * jnp.e) / (jnp.e - 1)
    return (nn.sigmoid(x) - 1 / 2) * (2 + 2 * jnp.e) / (jnp.e - 1)


# --------------------------------------------------------------------------
# modules (single-sample, feature-major (nfeat, N) like model/networks.py;
# Sequential vmaps over the batch)
# --------------------------------------------------------------------------

class HoneycombLocalCNN(nn.Module):
    """3-orientation-class, 5-slot masked local conv on links (no biases)."""

    nfeat_in: int
    nfeat_out: int
    link_class: Tuple[int, ...]          # (N,)
    slot_table: Tuple[Tuple[int, ...], ...]  # (N, 5), -1 = absent
    identity_init: bool = True
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        cls = np.asarray(self.link_class)
        slots = np.asarray(self.slot_table)
        N = slots.shape[0]
        init = _identity_init_local if self.identity_init else nn.initializers.normal()
        W = self.param('Wconv', init, (3, 5, self.nfeat_in, self.nfeat_out), self.dtype)

        x = x.reshape(self.nfeat_in, N)
        idx = jnp.asarray(np.where(slots == -1, 0, slots))
        mask = jnp.asarray((slots != -1).astype(float))
        xg = x[:, idx] * mask                     # (in, N, 5)
        Wl = W[cls]                               # (N, 5, in, out)
        x = jnp.einsum('ins,nsio->on', xg, Wl)    # (out, N)
        return _scaled_sigmoid(x, self.dtype)


class HoneycombInvariantCNN(nn.Module):
    """All-to-all displacement conv over the V vertex tokens, per-sublattice
    kernels (option (a)); elu activation (Re/Im split for complex)."""

    nfeat_in: int
    nfeat_out: int
    sublattice: Tuple[int, ...]              # (V,)
    tap_table: Tuple[Tuple[int, ...], ...]   # (V, V) tap index, no -1 by construction
    n_taps: int                              # padded per-sublattice tap count
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        sub = np.asarray(self.sublattice)
        tap = np.asarray(self.tap_table)
        V = tap.shape[0]
        W = self.param('WCNN', nn.initializers.normal(),
                       (2, self.n_taps, self.nfeat_in, self.nfeat_out), self.dtype)
        b = self.param('bCNN', nn.initializers.zeros_init(),
                       (self.nfeat_out,), self.dtype)

        x = x.reshape(self.nfeat_in, V)
        Wg = W[sub[:, None], tap]                 # (V, V, in, out)
        x = jnp.einsum('iu,vuio->ov', x, Wg) + b[:, None]
        if self.dtype == "complex":
            return nn.elu(jnp.real(x)) + 1j * nn.elu(jnp.imag(x))
        return nn.elu(x)


class SignedModel(nn.Module):
    """Phase-4 formulation A (equivalence witness -- production is the
    sign-framed operator, model/sign_frame.py): log psi = log A + 1j*pi*s(x),
    s = QECSignHead loop parity in {0,1}. Real params -> complex output; the
    head is a host callback on the INPUT only, so AD w.r.t. params never
    touches it (zero gradient by construction) and sampling still sees
    |psi| = A through Re(log psi). Wraps the WHOLE batched network (above
    Sequential's internal vmap) so the callback fires once per batch."""

    base: nn.Module
    head_s01: Any            # QECSignHead.s01 -- static (host) callable

    @nn.compact
    def __call__(self, x):
        log_a = self.base(x)
        s = jax.pure_callback(
            lambda xb: np.asarray(self.head_s01(xb), dtype=np.float64),
            jax.ShapeDtypeStruct(x.shape[:-1], jnp.float64),
            x, vmap_method='expand_dims')
        return log_a + 1j * jnp.pi * s


class ResidualSignedModel(nn.Module):
    """Phase-4b arm cnnqR: real trunk + QEC head + tie-gated residual phase.

    log psi = log A_theta(x) + 1j*(pi*s(x) + phi_chi(x)), with
    phi_chi = t(x) * w2 . tanh(W1 [s, d, r] + b1): a tiny real MLP over
    DECODER-DERIVED features (decoder-A parity s, syndrome bits d, ambiguity
    cycle r = eps_A XOR eps_B). Feeding s is ESSENTIAL (expressivity-oracle
    finding, 2026-08-28): r is a deterministic function of d (the decoder
    sees only the syndrome) and tie partners share d, so an MLP over (d, r)
    alone provably cannot split any tie pair -- s is exactly the bit that
    distinguishes them, lifting the class to the full tie-gate ceiling.
    The zero-init output layer makes phi == 0 at init (step 0 is
    exactly the head-only ansatz) and whose multiplicative tie gate t makes
    phi == 0 EXACTLY on every un-tied config forever -- the theorem regime
    (on-sector, the whole hz axis) is untouchable by training. Real params,
    complex output (non-holomorphic QGT); trained on the BARE Hamiltonian
    (head + residual both live in the model, formulation-A path).
    """

    base: nn.Module
    head_features: Any       # QECSignHead.features -- static (host) callable
    n_features: int          # QECSignHead.n_features (K = 1 + V + N + 1)
    hidden: int = 16

    @nn.compact
    def __call__(self, x):
        log_a = self.base(x)
        u = jax.pure_callback(
            lambda xb: np.asarray(self.head_features(xb), dtype=np.float64),
            jax.ShapeDtypeStruct(x.shape[:-1] + (self.n_features,), jnp.float64),
            x, vmap_method='expand_dims')
        s, sdr, t = u[..., 0], u[..., :-1], u[..., -1]
        h = nn.tanh(nn.Dense(self.hidden, param_dtype=jnp.float64)(sdr))
        phi = t * nn.Dense(1, kernel_init=nn.initializers.zeros,
                           param_dtype=jnp.float64)(h)[..., 0]
        return log_a + 1j * (jnp.pi * s + phi)


class _SignMLP(nn.Module):
    """``depth`` tanh Dense layers of width ``hidden`` then Dense(1); float64,
    default flax init (lecun-normal kernels, zero biases). Inputs are the raw
    {0,1} features and are recentred to {-1,+1} INSIDE (h = 2u - 1): with
    zero-bias init the all-zero feature vector (the all-up config, eps = x = 0)
    would otherwise give m = 0 exactly -- a node of tanh(m) sitting on the
    sign-gauge anchor. Instantiated under the fixed scope name 'mlp' so a
    pretrained tree can replace it wholesale (model/sign_mlp_io.py holds the
    flat key/shape spec; the pretrain script feeds the same raw {0,1} features
    and so inherits this recentring)."""

    hidden: int
    depth: int

    @nn.compact
    def __call__(self, u):
        h = 2.0 * u - 1.0
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden, param_dtype=jnp.float64)(h))
        # Last layer zero-kernel / bias +1: step 0 is the POSITIVE ansatz
        # (tanh(1) on every config, no sign information), not a random-sign
        # state (review 2026-09-24: lecun init gave 34% minority signs at init).
        return nn.Dense(1, param_dtype=jnp.float64,
                        kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.constant(1.0))(h)[..., 0]


class MLPSignModel(nn.Module):
    """Learned-sign arm M / M-pre (docs/signhead_benchmark_plan.md):

        psi = A_theta(sigma) * tanh(m_theta(u)),   u = features_ex(sigma) = [eps, x],

    log psi = log A + log|tanh m| + 1j*pi*[tanh m < 0] -- a complex log with
    REAL params (non-holomorphic QGT path, like ResidualSignedModel). The MLP
    sees exactly the decoder-derived variables the closed-form head uses
    ((eps, x) determines sigma), but must LEARN the sign from the energy alone
    (arm M, random init) or from a supervised warm start (arm M-pre, loaded
    with load_mlp_params). tanh -> 0 where the learned sign flips: psi has
    nodes there, sampled with vanishing weight. Host callback on the INPUT
    only, once per batch (wraps the whole batched trunk).

    Param tree: {'base': <Combo trunk>, 'mlp': {'Dense_0': {'kernel' (K, hidden),
    'bias' (hidden,)}, ..., 'Dense_depth': {'kernel' (hidden, 1), 'bias' (1,)}}},
    K = n_features = N + F; spec in model/sign_mlp_io.mlp_param_spec.
    """

    base: nn.Module
    head_features_ex: Any    # QECSignHead.features_ex -- static (host) callable
    n_features: int          # QECSignHead.n_features_ex (K = N + F)
    hidden: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, x):
        log_a = self.base(x)
        u = jax.pure_callback(
            lambda xb: np.asarray(self.head_features_ex(xb), dtype=np.float64),
            jax.ShapeDtypeStruct(x.shape[:-1] + (self.n_features,), jnp.float64),
            x, vmap_method='expand_dims')
        t = jnp.tanh(_SignMLP(self.hidden, self.depth, name='mlp')(u))
        return log_a + jnp.log(jnp.abs(t)) + 1j * jnp.pi * (t < 0).astype(jnp.float64)


class TwoBranchModel(nn.Module):
    """Two-branch arm T (docs/signhead_benchmark_plan.md):

        psi = a_{s} * A_triv(sigma) + s(sigma) * A_top(sigma),   s = (-1)^{s01} in {+-1},

    two INDEPENDENT positive Combo trunks (scopes 'base_triv' / 'base_top' =>
    distinct init draws) and TWO signed real scalars ('mix' = [a_+, a_-], both
    init mix_init), one per head sector. Stable complex log: with a1 = log A_triv,
    a2 = log A_top, m = max(a1, a2),
        w = a_s e^{a1-m} + s e^{a2-m},
        log psi = m + log|w| + 1j*pi*[w < 0]
    (a stays O(1) OUTSIDE the max/log, so a = 0 and a < 0 are both valid).
    a_+ = a_- = 0 is exactly SignedModel (the head-only arm). A single signed
    scalar pins one head sector to the global sign (a > 0: the s = +1 sector;
    a < 0: the s = -1 sector) and the exact ceilings say the better sign flips
    cell by cell (review 2026-09-24: h_z = 0 wants a < 0, h_z > 0 wants a > 0),
    so each sector gets its own scalar and no sector is pinned. mix_init =
    +0.05 keeps the trivial branch at ~5% amplitude at step 0. Real params, complex output; the
    head is a host callback on the input, once per batch (zero gradient by
    construction).
    """

    base_triv: nn.Module
    base_top: nn.Module
    head_s01: Any            # QECSignHead.s01 -- static (host) callable
    mix_init: float = 0.05
    positive_mix: bool = False   # arm T+: a = exp(c), ONE scalar 'log_mix' (init log mix_init)

    @nn.compact
    def __call__(self, x):
        if self.positive_mix:
            # The spec's original parameterisation (3D 'T+'): the trivial branch
            # can only ADD to the head-signed one (a > 0 always), so the s=+1
            # sector is pinned and the ceiling is T_gate_plus. Kept as a control
            # because the signed mix flips a < 0 early at every cell (2026-09-24).
            c = self.param('log_mix',
                           lambda key: jnp.asarray(jnp.log(self.mix_init), dtype=jnp.float64))
            mix = jnp.exp(c) * jnp.ones((2,), dtype=jnp.float64)
        else:
            mix = self.param('mix',
                             lambda key: jnp.full((2,), self.mix_init, dtype=jnp.float64))
        a1 = self.base_triv(x)
        a2 = self.base_top(x)
        s01 = jax.pure_callback(
            lambda xb: np.asarray(self.head_s01(xb), dtype=np.float64),
            jax.ShapeDtypeStruct(x.shape[:-1], jnp.float64),
            x, vmap_method='expand_dims')
        s = 1.0 - 2.0 * s01
        a = jnp.where(s > 0, mix[0], mix[1])      # per-sector mix a_{s(sigma)}
        m = jnp.maximum(a1, a2)
        w = a * jnp.exp(a1 - m) + s * jnp.exp(a2 - m)
        return m + jnp.log(jnp.abs(w)) + 1j * jnp.pi * (w < 0).astype(jnp.float64)


def load_mlp_params(params, npz_path):
    """Return a copy of the MLPSignModel param tree with its 'mlp' subtree
    replaced by the npz written by scripts/pretrain_sign_mlp.py (arm M-pre).
    Shapes are asserted against the live tree (model/sign_mlp_io spec)."""
    import flax
    from model.sign_mlp_io import load_mlp_npz, nest

    params = flax.core.unfreeze(params)          # mutable deep copy of the tree
    assert 'mlp' in params, "--mlp_init: the model has no 'mlp' subtree (need --sign_impl mlp)"
    spec = {f"{layer}/{leaf}": tuple(np.shape(a))
            for layer, sub in params['mlp'].items() for leaf, a in sub.items()}
    flat = load_mlp_npz(npz_path, spec)
    params['mlp'] = {layer: {leaf: jnp.asarray(a, dtype=jnp.float64)
                             for leaf, a in sub.items()}
                     for layer, sub in nest(flat).items()}
    return params


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def create_honeycomb_model(config, geometry) -> nn.Module:
    """Build the honeycomb Combo or PlainCNN from a config dict + geometry.

    config keys: 'architecture' ('Combo' | 'PlainCNN'), 'dtype' ('float64' |
    'complex'), 'channels_noninv' (Block-1 / PlainCNN channel list),
    'channels_inv' (Block-3 channel list, default [16, 8, 1]), optional
    'rescale' (complex-Wilson rescale, default 10**1.5).
    """
    arch = config['architecture']
    dtype = config['dtype']
    ch_non = [int(c) for c in config['channels_noninv']]
    rescale = config.get('rescale', 10 ** 1.5)

    cls, slots = block1_tables(geometry)
    cls_t = tuple(int(c) for c in cls)
    slots_t = tuple(tuple(int(s) for s in row) for row in slots)

    if arch == "Combo":
        ch_inv = [int(c) for c in config.get('channels_inv', [16, 8, 1])]
        assert ch_inv[0] == ch_non[-1], (
            f"channels_inv[0] ({ch_inv[0]}) must equal channels_noninv[-1] "
            f"({ch_non[-1]}): the Wilson tokens keep Block-1's channel count"
        )
        noninv = [
            HoneycombLocalCNN(i, o, cls_t, slots_t, identity_init=True, dtype=dtype)
            for i, o in zip(ch_non, ch_non[1:])
        ]
        vertex_all_t = tuple(tuple(int(e) for e in row) for row in geometry.vertex_all)
        wilson = WilsonNonlinearity(vertex_all_t, rescale, dtype)
        sub, tap, n_taps, _ = block3_tables(geometry)
        sub_t = tuple(int(s) for s in sub)
        tap_t = tuple(tuple(int(t) for t in row) for row in tap)
        inv = [
            HoneycombInvariantCNN(i, o, sub_t, tap_t, n_taps, dtype=dtype)
            for i, o in zip(ch_inv, ch_inv[1:])
        ]
        sequence = noninv + [wilson] + inv + [Final()]

    elif arch == "PlainCNN":
        # Unconstrained baseline: same local primitive, generic init, mean over
        # links. Matched-generous scale vs Combo: channels_noninv 1,32,24,8,2
        # gives 15 * (32 + 768 + 192 + 16) = 15,120 params.
        sequence = [
            HoneycombLocalCNN(i, o, cls_t, slots_t, identity_init=False, dtype=dtype)
            for i, o in zip(ch_non, ch_non[1:])
        ] + [Final()]

    else:
        raise ValueError(f"honeycomb architecture must be Combo or PlainCNN, got {arch!r}")

    return Sequential(tuple(sequence))


def create_two_branch_model(config, geometry, head):
    """Arm T: two independent Combo trunks (same config, distinct scopes) +
    the closed-form head. Returns (model, base) with ``base`` the trivial-branch
    trunk (either trunk serves the init-only hexflip gate; main.py runs it on
    both param subtrees 'base_triv' / 'base_top')."""
    base_triv = create_honeycomb_model(config, geometry)
    base_top = create_honeycomb_model(config, geometry)
    model = TwoBranchModel(base_triv, base_top, head.s01,
                           mix_init=float(config.get('mix_init', 0.05)),
                           positive_mix=bool(config.get('mix_positive', False)))
    return model, base_triv
