"""
Full-transformer pipeline (v2) for the 2D toric-code NQS.

Replaces the ENTIRE Combo stack (non-invariant edge CNN -> Wilson nonlinearity ->
invariant plaquette CNN) with transformers:

    sigma -> OddEdgeEmbedding
          -> n1 x SpatialEncoderBlock          (Block 1, edge tokens)
          -> WilsonFusion                       (edge -> plaquette, product)
          -> n2 x EncoderBlock (v1, reused)     (Block 2, plaquette tokens)
          -> sum-pool -> log-cosh head          -> log psi

Exact vertex/gauge symmetry `A_v` is preserved AT INIT and deliberately broken
during training (that is how the h_z field is captured). The recipe (proven in
scratchpad/validate_v2.py):

  * ODD, bias-free embedding  v_e = sigma_e * w_{orient(e)}  ->  a spin flip becomes
    a sign flip of the whole feature vector;
  * Block-1 blocks IDENTITY at init (zero-init the attention output proj `W` and the
    FFN's second layer `ffn1`) -> pre-Wilson feature stays exactly the odd embedding;
  * WilsonFusion = channelwise (Hadamard) product over the 4 edges of each plaquette,
    after the ODD `bound` map -> `W_p[c] = B_p * const_p[c]`, invariant since each
    plaquette shares an even number (0/2) of edges with any vertex star.

Anything downstream of the (invariant) Wilson output is invariant regardless of its
parameters, so Block 2 / the head reuse v1 verbatim (no zero-init needed there).

OBC (critical): edge-midpoint distances d(i,j) and the orientation-resolved alpha
displacement table use plain signed subtraction -- NO wraparound / modulo.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

from model.transformer_block import EncoderBlock, _log_cosh  # v1 Block-2 + stable log-cosh

# Odd bounded activation == the CNN's calibrated sigmoid map (model/networks.py:375):
# maps +-1 -> +-1, bounds to ~(-2.16, 2.16), and is ODD (sigmoid(-x)-1/2 = -(sigmoid(x)-1/2)),
# so bound(sigma_e * w) = sigma_e * bound(w) for sigma_e in {+-1} -> preserves A_v invariance.
_E = float(np.e)
_BOUND_SCALE = (2.0 + 2.0 * _E) / (_E - 1.0)


def _bound(x):
    return (nn.sigmoid(x) - 0.5) * _BOUND_SCALE


def edge_displacement_table(arr_coord) -> tuple:
    """Static orientation-resolved edge displacement table for Block-1 alpha.

    ``arr_coord`` = edge midpoints (``geometry.arr_coord``), half-integer coords.
    Orientation ``o=0`` horizontal (x half-integer), ``o=1`` vertical (y half-integer).
    Returns ``(orient, D, n_disp)`` where ``D[i,j]`` is the index (into a dense set of
    distinct ``(o_i, o_j, dx, dy)`` tuples, ``dx,dy`` in half-integer units by plain
    signed subtraction -- NO wrap) of the displacement of j relative to i.
    ``n_disp == 16 L^2 - 32 L + 14`` (verified in scratchpad/validate_v2.py).
    """
    c2 = np.round(np.asarray(arr_coord) * 2).astype(np.int64)   # half-integer units
    orient = (c2[:, 1] % 2).astype(np.int64)                    # y odd -> vertical(1) else horizontal(0)
    N = len(c2)
    lut: dict = {}
    D = np.empty((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            k = (int(orient[i]), int(orient[j]),
                 int(c2[i, 0] - c2[j, 0]), int(c2[i, 1] - c2[j, 1]))
            D[i, j] = lut.setdefault(k, len(lut))
    return orient.tolist(), D, len(lut)


class OddEdgeEmbedding(nn.Module):
    """Bias-free, orientation-dependent ODD embedding: ``v_e = sigma_e * w_{orient(e)}``."""

    d_model: int
    orient: Any            # tuple length N of 0/1 (hashable)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, sigma):                 # sigma: (N,) spins in {+-1}
        w = self.param("w", nn.initializers.normal(stddev=1.0), (2, self.d_model), self.dtype)
        o = jnp.asarray(self.orient)
        return sigma.astype(self.dtype)[:, None] * w[o]        # (N, d), odd & bias-free


class SpatialFactoredMHA_OBC(nn.Module):
    """Distance-kernel factored attention on edge tokens (Viteritti-Rende).

    ``A_i^h = sum_j softmax_j(-gamma_h d(i,j)) * alpha^h_{Delta(i,j),o_i,o_j} (V^h x_j)``.
    Real position-only ``alpha`` (orientation-resolved), per-head inverse-range
    ``gamma_h = softplus(.)`` init local. Output projection ``W`` is ZERO-INIT so the
    whole sublayer contributes 0 at init (block = identity).
    """

    d_model: int
    n_heads: int
    n_disp: int
    D: Any                 # (N,N) tuple-of-tuples (hashable)
    positions: Any         # (N,2) tuple-of-tuples (hashable) -- edge midpoints
    gamma_init: float = 7.862
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                     # x: (N, d)
        n, d = x.shape
        h = self.n_heads
        de = d // h

        v = nn.Dense(d, param_dtype=self.dtype, name="V")(x)
        v = v.reshape(n, h, de).transpose(1, 0, 2)             # (h, n, de)

        alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                           (h, self.n_disp), self.dtype)
        alpha_full = alpha[:, jnp.asarray(self.D)]             # (h, n, n)

        gamma_raw = self.param("gamma_raw", nn.initializers.constant(self.gamma_init),
                               (h,), self.dtype)
        gamma = nn.softplus(gamma_raw)                         # (h,) > 0
        pos = jnp.asarray(self.positions, dtype=self.dtype)    # (n, 2)
        dist = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1) + 1e-12)
        kern = jax.nn.softmax(-gamma[:, None, None] * dist[None, :, :], axis=-1)   # (h,n,n) over keys
        A = kern * alpha_full                                  # (h, n, n)

        out = jnp.matmul(A, v)                                 # (h, n, de)
        out = out.transpose(1, 0, 2).reshape(n, d)             # concat heads
        # zero-init W => identity block at init (preserves the odd pre-Wilson feature)
        out = nn.Dense(d, param_dtype=self.dtype, name="W",
                       kernel_init=nn.initializers.zeros)(out)
        return out


class SpatialEncoderBlock(nn.Module):
    """Pre-LayerNorm encoder block for Block 1; sublayer outputs zero-init => identity at init."""

    d_model: int
    n_heads: int
    n_disp: int
    D: Any
    positions: Any
    gamma_init: float = 7.862
    ffn_mult: int = 2
    activation: str = "relu"
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                     # x: (N, d)
        act = nn.gelu if self.activation == "gelu" else nn.relu

        y = nn.LayerNorm(param_dtype=self.dtype, name="ln1")(x)
        y = SpatialFactoredMHA_OBC(d_model=self.d_model, n_heads=self.n_heads,
                                   n_disp=self.n_disp, D=self.D, positions=self.positions,
                                   gamma_init=self.gamma_init, dtype=self.dtype, name="attn")(y)
        x = x + y

        z = nn.LayerNorm(param_dtype=self.dtype, name="ln2")(x)
        z = nn.Dense(self.ffn_mult * self.d_model, param_dtype=self.dtype, name="ffn0")(z)
        z = act(z)
        # zero-init ffn1 => FFN contributes 0 at init
        z = nn.Dense(self.d_model, param_dtype=self.dtype, name="ffn1",
                     kernel_init=nn.initializers.zeros)(z)
        x = x + z
        return x


class WilsonFusion(nn.Module):
    """Edge -> plaquette fusion: bound -> channelwise 4-product over plaq_all -> P -> LayerNorm.

    Pure product, NO residual skip: any linear edge-pool is a spin-sum (A_v-breaking); all
    deformation comes from Block 1 leaving identity. ``bound`` is odd, so at init
    ``prod_i[c] = B_p * const_p[c]`` -- invariant.
    """

    d_model: int
    plaq_all: Any          # (n_plaq, 4) tuple-of-tuples (hashable)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                     # x: (N_edges, d)
        idx = jnp.asarray(self.plaq_all)                       # (n_plaq, 4)
        prod = jnp.prod(_bound(x)[idx], axis=1)                # (n_plaq, d) channelwise Hadamard
        tok = nn.Dense(self.d_model, param_dtype=self.dtype, name="P")(prod)
        tok = nn.LayerNorm(param_dtype=self.dtype, name="fuse_ln")(tok)
        return tok


class FullTransformer(nn.Module):
    """v2 pipeline: odd embed -> n1 spatial blocks -> Wilson fusion -> n2 factored blocks -> head.

    Input ``sigma`` is a single-sample spin vector ``(N_edges,)`` (the ``Sequential`` wrapper
    vmaps over samples). Output is a scalar ``log psi`` (real; complex if ``complex_output``).
    d1 = d2 = ``d_model`` and one head count ``n_heads`` for both blocks (see plan).
    """

    # Block 1 (edge tokens)
    n1: int
    d_model: int
    n_heads: int
    n_disp_e: int
    D_e: Any
    positions_e: Any
    orient_e: Any
    gamma_init: float
    # fusion
    plaq_all: Any
    # Block 2 (plaquette tokens; reuses v1 EncoderBlock with displacement-only alpha)
    n2: int
    n_disp_p: int
    D_p: Any
    # shared
    ffn_mult: int = 2
    activation: str = "relu"
    # head
    readout_K: int = 0
    complex_output: bool = False
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, sigma):                 # sigma: (N_edges,)
        K = self.readout_K or self.d_model

        x = OddEdgeEmbedding(self.d_model, self.orient_e, self.dtype, name="embed")(sigma)

        for l in range(self.n1):               # Block 1: edge tokens
            x = SpatialEncoderBlock(
                d_model=self.d_model, n_heads=self.n_heads, n_disp=self.n_disp_e,
                D=self.D_e, positions=self.positions_e, gamma_init=self.gamma_init,
                ffn_mult=self.ffn_mult, activation=self.activation, dtype=self.dtype,
                name=f"sblock{l}")(x)

        x = WilsonFusion(self.d_model, self.plaq_all, self.dtype, name="fusion")(x)

        for l in range(self.n2):               # Block 2: plaquette tokens (v1 factored attention)
            x = EncoderBlock(
                d_model=self.d_model, n_heads=self.n_heads, n_disp=self.n_disp_p,
                D=self.D_p, ffn_mult=self.ffn_mult, activation=self.activation,
                dtype=self.dtype, name=f"pblock{l}")(x)

        z = jnp.sum(x, axis=0)                  # sum-pool over tokens -> (d,)
        z = nn.LayerNorm(param_dtype=self.dtype, name="out_ln")(z)

        if self.complex_output:                 # v3 real-deep + complex-shallow readout
            re = nn.LayerNorm(param_dtype=self.dtype, name="out_ln_re")(
                nn.Dense(K, param_dtype=self.dtype, name="head_re")(z))
            im = nn.LayerNorm(param_dtype=self.dtype, name="out_ln_im")(
                nn.Dense(K, param_dtype=self.dtype, name="head_im")(z))
            out = re + 1j * im
        else:
            out = nn.Dense(K, param_dtype=self.dtype, name="head")(z)

        return jnp.sum(_log_cosh(out))          # scalar log psi
