"""
Factored-attention transformer encoder as a drop-in replacement for the
exactly-symmetric block (Block 3) of the Combo ansatz.

The block consumes the post-Wilson-nonlinearity plaquette representation
(shape ``(C, n_plaq)`` per sample, C = channels_noninv[-1]) and returns a scalar
``log psi``. Because the exact stabilizer (vertex/gauge) symmetry is established
upstream by the Wilson nonlinearity, any deterministic function of that
representation preserves it; this block only needs to be translation-equivariant
like the global-kernel CNN it replaces. That is realized with FACTORED attention
(Rende & Viteritti, arXiv:2405.18874, Eq. 6):

    A_i^mu = sum_j  alpha^mu_{Delta(i,j)}  (V^mu x_j)

with REAL, position-only weights ``alpha`` indexed by the relative displacement
Delta between plaquettes. Encoder / readout follow Viteritti et al.
(arXiv:2311.16889): pre-LayerNorm + residuals, 2-layer FFN, sum-pool over tokens,
log-cosh readout.

OBC (critical): the NetKet ViT tutorial expands ``alpha`` with ``jnp.roll`` (PBC /
circular). Here the plaquettes live on an (Lx-1) x (Ly-1) OPEN grid, so ``alpha``
is indexed by a plain signed displacement (no wraparound, no modular arithmetic)
via the static table built by :func:`plaquette_displacement_table`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple

# netket is always available in the run environment; its log_cosh is stable for
# both real (this sign-free run) and complex (later sign-full) arguments.
try:
    from netket.nn.activation import log_cosh as _log_cosh
except Exception:  # pragma: no cover - version fallback
    from netket.nn import log_cosh as _log_cosh


def plaquette_displacement_table(dg_p_positions, Lx: int, Ly: int
                                 ) -> Tuple[np.ndarray, int]:
    """Static ``(n_plaq, n_plaq)`` int table of relative-displacement indices.

    ``D[i, j]`` is the flattened index, into the ``(2*Lx-3) * (2*Ly-3)``
    displacement axis, of plaquette ``j`` relative to plaquette ``i``. Plaquette
    order matches ``dg_p.positions`` (== the order of ``geometry.plaq_all`` and the
    Wilson-nonlinearity output token axis).

    OBC: plaquette grid coordinates are ``floor(position)`` (positions sit at
    ``(r+1/2, c+1/2)`` with ``r, c in [0, L-2]``); displacements are a plain
    signed subtraction shifted into ``[0, 2L-3)``. No wraparound. Every ordered
    pair of plaquettes has an in-range displacement, so NO masking is needed
    (unlike the CNN kernel, which masks absolute shifts that fall off the grid).

    Returns ``(D, n_disp)`` with ``n_disp = (2*Lx-3) * (2*Ly-3)``.
    """
    coords = np.floor(np.asarray(dg_p_positions)).astype(np.int64)  # (n_plaq, 2)
    span_x = 2 * Lx - 3   # displacements in [-(Lx-2), Lx-2]
    span_y = 2 * Ly - 3
    dx = coords[:, 0][None, :] - coords[:, 0][:, None] + (Lx - 2)  # -> [0, 2Lx-3)
    dy = coords[:, 1][None, :] - coords[:, 1][:, None] + (Ly - 2)  # -> [0, 2Ly-3)
    D = (dx * span_y + dy).astype(np.int32)
    return D, span_x * span_y


class FactoredMHA_OBC(nn.Module):
    """Factored multi-head self-attention with OBC relative-position weights.

    No queries/keys: the per-head attention weight depends only on the relative
    plaquette displacement (an input-independent learnable parameter). The value
    projection ``V`` (Dense d->d, split into heads) and output projection ``W``
    (Dense d->d) are retained.
    """

    d_model: int
    n_heads: int
    n_disp: int
    D: Any            # (n_plaq, n_plaq) int, passed as tuple-of-tuples (hashable)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                       # x: (n_plaq, d_model)
        n, d = x.shape
        h = self.n_heads
        d_eff = d // h

        v = nn.Dense(d, param_dtype=self.dtype, name="V")(x)     # (n, d)
        v = v.reshape(n, h, d_eff).transpose(1, 0, 2)            # (h, n, d_eff)

        # alpha: (h, n_disp) free params -> gather to the full (h, n, n) matrix.
        alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                           (h, self.n_disp), self.dtype)
        Dj = jnp.asarray(self.D)                                 # (n, n) int
        alpha_full = alpha[:, Dj]                                # (h, n, n)

        out = jnp.matmul(alpha_full, v)                          # (h, n, d_eff)
        out = out.transpose(1, 0, 2).reshape(n, d)              # concat heads
        out = nn.Dense(d, param_dtype=self.dtype, name="W")(out)
        return out


class EncoderBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block with residual connections."""

    d_model: int
    n_heads: int
    n_disp: int
    D: Any
    ffn_mult: int = 2
    activation: str = "relu"
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                        # x: (n_plaq, d_model)
        act = nn.gelu if self.activation == "gelu" else nn.relu

        y = nn.LayerNorm(param_dtype=self.dtype, name="ln1")(x)
        y = FactoredMHA_OBC(d_model=self.d_model, n_heads=self.n_heads,
                            n_disp=self.n_disp, D=self.D, dtype=self.dtype,
                            name="attn")(y)
        x = x + y

        z = nn.LayerNorm(param_dtype=self.dtype, name="ln2")(x)
        z = nn.Dense(self.ffn_mult * self.d_model, param_dtype=self.dtype, name="ffn0")(z)
        z = act(z)
        z = nn.Dense(self.d_model, param_dtype=self.dtype, name="ffn1")(z)
        x = x + z
        return x


class TransformerSymmetric(nn.Module):
    """Block-3 replacement: embed plaquette tokens -> encoder stack -> sum -> log-cosh.

    Input ``x`` is a single-sample Wilson output of shape ``(C, n_plaq)`` (the
    ``Sequential`` wrapper vmaps over samples). Output is a scalar ``log psi``.

    For the sign-free cut the whole block is real (``ψ = exp(log ψ)`` may be
    negative through log-cosh but the amplitude is real, matching the paper's
    real ansatz). Setting ``complex_output=True`` splits the readout into two real
    heads combined as ``real + 1j*imag`` (the Viteritti complex-W readout) WITHOUT
    touching the encoder -- the only change needed for the later sign-full case.
    """

    n_layers: int
    d_model: int
    n_heads: int
    n_disp: int
    D: Any
    readout_K: int
    ffn_mult: int = 2
    activation: str = "relu"
    complex_output: bool = False
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                         # x: (C, n_plaq)
        x = x.T                                     # tokens: (n_plaq, C)
        x = nn.Dense(self.d_model, param_dtype=self.dtype, name="embed")(x)

        for l in range(self.n_layers):
            x = EncoderBlock(d_model=self.d_model, n_heads=self.n_heads,
                             n_disp=self.n_disp, D=self.D, ffn_mult=self.ffn_mult,
                             activation=self.activation, dtype=self.dtype,
                             name=f"block{l}")(x)

        z = jnp.sum(x, axis=0)                      # sum-pool over tokens -> (d,)
        z = nn.LayerNorm(param_dtype=self.dtype, name="out_ln")(z)

        if self.complex_output:
            re = nn.LayerNorm(param_dtype=self.dtype, name="out_ln_re")(
                nn.Dense(self.readout_K, param_dtype=self.dtype, name="head_re")(z))
            im = nn.LayerNorm(param_dtype=self.dtype, name="out_ln_im")(
                nn.Dense(self.readout_K, param_dtype=self.dtype, name="head_im")(z))
            out = re + 1j * im
        else:
            out = nn.Dense(self.readout_K, param_dtype=self.dtype, name="head")(z)

        return jnp.sum(_log_cosh(out))             # scalar log psi
