"""
Variant 3: standalone pure-plaquette transformer with gated content attention.

Tokens are the classical stabilizer values t_p = B_p(s) = prod_{i in dp} s_i
in {+-1} -- the network never sees the edge spins, so exact A_v (vertex/gauge)
invariance holds by change of variables, with NO architectural constraint
downstream (no odd nonlinearities, no zero-init, no bias bans needed for
symmetry; the bias-free choice below is a cleanliness choice, not a symmetry
requirement). Valid standalone when the perturbation commutes with every
A_v = XXXX, i.e. the pure h_x cut; h_z / h_y anticommute with A_v and need the
Variant-1 cleaning block (future work).

Attention (per head h) is FULL, input-dependent, gated:

    A_ij = softmax_j( b_h(Delta_ij) + alpha_h * q_i . k_j / sqrt(d_head) )

* b_h(Delta): learned scalar per (head, signed displacement) -- T5-style
  relative-position bias over the same (2L-3)^2 OBC displacement table as the
  factored v1 block (distance AND direction; no wraparound, no absolute PE).
* alpha_h: per-head scalar gate, INITIALIZED AT ZERO -- at init this is pure
  (normalized) position-based attention; content routing is bought only where
  gradients demand it. The trained alpha_h per layer/head is the first
  content-routing diagnostic. `use_content=False` freezes alpha_h at 0
  (stop-gradient) => normalized-factored ablation, one flag.

Everything is bias-free and pre-RMSNorm (no LayerNorm). Readout: sum-pool ->
RMSNorm -> Dense_K -> sum log-cosh (single real head; the h_x cut is
sign-free/stoquastic, dtype float64).
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

from model.transformer_block import plaquette_displacement_table, _log_cosh  # noqa: F401

_ACTIVATIONS = {"relu": nn.relu, "gelu": nn.gelu, "tanh": nn.tanh}


class GatedMHA_OBC(nn.Module):
    """Full multi-head self-attention with OBC relative-position bias and a
    zero-init per-head content gate."""

    d_model: int
    n_heads: int
    n_disp: int
    D: Any                 # (n_plaq, n_plaq) int, tuple-of-tuples (hashable)
    use_content: bool = True
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                                   # x: (n, d)
        n, d = x.shape
        h = self.n_heads
        d_head = d // h

        def heads(name):
            y = nn.Dense(d, use_bias=False, param_dtype=self.dtype, name=name)(x)
            return y.reshape(n, h, d_head).transpose(1, 0, 2)          # (h, n, d_head)

        q, k, v = heads("Wq"), heads("Wk"), heads("Wv")

        b = self.param("b_pos", nn.initializers.xavier_uniform(),
                       (h, self.n_disp), self.dtype)
        b_full = b[:, jnp.asarray(self.D)]                             # (h, n, n)

        alpha = self.param("alpha", nn.initializers.zeros, (h,), self.dtype)
        if not self.use_content:
            alpha = jax.lax.stop_gradient(alpha)   # frozen at 0 => pure position attention

        content = jnp.einsum("hid,hjd->hij", q, k) / jnp.sqrt(
            jnp.asarray(d_head, dtype=q.dtype))                        # (h, n, n)
        logits = b_full + alpha[:, None, None] * content
        A = jax.nn.softmax(logits, axis=-1)                            # over keys j

        out = jnp.einsum("hij,hjd->hid", A, v)                         # (h, n, d_head)
        out = out.transpose(1, 0, 2).reshape(n, d)                     # concat heads
        return nn.Dense(d, use_bias=False, param_dtype=self.dtype, name="Wo")(out)


class EncoderBlockV3(nn.Module):
    """Pre-RMSNorm encoder block: x += MHA(norm(x)); x += FFN(norm(x)). Bias-free."""

    d_model: int
    n_heads: int
    n_disp: int
    D: Any
    ffn_mult: int = 2
    activation: str = "gelu"
    use_content: bool = True
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                                   # x: (n, d)
        act = _ACTIVATIONS[self.activation]

        y = nn.RMSNorm(param_dtype=self.dtype, name="norm1")(x)
        y = GatedMHA_OBC(d_model=self.d_model, n_heads=self.n_heads,
                         n_disp=self.n_disp, D=self.D,
                         use_content=self.use_content, dtype=self.dtype,
                         name="attn")(y)
        x = x + y

        z = nn.RMSNorm(param_dtype=self.dtype, name="norm2")(x)
        z = nn.Dense(self.ffn_mult * self.d_model, use_bias=False,
                     param_dtype=self.dtype, name="ffn0")(z)
        z = act(z)
        z = nn.Dense(self.d_model, use_bias=False,
                     param_dtype=self.dtype, name="ffn1")(z)
        return x + z


class PlaquetteTransformer(nn.Module):
    """Variant 3: s -> t_p = B_p(s) -> 2-entry embed -> encoder stack -> log psi.

    Input is a single-sample spin vector ``(N_edges,)`` in {+-1} (the
    ``Sequential`` wrapper vmaps over samples). Output is a scalar real log psi.
    """

    plaq_all: Any          # (n_plaq, 4) tuple-of-tuples, all edges valid (no -1)
    n_layers: int
    d_model: int
    n_heads: int
    n_disp: int
    D: Any
    readout_K: int
    ffn_mult: int = 2
    activation: str = "gelu"
    use_content: bool = True
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, sigma):                               # sigma: (N_edges,)
        # Tokenize: t_p = prod of the 4 edge spins of each plaquette, in {+-1}.
        t = jnp.prod(sigma[jnp.asarray(self.plaq_all)], axis=-1)       # (n_plaq,)

        # 2-entry lookup embedding (2*d params): t=-1 -> E[0], t=+1 -> E[1].
        E = self.param("embed", nn.initializers.normal(stddev=1.0),
                       (2, self.d_model), self.dtype)
        idx = ((t + 1.0) / 2.0).astype(jnp.int32)                      # {0, 1}
        x = E[idx]                                                     # (n_plaq, d)

        for l in range(self.n_layers):
            x = EncoderBlockV3(d_model=self.d_model, n_heads=self.n_heads,
                               n_disp=self.n_disp, D=self.D,
                               ffn_mult=self.ffn_mult, activation=self.activation,
                               use_content=self.use_content, dtype=self.dtype,
                               name=f"block{l}")(x)

        z = jnp.sum(x, axis=0)                                         # sum-pool -> (d,)
        z = nn.RMSNorm(param_dtype=self.dtype, name="out_norm")(z)
        out = nn.Dense(self.readout_K, use_bias=False,
                       param_dtype=self.dtype, name="head")(z)
        return jnp.sum(_log_cosh(out))                                 # scalar log psi
