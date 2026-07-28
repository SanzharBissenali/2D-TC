"""
Variant 1: gauge-combo transformer (the transformer upgrade of Kufel et al.'s Combo).

    sigma -> odd edge embedding  x_e = sigma_e * w_{orient(e)}      (bias-free, C channels)
          -> n1 x chi blocks     (gated FULL attention on edge tokens, IDENTITY at init)
          -> fixed Wilson step   P_{p,c} = prod_{e in dp} tanh(x_{e,c})   (channel-wise)
          -> Dense(C -> d)       (Omega's embedding: learned projection, bias-free)
          -> n2 x EncoderBlockV3 (the validated Variant-3 backbone, plaquette tokens)
          -> sum-pool -> RMSNorm -> Dense_K -> sum log-cosh          -> log psi

Exact A_v invariance holds AT INIT and is deliberately broken by training (that is how
an A_v-non-commuting field like h_z is captured). The init guarantee needs three pieces:
  * ODD, bias-free embedding: a spin flip = a sign flip of the whole feature vector;
  * chi blocks == identity at init: `zero_init_out` zeroes the attention output
    projection Wo and the FFN's second layer, so every residual branch contributes 0;
  * an ODD chain into the product: pre-RMSNorm has no centering/bias (RMSNorm(-x) =
    -RMSNorm(x), unlike LayerNorm) and the fixed `tanh` before the product is odd, so
    at init  prod tanh(sigma_e w_c) = B_p(s) * prod tanh(w_c)  = B_p x const per channel
    (every plaquette shares 0 or 2 edges with any vertex star). chi's FFN uses tanh too
    (the spec's odd-nonlinearity budget), keeping the learned dressing sign-covariant.

At init every channel carries B_p(s) times a constant, i.e. the model IS Variant 3 with
a linear embedding -- training Variant 1 at the symmetric point is training Variant 3.
Downstream of the Wilson step nothing is constrained (params belong in Omega).

Position tables are OBC signed-displacement (no wrap): chi uses the orientation-resolved
edge table (`spatial_attention_block.edge_displacement_table`), Omega the plaquette table
(`transformer_block.plaquette_displacement_table`).
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Any

from model.plaquette_transformer import EncoderBlockV3
from model.transformer_block import _log_cosh
from model.spatial_attention_block import edge_displacement_table  # noqa: F401 (re-export)


class GaugeComboTransformer(nn.Module):
    """Variant 1: chi (edge cleaning) -> fixed Wilson product -> Omega (Variant-3 backbone)."""

    # chi (edge tokens)
    chi_layers: int
    chi_dim: int           # C: channels per edge token
    chi_heads: int
    orient_e: Any          # tuple length N of 0/1 (hashable)
    n_disp_e: int
    D_e: Any               # (N, N) tuple-of-tuples (hashable)
    # fixed Wilson step
    plaq_all: Any          # (n_plaq, 4) tuple-of-tuples, all edges valid (no -1)
    # Omega (plaquette tokens; the Variant-3 backbone)
    n_layers: int
    d_model: int
    n_heads: int
    n_disp: int
    D: Any                 # (n_plaq, n_plaq) tuple-of-tuples (hashable)
    readout_K: int
    ffn_mult: int = 2
    activation: str = "gelu"
    use_content: bool = True
    remat: bool = False
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, sigma):                               # sigma: (N_edges,)
        # --- chi: odd bias-free embedding + identity-at-init gated-attention blocks ---
        w = self.param("embed_w", nn.initializers.normal(stddev=1.0),
                       (2, self.chi_dim), self.dtype)
        x = sigma.astype(self.dtype)[:, None] * w[jnp.asarray(self.orient_e)]  # (N, C), odd

        ChiBlock = nn.remat(EncoderBlockV3) if self.remat else EncoderBlockV3
        for l in range(self.chi_layers):
            x = ChiBlock(d_model=self.chi_dim, n_heads=self.chi_heads,
                         n_disp=self.n_disp_e, D=self.D_e,
                         ffn_mult=self.ffn_mult, activation="tanh",   # odd budget for chi
                         use_content=self.use_content,
                         zero_init_out=True,                          # identity at init
                         dtype=self.dtype, name=f"chi{l}")(x)

        # --- fixed Wilson step: odd bound + channel-wise 4-product (nothing learned) ---
        t = jnp.prod(jnp.tanh(x)[jnp.asarray(self.plaq_all)], axis=1)  # (n_plaq, C)

        # --- Omega: learned projection + the Variant-3 encoder stack ---
        z = nn.Dense(self.d_model, use_bias=False, param_dtype=self.dtype,
                     name="proj")(t)                                   # (n_plaq, d)

        Block = nn.remat(EncoderBlockV3) if self.remat else EncoderBlockV3
        for l in range(self.n_layers):
            z = Block(d_model=self.d_model, n_heads=self.n_heads,
                      n_disp=self.n_disp, D=self.D,
                      ffn_mult=self.ffn_mult, activation=self.activation,
                      use_content=self.use_content, dtype=self.dtype,
                      name=f"block{l}")(z)

        z = jnp.sum(z, axis=0)                                         # sum-pool -> (d,)
        z = nn.RMSNorm(param_dtype=self.dtype, name="out_norm")(z)
        out = nn.Dense(self.readout_K, use_bias=False,
                       param_dtype=self.dtype, name="head")(z)
        return jnp.sum(_log_cosh(out))                                 # scalar log psi
