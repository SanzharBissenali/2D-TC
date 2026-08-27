"""QEC sign head for the doubled semion (Phase 4).

Deterministic, parameter-free sign function on z-basis configurations:

    s(sigma) = #loops( eps . sigma ) mod 2,   sign = (-1)^s,

where eps is the minimum-weight (MWPM) recovery of the Q_v syndrome of sigma
and #loops is counted on the recovered zero-charge configuration. Exactly the
head validated by the gate-0 diagnostic (scripts/sign_fidelity.py, results in
results/diagnostics/): F_s = 1 exactly at hx = 0 (any hz); at hx > 0 the only
error channel is tie-degenerate recoveries (semion interference), ~hx^10 and
size-independent.

Production convention (frozen): pymatching sparse blossom on the link graph
with sign_fidelity's DECODER-A weights 1 + 1e-4*arange(N) -- a deterministic
tie-break that never changes the optimal cardinality (asserted; adversarially
verified, incl. vs fusion-blossom). ONE Matching object per run so the head is
a fixed function of sigma for the whole optimization.

Host/numpy only (pymatching is not JAX-traceable). Consumers:
  - model/sign_frame.SignFramedOperator (formulation B, production): multiplies
    Hamiltonian matrix elements by sign(sigma)*sign(sigma') in get_conn_padded.
  - model/honeycomb_networks.SignedModel (formulation A, equivalence witness):
    adds 1j*pi*s(sigma) to log psi via jax.pure_callback.

Bit convention matches exact/lanczos_ed._honeycomb_direct_ed: site i <-> bit i,
bit 1 = spin DOWN (sigma^z = -1), all-up = 0. NetKet samples are +-1 floats;
bit = (spin < 0).
"""

import numpy as np

TIE_ETA = 1e-4  # decoder-A tie-break; minimality preserved while eta*N^2 < 1


class QECSignHead:
    """Build once per run; ``s01``/``sign_pm1`` map batches of +-1 spin
    configurations to {0,1} loop parities / +-1 signs."""

    def __init__(self, geometry):
        import scipy.sparse as sp
        from pymatching import Matching

        from exact.loops import count_loops

        self.N = int(geometry.N)
        self.V = int(geometry.n_vertices)
        self.F = int(geometry.n_plaqs)
        self._vertex_all = np.asarray(geometry.vertex_all)
        self._bit_values = (1 << np.arange(self.N, dtype=np.int64))

        rows = np.asarray(geometry.link_endpoints).T.ravel()
        cols = np.tile(np.arange(self.N), 2)
        Hchk = sp.csc_matrix((np.ones(2 * self.N, dtype=np.uint8), (rows, cols)),
                             shape=(self.V, self.N))
        pert = TIE_ETA * np.arange(self.N)
        assert pert.max() * self.N < 0.5, "tie perturbation could reorder cardinalities"
        self._matching = Matching.from_check_matrix(Hchk, weights=1.0 + pert)

        # (-1)^{#loops} on the whole zero-charge sector, keyed by config int
        # (2^F entries -- the hexagon flips enumerate the sector bijectively).
        masks = [int(np.bitwise_or.reduce(1 << geometry.plaq_all[p].astype(np.int64)))
                 for p in range(self.F)]
        table = {}
        for S in range(1 << self.F):
            x = 0
            for p in range(self.F):
                if (S >> p) & 1:
                    x ^= masks[p]
            z = 1 - 2 * ((x >> np.arange(self.N)) & 1)
            table[x] = count_loops(z, geometry.link_endpoints) % 2
        assert len(table) == 1 << self.F, "hexagon flips are not independent?!"
        self._s01_table = table

    def _bits(self, states):
        """(B, N) +-1 spins (any float/int dtype) -> (B, N) uint8 bits."""
        states = np.asarray(states)
        assert states.shape[-1] == self.N, \
            f"expected {self.N} sites, got {states.shape[-1]}"
        return (states.reshape(-1, self.N) < 0).astype(np.uint8)

    def s01(self, states):
        """Loop parity s(sigma) in {0., 1.} for a batch of +-1 configs.

        Accepts any leading shape; returns float64 with the leading shape.
        """
        states = np.asarray(states)
        lead = states.shape[:-1]
        bits = self._bits(states)
        synd = np.zeros((bits.shape[0], self.V), dtype=np.uint8)
        for v in range(self.V):
            for l in self._vertex_all[v]:
                if l != -1:
                    synd[:, v] ^= bits[:, int(l)]
        corr = self._matching.decode_batch(synd).astype(np.int64)
        recovered = (bits.astype(np.int64) @ self._bit_values) \
            ^ (corr @ self._bit_values)
        out = np.empty(bits.shape[0], dtype=np.float64)
        for i, x in enumerate(recovered):
            out[i] = self._s01_table[int(x)]   # KeyError == recovery failed
        return out.reshape(lead)

    def sign_pm1(self, states):
        """(-1)^{s(sigma)} in {+1., -1.} with the input's leading shape."""
        return 1.0 - 2.0 * self.s01(states)
