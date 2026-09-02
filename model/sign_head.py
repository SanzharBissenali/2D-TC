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
with sign_fidelity's DECODER-A weights 1 + eta*arange(N), eta = TIE_ETA = 1e-4
at every size graded so far (N <= 63; model/decoders.tie_eta shrinks it as
0.4/N^2 beyond, where 1e-4 would break the minimality assert -- 6x6+) -- a
deterministic tie-break that never changes the optimal cardinality (asserted;
adversarially verified, incl. vs fusion-blossom). ONE Matching object per run
so the head is a fixed function of sigma for the whole optimization.

Scalable implementation (2026-09-02, docs/decoder_scaling.md): the loop parity
is a compiled union-find component counter over the recovered link bits
(model/decoders.loop_parity01 -- the same count as exact/loops.count_loops, so
bit-identical), replacing the 2^F sector table + int64 config packing that
capped the head at F <= ~20 / N <= 62. Construction is O(N) (no 2^F loop),
the syndrome is a vectorized gather, and every call is wall-clock accounted
(``t_head`` / ``n_head_configs``, drained by ``pop_head_stats`` for the
optimizer's per-step logging). scripts/decoder_regress.py asserts s01() is
bitwise identical to the table-based head for every decoder.

Row parallelism + connected-row reuse (2026-09-02, both bit-identical to the
flat head -- scripts/decoder_regress.py proves it at 1x2..12x12 for all five
decoders, 1 and 8 threads):
  * the numba kernels (loop parity, greedy, union-find, the conn classifier
    below) run their row loops under ``prange``; the thread count follows
    numba's default (``NUMBA_NUM_THREADS``) or ``QECSignHead(n_threads=...)``
    and is exposed as ``head.n_threads`` for run logging. pymatching's
    ``decode_batch`` holds the GIL (measured), so MWPM stays single-threaded.
  * ``sign_pm1_conn(x, xp)`` / ``s01_conn`` take the (B, N) samples together
    with their (B, C, N) connected configurations -- exactly what
    SignFramedOperator.get_conn_padded receives -- and decode each sample ONCE:
    a connected row whose difference from its sample is empty (the diagonal /
    padding rows) shares the sample's parity; one that is exactly a hexagon
    flip (all 6 links of plaquette p, identified via a link->hexagon table and
    verified exactly, no hashing) shares the sample's SYNDROME, hence -- every
    decoder being a deterministic function of the syndrome -- its correction,
    so r' = r XOR mask_p and the loop parity follows the LOCAL rule
        parity(r XOR mask_p) = parity(r) XOR 1 XOR ((n_down_legs_p(r) // 2) mod 2),
    n_down_legs_p(r) = #down links among hexagon p's existing legs on the
    recovered config r (the doubled-semion plaquette phase D_p = i^{n_down_legs},
    real on the zero-syndrome sector; the 2D "dressed-flip covariance"
    recursion). Verified against exact/loops.count_loops on 10^5 random
    (sector config, hexagon) pairs at each of 1x2..12x12 (700k checks, 0
    mismatches) before shipping; the harness re-checks it every run. Every
    other row (single-link flips change the syndrome) takes the full path;
    tie_sum (class sign, not a uniform flip) always takes the full path.
    ``t_head`` / ``n_head_configs`` still count every row, B*(1+C);
    ``n_shortcut_configs`` counts the rows served by the rule.

Host/numpy only (pymatching is not JAX-traceable). Consumers:
  - model/sign_frame.SignFramedOperator (formulation B, production): multiplies
    Hamiltonian matrix elements by sign(sigma)*sign(sigma') in get_conn_padded
    (via sign_pm1_conn).
  - model/honeycomb_networks.SignedModel (formulation A, equivalence witness):
    adds 1j*pi*s(sigma) to log psi via jax.pure_callback.

Bit convention matches exact/lanczos_ed._honeycomb_direct_ed: site i <-> bit i,
bit 1 = spin DOWN (sigma^z = -1), all-up = 0. NetKet samples are +-1 floats;
bit = (spin < 0).
"""

import time

import numpy as np

from model.decoders import _njit, _prange

TIE_ETA = 1e-4  # decoder-A tie-break slope for N <= 63 (== decoders.tie_eta there)


@_njit(parallel=True, cache=True)
def _conn_shortcut_kernel(xb, xpb, rb, s_x, link2plaq, plaq_all, legs_all, out):
    """Classify every connected row and apply the local parity rule.

    xb (B, N) sample bits, xpb (B, C, N) connected-config bits, rb (B, N)
    recovered sample configs (bits XOR correction, zero syndrome), s_x (B,)
    sample loop parities (int8, 0/1). out (B, C) int8: the parity of every
    row that is the identity (no flipped link) or exactly one hexagon flip
    (6 flipped links == the 6 edges of a plaquette, verified link by link);
    -1 for every other row (caller runs the full path there). Rows run in
    parallel (prange over samples)."""
    B, C, N = xpb.shape
    for b in _prange(B):
        for c in range(C):
            cnt = 0
            first = -1
            for l in range(N):
                if xb[b, l] != xpb[b, c, l]:
                    cnt += 1
                    if first < 0:
                        first = l
                    if cnt > 6:
                        break
            if cnt == 0:                          # identity / padding row
                out[b, c] = s_x[b]
                continue
            if cnt != 6:
                out[b, c] = -1
                continue
            hit = -1
            for j in range(2):                    # <= 2 hexagons share a link
                p = link2plaq[first, j]
                if p < 0:
                    continue
                full = True
                for k in range(6):
                    e = plaq_all[p, k]
                    if xb[b, e] == xpb[b, c, e]:
                        full = False
                        break
                if full:                          # 6 flips, all on hexagon p
                    hit = p
                    break
            if hit < 0:
                out[b, c] = -1
                continue
            nd = 0
            for k in range(6):
                e = legs_all[hit, k]
                if e >= 0 and rb[b, e]:
                    nd += 1
            if nd & 1:                            # impossible on-sector; be safe
                out[b, c] = -1
            else:
                out[b, c] = s_x[b] ^ 1 ^ ((nd >> 1) & 1)


class QECSignHead:
    """Build once per run; ``s01``/``sign_pm1`` map batches of +-1 spin
    configurations to {0,1} loop parities / +-1 signs.

    ``decoder`` selects the recovery rule (Phase 4c decoder ladder,
    model/decoders.py): 'mwpm' (default) is the frozen production head,
    byte-identical to Phase 4/4b; 'anchor'/'greedy'/'unionfind' swap the
    correction while keeping syndrome + loop-parity machinery identical;
    'tie_sum' replaces the single recovery by the coherent sign of the
    degenerate minimal class (mwpm fallback beyond its caps).

    ``n_threads``: numba thread count for the row-parallel kernels (None =
    numba's default, i.e. ``NUMBA_NUM_THREADS`` or the core count; clamped to
    that maximum). The effective value is ``self.n_threads``."""

    def __init__(self, geometry, decoder='mwpm', n_threads=None):
        import scipy.sparse as sp
        from pymatching import Matching

        from model.decoders import effective_threads, loop_parity01, tie_eta

        self.n_threads = effective_threads(n_threads)
        self.N = int(geometry.N)
        self.V = int(geometry.n_vertices)
        self.F = int(geometry.n_plaqs)
        self._vertex_all = np.asarray(geometry.vertex_all)
        self._ends = np.ascontiguousarray(geometry.link_endpoints, dtype=np.int32)
        # syndrome gather table: -1 slots point at a zero pad column (index N)
        self._synd_idx = np.where(self._vertex_all < 0, self.N,
                                  self._vertex_all).astype(np.int64)
        self._parity = loop_parity01
        # conn-shortcut tables: hexagon edges, legs (-1 = missing), link -> (<=2) hexagons
        self._plaq_all = np.ascontiguousarray(geometry.plaq_all, dtype=np.int32)
        self._legs_all = np.ascontiguousarray(geometry.legs_all, dtype=np.int32)
        assert self._plaq_all.shape == (self.F, 6) and self._legs_all.shape == (self.F, 6)
        link2plaq = np.full((self.N, 2), -1, dtype=np.int32)
        for p in range(self.F):
            for e in self._plaq_all[p]:
                slot = 0 if link2plaq[e, 0] < 0 else 1
                assert link2plaq[e, slot] < 0, "link on > 2 hexagons?!"
                link2plaq[e, slot] = p
        self._link2plaq = link2plaq

        rows = np.asarray(geometry.link_endpoints).T.ravel()
        cols = np.tile(np.arange(self.N), 2)
        Hchk = sp.csc_matrix((np.ones(2 * self.N, dtype=np.uint8), (rows, cols)),
                             shape=(self.V, self.N))
        eta = tie_eta(self.N)                    # == TIE_ETA (1e-4) for N <= 63
        assert self.N > 63 or eta == TIE_ETA
        pert = eta * np.arange(self.N)
        assert pert.max() * self.N < 0.5, "tie perturbation could reorder cardinalities"
        self._matching = Matching.from_check_matrix(Hchk, weights=1.0 + pert)
        # decoder B (reversed tie-break, sign_fidelity's second decoder):
        # used ONLY by features() to flag tie-degenerate configs -- the
        # production sign s01/sign_pm1 stays pure decoder-A.
        self._matching_b = Matching.from_check_matrix(Hchk, weights=1.0 + pert[::-1])

        # wall-clock accounting (drained per optimizer step via pop_head_stats)
        self.t_head = 0.0
        self.n_head_configs = 0
        self.n_shortcut_configs = 0      # conn rows served by the local rule
        self._t_popped = 0.0
        self._n_popped = 0

        self.decoder_name = str(decoder)
        if self.decoder_name != 'mwpm':
            from model.decoders import DECODER_NAMES, make_decoder
            assert self.decoder_name in DECODER_NAMES, \
                f"unknown decoder '{self.decoder_name}'"
            # production: no per-call validity matmul (the parity kernel
            # flags any off-sector recovery with -1, asserted below)
            self._alt = make_decoder(self.decoder_name, geometry, check=False)
        else:
            self._alt = None

    # -- instrumentation ---------------------------------------------------
    def pop_head_stats(self):
        """(seconds, #configurations) spent in the head since the last pop."""
        dt = self.t_head - self._t_popped
        n = self.n_head_configs - self._n_popped
        self._t_popped, self._n_popped = self.t_head, self.n_head_configs
        return dt, n

    # -- primitives ----------------------------------------------------------
    def _bits(self, states):
        """(B, N) +-1 spins (any float/int dtype) -> (B, N) uint8 bits."""
        states = np.asarray(states)
        assert states.shape[-1] == self.N, \
            f"expected {self.N} sites, got {states.shape[-1]}"
        return (states.reshape(-1, self.N) < 0).astype(np.uint8)

    def _syndrome(self, bits):
        """(B, N) bits -> (B, V) Q_v syndrome bits (XOR over each vertex's
        existing links; vectorized gather, -1 slots hit a zero column)."""
        pad = np.zeros((bits.shape[0], self.N + 1), dtype=np.uint8)
        pad[:, :self.N] = bits
        return np.bitwise_xor.reduce(pad[:, self._synd_idx], axis=2)

    def _parity01_bits(self, bits):
        """(K, N) uint8 zero-syndrome config bits -> {0., 1.} loop parities
        (compiled union-find counter; an off-sector row is a decoder bug)."""
        s = self._parity(bits, self._ends, self.V)
        assert (s >= 0).all(), "recovery left the sector?!"
        return s.astype(np.float64)

    def _decode(self, matching, bits, synd):
        """(loop parity s01, correction bit rows (B, N) uint8) for one decoder."""
        corr = matching.decode_batch(synd).astype(np.uint8)
        return self._parity01_bits(bits ^ corr), corr

    def _s01_bits(self, bits, want_corr=False):
        """The head on (K, N) uint8 rows (no timing): s01 float64 (K,), and
        the (K, N) uint8 corrections when ``want_corr`` (never for tie_sum,
        whose sign is a class property -- callers must not ask)."""
        synd = self._syndrome(bits)
        if self._alt is None:                       # production mwpm path
            s, corr = self._decode(self._matching, bits, synd)
        elif self.decoder_name == 'tie_sum':
            assert not want_corr, "tie_sum has no single correction"
            s, _, _, _ = self._alt.sign01(bits, synd, self._parity01_bits)
            corr = None
        else:
            corr = self._alt.corrections(synd).astype(np.uint8)
            s = self._parity01_bits(bits ^ corr)
        return (s, corr) if want_corr else s

    # -- public ------------------------------------------------------------
    def s01(self, states):
        """Loop parity s(sigma) in {0., 1.} for a batch of +-1 configs.

        Accepts any leading shape; returns float64 with the leading shape.
        """
        t0 = time.perf_counter()
        states = np.asarray(states)
        lead = states.shape[:-1]
        bits = self._bits(states)
        s = self._s01_bits(bits)
        self.t_head += time.perf_counter() - t0
        self.n_head_configs += int(bits.shape[0])
        return s.reshape(lead)

    def sign_pm1(self, states):
        """(-1)^{s(sigma)} in {+1., -1.} with the input's leading shape."""
        return 1.0 - 2.0 * self.s01(states)

    def s01_conn(self, x, xp):
        """Loop parities for samples x (..., N) AND their connected configs
        xp (..., C, N) (get_conn_padded's layout), reusing each sample's
        decode for its identity / hexagon-flip rows (module docstring).
        Returns (s (...,), sp (..., C)) float64 in {0., 1.} -- bitwise equal
        to s01(x) and s01(xp)."""
        t0 = time.perf_counter()
        x = np.asarray(x)
        xp = np.asarray(xp)
        lead = x.shape[:-1]
        assert xp.shape[:-2] == lead and xp.shape[-1] == self.N, \
            f"xp {xp.shape} does not match x {x.shape}"
        C = int(xp.shape[-2])
        xb = self._bits(x)                                   # (B, N)
        B = xb.shape[0]
        xpb = np.ascontiguousarray(self._bits(xp).reshape(B, C, self.N))
        if self.decoder_name == 'tie_sum' or C == 0:         # no shortcut
            s_x = self._s01_bits(xb)
            s_xp = (self._s01_bits(xpb.reshape(-1, self.N)) if C
                    else np.zeros(0, dtype=np.float64))
        else:
            s_x, corr = self._s01_bits(xb, want_corr=True)
            out = np.empty((B, C), dtype=np.int8)
            _conn_shortcut_kernel(xb, xpb, xb ^ corr, s_x.astype(np.int8),
                                  self._link2plaq, self._plaq_all, self._legs_all, out)
            full = np.flatnonzero(out.reshape(-1) < 0)
            s_xp = out.reshape(-1).astype(np.float64)
            if full.size:
                s_xp[full] = self._s01_bits(xpb.reshape(-1, self.N)[full])
            self.n_shortcut_configs += int(B * C - full.size)
        self.t_head += time.perf_counter() - t0
        self.n_head_configs += int(B * (1 + C))
        return s_x.reshape(lead), s_xp.reshape(lead + (C,))

    def sign_pm1_conn(self, x, xp):
        """(sign(x) (...,), sign(xp) (..., C)) in {+1., -1.}; see s01_conn."""
        s, sp = self.s01_conn(x, xp)
        return 1.0 - 2.0 * s, 1.0 - 2.0 * sp

    # -- residual-arm features (Phase 4b) --------------------------------
    @property
    def n_features(self):
        """Width K of features(): [s, d (V), r (N), t]."""
        return 1 + self.V + self.N + 1

    def features(self, states):
        """Decoder-derived features, shape lead + (K,), float64 in {0,1}.

        Columns [s, d, r, t]: s = decoder-A loop parity (== s01, the
        production head); d = Q_v syndrome bits; r = eps_A XOR eps_B, the
        ambiguity cycle where the two minimal recoveries disagree (zeros when
        they coincide); t = 1 iff the two decoders' loop PARITIES differ --
        the tie flag gating the residual phase. t is a LOWER-bound tie
        detector (linear-in-index perturbations miss sum-degenerate ties,
        measured ~0.003% of tie weight -- documented in sign_fidelity).
        """
        assert self._alt is None, \
            "features() is decoder-A/B MWPM machinery (residual arm); " \
            "pair --sign_impl residual with --decoder mwpm"
        t0 = time.perf_counter()
        states = np.asarray(states)
        lead = states.shape[:-1]
        bits = self._bits(states)
        synd = self._syndrome(bits)
        s_a, corr_a = self._decode(self._matching, bits, synd)
        s_b, corr_b = self._decode(self._matching_b, bits, synd)
        r = np.bitwise_xor(corr_a, corr_b).astype(np.float64)
        t = (s_a != s_b).astype(np.float64)
        out = np.concatenate(
            [s_a[:, None], synd.astype(np.float64), r, t[:, None]], axis=1)
        self.t_head += time.perf_counter() - t0
        self.n_head_configs += int(bits.shape[0])
        return out.reshape(lead + (self.n_features,))
