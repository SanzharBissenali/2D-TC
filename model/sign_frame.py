"""Sign-framed operator (Phase-4 formulation B, production).

For the diagonal unitary S|sigma> = (-1)^{s(sigma)}|sigma> built from a
QECSignHead, training a POSITIVE ansatz A_theta on

    H~ = S H S,   H~_{sigma sigma'} = (-1)^{s(sigma)+s(sigma')} H_{sigma sigma'}

is EXACTLY the same optimization as training the signed ansatz
psi = (-1)^s A_theta on H: the sampling distribution (|psi|^2 = A^2), every
local energy, and every log-derivative O_k = d log A / d theta coincide
number-for-number (S carries no parameters). The equivalence witness is
formulation A (honeycomb_networks.SignedModel) + scripts/ab_equivalence.py.

Implementation: a thin DiscreteOperator wrapper whose get_conn_padded delegates
to the wrapped operator and multiplies the returned matrix elements by
sign(sigma)*sign(sigma') -- via ``QECSignHead.sign_pm1_conn(x, xp)``, which
reuses each sample's decode for its diagonal and hexagon-flip connected rows
(local parity rule; see model/sign_head.py). Same numbers as the flat
``sign_pm1`` on x and on every xp row. This runs on the host numpy path netket already uses
for numba operators (PauliStrings), which is also where pymatching lives -- the
sampler never sees the head and stays fully on-device. Verified against netket
3.16.1.post1: MCState.expect on a trivial-sign wrapper reproduces the wrapped
operator BITWISE, and VMC_SRt drives the wrapper directly.

The wrapped head is exposed as ``_head`` so the optimizer loops
(simulation/optimizer.py) can drain its per-step wall-clock / configuration
counters into the run JSON (``t_head`` / ``n_head_configs``) -- the head runs
on the host inside get_conn_padded, so this is the only place its cost shows.

Real in, real out at hy=0: the wrapped honeycomb Hamiltonians are exactly real
and the signs are +-1, so E_loc stays float64 (no complex-JIT tax). At hy != 0
(Phase 4d) the wrapped H is complex Hermitian; the framing is dtype-agnostic
(mels * s * s' with s = +-1) and dtype delegates to the wrapped operator.
"""

import numpy as np
import netket as nk


class SignFramedOperator(nk.operator.DiscreteOperator):
    """S @ op @ S for a diagonal sign function sigma -> +-1."""

    def __init__(self, op, head):
        super().__init__(op.hilbert)
        self._op = op
        self._head = head

    @property
    def dtype(self):
        return self._op.dtype

    @property
    def is_hermitian(self):
        return True

    @property
    def max_conn_size(self):
        return self._op.max_conn_size

    def get_conn_padded(self, x):
        xp, mels = self._op.get_conn_padded(x)
        # one call with samples AND connected rows: the head decodes each sample
        # once and serves its identity / hexagon-flip rows from that decode
        # (bit-identical to sign_pm1 on x and on every xp row -- harness-proven)
        s, sp = self._head.sign_pm1_conn(np.asarray(x), np.asarray(xp))
        return xp, mels * s[..., None] * sp

    def __repr__(self):
        return f"SignFramedOperator(S @ {self._op!r} @ S)"
