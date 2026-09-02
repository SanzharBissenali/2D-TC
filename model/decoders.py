"""Decoder ladder for the QEC sign head (doubled semion, Phase 4c).

Five deterministic syndrome->recovery rules, ordered by expected sign quality
(pre-registered predictions, physics discussion 2026-08-30):

  anchor     -- every defect routed to a fixed central vertex ("move all anyons
                to a common location"). Valid recovery, zero optimality: its
                recovery = minimal XOR a fundamental cycle on the co-tree
                links, so signs break at LEADING perturbative order:
                1 - F_s ~ F*hx^2 (co-tree/cycle count -- still extensive;
                swarm-verified to 0.6% at 1x2, exponent fit k=2.03).
  greedy     -- nearest-pair matching: repeatedly pair the two closest
                unmatched defects, connect by the canonical shortest path.
                Near-minimal at low defect density; errors where greedy
                pairing is non-optimal (plus the tie channel).
  unionfind  -- Delfosse--Nickerson cluster-grow-and-peel (arXiv:1709.06218),
                unweighted growth. Valid, near-MWPM accuracy in QEC practice,
                not always minimal.
  mwpm       -- exact minimum-weight matching == the frozen production head
                (model/sign_head.py: pymatching sparse blossom, decoder-A
                tie-break). Errors = tie-degenerate recoveries only, ~hx^10,
                size-independent (gate 0).
  tie_sum    -- coherent upgrade with no literature precedent: enumerate the
                DEGENERATE minimal-recovery class (all minimal-total-length
                defect pairings x all shortest paths per pair, XOR-deduped)
                and take sign( sum_class (-1)^{#loops(eps.sigma)} ) -- the
                signed-amplitude analogue of maximum-likelihood (coset-sum)
                decoding. MEASURED OUTCOME (2026-08-30, falsifying the naive
                expectation): ~= mwpm. The dominant tied classes sum to
                exactly zero (=> mwpm fallback right where mwpm errs); it
                separates only via odd-multiplicity classes on the pure-hx
                axis (-40% at 2x2 (0.8,0)) and is ~0.3% WORSE at hz != 0
                (democratic counting anti-correlates with the hz-reweighted
                amplitudes on 2:1 classes). The true tied signs are decided
                by UNEQUAL resolvent weights of the minimal recoveries (DS
                projectors split the path environments) -- a working v2 must
                weight the class by leading-order perturbation theory and be
                hz-aware, NOT count deeper (min+2) classes (min+2 is empty on
                99.6% of tied weight; counting-v2 measured 2x worse than mwpm
                at hz != 0). Swarm S4 report, session 2026-08-30. Caps
                (defect count d_max, pairing/path/class budgets) make it a
                well-defined deterministic decoder; capped or cancelled
                configs fall back to the mwpm sign and are flagged, so the
                grading stays honest.

Interface: correction decoders expose ``corrections(synd)`` mapping (B, V)
uint8 syndrome bit arrays to (B, N) uint8 link corrections whose boundary
equals the syndrome. ``TieSumDecoder`` instead exposes
``sign01(bits, synd, parity01_fn)`` because its sign is a property of the
whole degenerate class, not of one correction. Fixed vertex/link orderings,
no RNG: every decoder is a frozen deterministic function of sigma for a
whole optimization -- the same requirement the production head satisfies.

Performance layer (2026-09-02, re-implementing the verified prototypes of the
Phase-4c scaling study, docs/decoder_scaling.md): the hot paths are numba
kernels over (B, V) / (K, N) uint8 rows -- ``loop_parity01`` (union-find
component counter, the head's shared cost), ``_greedy_kernel``,
``_uf_kernel`` and the all-pairs BFS of ``_VertexGraph``; the anchor decode
is a float32 GEMM (exact while counts < 2^24). Every kernel is BIT-IDENTICAL
to the pure-Python implementation it replaces, INCLUDING tie-breaks -- the
Python versions are retained as ``*_py`` reference methods and
``scripts/decoder_regress.py`` asserts equality on exhaustive + random
workloads. Validity checks (``_check_valid``) are opt-in (``check=True``; the
harness and scripts/sign_fidelity.py's grading tables turn them on) because
the per-call boundary matmul was a hidden 60-77 us/row tax at 12x12. The
per-syndrome memo caches of greedy/unionfind were dropped: the compiled
kernels beat the dict lookups they were hiding, and production flip
neighbourhoods are provably cache-hostile (docs/decoder_scaling.md Sec. 2).
Row parallelism (2026-09-02): every per-row kernel loop is a ``numba.prange``
(``parallel=True``) -- rows are independent, all scratch is allocated inside
the loop body, and the only cross-row state is the ``nbad`` counter (a numba
reduction), so per-row results are identical at any thread count. The thread
count follows numba's default (the ``NUMBA_NUM_THREADS`` env var, default =
core count) or ``effective_threads(n)``; ``NUMBA_NUM_THREADS=1`` is the
reproducible-timing mode. pymatching's ``decode_batch`` HOLDS the GIL
(measured: 8 Python threads with one Matching each are 1.7x SLOWER than a
single call at 12x12 / 500k rows), so the MWPM decode stays single-threaded.
If numba is unavailable the kernels run as plain Python (slow, identical
results).

Bit convention matches model/sign_head.py: bit 1 = spin down, site i <-> bit i.
"""

import itertools

import numpy as np

try:
    import numba as _numba
    from numba import njit as _njit, prange as _prange
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    import warnings
    HAVE_NUMBA = False
    _numba = None
    _prange = range
    warnings.warn("model.decoders: numba not importable -- decoder/parity kernels "
                  "run as plain Python (identical results, ~100x slower)",
                  RuntimeWarning, stacklevel=2)

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

DECODER_NAMES = ("mwpm", "anchor", "greedy", "unionfind", "tie_sum")
_INF32 = np.iinfo(np.int32).max


def effective_threads(n_threads=None):
    """Numba thread count used by the prange kernels on the CALLING thread.

    ``n_threads=None`` leaves numba's default alone (``NUMBA_NUM_THREADS`` env
    var, else the core count); an integer is applied via
    ``numba.set_num_threads`` after clamping to ``[1, NUMBA_NUM_THREADS]``
    (numba cannot exceed its launch-time maximum). Returns the effective
    count (1 without numba). numba's setting is thread-local, so call this on
    the thread that runs the kernels (the head does so in its constructor;
    production runs the head on netket's main host thread)."""
    if not HAVE_NUMBA:
        return 1
    if n_threads is not None:
        n = max(1, min(int(n_threads), int(_numba.config.NUMBA_NUM_THREADS)))
        _numba.set_num_threads(n)
    return int(_numba.get_num_threads())


def tie_eta(N):
    """Decoder-A tie-break weight slope. Equals the frozen production
    TIE_ETA=1e-4 for all sizes graded so far (N <= 63, i.e. up to 4x4) and
    shrinks as 0.4/N^2 beyond, where 1e-4 would violate the minimality
    assertion eta*N*(N-1) < 0.5 (benchmark finding, 2026-08-30). Used by both
    MWPMDecoder and the production QECSignHead."""
    return min(1e-4, 0.4 / max(N * (N - 1), 1))


# --------------------------------------------------------------------------
# compiled kernels
# --------------------------------------------------------------------------

@_njit(cache=True)
def _find_root(parent, a):
    """Union-find root with path halving (root semantics only matter)."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@_njit(parallel=True, cache=True)
def _loop_parity_kernel(bits, ends, V, out):
    """(K, N) uint8 down-link bits -> out[k] = (#connected components of the
    down subgraph) mod 2, or -1 if some vertex has odd down-degree (== the
    strict-input assert of exact/loops.count_loops). Union-find over the
    vertices touched by down links: +1 component per newly seen vertex, -1
    per merging union -- the same count as count_loops, so the parity is
    bit-identical by construction (a component count is algorithm-free).
    Rows run in parallel (prange); scratch is per row."""
    K, N = bits.shape
    for k in _prange(K):
        parent = np.empty(V, np.int32)
        deg = np.empty(V, np.int32)
        for v in range(V):
            parent[v] = -1
            deg[v] = 0
        ncomp = 0
        for l in range(N):
            if bits[k, l]:
                u = ends[l, 0]
                w = ends[l, 1]
                deg[u] += 1
                deg[w] += 1
                if parent[u] == -1:
                    parent[u] = u
                    ncomp += 1
                if parent[w] == -1:
                    parent[w] = w
                    ncomp += 1
                ru = _find_root(parent, u)
                rw = _find_root(parent, w)
                if ru != rw:
                    parent[ru] = rw
                    ncomp -= 1
        odd = False
        for v in range(V):
            if deg[v] & 1:
                odd = True
                break
        if odd:
            out[k] = -1
        else:
            out[k] = ncomp & 1


def loop_parity01(bits, link_endpoints, V):
    """(K, N) uint8 zero-charge config bits -> (K,) int8 loop parities
    (#loops mod 2); -1 flags an off-sector row (odd down-degree somewhere).
    Same numbers as ``exact.loops.count_loops(z) % 2`` row by row."""
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    ends = np.ascontiguousarray(link_endpoints, dtype=np.int32)
    out = np.empty(bits.shape[0], dtype=np.int8)
    if bits.shape[0]:
        _loop_parity_kernel(bits, ends, int(V), out)
    return out


@_njit(parallel=True, cache=True)
def _bfs_all_pairs(indptr, nbr_v, nbr_l, dist, pvert, plink):
    """FIFO BFS from every source over the CSR adjacency (sorted by
    (vertex, link)) -- identical discovery order to the level-list Python
    BFS of _VertexGraph._build_py, hence identical parent tables. Sources
    run in parallel (each writes only its own table rows)."""
    V = indptr.shape[0] - 1
    for s in _prange(V):
        queue = np.empty(V, np.int32)
        dist[s, s] = 0
        queue[0] = s
        qh = 0
        qt = 1
        while qh < qt:
            u = queue[qh]
            qh += 1
            du = dist[s, u]
            for j in range(indptr[u], indptr[u + 1]):
                v = nbr_v[j]
                if dist[s, v] == _INF32:
                    dist[s, v] = du + 1
                    pvert[s, v] = u
                    plink[s, v] = nbr_l[j]
                    queue[qt] = v
                    qt += 1


@_njit(parallel=True, cache=True)
def _greedy_kernel(synd, dist, pvert, plink, out):
    """GreedyDecoder._one_py for every row: repeatedly take the pair (u, v),
    u < v, minimizing (dist, u, v) lexicographically -- scanning the ascending
    defect list with a strict '<' on dist reproduces the tuple compare exactly
    -- and XOR the canonical BFS path rooted at u. Returns #rows left with an
    unpaired defect (odd syndrome weight; 0 for valid input). Rows run in
    parallel (prange; ``nbad`` is a reduction, scratch is per row)."""
    B, V = synd.shape
    nbad = 0
    for b in _prange(B):
        defects = np.empty(V, np.int32)
        alive = np.empty(V, np.uint8)
        d = 0
        for v in range(V):
            if synd[b, v]:
                defects[d] = v
                alive[d] = 1
                d += 1
        nrem = d
        while nrem > 1:
            best = _INF32
            ba = -1
            bb = -1
            for a in range(d):
                if alive[a] == 0:
                    continue
                ua = defects[a]
                for c in range(a + 1, d):
                    if alive[c] == 0:
                        continue
                    dd = dist[ua, defects[c]]
                    if dd < best:
                        best = dd
                        ba = a
                        bb = c
            s = defects[ba]
            x = defects[bb]
            while x != s:
                out[b, plink[s, x]] ^= 1
                x = pvert[s, x]
            alive[ba] = 0
            alive[bb] = 0
            nrem -= 2
        if nrem != 0:
            nbad += 1
    return nbad


@_njit(parallel=True, cache=True)
def _uf_kernel(synd, indptr, nbr_v, nbr_l, ends, out):
    """UnionFindDecoder._one_py for every row (Delfosse--Nickerson,
    unweighted synchronous growth, root = min vertex index, then DFS-forest
    peeling). Per round: the odd clusters are frozen first, every link
    incident to one of their vertices joins the erasure, its endpoints join
    the clusters and are unioned. The end state of a round is independent of
    the processing order (set union / union-find partition / XOR parity), so
    ascending link order == the reference's sorted (l, u, v) order. The peel
    replays the reference's iterative DFS verbatim (pop, scan sorted
    adjacency, mark-on-push). Returns #rows whose peel left parity behind or
    whose growth stalled (odd total parity; 0 for valid input). Rows run in
    parallel (prange; ``nbad`` is a reduction, scratch is per row)."""
    B, V = synd.shape
    N = out.shape[1]
    nbad = 0
    for b in _prange(B):
        parent = np.empty(V, np.int32)
        par = np.empty(V, np.uint8)
        in_cl = np.empty(V, np.uint8)
        odd_v = np.empty(V, np.uint8)
        erasure = np.empty(N, np.uint8)
        new_l = np.empty(N, np.uint8)
        seen = np.empty(V, np.uint8)
        parity = np.empty(V, np.uint8)
        stack = np.empty(V, np.int32)
        ord_child = np.empty(V, np.int32)
        ord_par = np.empty(V, np.int32)
        ord_link = np.empty(V, np.int32)
        any_def = False
        for v in range(V):
            parent[v] = v
            par[v] = synd[b, v]
            in_cl[v] = synd[b, v]
            if synd[b, v]:
                any_def = True
        for l in range(N):
            erasure[l] = 0
            out[b, l] = 0
        if not any_def:
            continue

        # --- growth rounds ---
        stalled = False
        while True:
            any_odd = False
            for v in range(V):
                if in_cl[v]:
                    odd_v[v] = par[_find_root(parent, v)]
                    if odd_v[v]:
                        any_odd = True
                else:
                    odd_v[v] = 0
            if not any_odd:
                break
            for l in range(N):
                new_l[l] = 0
            n_new = 0
            for u in range(V):
                if odd_v[u]:
                    for j in range(indptr[u], indptr[u + 1]):
                        l = nbr_l[j]
                        if new_l[l] == 0:
                            new_l[l] = 1
                            if erasure[l] == 0:
                                n_new += 1
            if n_new == 0:                 # odd cluster == whole graph
                stalled = True
                break
            for l in range(N):
                if new_l[l]:
                    erasure[l] = 1
                    u = ends[l, 0]
                    w = ends[l, 1]
                    in_cl[u] = 1
                    in_cl[w] = 1
                    ru = _find_root(parent, u)
                    rw = _find_root(parent, w)
                    if ru != rw:
                        if ru < rw:
                            parent[rw] = ru
                            par[ru] ^= par[rw]
                        else:
                            parent[ru] = rw
                            par[rw] ^= par[ru]
        if stalled:
            nbad += 1
            continue

        # --- peel: spanning forest of the erasure, leaves first ---
        for v in range(V):
            seen[v] = 0
            parity[v] = synd[b, v]
        for root in range(V):
            if seen[root] or in_cl[root] == 0:
                continue
            n_ord = 0
            seen[root] = 1
            sp = 1
            stack[0] = root
            while sp > 0:
                sp -= 1
                u = stack[sp]
                for j in range(indptr[u], indptr[u + 1]):
                    v = nbr_v[j]
                    l = nbr_l[j]
                    if erasure[l] and seen[v] == 0:
                        seen[v] = 1
                        ord_child[n_ord] = v
                        ord_par[n_ord] = u
                        ord_link[n_ord] = l
                        n_ord += 1
                        stack[sp] = v
                        sp += 1
            for i in range(n_ord - 1, -1, -1):
                c = ord_child[i]
                if parity[c]:
                    out[b, ord_link[i]] ^= 1
                    parity[c] ^= 1
                    parity[ord_par[i]] ^= 1
        for v in range(V):
            if parity[v]:
                nbad += 1
                break
    return nbad


# --------------------------------------------------------------------------
# shared geometry
# --------------------------------------------------------------------------

class _VertexGraph:
    """Shared geometry cache: BFS distances, canonical shortest paths, and
    shortest-path DAG predecessors on the vertex graph (links = edges).
    Tables come from the compiled FIFO BFS (``_bfs_all_pairs``); the original
    level-list Python BFS is kept as ``_build_py`` for the regression harness."""

    def __init__(self, geometry, _py=False):
        self.N, self.V = int(geometry.N), int(geometry.n_vertices)
        self.link_endpoints = np.asarray(geometry.link_endpoints, dtype=int)
        self.ends32 = np.ascontiguousarray(self.link_endpoints, dtype=np.int32)
        adj = [[] for _ in range(self.V)]
        for l, (u, v) in enumerate(self.link_endpoints):
            adj[int(u)].append((int(v), int(l)))
            adj[int(v)].append((int(u), int(l)))
        self.adj = [sorted(a) for a in adj]      # ascending (vertex, link)
        # CSR mirror of adj (same order) for the kernels
        self.indptr = np.zeros(self.V + 1, dtype=np.int32)
        for u in range(self.V):
            self.indptr[u + 1] = self.indptr[u] + len(self.adj[u])
        flat = [vl for a in self.adj for vl in a]
        self.nbr_v = np.array([vl[0] for vl in flat], dtype=np.int32)
        self.nbr_l = np.array([vl[1] for vl in flat], dtype=np.int32)

        # incidence (V, N) uint8 for boundary/validity checks
        self.incidence = np.zeros((self.V, self.N), dtype=np.uint8)
        for l, (u, v) in enumerate(self.link_endpoints):
            self.incidence[int(u), l] = 1
            self.incidence[int(v), l] = 1
        self._incidence_T32 = np.ascontiguousarray(self.incidence.T,
                                                   dtype=np.float32)

        self.dist = np.full((self.V, self.V), _INF32, dtype=np.int32)
        self._parent_vert = np.full((self.V, self.V), -1, dtype=np.int32)
        self._parent_link = np.full((self.V, self.V), -1, dtype=np.int32)
        if _py:
            self._build_py()
        else:
            _bfs_all_pairs(self.indptr, self.nbr_v, self.nbr_l,
                           self.dist, self._parent_vert, self._parent_link)
        assert int(self.dist.max()) < _INF32, "vertex graph is disconnected?!"

    def _build_py(self):
        """Reference level-synchronous BFS (the original implementation)."""
        INF = _INF32
        for s in range(self.V):
            self.dist[s, s] = 0
            frontier = [s]
            while frontier:
                nxt = []
                for u in frontier:
                    du = self.dist[s, u]
                    for v, l in self.adj[u]:
                        if self.dist[s, v] == INF:
                            self.dist[s, v] = du + 1
                            self._parent_vert[s, v] = u
                            self._parent_link[s, v] = l
                            nxt.append(v)
                frontier = nxt

    def path_bits(self, s, v):
        """Canonical shortest path s -> v as an (N,) uint8 link-bit row
        (BFS parent chain; deterministic via the sorted adjacency)."""
        row = np.zeros(self.N, dtype=np.uint8)
        v = int(v)
        while v != s:
            row[self._parent_link[s, v]] ^= 1
            v = int(self._parent_vert[s, v])
        return row

    def boundary(self, corr):
        """(B, N) corrections -> (B, V) syndrome they produce (float32 GEMM,
        exact: per-vertex counts <= 3)."""
        prod = np.ascontiguousarray(corr, dtype=np.float32) @ self._incidence_T32
        return (prod % 2).astype(np.int64)

    def all_shortest_paths(self, s, v, cap):
        """All shortest paths s -> v as a list of (N,) uint8 rows, in a
        canonical DFS order over the BFS-level DAG; at most ``cap`` paths
        (returns (paths, truncated_flag))."""
        s, v = int(s), int(v)
        out, trunc = [], False

        def rec(node, row):
            nonlocal trunc
            if len(out) >= cap:
                trunc = True
                return
            if node == s:
                out.append(row.copy())
                return
            dn = self.dist[s, node]
            for u, l in self.adj[node]:          # ascending order = canonical
                if self.dist[s, u] == dn - 1:
                    row[l] ^= 1
                    rec(u, row)
                    row[l] ^= 1
                    if trunc and len(out) >= cap:
                        return

        rec(v, np.zeros(self.N, dtype=np.uint8))
        return out, trunc


def _check_valid(g, corr, synd):
    assert (g.boundary(corr) == synd.astype(np.int64)).all(), \
        "decoder produced an invalid recovery (boundary != syndrome)"


def _as_synd(synd):
    return np.ascontiguousarray(synd, dtype=np.uint8)


# --------------------------------------------------------------------------
# decoders
# --------------------------------------------------------------------------

class AnchorDecoder:
    """Route every defect to one fixed vertex: the vertex nearest the patch
    centroid (ties -> lowest index; deterministic). Correction = XOR of the
    canonical anchor->defect paths; even defect count (prod Q_v = 1) makes
    the anchor's own parity cancel. Decode = float32 GEMM then mod 2 (exact:
    per-link counts <= #defects <= V << 2^24)."""

    def __init__(self, geometry, check=False):
        self.g = _VertexGraph(geometry)
        self.check = bool(check)
        pos = np.asarray(geometry.vertex_positions, dtype=float)
        d2 = ((pos - pos.mean(axis=0)) ** 2).sum(axis=1)
        self.anchor = int(np.argmin(np.round(d2, 9)))
        self._paths = np.stack([self.g.path_bits(self.anchor, v)
                                for v in range(self.g.V)])   # (V, N)
        self._paths32 = np.ascontiguousarray(self._paths, dtype=np.float32)

    def corrections_py(self, synd):
        """Reference int64 matmul path (original implementation)."""
        synd = _as_synd(synd)
        return ((synd.astype(np.int64) @ self._paths.astype(np.int64)) % 2
                ).astype(np.uint8)

    def corrections(self, synd):
        synd = _as_synd(synd)
        corr = ((synd.astype(np.float32) @ self._paths32) % 2).astype(np.uint8)
        if self.check:
            _check_valid(self.g, corr, synd)
        return corr


class GreedyDecoder:
    """Nearest-pair matching: repeatedly match the two closest unmatched
    defects (tie-break lexicographic on (dist, u, v), u < v), connect by the
    canonical shortest path. Compiled batch kernel; ``_one_py`` is the
    original per-syndrome Python implementation (regression reference)."""

    def __init__(self, geometry, check=False):
        self.g = _VertexGraph(geometry)
        self.check = bool(check)

    def _one_py(self, defects):
        row = np.zeros(self.g.N, dtype=np.uint8)
        rem = [int(d) for d in defects]          # ascending (flatnonzero)
        while rem:
            best = None
            for a in range(len(rem)):
                for b in range(a + 1, len(rem)):
                    key = (int(self.g.dist[rem[a], rem[b]]), rem[a], rem[b])
                    if best is None or key < best[0]:
                        best = (key, a, b)
            _, a, b = best
            row ^= self.g.path_bits(rem[a], rem[b])
            del rem[b], rem[a]                   # b > a: delete b first
        return row

    def corrections_py(self, synd):
        """Reference: per-row _one_py (original corrections minus the cache)."""
        synd = _as_synd(synd)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        for i in range(synd.shape[0]):
            d = np.flatnonzero(synd[i])
            if d.size:
                corr[i] = self._one_py(d)
        return corr

    def corrections(self, synd):
        synd = _as_synd(synd)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        if synd.shape[0]:
            nbad = _greedy_kernel(synd, self.g.dist, self.g._parent_vert,
                                  self.g._parent_link, corr)
            assert nbad == 0, f"greedy: {nbad} odd-weight syndromes (prod Q_v != 1)"
        if self.check:
            _check_valid(self.g, corr, synd)
        return corr


class UnionFindDecoder:
    """Delfosse--Nickerson union-find decoder, unweighted synchronous growth:
    grow all odd clusters by one full edge per round (deterministic sorted
    order), merge on contact, then peel a spanning forest of the grown
    erasure to extract a valid correction. Compiled batch kernel; ``_one_py``
    is the original per-syndrome Python implementation (regression reference)."""

    def __init__(self, geometry, check=False):
        self.g = _VertexGraph(geometry)
        self.check = bool(check)

    def _one_py(self, synd_row):
        g = self.g
        parent = list(range(g.V))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                lo, hi = min(ra, rb), max(ra, rb)   # root = min index
                parent[hi] = lo
                par[lo] = par.get(lo, 0) ^ par.pop(hi, 0)

        defects = [int(d) for d in np.flatnonzero(synd_row)]
        par = {d: 1 for d in defects}            # root -> defect parity
        in_cluster = np.zeros(g.V, dtype=bool)
        in_cluster[defects] = True
        erasure = np.zeros(g.N, dtype=bool)

        while any(par.get(find(r), 0) for r in list(par)):
            grow = set()
            odd_roots = {r for r in par if find(r) == r and par[r]}
            for u in range(g.V):
                if in_cluster[u] and find(u) in odd_roots:
                    for v, l in g.adj[u]:
                        grow.add((l, u, v))
            for l, u, v in sorted(grow):
                erasure[l] = True
                if not in_cluster[v]:
                    in_cluster[v] = True
                    par.setdefault(find(v), 0)   # lone vertex joins as even
                union(u, v)

        # peel: spanning forest of the erasure, leaves first
        row = np.zeros(g.N, dtype=np.uint8)
        parity = np.asarray(synd_row, dtype=np.uint8).copy()
        seen = np.zeros(g.V, dtype=bool)
        for root in range(g.V):
            if seen[root] or not in_cluster[root]:
                continue
            order = []                            # (child, parent, link) DFS
            seen[root] = True
            stack = [root]
            while stack:
                u = stack.pop()
                for v, l in g.adj[u]:
                    if erasure[l] and not seen[v]:
                        seen[v] = True
                        order.append((v, u, l))
                        stack.append(v)
            for child, par_v, l in reversed(order):
                if parity[child]:
                    row[l] ^= 1
                    parity[child] ^= 1
                    parity[par_v] ^= 1
        assert not parity.any(), "union-find peeling left unpaired defects"
        return row

    def corrections_py(self, synd):
        """Reference: per-row _one_py (original corrections minus the cache)."""
        synd = _as_synd(synd)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        for i in range(synd.shape[0]):
            if synd[i].any():
                corr[i] = self._one_py(synd[i])
        return corr

    def corrections(self, synd):
        synd = _as_synd(synd)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        if synd.shape[0]:
            nbad = _uf_kernel(synd, self.g.indptr, self.g.nbr_v, self.g.nbr_l,
                              self.g.ends32, corr)
            assert nbad == 0, \
                f"union-find: {nbad} rows left unpaired defects (prod Q_v != 1)"
        if self.check:
            _check_valid(self.g, corr, synd)
        return corr


class MWPMDecoder:
    """Exact MWPM via pymatching with the production decoder-A tie-break
    (weights 1 + tie_eta(N)*arange(N)); byte-identical to the frozen head
    at all current sizes. Single-threaded on purpose: ``decode_batch`` holds
    the GIL (measured 2026-09-02), so Python threads cannot parallelize it."""

    def __init__(self, geometry, check=False):
        import scipy.sparse as sp
        from pymatching import Matching

        self.g = _VertexGraph(geometry)
        self.check = bool(check)
        rows = self.g.link_endpoints.T.ravel()
        cols = np.tile(np.arange(self.g.N), 2)
        H = sp.csc_matrix((np.ones(2 * self.g.N, dtype=np.uint8),
                           (rows, cols)), shape=(self.g.V, self.g.N))
        pert = tie_eta(self.g.N) * np.arange(self.g.N)
        assert pert.max() * self.g.N < 0.5, \
            "tie perturbation could reorder cardinalities"
        self._matching = Matching.from_check_matrix(H, weights=1.0 + pert)

    def corrections(self, synd):
        synd = _as_synd(synd)
        corr = self._matching.decode_batch(synd).astype(np.uint8)
        if self.check:
            _check_valid(self.g, corr, synd)
        return corr


def _group_rows(synd):
    """Group identical (B, V) uint8 rows: (unique_rows, inverse) via a
    bytes-view unique -- no integer packing, so any V works; the grouping
    order is irrelevant to the (row-order independent) consumers."""
    synd = np.ascontiguousarray(synd, dtype=np.uint8)
    B, V = synd.shape
    if B == 0:
        return synd, np.zeros(0, dtype=np.int64)
    keys = synd.view(np.dtype((np.void, V))).ravel()
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    return synd[first], np.asarray(inv).ravel()


class TieSumDecoder:
    """Coherent sum over the degenerate minimal-recovery class.

    Per syndrome, ``classes`` enumerates every distinct minimal recovery:
    branch-and-bound over defect pairings achieving the (pymatching-certified)
    minimal total distance, times all shortest paths per pair, XOR-combined
    and deduped (within the minimal class, path systems are link-disjoint, so
    the XOR cardinality always equals the minimal total -- asserted). ``sign01``
    then evaluates sign( sum (-1)^{parity(eps.sigma)} ).

    Caps keep it a total, deterministic function: syndromes with more than
    ``d_max`` defects use the mwpm sign (fallback flag); pairing/path/class
    budgets truncate in canonical order (truncated flag); an exactly
    cancelled sum uses the mwpm sign (cancelled flag -- physically these
    configs have zero leading-order amplitude, so any sign is optimal).

    ``check`` defaults to True here (diagnostic decoder; the per-class
    validity check is amortized by the class memo)."""

    def __init__(self, geometry, d_max=10, max_pairings=20000,
                 max_paths_per_pair=256, max_class=4096, check=True):
        self.g = _VertexGraph(geometry)
        self.check = bool(check)
        self.mwpm = MWPMDecoder(geometry, check=check)
        self.d_max = int(d_max)
        self.max_pairings = int(max_pairings)
        self.max_paths_per_pair = int(max_paths_per_pair)
        self.max_class = int(max_class)
        self._path_cache = {}
        self._class_cache = {}

    def _paths(self, u, v):
        key = (int(u), int(v))
        if key not in self._path_cache:
            self._path_cache[key] = self.g.all_shortest_paths(
                u, v, self.max_paths_per_pair)
        return self._path_cache[key]

    def _min_pairings(self, defects, min_total):
        """All perfect pairings of ``defects`` with total distance ==
        min_total, canonical order; (pairings, truncated)."""
        dist = self.g.dist
        out, trunc = [], False

        def lb(rem):
            if not rem:
                return 0
            near = [min(int(dist[a, b]) for b in rem if b != a) for a in rem]
            return (sum(near) + 1) // 2

        def rec(rem, partial, pairs):
            nonlocal trunc
            if len(out) >= self.max_pairings:
                trunc = True
                return
            if not rem:
                if partial == min_total:
                    out.append(list(pairs))
                return
            a = rem[0]
            for b in rem[1:]:
                d = partial + int(dist[a, b])
                rest = [x for x in rem if x not in (a, b)]
                if d + lb(rest) <= min_total:
                    pairs.append((a, b))
                    rec(rest, d, pairs)
                    pairs.pop()
                    if trunc:
                        return

        rec([int(d) for d in defects], 0, [])
        return out, trunc

    def class_for_syndrome(self, synd_row):
        """(corr_rows (K, N) uint8 or None, fallback, truncated) for ONE
        syndrome row. None => mwpm fallback (too many defects). Memoized
        per syndrome (pure memoization -- outputs unchanged)."""
        synd_row = np.asarray(synd_row, dtype=np.uint8)
        key = synd_row.tobytes()
        hit = self._class_cache.get(key)
        if hit is None:
            hit = self._class_uncached(synd_row)
            self._class_cache[key] = hit
        return hit

    def _class_uncached(self, synd_row):
        defects = np.flatnonzero(synd_row)
        if defects.size == 0:
            return np.zeros((1, self.g.N), dtype=np.uint8), False, False
        if defects.size > self.d_max:
            return None, True, False

        min_total = int(self.mwpm.corrections(
            np.asarray(synd_row, dtype=np.uint8)[None, :])[0].sum())
        pairings, trunc = self._min_pairings(defects, min_total)
        assert pairings, "MWPM-certified minimum has no pairing?!"

        seen, rows = set(), []
        for pairing in pairings:
            per_pair = []
            for a, b in pairing:
                paths, tr = self._paths(a, b)
                trunc |= tr
                per_pair.append(paths)
            for combo in itertools.product(*per_pair):
                if len(rows) >= self.max_class:
                    trunc = True
                    break
                eps = np.zeros(self.g.N, dtype=np.uint8)
                for p in combo:
                    eps ^= p
                assert int(eps.sum()) == min_total, \
                    "minimal path system shares links (should be impossible)"
                key = eps.tobytes()
                if key not in seen:
                    seen.add(key)
                    rows.append(eps)
            if trunc and len(rows) >= self.max_class:
                break
        corr = np.stack(rows)
        if self.check:
            _check_valid(self.g, corr,
                         np.tile(np.asarray(synd_row, dtype=np.uint8),
                                 (corr.shape[0], 1)))
        return corr, False, trunc

    def sign01(self, bits, synd, parity01_fn):
        """Batch signs: (s01 float64 (B,), fallback bool (B,), truncated
        bool (B,), cancelled bool (B,)). ``parity01_fn`` maps (K, N) uint8
        recovered-config bits to {0,1} loop parities. Rows are grouped by
        syndrome (bytes-view unique, any V) so each class is fetched once per
        batch (memoized across batches too) -- outputs are row-order
        independent."""
        bits = np.asarray(bits, dtype=np.uint8)
        synd = np.asarray(synd, dtype=np.uint8)
        B = bits.shape[0]
        s01 = np.zeros(B, dtype=np.float64)
        fallback = np.zeros(B, dtype=bool)
        truncated = np.zeros(B, dtype=bool)
        cancelled = np.zeros(B, dtype=bool)

        uniq, inv = _group_rows(synd)
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(np.concatenate(
            [[-1], inv[order], [-1]])))                      # group starts + end
        for gi in range(len(bounds) - 1):
            rows = order[bounds[gi]:bounds[gi + 1]]
            cls, fb, tr = self.class_for_syndrome(uniq[inv[rows[0]]])
            fallback[rows], truncated[rows] = fb, tr
            if not fb:
                rec = bits[rows][:, None, :] ^ cls[None, :, :]
                tot = np.asarray(
                    1.0 - 2.0 * parity01_fn(rec.reshape(-1, self.g.N))
                ).reshape(len(rows), -1).sum(axis=1)
                s01[rows[tot > 0]] = 0.0
                s01[rows[tot < 0]] = 1.0
                cancelled[rows[tot == 0]] = True
        need = fallback | cancelled
        if need.any():
            corr = self.mwpm.corrections(synd[need])
            s01[need] = parity01_fn(bits[need] ^ corr)
        return s01, fallback, truncated, cancelled


def make_decoder(name, geometry, check=None, **kw):
    """Factory. Correction decoders share ``.corrections``; ``tie_sum``
    returns a TieSumDecoder (sign-level interface). ``check`` turns the
    per-call boundary==syndrome validity assert on (harness / grading);
    None keeps each decoder's default (off for the correction decoders, on
    for tie_sum's memoized per-class check)."""
    name = str(name)
    ck = {} if check is None else {"check": bool(check)}
    if name == "anchor":
        return AnchorDecoder(geometry, **ck)
    if name == "greedy":
        return GreedyDecoder(geometry, **ck)
    if name == "unionfind":
        return UnionFindDecoder(geometry, **ck)
    if name == "mwpm":
        return MWPMDecoder(geometry, **ck)
    if name == "tie_sum":
        return TieSumDecoder(geometry, **ck, **kw)
    raise ValueError(f"unknown decoder '{name}' (known: {DECODER_NAMES})")
