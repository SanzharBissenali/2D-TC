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
equals the syndrome (always asserted). ``TieSumDecoder`` instead exposes
``sign01(bits, synd, parity01_fn)`` because its sign is a property of the
whole degenerate class, not of one correction. Everything is pure
numpy/python (+ pymatching), with fixed vertex/link orderings and no RNG,
so every decoder is a frozen deterministic function of sigma for a whole
optimization -- the same requirement the production head satisfies.

Bit convention matches model/sign_head.py: bit 1 = spin down, site i <-> bit i.
"""

import itertools

import numpy as np

DECODER_NAMES = ("mwpm", "anchor", "greedy", "unionfind", "tie_sum")


def tie_eta(N):
    """Decoder-A tie-break weight slope. Equals the frozen production
    TIE_ETA=1e-4 for all current sizes (N <= 71) and shrinks ~0.4/N^2 beyond,
    where 1e-4 would violate the minimality assertion (benchmark finding,
    2026-08-30)."""
    return min(1e-4, 0.4 / max(N * (N - 1), 1))


class _VertexGraph:
    """Shared geometry cache: BFS distances, canonical shortest paths, and
    shortest-path DAG predecessors on the vertex graph (links = edges)."""

    def __init__(self, geometry):
        self.N, self.V = int(geometry.N), int(geometry.n_vertices)
        self.link_endpoints = np.asarray(geometry.link_endpoints, dtype=int)
        adj = [[] for _ in range(self.V)]
        for l, (u, v) in enumerate(self.link_endpoints):
            adj[int(u)].append((int(v), int(l)))
            adj[int(v)].append((int(u), int(l)))
        self.adj = [sorted(a) for a in adj]      # ascending (vertex, link)

        # incidence (V, N) uint8 for boundary/validity checks
        self.incidence = np.zeros((self.V, self.N), dtype=np.uint8)
        for l, (u, v) in enumerate(self.link_endpoints):
            self.incidence[int(u), l] = 1
            self.incidence[int(v), l] = 1

        INF = np.iinfo(np.int32).max
        self.dist = np.full((self.V, self.V), INF, dtype=np.int32)
        self._parent_vert = np.full((self.V, self.V), -1, dtype=np.int32)
        self._parent_link = np.full((self.V, self.V), -1, dtype=np.int32)
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
        assert int(self.dist.max()) < INF, "vertex graph is disconnected?!"

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
        """(B, N) corrections -> (B, V) syndrome they produce."""
        return (corr.astype(np.int64) @ self.incidence.T.astype(np.int64)
                ).astype(np.int64) % 2

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


class AnchorDecoder:
    """Route every defect to one fixed vertex: the vertex nearest the patch
    centroid (ties -> lowest index; deterministic). Correction = XOR of the
    canonical anchor->defect paths; even defect count (prod Q_v = 1) makes
    the anchor's own parity cancel."""

    def __init__(self, geometry):
        self.g = _VertexGraph(geometry)
        pos = np.asarray(geometry.vertex_positions, dtype=float)
        d2 = ((pos - pos.mean(axis=0)) ** 2).sum(axis=1)
        self.anchor = int(np.argmin(np.round(d2, 9)))
        self._paths = np.stack([self.g.path_bits(self.anchor, v)
                                for v in range(self.g.V)])   # (V, N)

    def corrections(self, synd):
        synd = np.asarray(synd, dtype=np.uint8)
        corr = ((synd.astype(np.int64) @ self._paths.astype(np.int64)) % 2
                ).astype(np.uint8)
        _check_valid(self.g, corr, synd)
        return corr


class GreedyDecoder:
    """Nearest-pair matching: repeatedly match the two closest unmatched
    defects (tie-break lexicographic on (dist, u, v), u < v), connect by the
    canonical shortest path. Per-syndrome memo cache (pure memoization --
    outputs unchanged; NQS batches revisit few distinct syndromes)."""

    def __init__(self, geometry):
        self.g = _VertexGraph(geometry)
        self._cache = {}

    def _one(self, defects):
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

    def corrections(self, synd):
        synd = np.asarray(synd, dtype=np.uint8)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        for i in range(synd.shape[0]):
            key = synd[i].tobytes()
            row = self._cache.get(key)
            if row is None:
                d = np.flatnonzero(synd[i])
                row = self._one(d) if d.size else \
                    np.zeros(self.g.N, dtype=np.uint8)
                self._cache[key] = row
            corr[i] = row
        _check_valid(self.g, corr, synd)
        return corr


class UnionFindDecoder:
    """Delfosse--Nickerson union-find decoder, unweighted synchronous growth:
    grow all odd clusters by one full edge per round (deterministic sorted
    order), merge on contact, then peel a spanning forest of the grown
    erasure to extract a valid correction. Per-syndrome memo cache (pure
    memoization -- outputs unchanged)."""

    def __init__(self, geometry):
        self.g = _VertexGraph(geometry)
        self._cache = {}

    def _one(self, synd_row):
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

    def corrections(self, synd):
        synd = np.asarray(synd, dtype=np.uint8)
        corr = np.zeros((synd.shape[0], self.g.N), dtype=np.uint8)
        for i in range(synd.shape[0]):
            key = synd[i].tobytes()
            row = self._cache.get(key)
            if row is None:
                row = self._one(synd[i]) if synd[i].any() else \
                    np.zeros(self.g.N, dtype=np.uint8)
                self._cache[key] = row
            corr[i] = row
        _check_valid(self.g, corr, synd)
        return corr


class MWPMDecoder:
    """Exact MWPM via pymatching with the production decoder-A tie-break
    (weights 1 + tie_eta(N)*arange(N)); byte-identical to the frozen head
    at all current sizes."""

    def __init__(self, geometry):
        import scipy.sparse as sp
        from pymatching import Matching

        self.g = _VertexGraph(geometry)
        rows = self.g.link_endpoints.T.ravel()
        cols = np.tile(np.arange(self.g.N), 2)
        H = sp.csc_matrix((np.ones(2 * self.g.N, dtype=np.uint8),
                           (rows, cols)), shape=(self.g.V, self.g.N))
        pert = tie_eta(self.g.N) * np.arange(self.g.N)
        assert pert.max() * self.g.N < 0.5, \
            "tie perturbation could reorder cardinalities"
        self._matching = Matching.from_check_matrix(H, weights=1.0 + pert)

    def corrections(self, synd):
        synd = np.asarray(synd, dtype=np.uint8)
        corr = self._matching.decode_batch(synd).astype(np.uint8)
        _check_valid(self.g, corr, synd)
        return corr


class TieSumDecoder:
    """Coherent sum over the degenerate minimal-recovery class.

    Per syndrome, ``classes`` enumerates every distinct minimal recovery:
    branch-and-bound over defect pairings achieving the (pymatching-certified)
    minimal total distance, times all shortest paths per matched pair,
    XOR-combined and deduped (within the minimal class, path systems are
    link-disjoint, so the XOR cardinality always equals the minimal total --
    asserted). ``sign01`` then evaluates sign( sum (-1)^{parity(eps.sigma)} ).

    Caps keep it a total, deterministic function: syndromes with more than
    ``d_max`` defects use the mwpm sign (fallback flag); pairing/path/class
    budgets truncate in canonical order (truncated flag); an exactly
    cancelled sum uses the mwpm sign (cancelled flag -- physically these
    configs have zero leading-order amplitude, so any sign is optimal)."""

    def __init__(self, geometry, d_max=10, max_pairings=20000,
                 max_paths_per_pair=256, max_class=4096):
        self.g = _VertexGraph(geometry)
        self.mwpm = MWPMDecoder(geometry)
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
        _check_valid(self.g, corr,
                     np.tile(np.asarray(synd_row, dtype=np.uint8),
                             (corr.shape[0], 1)))
        return corr, False, trunc

    def sign01(self, bits, synd, parity01_fn):
        """Batch signs: (s01 float64 (B,), fallback bool (B,), truncated
        bool (B,), cancelled bool (B,)). ``parity01_fn`` maps (K, N) uint8
        recovered-config bits to {0,1} loop parities. Rows are grouped by
        syndrome so each class is fetched once per batch (memoized across
        batches too) -- outputs are row-order independent."""
        bits = np.asarray(bits, dtype=np.uint8)
        synd = np.asarray(synd, dtype=np.uint8)
        B = bits.shape[0]
        s01 = np.zeros(B, dtype=np.float64)
        fallback = np.zeros(B, dtype=bool)
        truncated = np.zeros(B, dtype=bool)
        cancelled = np.zeros(B, dtype=bool)

        assert self.g.V <= 62, "syndrome int packing needs V <= 62"
        skey = synd.astype(np.int64) @ (1 << np.arange(self.g.V,
                                                       dtype=np.int64))
        order = np.argsort(skey, kind="stable")
        lo = 0
        while lo < B:
            hi = lo
            while hi < B and skey[order[hi]] == skey[order[lo]]:
                hi += 1
            rows = order[lo:hi]
            cls, fb, tr = self.class_for_syndrome(synd[rows[0]])
            fallback[rows], truncated[rows] = fb, tr
            if not fb:
                rec = bits[rows][:, None, :] ^ cls[None, :, :]
                tot = np.asarray(
                    1.0 - 2.0 * parity01_fn(rec.reshape(-1, self.g.N))
                ).reshape(len(rows), -1).sum(axis=1)
                s01[rows[tot > 0]] = 0.0
                s01[rows[tot < 0]] = 1.0
                cancelled[rows[tot == 0]] = True
            lo = hi
        need = fallback | cancelled
        if need.any():
            corr = self.mwpm.corrections(synd[need])
            s01[need] = parity01_fn(bits[need] ^ corr)
        return s01, fallback, truncated, cancelled


def make_decoder(name, geometry, **kw):
    """Factory. Correction decoders share ``.corrections``; ``tie_sum``
    returns a TieSumDecoder (sign-level interface)."""
    name = str(name)
    if name == "anchor":
        return AnchorDecoder(geometry)
    if name == "greedy":
        return GreedyDecoder(geometry)
    if name == "unionfind":
        return UnionFindDecoder(geometry)
    if name == "mwpm":
        return MWPMDecoder(geometry)
    if name == "tie_sum":
        return TieSumDecoder(geometry, **kw)
    raise ValueError(f"unknown decoder '{name}' (known: {DECODER_NAMES})")
