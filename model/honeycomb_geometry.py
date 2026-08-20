"""Honeycomb-lattice geometry (brick-wall embedding, smooth OBC) for the
Levin-Gu toric code and doubled-semion models (arXiv:1202.3120, Sec. IV).

Conventions (locked in discussion, 2026-08-20):

- Patch = Lx x Ly array of COMPLETE hexagons (Lx per row, Ly rows), rows
  staggered brick-wall style: row hy holds hexagons with bottom-left corner
  (hx, hy), hx = (hy % 2) + 2*i, i = 0..Lx-1 (so hx + hy is always even).
- Brick-wall square grid: honeycomb vertices sit at integer (x, y).
  Horizontal links (x,y)-(x+1,y); vertical links (x,y)-(x,y+1) exist only
  where x + y is even. Hexagon (hx, hy) is the brick face spanning
  [hx, hx+2] x [hy, hy+1].
- Qubits live on links. Link index = rank of the link midpoint under
  lexicographic (x, y) ordering (x primary), mirroring the square-lattice
  convention in model/geometry.py.
- vertex_all: (V, 3) int with fixed slots [left-horizontal, right-horizontal,
  vertical], -1 where the patch has no such link. Bulk vertices have degree 3,
  boundary vertices degree 2 -- smooth OBC truncates ONLY vertex terms.
- plaq_all / plaq_vertices / legs_all: (F, 6) int, slot-aligned cyclically,
  counterclockwise from the bottom-left corner:
    corners[k]: (hx,hy), (hx+1,hy), (hx+2,hy), (hx+2,hy+1), (hx+1,hy+1), (hx,hy+1)
    edges[k] connects corners k and k+1 (mod 6):
      h(hx,hy), h(hx+1,hy), v(hx+2,hy), h(hx+1,hy+1), h(hx,hy+1), v(hx,hy)
    legs[k] = the third link at corner k (-1 if that corner has degree 2):
      h(hx-1,hy), v(hx+1,hy-1), h(hx+2,hy), h(hx+2,hy+1), v(hx+1,hy+1), h(hx-1,hy+1)
  Every hexagon keeps all 6 boundary links (plaquette terms are never
  truncated); only legs and vertex terms are cut at the boundary.
- Counting identities (asserted at construction): F = Lx*Ly,
  N = 3*F + 2*(Lx+Ly) - 1, V = 2*(F + Lx + Ly), degree-3 vertices = 2*(F-1),
  hexagon-shared links = (Lx-1)*Ly + (2*Lx-1)*(Ly-1), cycle-space dim = F.
"""

import numpy as np


class HoneycombGeometry:
    """Duck-typed geometry object for the honeycomb models (numpy only)."""

    def __init__(self, Lx, Ly, bc="OBC"):
        assert bc == "OBC", "honeycomb geometry: only the smooth-OBC patch is implemented"
        assert Lx >= 1 and Ly >= 1, "need at least one hexagon per direction"
        self.Lx, self.Ly, self.bc = int(Lx), int(Ly), bc

        self.hex_coords = [((hy % 2) + 2 * i, hy)
                           for hy in range(self.Ly) for i in range(self.Lx)]
        self.n_plaqs = len(self.hex_coords)

        edges = sorted({e for hc in self.hex_coords for e in self._hex_edges(*hc)},
                       key=self._midpoint)
        self._edge_index = {e: i for i, e in enumerate(edges)}
        self.N = len(edges)
        self.arr_coord = np.array([self._midpoint(e) for e in edges])
        self.link_type = np.array([0 if e[0] == "h" else 1 for e in edges])

        corners = sorted({c for hc in self.hex_coords for c in self._hex_corners(*hc)})
        self._vertex_index = {c: i for i, c in enumerate(corners)}
        self.n_vertices = len(corners)
        self.vertex_positions = np.array(corners, dtype=float)
        self.plaq_positions = np.array([(hx + 1.0, hy + 0.5)
                                        for hx, hy in self.hex_coords])

        look = lambda e: self._edge_index.get(e, -1)

        rows = []
        for (x, y) in corners:
            v_up, v_dn = look(("v", x, y)), look(("v", x, y - 1))
            assert v_up == -1 or v_dn == -1  # brick rule: at most one vertical link
            rows.append([look(("h", x - 1, y)), look(("h", x, y)),
                         v_up if v_up != -1 else v_dn])
        self.vertex_all = np.array(rows, dtype=int)

        self.plaq_all = np.array([[self._edge_index[e] for e in self._hex_edges(*hc)]
                                  for hc in self.hex_coords], dtype=int)
        self.plaq_vertices = np.array(
            [[self._vertex_index[c] for c in self._hex_corners(*hc)]
             for hc in self.hex_coords], dtype=int)
        self.legs_all = np.array([[look(e) for e in self._hex_legs(*hc)]
                                  for hc in self.hex_coords], dtype=int)

        self.link_endpoints = np.array(
            [[self._vertex_index[a], self._vertex_index[b]]
             for a, b in (self._endpoints(e) for e in edges)], dtype=int)

        self._validate()

    # -- static structure of a single brick/hexagon ---------------------------

    @staticmethod
    def _midpoint(e):
        t, x, y = e
        return (x + 0.5, float(y)) if t == "h" else (float(x), y + 0.5)

    @staticmethod
    def _endpoints(e):
        t, x, y = e
        return ((x, y), (x + 1, y)) if t == "h" else ((x, y), (x, y + 1))

    @staticmethod
    def _hex_edges(hx, hy):
        return [("h", hx, hy), ("h", hx + 1, hy), ("v", hx + 2, hy),
                ("h", hx + 1, hy + 1), ("h", hx, hy + 1), ("v", hx, hy)]

    @staticmethod
    def _hex_corners(hx, hy):
        return [(hx, hy), (hx + 1, hy), (hx + 2, hy),
                (hx + 2, hy + 1), (hx + 1, hy + 1), (hx, hy + 1)]

    @staticmethod
    def _hex_legs(hx, hy):
        return [("h", hx - 1, hy), ("v", hx + 1, hy - 1), ("h", hx + 2, hy),
                ("h", hx + 2, hy + 1), ("v", hx + 1, hy + 1), ("h", hx - 1, hy + 1)]

    # -- construction-time consistency net ------------------------------------

    def _validate(self):
        Lx, Ly, F = self.Lx, self.Ly, self.n_plaqs
        assert F == Lx * Ly
        assert self.N == 3 * F + 2 * (Lx + Ly) - 1
        assert self.n_vertices == 2 * (F + Lx + Ly)
        for (t, x, y) in self._edge_index:
            if t == "v":
                assert (x + y) % 2 == 0  # brick rule

        deg = (self.vertex_all >= 0).sum(axis=1)
        assert set(deg.tolist()) <= {2, 3}
        assert (deg == 3).sum() == 2 * (F - 1)
        assert deg.sum() == 2 * self.N

        assert (self.plaq_all >= 0).all()
        counts = np.zeros(self.N, dtype=int)
        for p in range(F):
            assert len(set(self.plaq_all[p].tolist())) == 6
            counts[self.plaq_all[p]] += 1
        assert counts.min() >= 1 and counts.max() <= 2
        assert (counts == 2).sum() == (Lx - 1) * Ly + (2 * Lx - 1) * (Ly - 1)
        assert self.N - self.n_vertices + 1 == F  # cycle-space dim = F

        # slot alignment: corner k touches hexagon edges k-1 and k; its third
        # link, when present, is exactly legs_all[p, k]
        for p in range(F):
            for k in range(6):
                v = self.plaq_vertices[p, k]
                vlinks = set(self.vertex_all[v][self.vertex_all[v] >= 0].tolist())
                hex_pair = {int(self.plaq_all[p, (k - 1) % 6]),
                            int(self.plaq_all[p, k])}
                assert hex_pair <= vlinks
                third = vlinks - hex_pair
                leg = int(self.legs_all[p, k])
                if leg == -1:
                    assert third == set() and deg[v] == 2
                else:
                    assert third == {leg} and deg[v] == 3

        # slot SEMANTICS by coordinates (Phase-2 kernel code relies on slot order;
        # the set-based check above would not catch a left/right-horizontal swap)
        for v, (x, y) in enumerate(self.vertex_positions):
            lh, rh, vv = (int(a) for a in self.vertex_all[v])
            if lh != -1:
                assert tuple(self.arr_coord[lh]) == (x - 0.5, y)
            if rh != -1:
                assert tuple(self.arr_coord[rh]) == (x + 0.5, y)
            if vv != -1:
                vx, vy = self.arr_coord[vv]
                assert vx == x and abs(abs(vy - y) - 0.5) < 1e-12

        # any two hexagons share at most one link (the total shared count above
        # could in principle be satisfied by a pathological pairing)
        for p in range(F):
            sp = set(self.plaq_all[p].tolist())
            for q in range(p):
                assert len(sp & set(self.plaq_all[q].tolist())) <= 1
