"""Reference loop counter for the honeycomb patch — the sign ground truth for
the doubled semion (Phase-1 gate: DS ground-state amplitudes == (-1)**count_loops).

Input convention
----------------
z : (N,) array-like of sigma^z eigenvalues, +1 = up / empty, -1 = down / string.
    Must lie in the vertex-constrained sector: every vertex touches an even
    number (0 or 2) of down links. The down links then form disjoint simple
    closed loops; loops hugging the open boundary are ordinary cycles and need
    no special treatment.
link_endpoints : (N, 2) int array — the two vertex indices of every link
    (use HoneycombGeometry.link_endpoints).

Returns: int — the number of closed down-spin loops (0 for the all-up config).

Author: user (Phase-1 contribution, cf. the _generate_dressed_stars precedent).
The validator and the adversarial swarm carry independent counters; all
implementations must agree on every tested configuration.

Notes from the design discussion:
- In the constrained sector each vertex has down-degree 0 or 2, so the down
  subgraph is a disjoint union of simple cycles: counting loops == counting
  connected components of the down subgraph.
- Your two real choices: (a) union-find over vertices vs walk-each-loop-and-
  mark; (b) strictness — assert the even-down-degree constraint on input
  (catches bad callers early) vs trust the caller.
"""

import numpy as np


def count_loops(z, link_endpoints):
    """Count closed loops of down (-1) links. See module docstring."""
    # TODO(user): ~10 lines.
    raise NotImplementedError


def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from model.honeycomb_geometry import HoneycombGeometry

    def cfg(g, plaqs):
        z = np.ones(g.N, dtype=int)
        for p in plaqs:
            z[g.plaq_all[p]] *= -1  # XOR semantics: shared links flip twice
        return z

    cases = []
    g12 = HoneycombGeometry(1, 2)
    cases += [(g12, (), 0), (g12, (0,), 1), (g12, (1,), 1), (g12, (0, 1), 1)]
    # 2x2 hexagon order: p0=(0,0) p1=(2,0) p2=(1,1) p3=(3,1); only p0-p3 not adjacent
    g22 = HoneycombGeometry(2, 2)
    cases += [(g22, (0,), 1),
              (g22, (0, 3), 2),        # disjoint pair -> 2 loops
              (g22, (0, 1), 1),        # fused within-row pair -> 1 loop
              (g22, (0, 2), 1),        # fused between-row pair -> 1 loop
              (g22, (0, 1, 2), 1),
              (g22, (0, 1, 2, 3), 1)]  # simply-connected union -> 1 perimeter loop
    ok = True
    for g, plaqs, want in cases:
        got = count_loops(cfg(g, plaqs), g.link_endpoints)
        good = got == want
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {g.Lx}x{g.Ly} flipped hexagons "
              f"{plaqs}: count_loops={got} expected={want}")
    print("ALL PASS" if ok else "FAILURES — fix before the Phase-1 sign gates run")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
