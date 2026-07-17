#!/usr/bin/env python3
"""
When did a TDVP run converge? Helps pick the step budget (sim_time) for the
larger-L sweeps, where each step is expensive (L=12 ~57 s/step).

Reads the G-equiv_*.json run files written by main.py. Energy/Vscore are stored
as str(...) per metric (see utils.config.update_data); complex().real parses both
"-25.06" (float64 runs) and "(-25.06+0j)" (complex runs).

Convergence criterion (descriptive, no free judgment): the "convergence step" at
tolerance `tol` is the FIRST step from which the relative deviation of the energy
from its converged value stays below `tol` for ALL remaining steps — i.e. the start
of the final plateau, robust to a single lucky step dipping into tolerance early.
The converged value E* is the median of the last `tail` steps.

Usage:
    python scripts/convergence.py results/nqs/G-equiv_1_L4_*.json
"""
import sys, os, glob, json
import numpy as np

TOLS = (1e-3, 3e-4, 1e-4)   # relative-energy tolerances to report
TAIL = 10                   # steps used to define the converged value E*


def load(path):
    with open(path) as f:
        d = json.load(f)
    E = np.array([complex(x).real for x in d["energy"]])
    V = np.array([complex(x).real for x in d["Vscore"]]) if d.get("Vscore") else None
    return E, V


def plateau_step(E, tol, tail=TAIL):
    """First step i such that |E[j]-E*|/|E*| < tol for every j >= i (E* = median tail)."""
    Estar = np.median(E[-tail:])
    within = np.abs(E - Estar) <= tol * abs(Estar)
    # walk back from the end while the plateau holds; the plateau start is where it breaks +1
    i = len(E)
    while i > 0 and within[i - 1]:
        i -= 1
    return (i if i < len(E) else None), Estar


def main():
    paths = sorted({p for arg in sys.argv[1:] for p in glob.glob(arg)})
    if not paths:
        print("no files matched; pass e.g. results/nqs/G-equiv_1_L4_*.json")
        return
    hdr = f"{'run':42s} {'nstep':>5s} {'E*':>12s} " + " ".join(f"conv@{t:g}".rjust(11) for t in TOLS) + f" {'Vscore*':>10s}"
    print(hdr)
    print("-" * len(hdr))
    worst = {t: 0 for t in TOLS}
    for p in paths:
        E, V = load(p)
        steps = [plateau_step(E, t)[0] for t in TOLS]
        Estar = plateau_step(E, TOLS[0])[1]
        Vstar = np.median(V[-TAIL:]) if V is not None else float("nan")
        cells = " ".join(("-" if s is None else str(s)).rjust(11) for s in steps)
        print(f"{os.path.basename(p):42s} {len(E):5d} {Estar:12.5f} {cells} {Vstar:10.2e}")
        for t, s in zip(TOLS, steps):
            if s is not None:
                worst[t] = max(worst[t], s)
    print("-" * len(hdr))
    print("worst-case convergence step across runs:",
          "  ".join(f"{t:g}->{worst[t]}" for t in TOLS))
    print(f"(=> a step budget at each tolerance that covers every hz point in this set)")


if __name__ == "__main__":
    main()
