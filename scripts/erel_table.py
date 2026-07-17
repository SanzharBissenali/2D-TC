#!/usr/bin/env python3
"""Relative-energy error table for the transformer-vs-CNN L=4 comparison.

For every NQS run file results/nqs/G-equiv_1_L<L>_hx0.00_hz<hz>_<arm>.json, take the
converged energy (last "energy" entry) and, when an ED benchmark exists for that
(L, hz), report rel_err = |E_NQS - E0_ED| / |E0_ED| and whether it clears the 1e-5
kill criterion. L=6 rows print the converged energy only (no ED ground truth).

Usage:
    python scripts/erel_table.py
    python scripts/erel_table.py results/nqs      # custom NQS dir
"""
import glob
import json
import os
import re
import sys

KILL = 1e-5
NQS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/nqs"
ED_DIR = "results/ed"
PAT = re.compile(r"G-equiv_1_L(\d+)_hx([0-9.]+)_hz([0-9.]+)_([a-z]+)\.json$")


def final_energy(path):
    with open(path) as f:
        e = json.load(f).get("energy", [])
    return complex(e[-1]).real if e else None


def ed_energy(L, hz):
    p = os.path.join(ED_DIR, f"ed_L{L}_hx0.00_hz{hz:.2f}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["E0"]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(NQS_DIR, "G-equiv_1_L*_hz*_*.json"))):
        m = PAT.search(os.path.basename(path))
        if not m:
            continue
        L, _, hz_s, arm = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        hz = float(hz_s)
        e_nqs = final_energy(path)
        if e_nqs is None:
            continue
        e_ed = ed_energy(L, hz)
        rel = abs(e_nqs - e_ed) / abs(e_ed) if e_ed else None
        rows.append((L, hz, arm, e_nqs, e_ed, rel))

    if not rows:
        print(f"no matching run files in {NQS_DIR}")
        return

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    hdr = f"{'L':>2} {'hz':>5} {'arm':>4} {'E_NQS':>13} {'E_ED':>13} {'rel_err':>10} {'<=1e-5':>7}"
    print(hdr)
    print("-" * len(hdr))
    for L, hz, arm, e_nqs, e_ed, rel in rows:
        ed_s = f"{e_ed:13.6f}" if e_ed is not None else f"{'--':>13}"
        rel_s = f"{rel:10.2e}" if rel is not None else f"{'--':>10}"
        pass_s = ("PASS" if rel <= KILL else "FAIL") if rel is not None else "--"
        print(f"{L:>2} {hz:>5.2f} {arm:>4} {e_nqs:13.6f} {ed_s} {rel_s} {pass_s:>7}")


if __name__ == "__main__":
    main()
