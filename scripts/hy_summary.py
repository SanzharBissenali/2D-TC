#!/usr/bin/env python3
"""Summary of the pure-Y-field (hx=hz=0) sweep: locate the first-order transition.

For every NQS run results/nqs/G-equiv_1_L<L>_hx0.00_hy<hy>_<arm>.json this prints,
per (L, arm), a table over hy of:
  - E        : converged energy (last "energy" entry, real part)
  - <sy>     : order parameter = mean over bulk qubits of the final magnetization_Ymean
  - <Bp>     : final plaquette-stabilizer mean (drops as topological order dies)
  - d<sy>    : forward jump in <sy> to the next hy (first-order => a sharp spike)
and flags the hy with the largest |d<sy>| as the finite-L transition estimate.

When an L=4 ED benchmark exists (results/ed/ed_L4_hx0.00_hy<hy>.json) it also reports
E0, the gap, ED <sy>, and the NQS-vs-ED relative energy error.

Also emits a WINDOW check: whether the located jump sits at an edge of the scanned
hy range (=> the sweep window may need widening).

Usage:
    python scripts/hy_summary.py
    python scripts/hy_summary.py results/nqs        # custom NQS dir
"""
import glob
import json
import os
import re
import sys

NQS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/nqs"
ED_DIR = "results/ed"
PAT = re.compile(r"G-equiv_1_L(\d+)_hx([0-9.]+)_hy([0-9.]+)_([a-z]+)\.json$")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def final_energy(d):
    e = d.get("energy", [])
    if not e:
        return None
    try:
        return complex(e[-1]).real
    except (ValueError, TypeError):
        return None


def final_sy(d):
    """Mean over bulk qubits of the last magnetization_Ymean entry."""
    seq = d.get("order_params", {}).get("magnetization_Ymean", [])
    if not seq or not seq[-1]:
        return None
    vals = seq[-1]
    return sum(float(v) for v in vals) / len(vals)


def final_bp(d):
    seq = d.get("order_params", {}).get("Bp_mean", [])
    return float(seq[-1]) if seq else None


def ed_lookup(L, hy):
    p = os.path.join(ED_DIR, f"ed_L{L}_hx0.00_hy{hy:.2f}.json")
    d = _load(p)
    if d is None:
        return None
    return {
        "E0": d.get("E0"),
        "gap": d.get("gap"),
        "sy": d.get("magnetization_Y_mean"),
    }


def cell(x, width, prec=None):
    """Right-justified numeric cell, or '--' when x is None."""
    if x is None:
        return f"{'--':>{width}}"
    if prec is None:
        return f"{x:>{width}}"
    if prec == "e":
        return f"{x:>{width}.2e}"
    return f"{x:>{width}.{prec}f}"


def main():
    # Collect: rows[(L, arm)] = list of (hy, E, sy, bp)
    # Only COMPLETE runs (the .mpack is written after the full TDVP loop finishes,
    # per main.py) — a walltime-killed partial run has a garbage, unconverged energy
    # (e.g. E off by 20, <B_p> negative) that would corrupt the transition read.
    rows = {}
    skipped = 0
    for path in sorted(glob.glob(os.path.join(NQS_DIR, "G-equiv_1_L*_hy*_*.json"))):
        m = PAT.search(os.path.basename(path))
        if not m:
            continue
        if not os.path.exists(path[:-5] + ".mpack"):
            skipped += 1
            continue
        L, arm, hy = int(m.group(1)), m.group(4), float(m.group(3))
        d = _load(path)
        if d is None:
            continue
        rows.setdefault((L, arm), []).append(
            (hy, final_energy(d), final_sy(d), final_bp(d))
        )
    if skipped:
        print(f"(skipped {skipped} incomplete run(s) with no .mpack — still training / walltime-killed)")

    if not rows:
        print(f"no hy run files in {NQS_DIR}")
        return

    for (L, arm) in sorted(rows):
        data = sorted(rows[(L, arm)], key=lambda r: r[0])
        hys = [r[0] for r in data]
        sys_ = [r[2] for r in data]

        has_ed = L == 4
        print(f"\n=== L={L}  arm={arm}  ({len(data)} points) ===")
        hdr = f"{'hy':>5} {'E':>13} {'<sy>':>9} {'<Bp>':>8} {'d<sy>':>8}"
        if has_ed:
            hdr += f" {'E0_ED':>13} {'gap':>7} {'<sy>_ED':>8} {'relE':>9}"
        print(hdr)
        print("-" * len(hdr))

        jumps = []  # (|d<sy>|, interval_left_index, hy_at_jump)
        for i, (hy, E, sy, bp) in enumerate(data):
            # forward difference in <sy>
            dsy = None
            if sy is not None and i + 1 < len(data) and sys_[i + 1] is not None:
                dsy = sys_[i + 1] - sy
                jumps.append((abs(dsy), i, hy))
            line = (f"{hy:>5.2f} "
                    f"{cell(E, 13, 6)} {cell(sy, 9, 4)} "
                    f"{cell(bp, 8, 4)} {cell(dsy, 8, 4)}")
            if has_ed:
                ed = ed_lookup(L, hy)
                if ed and ed["E0"] is not None:
                    rel = (abs(E - ed["E0"]) / abs(ed["E0"])
                           if E is not None else None)
                    line += (f" {cell(ed['E0'], 13, 6)} {cell(ed['gap'], 7, 4)}"
                             f" {cell(ed['sy'], 8, 4)} {cell(rel, 9, 'e')}")
                else:
                    line += f" {cell(None, 13)} {cell(None, 7)} {cell(None, 8)} {cell(None, 9)}"
            print(line)

        if jumps:
            dmax, imax, hy_star = max(jumps, key=lambda t: t[0])
            last_interval = len(data) - 2  # index of the final forward-diff interval
            edge = ""
            if imax == 0:
                edge = "  <-- jump in FIRST interval: extend window DOWN"
            elif imax == last_interval:
                edge = "  <-- jump in LAST interval: extend window UP"
            print(f"  transition estimate: largest d<sy>={dmax:.4f} between "
                  f"hy={hy_star:.2f} and {hys[imax + 1]:.2f}{edge}")


if __name__ == "__main__":
    main()
