#!/usr/bin/env python3
"""Relative-energy error table for the transformer-vs-CNN L=4 comparison.

For every NQS run file results/nqs/G-equiv_1_L<L>_hx0.00_hz<hz>_<arm>.json, take the
converged energy (median of the last `TAIL` steps — noise-robust vs a single last
step) and, when an ED benchmark exists for that (L, hz), report
rel_err = |E - E0_ED| / |E0_ED| and whether it clears the 1e-5 kill criterion.
`done` = the run's .mpack exists (training finished); rows with done=N are still
in progress — ignore their rel_err. L=6+ rows print the converged energy only.

Usage:
    python scripts/erel_table.py [results/nqs]
"""
import glob
import json
import os
import re
import statistics
import sys

KILL = 1e-5
TAIL = 10
NQS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/nqs"
ED_DIR = "results/ed"
PAT = re.compile(r"G-equiv_1_L(\d+)_hx([0-9.]+)_hz([0-9.]+)_([a-z0-9]+)\.json$")


def converged_energy(path):
    with open(path) as f:
        e = json.load(f).get("energy", [])
    if not e:
        return None
    return statistics.median(complex(x).real for x in e[-TAIL:])


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
        L, hz, arm = int(m.group(1)), float(m.group(3)), m.group(4)
        e_nqs = converged_energy(path)
        if e_nqs is None:
            continue
        done = os.path.exists(path[:-5] + ".mpack")   # .json -> .mpack
        e_ed = ed_energy(L, hz)
        rel = abs(e_nqs - e_ed) / abs(e_ed) if e_ed else None
        rows.append((L, hz, arm, e_nqs, e_ed, rel, done))

    if not rows:
        print(f"no matching run files in {NQS_DIR}")
        return

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    hdr = f"{'L':>2} {'hz':>5} {'arm':>4} {'done':>4} {'E_NQS':>13} {'E_ED':>13} {'rel_err':>10} {'<=1e-5':>7}"
    print(hdr)
    print("-" * len(hdr))
    for L, hz, arm, e_nqs, e_ed, rel, done in rows:
        ed_s = f"{e_ed:13.6f}" if e_ed is not None else f"{'--':>13}"
        rel_s = f"{rel:10.2e}" if rel is not None else f"{'--':>10}"
        pass_s = ("PASS" if rel <= KILL else "FAIL") if rel is not None else "--"
        print(f"{L:>2} {hz:>5.2f} {arm:>4} {('Y' if done else 'N'):>4} "
              f"{e_nqs:13.6f} {ed_s} {rel_s} {pass_s:>7}")

    # tf-vs-cnn readout at each hz among COMPLETED runs (the real experiment signal):
    print("\ntf-vs-cnn (completed runs; tf≈cnn ⇒ architecture matches, no tweak needed):")
    by = {}
    for L, hz, arm, e_nqs, e_ed, rel, done in rows:
        if done and rel is not None:
            by.setdefault((L, hz), {})[arm] = rel
    for (L, hz), d in sorted(by.items()):
        if "cnn" in d and "tf" in d:
            ratio = d["tf"] / d["cnn"] if d["cnn"] else float("inf")
            # Tweak ONLY if tf ABSOLUTELY fails the kill criterion AND is >2x worse than
            # cnn. A large ratio at the sub-1e-5 floor is just MC noise (both converged).
            if d["tf"] <= KILL:
                verdict = "tf PASSES <=1e-5 (converged) -> NO tweak"
            elif ratio > 2.0:
                verdict = "tf >1e-5 AND >2x cnn -> CONSIDER TWEAK"
            else:
                verdict = "both >1e-5, tf~cnn -> protocol (no tweak)"
            print(f"  L{L} hz{hz:.2f}: tf={d['tf']:.2e} cnn={d['cnn']:.2e} "
                  f"ratio={ratio:.2f}  [{verdict}]")


if __name__ == "__main__":
    main()
