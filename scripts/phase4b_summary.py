"""Phase-4b grading table: three sign architectures vs exact references.

Stdlib-only. Scans results/nqs/G-equiv_1_hc{P}x{Q}_ds_*_{arm}.json runs
(arms cnnqB / cnnqC / cnnqR / cnnqA / cnn; h0-named runs map to (0,0)),
grades each against the exact E0 references collected from
results/diagnostics/signfid_*.json (which also carry the head-ceiling F_s and
positive-ceiling F_plus per point) and results/ed/ed_hc*_ds_*.json, and prints
one table per patch size: per point, per arm, the median-tail energy, rel-err,
V-score, plus the ceiling columns 1-F_s / 1-F_plus for context.

No KILL constant: this experiment MEASURES where each arm stands relative to
the head ceiling; failure modes are data, not gates.

Usage:  python scripts/phase4b_summary.py [--dir results/nqs]
"""

import argparse
import glob
import json
import os
import re
import statistics


def load_refs():
    """(size, hx, hz) -> {'E0': float, 'F_s': float|None, 'F_plus': float|None,
    'T_gate': float|None, 'T_gate_branch': str|None} (T_gate keys are the
    two-branch arm's ceiling, plan Sec 7; None on signfid JSONs predating it)."""
    refs = {}
    for f in glob.glob("results/diagnostics/signfid_hc*_ds*.json"):
        d = json.load(open(f))
        size = f"{d['Lx']}x{d['Ly']}"
        for p in d["points"]:
            key = (size, round(p["hx"], 6), round(p["hz"], 6))
            refs[key] = {"E0": p["E0"], "F_s": p.get("F_s"),
                         "F_plus": p.get("F_plus"),
                         "T_gate": p.get("T_gate"),
                         "T_gate_branch": p.get("T_gate_branch")}
    for f in glob.glob("results/ed/ed_hc*_ds_*.json"):
        m = re.search(r"ed_hc(\d+x\d+)_ds_hx([\d.]+)_hz([\d.]+)\.json", f)
        if not m:
            continue
        key = (m.group(1), round(float(m.group(2)), 6), round(float(m.group(3)), 6))
        refs.setdefault(key, {"F_s": None, "F_plus": None, "T_gate": None,
                              "T_gate_branch": None})["E0"] = \
            json.load(open(f))["E0"]
    return refs


def tail_stats(run_json):
    d = json.load(open(run_json))
    E = [complex(e).real for e in d["energy"]]
    V = [complex(v).real for v in d["Vscore"]]
    k = max(1, min(20, len(E)))
    op = d.get("order_params", {})
    return (statistics.median(E[-k:]), statistics.median(V[-k:]),
            (op.get("plaq_term_mean") or [None])[-1],
            (op.get("Qv_mean") or [None])[-1],
            any(e != e for e in E[-1:]))          # NaN tail flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/nqs")
    args = ap.parse_args()

    refs = load_refs()
    pat = re.compile(
        r"G-equiv_1_hc(\d+x\d+)_ds_(?:h0|hx([\d.]+)_hz([\d.]+))_(\w+)\.json$")
    runs = {}
    for f in glob.glob(os.path.join(args.dir, "G-equiv_1_hc*_ds_*.json")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        size, hx, hz, arm = m.group(1), m.group(2), m.group(3), m.group(4)
        key = (size, round(float(hx or 0), 6), round(float(hz or 0), 6))
        runs.setdefault(key, {})[arm] = f

    for size in sorted({k[0] for k in runs}):
        print(f"\n=== hc{size} (ds) ===")
        print(f"{'hx':>5} {'hz':>5} {'arm':>6} {'E_tail':>15} {'rel-err':>9} "
              f"{'Vscore':>8} {'plaq':>9} {'1-F_s':>9} {'1-F_plus':>9}")
        for key in sorted(k for k in runs if k[0] == size):
            ref = refs.get(key, {})
            e0, fs, fp = ref.get("E0"), ref.get("F_s"), ref.get("F_plus")
            for arm in sorted(runs[key]):
                E, V, plaq, _, nan = tail_stats(runs[key][arm])
                rel = abs(E - e0) / abs(e0) if e0 else float("nan")
                print(f"{key[1]:5.2f} {key[2]:5.2f} {arm:>6} {E:15.9f} "
                      f"{rel:9.2e} {V:8.1e} "
                      f"{plaq if plaq is None else round(plaq, 5)!s:>9} "
                      f"{'' if fs is None else f'{1 - fs:9.2e}':>9} "
                      f"{'' if fp is None else f'{1 - fp:9.2e}':>9}"
                      + ("  NaN!" if nan else ""))


if __name__ == "__main__":
    main()
