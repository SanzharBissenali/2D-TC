#!/usr/bin/env python3
"""Fermionic-toric-code (--ftc) hx-sweep summary: Combo CNN vs PlainCNN vs ED.

Purely descriptive/diagnostic — NOT a pass/fail gate. The fTC Hamiltonian is
non-stoquastic (signful GS), so both NQS arms are *expected* to fall short of
ED to some degree; the point of this script is to quantify exactly how far
short each one falls, and how much worse the unconstrained PlainCNN baseline
does relative to the approximately-symmetric Combo CNN under the sign
problem. There is no KILL constant here (contrast scripts/erel_table.py) —
report the numbers faithfully even when they show failure.

For every NQS run file
    results/nqs/G-equiv_1_ftc_L<Lx>_hx<hx>_hz<hz>_<arm>.json
this computes the converged energy (median of the last TAIL "energy" entries,
real part — JSON stores energies as stringified complex numbers e.g.
"(-25.06+0j)"), the tail std (a convergence-quality signal — large spread
means the run hasn't settled), the final Vscore, and the final fTC order
params (Bp_mean, Av_mean, Avp_mean). When a matching ED companion
results/ed/ed_ftc_L<Lx>_hx<hx>_hz<hz>.json exists, it also reports E0, gap,
the sign-structure diagnostics (neg_amp_fraction/neg_amp_weight, present only
for real/float64 ED Hamiltonians), and rel_err = |E_nqs - E0| / |E0|.

A run is flagged done=N (incomplete) if its .mpack is missing, or if its
order_params has no recorded Av_mean/Avp_mean (whichever the arm's Hamiltonian
uses) at the end of the run — that combination is the same convergence
signal used elsewhere in this repo (scripts/erel_table.py's .mpack check +
scripts/is_complete.py's order_params non-empty check). Incomplete rows are
still printed (flagged), never silently dropped, since a partial/diverged run
is itself part of "how hard is this for NQS."

Usage:
    python scripts/ftc_summary.py
    python scripts/ftc_summary.py --glob "results/nqs/G-equiv_1_ftc_L4_*.json"
    python scripts/ftc_summary.py --ed_dir results/ed --tail 10
"""
import argparse
import glob
import json
import os
import re
import statistics

DEFAULT_GLOB = "results/nqs/G-equiv_1_ftc_L4_*.json"
DEFAULT_ED_DIR = "results/ed"
DEFAULT_TAIL = 10
PAT = re.compile(r"ftc_L(\d+)_hx([0-9.]+)_hz([0-9.]+)_([a-z0-9]+)\.json$")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def tail_energy_stats(d, tail):
    """(median, std) of the real part of the last `tail` "energy" entries."""
    e = d.get("energy", [])
    if not e:
        return None, None
    vals = []
    for x in e[-tail:]:
        try:
            vals.append(complex(x).real)
        except (ValueError, TypeError):
            continue
    if not vals:
        return None, None
    med = statistics.median(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return med, std


def final_vscore(d):
    seq = d.get("Vscore", [])
    return float(seq[-1]) if seq else None


def final_order_param(d, key):
    seq = d.get("order_params", {}).get(key, [])
    return float(seq[-1]) if seq else None


def ed_lookup(ed_dir, Lx, hx, hz):
    p = os.path.join(ed_dir, f"ed_ftc_L{Lx}_hx{hx:.2f}_hz{hz:.2f}.json")
    d = _load(p)
    if d is None:
        return None
    return {
        "E0": d.get("E0"),
        "gap": d.get("gap"),
        "neg_amp_fraction": d.get("neg_amp_fraction"),
        "neg_amp_weight": d.get("neg_amp_weight"),
    }


def cell(x, width, prec=None):
    """Right-justified numeric cell, or '--'/'—' when x is None."""
    if x is None:
        return f"{'--':>{width}}"
    if prec is None:
        return f"{x!s:>{width}}"
    if prec == "e":
        return f"{x:>{width}.2e}"
    return f"{x:>{width}.{prec}f}"


def main():
    p = argparse.ArgumentParser(
        description="Descriptive summary of the fermionic-toric-code (--ftc) "
                     "hx-sweep: Combo CNN vs PlainCNN vs ED. No pass/fail gate."
    )
    p.add_argument("--glob", default=DEFAULT_GLOB,
                    help=f"glob pattern for NQS run JSONs (default: {DEFAULT_GLOB})")
    p.add_argument("--ed_dir", default=DEFAULT_ED_DIR,
                    help=f"directory with ed_ftc_L*.json companions (default: {DEFAULT_ED_DIR})")
    p.add_argument("--tail", type=int, default=DEFAULT_TAIL,
                    help=f"number of trailing 'energy' entries to use for the "
                         f"converged-energy median/std (default: {DEFAULT_TAIL})")
    args = p.parse_args()

    # rows[(Lx, hx, hz)] = list of per-arm dicts
    rows = {}
    skipped = 0
    for path in sorted(glob.glob(args.glob)):
        m = PAT.search(os.path.basename(path))
        if not m:
            continue
        Lx = int(m.group(1))
        hx = float(m.group(2))
        hz = float(m.group(3))
        arm = m.group(4)

        d = _load(path)
        if d is None:
            skipped += 1
            continue

        e_med, e_std = tail_energy_stats(d, args.tail)
        vscore = final_vscore(d)
        bp = final_order_param(d, "Bp_mean")
        av = final_order_param(d, "Av_mean")
        avp = final_order_param(d, "Avp_mean")

        has_mpack = os.path.exists(path[:-5] + ".mpack")
        # convergence signal: whichever vertex-star order param this arm's H
        # actually uses (Avp for --ftc, but a run that predates the flag or
        # was misconfigured would only have Av) must be non-empty.
        has_star_op = (av is not None) or (avp is not None)
        done = has_mpack and has_star_op

        ed = ed_lookup(args.ed_dir, Lx, hx, hz)
        e0 = ed["E0"] if ed else None
        rel_err = abs(e_med - e0) / abs(e0) if (e_med is not None and e0) else None

        rows.setdefault((Lx, hx, hz), []).append({
            "arm": arm,
            "e_nqs": e_med,
            "e_std": e_std,
            "vscore": vscore,
            "bp": bp,
            "av": av,
            "avp": avp,
            "e0": e0,
            "gap": ed.get("gap") if ed else None,
            "neg_amp_fraction": ed.get("neg_amp_fraction") if ed else None,
            "neg_amp_weight": ed.get("neg_amp_weight") if ed else None,
            "rel_err": rel_err,
            "done": done,
        })

    if skipped:
        print(f"(skipped {skipped} unreadable/malformed run file(s))")

    if not rows:
        print(f"no matching run files for glob {args.glob!r}")
        return

    hdr = (f"{'hx':>5} {'arm':>9} {'E_nqs':>13} {'E_ed':>13} {'rel_err':>10} "
           f"{'Vscore':>10} {'Bp':>8} {'Av':>8} {'Avp':>8} {'done':>5}")
    print(hdr)
    print("=" * len(hdr))

    for (Lx, hx, hz) in sorted(rows):
        arms = sorted(rows[(Lx, hx, hz)], key=lambda r: r["arm"])
        print(f"\n--- Lx={Lx}  hx={hx:.2f}  hz={hz:.2f} "
              f"({len(arms)} arm(s)) ---")
        printed_hdr_extra = False
        for r in arms:
            line = (f"{hx:>5.2f} {r['arm']:>9} "
                    f"{cell(r['e_nqs'], 13, 6)} {cell(r['e0'], 13, 6)} "
                    f"{cell(r['rel_err'], 10, 'e')} {cell(r['vscore'], 10, 'e')} "
                    f"{cell(r['bp'], 8, 4)} {cell(r['av'], 8, 4)} {cell(r['avp'], 8, 4)} "
                    f"{('Y' if r['done'] else 'N'):>5}")
            print(line)
            # once per (Lx,hx,hz) group, show ED extras + energy-std alongside
            if r["e0"] is not None and not printed_hdr_extra:
                extra = f"       ED: gap={cell(r['gap'], 7, 4).strip()}"
                if r["neg_amp_fraction"] is not None:
                    extra += (f"  neg_amp_fraction={r['neg_amp_fraction']:.4f}"
                              f"  neg_amp_weight={r['neg_amp_weight']:.4f}")
                else:
                    extra += "  neg_amp_fraction=-- (complex ED, no sign diagnostic)"
                print(extra)
                printed_hdr_extra = True
            if r["e_nqs"] is not None:
                print(f"       {r['arm']}: E tail-std (last-{args.tail} spread) = "
                      f"{cell(r['e_std'], 8, 'e').strip()}"
                      + ("" if r["done"] else "  <-- INCOMPLETE (no .mpack or empty star order param)"))

    # --- comparative summary: how much worse does the unconstrained arm do? ---
    print("\n" + "=" * len(hdr))
    print("Combo vs PlainCNN under the sign problem (completed runs only):")
    any_comparison = False
    for (Lx, hx, hz) in sorted(rows):
        by_arm = {r["arm"]: r for r in rows[(Lx, hx, hz)] if r["done"]}
        if not by_arm:
            print(f"  Lx={Lx} hx={hx:.2f} hz={hz:.2f}: no completed arms yet")
            continue
        with_rel = {a: r["rel_err"] for a, r in by_arm.items() if r["rel_err"] is not None}
        if not with_rel:
            print(f"  Lx={Lx} hx={hx:.2f} hz={hz:.2f}: no ED match / no rel_err available "
                  f"(arms present: {', '.join(sorted(by_arm))})")
            continue
        any_comparison = True
        best_arm = min(with_rel, key=lambda a: with_rel[a])
        parts = ", ".join(f"{a}={with_rel[a]:.2e}" for a in sorted(with_rel))
        print(f"  Lx={Lx} hx={hx:.2f} hz={hz:.2f}: closest to ED = {best_arm}  ({parts})")
        if len(with_rel) >= 2:
            arms_sorted = sorted(with_rel)
            for i in range(len(arms_sorted)):
                for j in range(i + 1, len(arms_sorted)):
                    a, b = arms_sorted[i], arms_sorted[j]
                    gap = abs(with_rel[a] - with_rel[b])
                    ratio = (max(with_rel[a], with_rel[b]) / min(with_rel[a], with_rel[b])
                             if min(with_rel[a], with_rel[b]) > 0 else float("inf"))
                    print(f"      {a} vs {b}: |rel_err_{a} - rel_err_{b}| = {gap:.2e}"
                          f"  (ratio {ratio:.2f}x)")
    if not any_comparison:
        print("  (no hx point yet has >=1 completed arm with a matching ED file)")


if __name__ == "__main__":
    main()
