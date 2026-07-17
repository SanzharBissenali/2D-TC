#!/usr/bin/env python3
"""Night-shift summary of the phase-transition sweep.

For every completed run (G-equiv_1_L*_hx0.00_hz*.json) reports convergence and
stability, and aggregates per L: final Vscore range, convergence step, per-point
runtime, whether jobs fit their walltime, and any red flags (NaN / short run /
noisy tail). Reuses the plateau criterion from scripts/convergence.py.

Usage: python scripts/sweep_summary.py [results/nqs]
"""
import os, sys, glob, json, re
import numpy as np

TAIL = 10
CONV_TOL = 1e-3
NSTEP_EXPECTED = 200
# per-L SLURM walltime (s) and how the 15 points were chunked across array tasks
WALL = {6: 5400, 8: 10800, 10: 7200, 12: 18000}      # 1:30 / 3:00 / 2:00 / 5:00
NTASKS = {6: 2, 8: 3, 10: 15, 12: 15}                # --array size (points per task = ceil(15/N))


def load(path):
    d = json.load(open(path))
    E = np.array([complex(x).real for x in d["energy"]])
    V = np.array([complex(x).real for x in d["Vscore"]])
    rt = float(d["sim_params"]["runtime"][0]) if d["sim_params"].get("runtime") else np.nan
    return E, V, rt


def conv_step(E, tol=CONV_TOL):
    Estar = np.median(E[-TAIL:])
    within = np.abs(E - Estar) <= tol * abs(Estar)
    i = len(E)
    while i > 0 and within[i - 1]:
        i -= 1
    return i if i < len(E) else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/nqs"
    rows = {}   # L -> list of dicts
    for p in sorted(glob.glob(os.path.join(root, "G-equiv_1_L*_hx0.00_hz*.json"))):
        m = re.search(r"_L(\d+)_hx[\d.]+_hz([\d.]+)\.json", p)
        if not m:
            continue
        L, hz = int(m.group(1)), float(m.group(2))
        if L not in WALL:      # skip the L=4 reference sweep (different step budget/jobs)
            continue
        E, V, rt = load(p)
        nan = bool(np.any(np.isnan(E)))
        Estar = np.median(E[-TAIL:])
        tail_noise = np.std(E[-50:]) / abs(Estar) if len(E) >= 50 else np.nan
        rows.setdefault(L, []).append(dict(
            hz=hz, n=len(E), Estar=Estar, Vstar=np.median(V[-TAIL:]),
            conv=conv_step(E), rt=rt, nan=nan, noise=tail_noise))

    for L in sorted(rows):
        R = sorted(rows[L], key=lambda r: r["hz"])
        Vs = [r["Vstar"] for r in R]
        convs = [r["conv"] for r in R if r["conv"] is not None]
        rts = [r["rt"] for r in R if not np.isnan(r["rt"])]
        short = [r["hz"] for r in R if r["n"] < NSTEP_EXPECTED - 1]
        nans = [r["hz"] for r in R if r["nan"]]
        # walltime fit: sum runtimes within each contiguous array-task chunk
        ppt = -(-15 // NTASKS[L])
        by_hz = {r["hz"]: r["rt"] for r in R}
        hzs = sorted(by_hz)
        task_tot = [sum(by_hz[h] for h in hzs[i*ppt:(i+1)*ppt] if not np.isnan(by_hz[h]))
                    for i in range(NTASKS[L])]
        worst_task = max(task_tot) if task_tot else np.nan
        print(f"\n=== L={L}  ({len(R)}/15 points, N={2*L*(L-1)} qubits) ===")
        print(f"  Vscore(final):  min {min(Vs):.2e}   max {max(Vs):.2e}")
        print(f"  conv step@1e-3: min {min(convs)}  max {max(convs)}  (of {R[0]['n']})")
        print(f"  runtime/point:  {min(rts)/60:.1f}–{max(rts)/60:.1f} min"
              f"   | worst task total {worst_task/60:.1f} min vs {WALL[L]/60:.0f} min wall"
              f"  ({'OK' if worst_task < WALL[L] else 'OVER'})")
        print(f"  tail noise σ(E)/|E|: max {max(r['noise'] for r in R):.1e}")
        flags = []
        if nans: flags.append(f"NaN at hz={nans}")
        if short: flags.append(f"short run (<{NSTEP_EXPECTED} steps) at hz={short}")
        hi_v = [r["hz"] for r in R if r["Vstar"] > 1e-2]
        if hi_v: flags.append(f"Vscore>1e-2 at hz={hi_v}")
        print(f"  flags: {', '.join(flags) if flags else 'none'}")
        # per-point Vscore line so the h_c region is visible
        print("  hz→Vscore: " + "  ".join(f"{r['hz']:.3f}:{r['Vstar']:.1e}" for r in R))


if __name__ == "__main__":
    main()
