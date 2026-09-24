#!/usr/bin/env python3
"""Learned-vs-gated sign head benchmark grading table
(docs/signhead_benchmark_plan.md Sec 6).

Stdlib-only. Reuses scripts.phase4b_summary.load_refs() for the E0 / F_s
(mwpm head ceiling) / F_plus references (same JSON sources: results/
diagnostics/signfid_hc*_ds*.json + results/ed/ed_hc*_ds_*.json) rather than
re-scanning them -- no duplicated parsing.

For the expected (Lx, Ly) x hx-grid x hz-grid x arms grid (default: 2x3,
{0.4,0.8,1.2} x {0,0.2,0.4}, cnnqM/cnnqMp/cnnqT -- the plan's Sec 2 field
points and Sec 1 arms), prints one row per (point, arm, seed) with:

    E_tail    median energy, last --tail training steps (real part)
    rel-err   |E_tail - E0| / |E0|                    (E0 from load_refs())
    1-F       1 - fidelity, from results/diagnostics/fidelity_*.json
              (scripts/nqs_fidelity.py output; F = |<psi_ED|psi_NQS>|^2)
    ceiling   arm T: T_gate (plan Sec 7 -- min over the pinned head-sector
              of the ED weight on the wrong global sign; signfid JSONs) when
              the signfid JSON carries it, else 1 - F_s of the mwpm head
              flagged '(1-Fs)' in the table / ceiling_kind '1-Fs' in the
              JSON; EXACTLY 0 for M / M-pre ((eps, x) = features_ex
              determines sigma bijectively, so there is no head information
              loss to floor against -- see docs/signhead_benchmark_plan.md Sec 6)
    std       tail energy std (convergence signal; large => not converged)
    mix       trained signed mix a (arm T only; blank for M/M-pre)
    n_params  the run's parameter count (sim_params.n_params of the run JSON)
    prior     what the sign started from (fairness column): cnnqM 'random';
              cnnqMp 'ED-pretrained@point (warm 1-Fs=<best>)' with <best>
              read from results/pretrain/mlp_hc<size>_hx<hx:g>_hz<hz:g>_
              h<hidden>d<depth>.json (best_1_minus_Fs; '?' if absent);
              cnnqT 'head-only init (mix=<mix_init>)'

A run is "missing" if its --Lx/--Ly/hx/hz/arm/seed=0 checkpoint json isn't
in --dir (still printed, flagged, never silently dropped -- ftc_summary.py's
philosophy). A tail with a NaN last energy is flagged "NaN!". Seed-1
replicas (jobs/nersc_signbench.sh's "_s<seed>" jobid suffix, seed 0 never
suffixed) are picked up opportunistically by globbing --dir / the fidelity
JSONs -- not "missing" if absent, since stage f (seed replicas) is optional
and only run for cells close to the 3x threshold.

Verdict rule (plan Sec 6, applied only when BOTH arms in a pair have 1-F for
BOTH seed 0 and a seed-1 replica at the same point): arm X "wins" over arm Y
if X's 1-F is >= 3x lower than Y's on both seeds.

Usage:
    python scripts/signbench_summary.py
    python scripts/signbench_summary.py --Lx 2 --Ly 3 \
        --hx_list 0.4,0.8,1.2 --hz_list 0,0.2,0.4 \
        --out results/diagnostics/signbench_summary.json
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase4b_summary import load_refs  # noqa: E402  (stdlib-only)

ARMS = ('cnnqM', 'cnnqMp', 'cnnqT', 'cnnqTp')
ARM_LABEL = {'cnnqM': 'M', 'cnnqMp': 'M-pre', 'cnnqT': 'T', 'cnnqTp': 'T+'}
_SEED_SUFFIX_RE = re.compile(r'_s(\d+)$')
RUN_PAT = re.compile(
    r"G-equiv_1_hc(\d+x\d+)_ds_(?:h0|hx([\d.]+)_hz([\d.]+))_"
    r"(cnnqMp|cnnqM|cnnqT)(?:_s(\d+))?\.json$")


def _strip_seed(token):
    """('cnnqM_s1', 1); ('cnnqM', 0) -- mirrors nqs_fidelity.parse_arm's
    seed-suffix convention without importing that (netket-requiring) module."""
    m = _SEED_SUFFIX_RE.search(token)
    return (token[:m.start()], int(m.group(1))) if m else (token, 0)


def _tail_stats(path, tail):
    """(E_tail median, E_tail std, Vscore_tail median, nan_flag) over the
    real part of the last ``tail`` steps."""
    d = json.load(open(path))
    E = [complex(e).real for e in d.get("energy", [])]
    V = [complex(v).real for v in d.get("Vscore", [])]
    if not E:
        return None, None, None, True
    k = max(1, min(tail, len(E)))
    e_tail = E[-k:]
    v_tail = V[-k:] if V else []
    nan = any(e != e for e in E[-1:])
    if len(e_tail) > 1:
        mean = sum(e_tail) / len(e_tail)
        # plain population stdev (not statistics.pstdev: that module raises
        # on a NaN-containing sample via an internal exact-Fraction path --
        # https://github.com/python/cpython/issues -- a NaN tail is exactly
        # the case this function must handle gracefully, propagating NaN
        # rather than crashing)
        std = math.sqrt(sum((e - mean) ** 2 for e in e_tail) / len(e_tail))
    else:
        std = 0.0
    return (statistics.median(e_tail), std,
            statistics.median(v_tail) if v_tail else None,
            nan)


def load_runs(run_dir):
    """(size, hx, hz, arm, seed) -> run json path, for every checkpointed
    (.mpack present) run under run_dir."""
    runs = {}
    for f in glob.glob(os.path.join(run_dir, "G-equiv_1_hc*_ds_*.json")):
        if not os.path.exists(f[:-len(".json")] + ".mpack"):
            continue                      # training not finished
        m = RUN_PAT.search(os.path.basename(f))
        if not m:
            continue
        size, hx, hz, arm, seed = m.groups()
        key = (size, round(float(hx or 0), 6), round(float(hz or 0), 6),
               arm, int(seed or 0))
        runs[key] = f
    return runs


def load_fidelity(diag_dir):
    """(size, hx, hz, hy, arm, seed) -> {'F', 'F_trunk', 'mix'} from every
    results/diagnostics/fidelity_*.json (scripts/nqs_fidelity.py output;
    'arm' in each record is the raw --arms token, e.g. 'cnnqM_s1'). hy is
    part of the key so a hy != 0 record can never shadow the hy = 0 run
    (this benchmark is hy = 0 only; records default hy to 0 if absent)."""
    fid = {}
    for f in glob.glob(os.path.join(diag_dir, "fidelity_*.json")):
        d = json.load(open(f))
        lx, ly = d.get("Lx"), d.get("Ly")
        if lx is None or ly is None:
            continue
        size = f"{lx}x{ly}"
        for r in d.get("records", []):
            arm, seed = _strip_seed(r["arm"])
            if arm not in ARMS:
                continue
            key = (size, round(float(r["hx"]), 6), round(float(r["hz"]), 6),
                   round(float(r.get("hy", 0.0)), 6), arm, seed)
            fid[key] = {"F": r.get("F"), "F_trunk": r.get("F_trunk"),
                        "mix": r.get("mix")}
    return fid


def _sim_param(d, key, default=None):
    """sim_params values are 1-element lists in the run JSON."""
    v = d.get("sim_params", {}).get(key, default)
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def _prior(arm, d, size, hx, hz, pretrain_dir):
    """Fairness column: what the sign was initialised from (see module doc)."""
    if arm == 'cnnqM':
        return "random", None
    if arm == 'cnnqT':
        return f"head-only init (mix={_sim_param(d, 'mix_init', 0.05):g})", None
    if arm == 'cnnqTp':
        return f"head-only init (positive mix a=exp(c), a0={_sim_param(d, 'mix_init', 0.05):g})", None
    if arm == 'cnnqMp':
        hidden, depth = _sim_param(d, 'mlp_hidden', 64), _sim_param(d, 'mlp_depth', 2)
        pj = os.path.join(pretrain_dir,
                          f"mlp_hc{size}_hx{hx:g}_hz{hz:g}_h{hidden}d{depth}.json")
        warm = None
        if os.path.exists(pj):
            pd = json.load(open(pj))
            warm = pd.get("best_1_minus_Fs", pd.get("min_1_minus_Fs",
                                                     pd.get("final_1_minus_Fs")))
        return (f"ED-pretrained@point (warm 1-Fs="
                f"{warm:.2e})" if warm is not None else
                "ED-pretrained@point (warm 1-Fs=?)"), warm
    return None, None


def build_records(size, points, arms, runs, fid, refs, tail,
                  pretrain_dir="results/pretrain"):
    """One record per (point, arm, seed present in runs/fid); seed 0 is
    always emitted (missing=True if no checkpoint), extra seeds only if
    found on disk."""
    records = []
    for hx, hz in points:
        ref = refs.get((size, hx, hz), {})
        e0, f_s = ref.get("E0"), ref.get("F_s")
        t_gate = ref.get("T_gate")
        t_plus, t_minus = ref.get("T_gate_plus"), ref.get("T_gate_minus")
        for arm in arms:
            seeds = {0} | {s for (sz, hhx, hhz, a, s) in runs
                          if (sz, hhx, hhz, a) == (size, hx, hz, arm)}
            for seed in sorted(seeds):
                key = (size, hx, hz, arm, seed)
                path = runs.get(key)
                # fixed schema regardless of missing/found, so every JSON
                # consumer (verdicts() here, analysis/07's notebook) can use
                # plain .get()/[...] without a presence check.
                rec = {"size": size, "hx": hx, "hz": hz, "arm": arm,
                       "seed": seed, "missing": path is None,
                       "E_tail": None, "E_tail_std": None, "Vscore_tail": None,
                       "nan_tail": False, "E0": e0, "rel_err": None,
                       "F": None, "F_trunk": None, "one_minus_F": None,
                       "ceiling": None, "ceiling_kind": None, "mix": None,
                       "family": None, "ceiling_used": None,
                       "n_params": None, "prior": None, "warm_1mFs": None}
                if path is None:
                    records.append(rec)
                    continue
                e_med, e_std, v_med, nan = _tail_stats(path, tail)
                rel_err = (abs(e_med - e0) / abs(e0)
                          if e_med is not None and e0 not in (None, 0) else None)
                fr = fid.get((size, hx, hz, 0.0, arm, seed), {})
                one_minus_f = (1.0 - fr["F"]) if fr.get("F") is not None else None
                if arm in ('cnnqT', 'cnnqTp'):
                    # plan Sec 7: T's ceiling is T_gate; fall back to the
                    # mwpm 1-F_s (flagged) on signfid JSONs predating it
                    if t_gate is not None:
                        ceiling, ceiling_kind = t_gate, "T_gate"
                    elif f_s is not None:
                        ceiling, ceiling_kind = 1.0 - f_s, "1-Fs"
                    else:
                        ceiling, ceiling_kind = None, None
                elif arm in ('cnnqM', 'cnnqMp'):
                    ceiling, ceiling_kind = 0.0, "exact-0"
                else:
                    ceiling, ceiling_kind = None, None
                # Which head sector the trained [a+, a-] leaves flippable: with
                # a+ < 0 the s=+1 sector can flip (s=-1 pinned => T_gate_minus
                # is the operative ceiling), a- > 0 frees s=-1; both free => 0.
                family, ceiling_used = None, None
                mix = fr.get("mix") if arm in ('cnnqT', 'cnnqTp') else None
                if arm in ('cnnqT', 'cnnqTp') and mix is not None:
                    a_p, a_m = (mix[0], mix[-1]) if isinstance(mix, list) else (mix, mix)
                    if a_p < 0 and a_m < 0:
                        family, ceiling_used = "minus-pinned", t_minus
                    elif a_p > 0 and a_m > 0:
                        family, ceiling_used = "plus-pinned", t_plus
                    elif a_p < 0 and a_m > 0:
                        family, ceiling_used = "both-free", 0.0
                    else:
                        family = "both-pinned"
                        ceiling_used = (max(t_plus, t_minus)
                                        if None not in (t_plus, t_minus) else None)
                d = json.load(open(path))
                n_params = _sim_param(d, "n_params")
                prior, warm = _prior(arm, d, size, hx, hz, pretrain_dir)
                rec.update({
                    "E_tail": e_med, "E_tail_std": e_std, "Vscore_tail": v_med,
                    "nan_tail": nan, "E0": e0, "rel_err": rel_err,
                    "F": fr.get("F"), "F_trunk": fr.get("F_trunk"),
                    "one_minus_F": one_minus_f, "ceiling": ceiling,
                    "ceiling_kind": ceiling_kind,
                    "family": family, "ceiling_used": ceiling_used,
                    "mix": mix,
                    "n_params": int(n_params) if n_params is not None else None,
                    "prior": prior, "warm_1mFs": warm,
                })
                records.append(rec)
    return records


def verdicts(records):
    """Plan Sec 6: X wins over Y at a point if 1-F(X) <= 1-F(Y)/3 on BOTH
    seed 0 and a seed-1 replica -- only evaluated when both arms have BOTH
    seeds' 1-F available."""
    by_point_arm = {}
    for r in records:
        if r["one_minus_F"] is None:
            continue
        by_point_arm.setdefault((r["size"], r["hx"], r["hz"], r["arm"]), {})[r["seed"]] = r["one_minus_F"]

    out = []
    points = sorted({(s, hx, hz) for (s, hx, hz, a) in by_point_arm})
    for size, hx, hz in points:
        present = {a: by_point_arm[(size, hx, hz, a)]
                  for a in ARMS if (size, hx, hz, a) in by_point_arm}
        arms_here = [a for a in present if {0, 1} <= present[a].keys()]
        for i, a in enumerate(arms_here):
            for b in arms_here[i + 1:]:
                fa0, fa1 = present[a][0], present[a][1]
                fb0, fb1 = present[b][0], present[b][1]
                a_wins = fa0 * 3 <= fb0 and fa1 * 3 <= fb1
                b_wins = fb0 * 3 <= fa0 and fb1 * 3 <= fa1
                winner = a if a_wins else (b if b_wins else None)
                out.append({
                    "size": size, "hx": hx, "hz": hz, "arm_a": a, "arm_b": b,
                    "one_minus_F_a_seed0": fa0, "one_minus_F_a_seed1": fa1,
                    "one_minus_F_b_seed0": fb0, "one_minus_F_b_seed1": fb1,
                    "winner": winner,
                    "rule": "1-F >= 3x lower on both seeds",
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, default=2)
    ap.add_argument("--Ly", type=int, default=3)
    ap.add_argument("--hx_list", default="0.4,0.8,1.2")
    ap.add_argument("--hz_list", default="0,0.2,0.4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dir", default="results/nqs")
    ap.add_argument("--diag_dir", default="results/diagnostics")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--pretrain_dir", default="results/pretrain",
                    help="scripts/pretrain_sign_mlp.py JSONs (M-pre warm-start 1-Fs)")
    ap.add_argument("--out", default=None, help="optional JSON dump for "
                    "analysis/07_signbench_heatmaps.ipynb")
    args = ap.parse_args()

    size = f"{args.Lx}x{args.Ly}"
    points = [(round(float(x), 6), round(float(z), 6))
              for x in args.hx_list.split(",") for z in args.hz_list.split(",")]
    arms = [a for a in args.arms.split(",") if a]

    refs = load_refs()
    runs = load_runs(args.dir)
    fid = load_fidelity(args.diag_dir)
    records = build_records(size, points, arms, runs, fid, refs, args.tail,
                            pretrain_dir=args.pretrain_dir)
    vdicts = verdicts(records)

    print(f"\n=== signbench hc{size} (ds) ===")
    print(f"{'hx':>5} {'hz':>5} {'arm':>6} {'seed':>4} {'E_tail':>15} "
          f"{'rel-err':>9} {'1-F':>10} {'ceiling':>17} {'std':>9} "
          f"{'mix':>8} {'params':>7}  prior")
    for r in records:
        if r["missing"]:
            print(f"{r['hx']:5.2f} {r['hz']:5.2f} {ARM_LABEL[r['arm']]:>6} "
                  f"{r['seed']:>4} {'MISSING':>15}")
            continue
        lm = (("%8s" % "/".join(f"{v:.3f}" for v in (r['mix'] if isinstance(r['mix'], list) else [r['mix']])))
              if r.get("mix") is not None else f"{'':>8}")
        omf = f"{r['one_minus_F']:10.3e}" if r["one_minus_F"] is not None else f"{'?':>10}"
        if r["ceiling"] is None:
            ceil = f"{'?':>17}"
        else:
            flag = " (1-Fs)" if r.get("ceiling_kind") == "1-Fs" else ""
            ceil = f"{r['ceiling']:10.3e}{flag:>7}"
        npar = f"{r['n_params']:7d}" if r.get("n_params") is not None else f"{'?':>7}"
        print(f"{r['hx']:5.2f} {r['hz']:5.2f} {ARM_LABEL[r['arm']]:>6} "
              f"{r['seed']:>4} {r['E_tail']:15.9f} "
              f"{(r['rel_err'] if r['rel_err'] is not None else float('nan')):9.2e} "
              f"{omf} {ceil} {r['E_tail_std']:9.2e} {lm} {npar}  {r.get('prior') or ''}"
              + ("  NaN!" if r["nan_tail"] else ""))

    if vdicts:
        print("\n--- verdicts (plan Sec 6: 1-F >= 3x lower on both seeds) ---")
        for v in vdicts:
            who = v["winner"] or "no winner (< 3x on both seeds)"
            print(f"  ({v['hx']:g},{v['hz']:g}) {ARM_LABEL[v['arm_a']]} vs "
                  f"{ARM_LABEL[v['arm_b']]}: {who}")
    else:
        print("\n(no verdicts: need a seed-1 replica with 1-F on BOTH arms "
              "at the same point -- stage f, not yet run)")

    n_missing = sum(r["missing"] for r in records)
    n_nan = sum(r.get("nan_tail", False) for r in records)
    print(f"\n{len(records)} rows, {n_missing} missing, {n_nan} NaN tails")

    if args.out:
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"Lx": args.Lx, "Ly": args.Ly, "size": size,
                       "points": points, "arms": arms, "tail": args.tail,
                       "records": records, "verdicts": vdicts}, f, indent=1)
        print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
