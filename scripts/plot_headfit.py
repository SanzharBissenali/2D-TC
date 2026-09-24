"""Learning-curve plots for scripts/pretrain_sign_mlp.py --target head
(docs/signhead_benchmark_plan.md addendum, 2026-09-24).

Reads one or more JSONs written by pretrain_sign_mlp.py and draws:
  (a) held-out sign error vs step (log-y), overall + per-k, one color per
      input JSON (overall = thick line, per-k = thin lines in the same
      color) -- from each JSON's top-level "curve" list (present for
      --target head fit runs; an --eval_only grading JSON has no top-level
      "curve" and only contributes to panel (b)).
  (b) if present, ED-weighted true-sign error vs step per field point (small
      multiples, one subplot per (hx,hz)), with the head's own ceiling
      (1-F_s, computed directly from the closed form during grading) as a
      dashed horizontal line -- from each JSON's "ed_points" key (set when a
      --target head fit run also passed --ed_points) or "points" key (an
      --eval_only *_edeval.json).

Multiple JSONs overlay on the SAME axes (e.g. hidden 64 vs 128 side by
side); pass a fit json together with its --eval_only *_edeval.json, several
fit jsons, or several eval jsons -- any mix works since each input is
classified independently by which keys it has.

Saves to figures/signbench/headfit_<tag>.png (<tag> auto-inferred from the
first json's Lx/Ly/k_max unless --tag/--out is given).

Example:
    python scripts/plot_headfit.py \\
        results/pretrain/mlp_hc2x3_head_h64d2_k12.json \\
        results/pretrain/mlp_hc2x3_head_h128d2_k12.json \\
        results/pretrain/mlp_hc2x3_head_h64d2_k12_snapshots_edeval.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _label_for(path, record):
    hidden, depth = record.get("hidden"), record.get("depth")
    if hidden is not None and depth is not None:
        return f"h{hidden}d{depth}"
    return os.path.splitext(os.path.basename(path))[0]


def _infer_tag(records):
    for r in records:
        Lx, Ly = r.get("Lx"), r.get("Ly")
        if Lx is not None and Ly is not None:
            tag = f"hc{Lx}x{Ly}"
            k_max = r.get("k_max")
            if k_max is not None:
                tag += f"_k{k_max}"
            return tag
    return "headfit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+",
                    help="one or more pretrain_sign_mlp.py --target head "
                         "fit JSONs and/or --eval_only *_edeval.json files")
    ap.add_argument("--labels", default="",
                    help="comma-separated overlay labels, one per json "
                         "(default: h{hidden}d{depth}, else the filename)")
    ap.add_argument("--tag", default="",
                    help="output filename tag (default: inferred from the "
                         "first json's Lx/Ly/k_max)")
    ap.add_argument("--out", default="",
                    help="explicit output path (overrides --tag)")
    args = ap.parse_args()

    records = []
    for p in args.jsons:
        with open(p) as f:
            records.append(json.load(f))

    labels_in = [s.strip() for s in args.labels.split(",")] if args.labels else []
    labels = [labels_in[i] if i < len(labels_in) else _label_for(p, r)
              for i, (p, r) in enumerate(zip(args.jsons, records))]
    # de-duplicate labels that collide (e.g. two eval jsons at same hidden/depth)
    seen = {}
    for i, lab in enumerate(labels):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            labels[i] = f"{lab}#{seen[lab]}"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_of = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

    have_curve = [r for r in records if r.get("curve")]
    ed_entries = []   # (label, {token: point_dict})
    for lab, r in zip(labels, records):
        pts = r.get("ed_points") or r.get("points")
        if pts:
            ed_entries.append((lab, pts))
    point_tokens = sorted({tok for _, pts in ed_entries for tok in pts})

    n_rows = 2 if ed_entries else 1
    n_cols = max(len(point_tokens), 1)
    fig = plt.figure(figsize=(max(6.5, 3.0 * n_cols), 4.5 + (3.2 if ed_entries else 0)))
    gs = fig.add_gridspec(n_rows, n_cols)

    ax0 = fig.add_subplot(gs[0, :]) if ed_entries else fig.add_subplot(gs[0, 0])
    for lab, r in zip(labels, records):
        curve = r.get("curve")
        if not curve:
            continue
        steps = [c["step"] for c in curve]
        val_err = [c["val_err"] for c in curve]
        c = color_of[lab]
        ax0.plot(steps, val_err, color=c, lw=2, label=f"{lab} (overall)")
        k_keys = sorted({k for c_ in curve for k in c_.get("val_err_per_k", {})},
                        key=lambda s: int(s))
        for kk in k_keys:
            ys = [c_.get("val_err_per_k", {}).get(kk, np.nan) for c_ in curve]
            ax0.plot(steps, ys, color=c, lw=0.6, alpha=0.35)
    ax0.set_yscale("log")
    ax0.set_xlabel("step")
    ax0.set_ylabel("held-out sign error (unweighted)")
    ax0.set_title("(a) held-out sign error vs step (thin = per-k)")
    if have_curve:
        ax0.legend(fontsize=8)
    else:
        ax0.text(0.5, 0.5, "no fit --curve found in the given json(s)",
                 ha="center", va="center", transform=ax0.transAxes, fontsize=9)

    if ed_entries:
        for j, tok in enumerate(point_tokens):
            ax = fig.add_subplot(gs[1, j])
            ceiling = None
            for lab, pts in ed_entries:
                if tok not in pts:
                    continue
                pt = pts[tok]
                curve = pt["curve"]
                steps = [c["step"] for c in curve]
                errs = [c["ed_err"] for c in curve]
                ax.plot(steps, errs, color=color_of[lab], lw=1.6, label=lab)
                if ceiling is None:
                    ceiling = pt["head_ceiling"]
            if ceiling is not None:
                ax.axhline(ceiling, color="k", ls="--", lw=1,
                          label="head ceiling (1-F_s)")
            ax.set_yscale("log")
            ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("ED-weighted sign error")
            hx_s, hz_s = tok.split(":")
            ax.set_title(f"hx={hx_s}, hz={hz_s}", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("(b) ED-weighted true-sign error vs step "
                     "(dashed = head ceiling)", y=0.47, fontsize=10)

    fig.tight_layout()

    if args.out:
        out_path = args.out
    else:
        tag = args.tag or _infer_tag(records)
        out_path = os.path.join(REPO, "figures", "signbench", f"headfit_{tag}.png")
    outdir = os.path.dirname(out_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
