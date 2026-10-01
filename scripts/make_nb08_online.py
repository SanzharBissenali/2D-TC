"""Regenerates analysis/08_sign_learnability.ipynb (online-learning version) -- unexecuted;
run `jupyter nbconvert --to notebook --execute --inplace analysis/08_sign_learnability.ipynb`."""
import nbformat as nbf

M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []
cells.append(M(r"""# Can an MLP learn the doubled-semion sign? — online learning, no fixed training set

Pure supervised learning, no VMC and no energy. The arm-M network `m_θ(ε, x)` (tanh MLP `N+F → 128 → 128 → 1`, float64) is trained by cross-entropy on the sign, **with a fresh i.i.d. minibatch (4,096 samples) at every Adam step (lr 1e-3)** — there is no fixed training pool. Where the configuration space is huge (F ≥ 16) a fresh batch is essentially all-new, so the training loss is itself a measurement on unseen data; where it is small (2×3, small F) repeats occur (see the caveat below). The validation set is a fixed draw that the training stream can never produce.

**Input** `σ → (ε, x)`: MWPM recovery bits `ε` (N) and hexagon-flip bits `x` (F, one per hexagon: was it flipped to reach the repaired configuration). `(ε, x)` determines `σ`. The sign is (to 1e-6…1e-3 of the weight) a function of `x` alone, `(−1)^poly(x)`, the Levin–Gu cubic.

**Two held-out rules (20% held out, fixed for the run, defined by a hash — no stored sets):**

| split | held out | what it tests |
|---|---|---|
| `random` | configurations | can it fit the distribution it is trained on (a held-out configuration usually shares its `x` with training rows) |
| `pattern` | whole `x` patterns | can it predict the sign of hexagon-flip patterns it has never seen |

**Data.** *Figure 1* (27 qubits, 2×3): configurations drawn i.i.d. from the exact `|ψ_ED|²` at three field points in the topological phase, labelled with the exact ground-state sign. *Figure 2* (3×3 … 6×6): no ED exists, so the label is the head sign (exact on the `h_x = 0` line, a proxy elsewhere); configurations are a uniformly random closed-loop set plus `k ≤ 2F` random link flips. Both: fresh i.i.d. draws, no repetition control.

**Reading the plots.** x-axis = fresh samples seen (= steps × 4096). Thin lines = the three network initialisations (they share the same data stream); thick = their mean. Train = mean over the steps since the previous point, measured on each batch *before* it is used for the update. Dotted grey = always answer the training-majority sign; dotted green (Figure 1) = the computed head sign.

**What the validation set is for ED (Figure 1).** The held-out side is a hash-defined 20% of configurations (or of `x` patterns), and the heaviest configuration (all-up, 50–70% of the mass) sits on the *training* side. The validation set is therefore the *tail* of `|ψ_ED|²`, and the printed table gives how much probability mass the held-out side carries (`val_side_weight`). Figure 1 measures generalisation to rarer configurations / unseen patterns, not error under the full `|ψ|²`.

**Caveat on the train curve.** It is a held-out estimate only while samples do not repeat. At 2×3 (`2²⁷` configurations, heavy ones drawn many times) and for small F under the pattern split (64 patterns at 2×3) the training stream repeats, so only the validation curve measures generalisation there. The train curve is a window mean of pre-update batch metrics, so it lags the validation curve by about half a window.

Code: `scripts/sign_learn_online.py`, `jobs/nersc_signonline.sh`; data: `results/signonline/`. The earlier fixed-pool (200k samples, full-batch) version is in git history (`0fb1fe6`) and `results/signlearn/`."""))
cells.append(C(r"""import json, glob, os, pathlib
import numpy as np
import matplotlib.pyplot as plt

os.chdir(next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'results').is_dir()))
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
os.makedirs('figures/signlearn', exist_ok=True)
data = [json.load(open(f)) for f in sorted(glob.glob('results/signonline/signonline_*.json')) if '_smoke' not in f]
ed = sorted((d for d in data if d['cfg']['source'] == 'ed'), key=lambda d: (d['cfg']['hx'], d['cfg']['hz']))
hd = sorted((d for d in data if d['cfg']['source'] == 'head'), key=lambda d: d['F'])
name = lambda d: (f"({d['cfg']['hx']:g}, {d['cfg']['hz']:g})" if d['cfg']['source'] == 'ed'
                  else f"{d['cfg']['Lx']}×{d['cfg']['Ly']}  (F={d['F']}, N={d['N']})")
for d in ed + hd:
    c = d['curve']
    print(f"{d['cfg']['source']:5s} {name(d):22s} {d['split']:8s} complete={d['complete']!s:5s} samples {c['samples'][-1]:.2e}/{d['budget_samples']:.2e} "
          f"val rows {d['n_val']} (held-out mass {100 * d['refs']['val_side_weight']:.1f}%, {d['refs']['val_distinct_patterns']} distinct x) "
          f"{d.get('throughput_samples_per_s', float('nan')):.0f} samples/s, starved {100 * d.get('starved_frac', float('nan')):.0f}%")"""))
cells.append(C(r"""LOSS = [('train_loss', 'C0', 'train loss (fresh batches)'), ('val_loss', 'C1', 'validation loss')]
ERR = [('train_err', 'C0', 'train'), ('val_err', 'C1', 'val')]


def panel(ax, d, keys, refs=()):
    c = d['curve']
    s = np.array(c['samples'], dtype=float)
    for key, col, lab in keys:
        a = np.array(c[key], dtype=float)                   # (points, seeds)
        ax.plot(s[1:], a[1:] if key.startswith('val') else a[1:], color=col, lw=0.6, alpha=0.3)
        ax.plot(s[1:], np.nanmean(a[1:], axis=1), color=col, lw=1.8, label=lab)
    for key, col, lab in refs:
        if d['refs'].get(key) is not None:
            ax.axhline(d['refs'][key], color=col, ls=':', lw=1.3, label=lab)
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-6); ax.grid(alpha=0.25)


def figure(ds, split, title, fname, refs):
    ds = [d for d in ds if d['split'] == split]
    if not ds:
        return print(f'{fname}: no data yet')
    fig, axes = plt.subplots(2, len(ds), figsize=(4.3 * len(ds), 6.2), sharex='col', squeeze=False)
    for j, d in enumerate(ds):
        panel(axes[0, j], d, LOSS); panel(axes[1, j], d, ERR, refs)
        axes[0, j].set_title(name(d) + ('' if d['complete'] else '  [partial]')); axes[1, j].set_xlabel('fresh samples seen')
        axes[1, j].set_ylim(-1e-7, 1.5)
    axes[0, 0].set_ylabel('cross-entropy loss'); axes[1, 0].set_ylabel('sign error')
    axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=7.5, loc='lower left')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(f'figures/signlearn/{fname}.png', dpi=150); plt.show()


REF_ED = [('trainmajority_val_err', 'grey', 'constant guess'), ('head_val_err', 'C2', 'computed head sign')]
REF_HD = [('trainmajority_val_err', 'grey', 'constant guess')]"""))
cells.append(M(r"""## Figure 1 — 27 qubits, exact ED labels, three points in the topological phase

Samples drawn from `|ψ_ED|²` (so the plain error is already the ground-state-weighted error). 2×3 has only 64 `x` patterns, so the pattern split holds out 13 of them."""))
cells.append(C("figure(ed, 'random', 'Online learning, random 20% of configurations held out (ED labels, 2×3)', 'fig1a_ed_random', REF_ED)"))
cells.append(C("figure(ed, 'pattern', 'Online learning, 20% of x patterns held out (ED labels, 2×3)', 'fig1b_ed_pattern', REF_ED)"))
cells.append(M(r"""## Figure 2 — beyond ED: 3×3 to 6×6, head-sign labels

Dotted grey = the training-majority constant guess. At 5×5 and 6×6 a fresh sample essentially never repeats an `x` pattern (2²⁵ and 2³⁶ patterns), so every validation row is a new pattern in both splits."""))
cells.append(C("figure(hd, 'random', 'Online learning, random 20% of configurations held out (head labels)', 'fig2a_head_random', REF_HD)"))
cells.append(C("figure(hd, 'pattern', 'Online learning, 20% of x patterns held out (head labels)', 'fig2b_head_pattern', REF_HD)"))
cells.append(M("## Summary — final validation error against system size"))
cells.append(C(r"""fin = lambda d, k: np.array(d['curve'][k][-1], dtype=float)
rows = []
for d in ed + hd:
    rows.append((d['cfg']['source'], name(d), d['split'], d['curve']['samples'][-1], fin(d, 'train_err').mean(), fin(d, 'val_err').mean(),
                 fin(d, 'val_err').min(), fin(d, 'val_err').max(), d['refs']['trainmajority_val_err'], d['refs'].get('head_val_err'), d['complete']))
print(f"{'src':5s} {'dataset':22s} {'split':8s} {'samples':>9s} {'train':>9s} {'val mean':>9s} {'val min':>9s} {'val max':>9s} {'const':>7s} {'head':>9s}  done")
for r in rows:
    h = f"{r[9]:9.2e}" if r[9] is not None else f"{'—':>9s}"
    print(f"{r[0]:5s} {r[1]:22s} {r[2]:8s} {r[3]:9.2e} {r[4]:9.2e} {r[5]:9.2e} {r[6]:9.2e} {r[7]:9.2e} {r[8]:7.3f} {h}  {r[10]}")

if hd:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for split, col in (('random', 'C0'), ('pattern', 'C3')):
        q = [d for d in hd if d['split'] == split]
        if q:
            ax.errorbar([d['F'] for d in q], [fin(d, 'val_err').mean() for d in q],
                        yerr=[[fin(d, 'val_err').mean() - fin(d, 'val_err').min() for d in q], [fin(d, 'val_err').max() - fin(d, 'val_err').mean() for d in q]],
                        color=col, marker='o', capsize=3, label=f'validation, {split} split')
    q = [d for d in hd if d['split'] == 'random']
    ax.plot([d['F'] for d in q], [fin(d, 'train_err').mean() for d in q], 'k--s', label='train')
    ax.plot([d['F'] for d in q], [d['refs']['trainmajority_val_err'] for d in q], color='grey', ls=':', label='constant guess')
    ax.set_yscale('symlog', linthresh=1e-4); ax.set_ylim(-1e-5, 1); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_xlabel('F (hexagons)'); ax.set_ylabel('final sign error'); ax.set_title('Online learning: final error vs size (bars = min/max over inits)')
    fig.tight_layout(); fig.savefig('figures/signlearn/fig3_error_vs_size.png', dpi=150); plt.show()"""))

cells.append(M(r"""---
# Part II — much higher capacity: deep MLP, hexagonal CNN, deep transformer

The 38k-parameter MLP above leaves open whether a failure at some size is just too little capacity. Here three much larger networks are trained on the **x-only** problem (the F hexagon-flip bits in, `(−1)^poly(x)` out; no ε), with fresh GPU-generated batches and the same hash-held-out `x` patterns (20% of patterns never trained on). Code: `scripts/sign_learn_arch.py`, `jobs/nersc_signarch.sh`; data: `results/signarch/`. All float32 (TF32 matmuls), Adam with 1000-step warmup, constant learning rate.

| | architecture | parameters (F=25 / F=100) |
|---|---|---|
| `mlp` | residual MLP, width 512, 12 blocks (LayerNorm, GELU) | 3.2M / 3.2M |
| `cnn` | hexagonal CNN: 7-point stencil with direction-specific weights (self + E, W, NE, NW, SE, SW), 12 residual layers, 128 channels, zero padding at the open boundary, mean-pool + MLP head | 1.4M / 1.4M |
| `tf` | transformer: one token per hexagon (bit embedding + learned position embedding), 10 pre-LN layers, d = 192, 6 heads, mean-pool + MLP head | 4.5M / 4.5M |

For scale: the target polynomial has 113 monomials at 5×5 and 171 at 6×6, and the earlier `(ε, x)` MLP had 32k–38k parameters. Learning rates 5e-4 (mlp, cnn) and 3e-4 (tf); batch 4096 (smaller at large F for memory)."""))
cells.append(C(r"""arch = [json.load(open(f)) for f in sorted(glob.glob('results/signarch/signarch_*.json')) if '_smoke' not in f]
arch = sorted(arch, key=lambda d: (d['cfg']['Lx'], d['arch']))
COL = {'mlp': 'C0', 'cnn': 'C1', 'tf': 'C3'}
def first(d, thr):
    s = np.array(d['curve']['samples']); v = np.array([x[0] for x in d['curve']['val_err']])
    i = np.where(v < thr)[0]
    return float(s[i[0]]) if len(i) else np.nan
print(f"{'size':6s} {'arch':4s} {'params':>10s} {'batch':>6s} {'samples':>10s} {'stopped':>12s} {'final val':>10s} {'final train':>11s} {'first <0.1':>11s} {'first <1e-3':>12s} {'first <1e-5':>12s}")
for d in arch:
    c = d['curve']
    print(f"{d['cfg']['Lx']}x{d['cfg']['Ly']:<4d} {d['arch']:4s} {d['n_params']:10,d} {d['batch']:6d} {c['samples'][-1]:10.2e} {(d['stopped'] or ('budget' if d['complete'] else 'running')):>12s} "
          f"{c['val_err'][-1][0]:10.2e} {c['train_err'][-1][0]:11.2e} {first(d, 0.1 * d['refs']['trainmajority_val_err']):11.2e} {first(d, 1e-3):12.2e} {first(d, 1e-5):12.2e}")"""))
cells.append(C(r"""sizes = sorted({d['cfg']['Lx'] for d in arch})
if sizes:
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.2 * len(sizes), 6.2), sharex='col', squeeze=False)
    for j, L in enumerate(sizes):
        for d in [d for d in arch if d['cfg']['Lx'] == L]:
            s = np.array(d['curve']['samples'], dtype=float)[1:]
            for i, key in enumerate(('val_loss', 'val_err')):
                axes[i, j].plot(s, np.array([x[0] for x in d['curve'][key]])[1:], color=COL[d['arch']], lw=1.6,
                                label=f"{d['arch']} ({d['n_params'] / 1e6:.1f}M)")
        for i in range(2):
            axes[i, j].set_xscale('log'); axes[i, j].set_yscale('symlog', linthresh=1e-6); axes[i, j].grid(alpha=0.25)
        axes[0, j].set_title(f"{L}×{L}  (F={L * L})"); axes[1, j].set_xlabel('fresh samples seen'); axes[1, j].set_ylim(-1e-7, 1.5)
    axes[0, 0].set_ylabel('validation loss'); axes[1, 0].set_ylabel('validation sign error (held-out x patterns)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('High-capacity architectures, x-only input, fresh batches, held-out patterns'); fig.tight_layout()
    fig.savefig('figures/signlearn/fig6_architectures.png', dpi=150); plt.show()"""))
cells.append(C(r"""# samples needed to reach validation error < 1e-3, per architecture and size, against the small (eps, x) MLP of Figure 2 (pattern split)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for a_ in ('mlp', 'cnn', 'tf'):
    pts = sorted([(d['F'], first(d, 1e-3)) for d in arch if d['arch'] == a_])
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=COL[a_], label=f'{a_} (x only)')
base = sorted([(d['F'], np.array(d['curve']['samples'], dtype=float)[np.where(np.mean(d['curve']['val_err'], axis=1) < 1e-3)[0][0]]) for d in hd
               if d['split'] == 'pattern' and (np.mean(d['curve']['val_err'], axis=1) < 1e-3).any()])
if base:
    ax.plot([b[0] for b in base], [b[1] for b in base], 's--', color='k', label='38k-param MLP, (ε, x) input')
ax.set_yscale('log'); ax.set_xlabel('F (hexagons)'); ax.set_ylabel('fresh samples to validation error < 1e-3')
ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_title('Samples to learn vs lattice size')
fig.tight_layout(); fig.savefig('figures/signlearn/fig7_samples_to_learn.png', dpi=150); plt.show()"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
nbf.write(nb, '/Users/sanzhar123/Desktop/2D-TC/analysis/08_sign_learnability.ipynb')
