"""Rebuild the sign-bench page: summary -> notebook (figures) -> HTML. Run from the 2D-TC root."""
import json, base64, html, os, subprocess, sys
S = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, 'scripts/signbench_summary.py', '--Lx', '2', '--Ly', '3',
                '--out', 'results/diagnostics/signbench_2d.json'], check=True, capture_output=True)
subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
                'analysis/07_signbench_heatmaps.ipynb'], check=True, capture_output=True)
nb = json.load(open('analysis/07_signbench_heatmaps.ipynb'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
L = {'cnnqM': 'M', 'cnnqMp': 'M-pre', 'cnnqT': 'T'}
def f(v, fmt='{:.2e}'): return '–' if v is None else fmt.format(v)
def table(summ):
    rows = []
    order = {a: i for i, a in enumerate(summ['arms'])}
    for r in sorted(summ['records'], key=lambda r: (r['hx'], r['hz'], order.get(r['arm'], 9))):
        a = L.get(r['arm'], r['arm'])
        if r.get('missing'):
            rows.append(f"<tr><td>{r['hx']:g}</td><td>{r['hz']:g}</td><td>{a}</td><td colspan=5 class=muted>queued</td></tr>"); continue
        mix = ('[' + ', '.join(f'{m:+.2f}' for m in (r['mix'] if isinstance(r['mix'], list) else [r['mix']])) + ']') if r.get('mix') is not None else '–'
        ceil = f(r.get('ceiling'), '{:.0e}') + (' (1−F_s)' if r.get('ceiling_kind') == '1-Fs' else '')
        rows.append(f"<tr><td>{r['hx']:g}</td><td>{r['hz']:g}</td><td>{a}</td><td>{f(r.get('rel_err'))}</td><td>{f(r.get('one_minus_F'))}</td><td>{ceil}</td><td>{mix}</td><td>{f(r.get('warm_1mFs'), '{:.1e}')}</td></tr>")
    return ('<div class=tbl><table><thead><tr><th>h<sub>x</sub></th><th>h<sub>z</sub></th><th>arm</th><th>rel-err</th><th>1−F</th>'
            '<th>ceiling</th><th>T mix [a₊, a₋]</th><th>M-pre warm 1−F<sub>s</sub></th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
def img(path, alt):
    return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(open(path, "rb").read()).decode()}">' if os.path.exists(path) else f'<p class=muted>{alt}: pending — the 2^27 fidelity enumerations are queued on Perlmutter.</p>'
head = open(os.path.join(S, 'page_head.html')).read()
two_d = json.load(open('results/diagnostics/signbench_2d.json'))
three_d = json.load(open('results/diagnostics/signbench_3d.json')) if os.path.exists('results/diagnostics/signbench_3d.json') else None
note3d = open(os.path.join(S, 'note_3d.txt')).read().strip() if os.path.exists(os.path.join(S, 'note_3d.txt')) else ''
body = f"""<section><h2>Relative energy error |E − E<sub>0</sub>| / |E<sub>0</sub>|</h2>{img('figures/signbench/rel_err.png', 'relative energy error heatmaps')}
<p class=muted>Row 1: 2D doubled semion 2×3. {'Row 2: 3D fermionic toric code (peer repo). ' + note3d if three_d else 'A second row (3D fermionic toric code, peer repo) is added when its data arrive.'}</p></section>
<section><h2>Exact fidelity 1 − F</h2>{img('figures/signbench/one_minus_F.png', 'exact-fidelity heatmap')}</section>
<section><h2>Per-cell table — 2D</h2>{table(two_d)}
<p class=muted>Ceiling = T<sub>gate</sub> for T (weight the two positive trunks cannot re-sign, min over sectors); exactly 0 for M/M-pre. Warm 1−F<sub>s</sub> = sign fidelity of M-pre's pretrained MLP before VMC (target 1e-5; 3000-epoch cap).</p></section>
{'<section><h2>Per-cell table — 3D</h2>' + table(three_d) + '</section>' if three_d else ''}
<section><h2>ED-free warm start: can an MLP learn the head's sign?</h2>{img('figures/signbench/headfit_sizes.png', 'head-fit learning curves')}
<p class=muted>Supervised fit of the sign MLP (33→64/128→64/128→1, tanh, Adam lr 1e-2, batch 4096, BCE on the logit) to the deterministic head sign (−1)<sup>poly(x)</sup> on synthetic configurations (random closed-loop set ⊕ k random spin flips, k ≤ 2F; 4×10<sup>5</sup> train / 4×10<sup>4</sup> held-out, de-duplicated; 2×10<sup>5</sup>/2×10<sup>4</sup> at 2×3). y = fraction of held-out configurations whose sign is predicted wrong (thin lines: per k). The sign depends only on the F hexagon-flip bits x: 2×3 (F=6) and 4×4 (F=16, training covers 99.8% of the 2<sup>16</sup> patterns) are memorisable and reach zero error; 6×6 (F=36) stays at chance for 40 000 steps — the plain MLP does not learn the cubic parity of 36 bits from 4×10<sup>5</sup> examples.</p></section>
<section><h2>Notebook code (all of it)</h2>{''.join(f'<pre><code>{html.escape(c)}</code></pre>' for c in code)}</section>
</main>"""
open(os.path.join(S, 'signbench_grid.html'), 'w').write(head + body)
print('page rebuilt; 3D:', bool(three_d))
