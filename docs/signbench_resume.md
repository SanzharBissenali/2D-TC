# Sign-bench overnight state + resume steps (2026-09-24, cert expired ~22:30 PDT 09-23)

The sshproxy cert lapsed mid-campaign. Everything already in the Slurm queue keeps
running; the steps below finish the campaign once the cert is renewed
(`sshproxy -u sanzharb`, then `bash scripts/cluster.sh status` to confirm).

## Done (results on the cluster, not yet fetched)
- 9-point ED/`T_gate` gradings (jobs 58796230/232/235) → `results/diagnostics/signfid_hc2x3_ds_<hx>_<hz>…json`.
- 18 production runs cnnqM + cnnqT × 9 points (350 steps, no NaN).
- 5 M-pre runs: (0.4,0) (0.4,0.2) (0.4,0.4) (0.8,0) (1.2,0).
- Pretrain npz: the same 5 points (`results/pretrain/mlp_hc2x3_*.npz`).

## Queued at cert expiry (regular queue; check with `cluster.sh status` / `sacct`)
- pretraining, single-point jobs: 58809322 (0.8,0.4), 58809323 (1.2,0.4),
  58811394 (0.8,0.2), 58811396 (1.2,0.2).
- fidelities: 58805841 (hx0.4 M+T), 58805856 (hx0.8 M+T), 58806177 (hx1.2 M+T),
  58811761 (hx0.4 Mp).

## Resume (in order; each `submit` is a short shared/regular job)
1. For each pretraining job that COMPLETED (npz present), launch its M-pre run:
   ```
   for p in 0.8:0.2 0.8:0.4 1.2:0.2 1.2:0.4; do
     bash scripts/cluster.sh submit jobs/nersc_signbench.sh -t 1:15:00 \
       --export=ALL,LX=2,LY=3,ARMS=cnnqMp,POINTS=$p,WANDB=1,WANDB_GROUP=hc-signbench
   done
   ```
   (the job fails loudly per point if its npz is missing — resubmit that point's
   `jobs/nersc_pretrain.sh -q regular -t 1:45:00 --export=ALL,LX=2,LY=3,POINTS=<p>` first).
2. When the M-pre runs of a column are complete, its fidelity job:
   ```
   bash scripts/cluster.sh submit jobs/nersc_fidelity.sh -q regular -t 2:00:00 \
     --export=ALL,LX=2,LY=3,HY=0,POINTS=0.8:0+0.8:0.2+0.8:0.4,ARMS=cnnqMp,TAG=sb_hx0.8_Mp
   bash scripts/cluster.sh submit jobs/nersc_fidelity.sh -q regular -t 2:00:00 \
     --export=ALL,LX=2,LY=3,HY=0,POINTS=1.2:0+1.2:0.2+1.2:0.4,ARMS=cnnqMp,TAG=sb_hx1.2_Mp
   ```
3. `bash scripts/cluster.sh fetch` (commits results on the cluster, pulls locally), then
   `python scripts/signbench_summary.py --Lx 2 --Ly 3 --out results/diagnostics/signbench_2d.json`
   and execute `analysis/07_signbench_heatmaps.ipynb` (figures → `figures/signbench/`).
   The 3D panel reads `results/diagnostics/signbench_3d.json` from the peer session.

## Energy-level results so far (tail mean of last 20 steps vs ED E0; fidelities pending)
| point | M-cold | M-pre | T |
|---|---|---|---|
| (0.4,0)   | 2.7e-2 | 1.9e-4 | 2.4e-4 |
| (0.4,0.2) | 5.9e-2 | —      | 3.0e-4 |
| (0.4,0.4) | 3.2e-2 | —      | 2.2e-4 |
| (0.8,0)   | 2.0e-2 | —      | 1.4e-3 |
| (0.8,0.2) | 5.1e-2 | —      | 2.0e-3 |
| (1.2,0)   | 3.0e-4 | —      | 2.5e-4 |
(M-pre at (0.4,0.2), (0.4,0.4), (0.8,0), (1.2,0) finished but were not read before the
cert lapsed; (0.8,0.4), (1.2,0.2), (1.2,0.4) have no committed E0 in this table yet — they
are in the new signfid JSONs.)
Pretrain warm-start quality: (0.4,·) ≤ 1e-5 reached; (0.8,0) 3.3e-5 and (1.2,0) 2.6e-4 hit
the 3000-epoch cap (recorded in `results/pretrain/*.json`, shown in the prior column).
