# Session-start prompt

Paste this at the top of each new session, then add your task-specific sentences.

---

This is the **2D-TC** project: approximately-symmetric neural quantum states (NQS)
for the 2D mixed-field toric code, reproducing/extending Kufel et al., PRL 135,
056702 (2025). Before doing anything, read `CLAUDE.md` and `docs/NERSC.md` in the
repo root — they contain the architecture overview, physics conventions,
operational gotchas, and the cluster workflow. The repo lives at
github.com/SanzharBissenali/2D-TC (private). `netket` is **not** installed on this
laptop; real simulations run on NERSC Perlmutter (allocation `m5340_g`, GPU-only,
conda env `2dtc`). The working model: develop code locally → commit/push → I pull
on NERSC and `sbatch` jobs → results are committed under `results/` and pushed back
→ pull locally to analyze. Keep all infrastructure clean, concise, and minimal;
put new tooling in `exact/`, `scripts/`, or `jobs/` and avoid modifying the upstream
paper code in `model/`, `simulation/`, `utils/` without a clear reason.

A few standing conventions: commit with the repo-local identity already configured
(`SanzharBissenali` + GitHub no-reply email — don't touch global git config);
confirm with me before submitting cluster jobs or anything hard to reverse; and
when we settle a non-obvious decision or hit a new gotcha, update `CLAUDE.md` (and
your memory) so the next session inherits it. Familiarize yourself with the current
state first, then help me with the following:
