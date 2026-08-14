"""Optional Weights & Biases logging for training runs (opt-in via ``--wandb``).

Design goals:
- **Zero-impact when off.** Every entry point is a no-op unless the run was
  launched with ``--wandb`` *and* wandb imports cleanly, so existing CNN / v2
  baselines stay byte-identical and a missing ``wandb`` install never crashes a
  run. State lives in one module-global handle, so ``optimizer.run_tdvp`` logs
  without threading a logger object through its signature.
- **Offline by default (NERSC).** Compute nodes have no outbound internet
  (docs/NERSC.md), so we default ``WANDB_MODE=offline``: the run streams to a
  local dir and is pushed to the dashboard later via ``wandb sync`` from a login
  node (``bash scripts/cluster.sh wandb-sync``). Set ``WANDB_MODE=online`` in the
  environment to override (only works if the compute node can reach the internet).
"""

import os

_run = None  # module-global handle; None => logging disabled (the common case)


def init_run(config):
    """Start a wandb run if ``config['wandb']`` is set. Safe to call unconditionally."""
    global _run
    if not config.get("wandb"):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] --wandb set but wandb is not installed; skipping "
              "(run `pip install wandb` on a NERSC login node).")
        return None

    # Offline unless the caller explicitly chose otherwise. setdefault so an
    # externally-exported WANDB_MODE=online (the tested-internet path) wins.
    os.environ.setdefault("WANDB_MODE", "offline")

    try:
        _run = wandb.init(
            project=config.get("wandb_project", "2d-tc"),
            entity=config.get("wandb_entity") or None,
            name=config.get("jobid"),
            group=config.get("wandb_group") or None,
            tags=_tags(config),
            config=config,
        )
        print(f"[wandb] logging enabled (mode={os.environ.get('WANDB_MODE')}, "
              f"project={config.get('wandb_project', '2d-tc')}, run={config.get('jobid')})")
    except Exception as e:  # never let telemetry setup kill a physics run
        print(f"[wandb] init failed ({e}); continuing without logging.")
        _run = None
    return _run


def _tags(config):
    """Tags let you slice the dashboard by arm / size / field without parsing names."""
    tags = [
        f"L{config.get('Lx')}",
        config.get("architecture", "Combo"),
        config.get("symmetric_block", "cnn"),
        f"hz{config.get('hz')}",
    ]
    if config.get("hy"):
        tags.append(f"hy{config.get('hy')}")
    if config.get("hx"):
        tags.append(f"hx{config.get('hx')}")
    if config.get("ftc"):
        tags.append("ftc")
    return tags


def log_step(step, metrics):
    """Log one training step's scalar metrics. No-op when logging is disabled."""
    if _run is None:
        return
    _run.log(metrics, step=step)


def log_summary(summary):
    """Set run-level summary values (final energy, runtime, observables)."""
    if _run is None:
        return
    _run.summary.update(summary)


def finish():
    """Close the run cleanly (flushes the offline dir)."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
