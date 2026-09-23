"""npz interchange for the learned sign MLP (arm M-pre, docs/signhead_benchmark_plan.md).

Numpy-only on purpose: the WRITER (scripts/pretrain_sign_mlp.py, supervised fit
to ED signs) and the READER (honeycomb_networks.load_mlp_params, warm start via
--mlp_init) share one spec, and the spec can be validated without jax
(scratchpad/validate_signbench.py).

Spec == the flax param tree of honeycomb_networks._SignMLP under its 'mlp' scope,
flattened with '/' (flax.traverse_util convention):

    Dense_0/kernel  (n_in, hidden)     Dense_0/bias  (hidden,)
    Dense_i/kernel  (hidden, hidden)   Dense_i/bias  (hidden,)   i = 1 .. depth-1
    Dense_D/kernel  (hidden, 1)        Dense_D/bias  (1,)        D = depth

n_in = QECSignHead.n_features_ex = N + F (features [eps, x] in {0,1}); float64.
depth = number of tanh hidden layers (depth 0 => a single Dense_0 (n_in, 1)).
Both writer and reader feed the RAW {0,1} features: _SignMLP recentres them to
{-1,+1} itself (h = 2u - 1) before Dense_0, so the saved kernels are expressed
in the recentred basis on both sides -- do NOT pre-recentre in the writer.
"""

import numpy as np


def mlp_param_spec(n_in, hidden, depth):
    """Ordered {flat key: shape} of the MLP param tree (see module docstring)."""
    assert n_in >= 1 and hidden >= 1 and depth >= 0
    spec = {}
    fan_in = n_in
    for i in range(depth):
        spec[f"Dense_{i}/kernel"] = (fan_in, hidden)
        spec[f"Dense_{i}/bias"] = (hidden,)
        fan_in = hidden
    spec[f"Dense_{depth}/kernel"] = (fan_in, 1)
    spec[f"Dense_{depth}/bias"] = (1,)
    return spec


def nest(flat):
    """{'Dense_0/kernel': a, ...} -> {'Dense_0': {'kernel': a, ...}, ...}."""
    tree = {}
    for k, v in flat.items():
        layer, leaf = k.split("/")
        tree.setdefault(layer, {})[leaf] = v
    return tree


def save_mlp_npz(path, flat):
    """Write a flat {key: array} MLP tree (float64) to ``path``."""
    np.savez(path, **{k: np.asarray(v, dtype=np.float64) for k, v in flat.items()})


def load_mlp_npz(path, spec):
    """Read ``path`` and return the flat {key: float64 array} tree, asserting it
    carries EXACTLY the keys of ``spec`` with matching shapes and finite values
    (a shape mismatch means the npz was fit for a different n_in/hidden/depth)."""
    with np.load(path) as z:
        flat = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    missing = set(spec) - set(flat)
    extra = set(flat) - set(spec)
    assert not missing and not extra, \
        f"MLP npz {path}: missing keys {sorted(missing)}, unexpected keys {sorted(extra)}"
    for k, shape in spec.items():
        assert flat[k].shape == tuple(shape), \
            f"MLP npz {path}: {k} has shape {flat[k].shape}, model expects {tuple(shape)}"
        assert np.isfinite(flat[k]).all(), f"MLP npz {path}: {k} has non-finite entries"
    return flat
