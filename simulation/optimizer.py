"""
Module for time-dependent variational principle (TDVP) optimization.
"""

import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from functools import partial
from tqdm import tqdm
from typing import Dict, Any, Callable, Optional, List, Tuple

from utils.config import update_data
from utils import wandb_logger


# --- instability instrumentation (arm-agnostic; used by both CNN and transformer arms) ---
_WRAP = ("Sequential", "FullTransformer", "TransformerSymmetric", "PlaquetteTransformer",
         "GaugeComboTransformer")


def _key_str(k):
    for attr in ("key", "name", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def _block_of(path):
    """First non-wrapper, non-'params' path component => the block name."""
    names = [_key_str(k) for k in path]
    for nm in names:
        if nm == "params" or any(nm.startswith(w) for w in _WRAP):
            continue
        return nm
    return names[0] if names else "root"


def _tree_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves)))


def _block_norms(tree):
    acc = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        b = _block_of(path)
        acc[b] = acc.get(b, 0.0) + float(jnp.sum(jnp.abs(leaf) ** 2))
    return {k: round(float(np.sqrt(v)), 6) for k, v in acc.items()}


def _gammas(params):
    """softplus(gamma_raw) per block -- attention range diagnostic (full_transformer only)."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if _key_str(path[-1]) == "gamma_raw":
            a = np.asarray(leaf)
            sp = np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))   # stable softplus
            out[_block_of(path)] = [round(float(v), 5) for v in np.ravel(sp)]
    return out


def _qgt_cond(S):
    """Conditioning of the (regularized) QGT: max/min singular value, cond, retained rank."""
    try:
        sv = np.linalg.svd(np.asarray(S.to_dense()), compute_uv=False)
        smax = float(sv[0]); smin = float(sv[-1])
        return {"sv_max": round(smax, 8), "sv_min": round(smin, 12),
                "cond": round(smax / max(smin, 1e-300), 2),
                "rank": int(np.sum(sv > 1e-12 * smax)), "P": int(sv.size)}
    except Exception as e:  # never let a diagnostic kill training
        return {"error": str(e)[:100]}

# --- QEC sign-head accounting (Phase 4 honeycomb arms; no-op otherwise) -------------
HEAD_KEYS = ["t_head", "n_head_configs", "step_wall"]   # per-step JSON fields (str-serialized like the rest)
# step_wall = wall-clock between consecutive per-step JSON appends (the full step incl.
# callbacks/observables) -- the denominator for the head-time share in analysis/06.


def _resolve_head(hamiltonian, head):
    """The QECSignHead driving this run, if any: passed explicitly by main.py
    (any --sign_impl), else discovered on a SignFramedOperator (impl operator)."""
    return head if head is not None else getattr(hamiltonian, '_head', None)


def _ensure_keys(filename, keys):
    """Add missing per-step lists to the run JSON (update_data appends blindly)."""
    with open(filename, 'r') as fh:
        d = json.load(fh)
    if any(k not in d for k in keys):
        for k in keys:
            d.setdefault(k, [])
        with open(filename, 'w') as fh:
            json.dump(d, fh)


def _head_stats(head):
    """(seconds, #configs) the head spent since the previous step; (0.0, 0) without a head."""
    if head is None or not hasattr(head, 'pop_head_stats'):
        return 0.0, 0
    dt, n = head.pop_head_stats()
    return float(dt), int(n)


def run_tdvp(
    hamiltonian: nk.operator.AbstractOperator,
    vstate: nk.vqs.VariationalState,
    config: Dict[str, Any],
    callbacks: Optional[List[Callable]] = None,
    head=None,
) -> nk.vqs.VariationalState:
    """
    Run the time-dependent variational principle optimization.
    
    Args:
        hamiltonian: Hamiltonian operator
        vstate: Variational state
        config: Configuration dictionary
        callbacks: List of callback functions to call after each optimization step
        head: optional QECSignHead (model/sign_head.py) -- its wall-clock and
            configuration counts are drained every step into the JSON fields
            ``t_head`` / ``n_head_configs`` (0 without a head; also auto-detected
            on a SignFramedOperator hamiltonian)
        
    Returns:
        Optimized variational state
    """
    dt = config['dt']
    t_start = 0.0
    t_end = config['sim_time']
    diag_shift = config['diag_shift']
    filename = config['filename']
    
    n_iter = int((t_end - t_start) / dt)
    diag_scale = 0.0
    rtol = 1e-30
    rtol_smooth = 1e-30
    K_diag = 8    # cadence for the cheap per-block grad-norm / gamma diagnostics
    K_qgt = 40    # coarser cadence for the O(P^3) host-side QGT-conditioning SVD (avoids inflating L>=6 wall-clock)

    # v3: real params + complex output = non-holomorphic => explicit mode='complex'.
    # Default (sign-free real, or the existing complex-CNN hy path) leaves mode unset (auto).
    qgt_mode = 'complex' if (config.get('tf_complex_output', False)
                             and config.get('dtype') == 'float64') else None

    # dt (learning-rate) schedule: cosine-decay to lr_final_frac*dt over the run damps the
    # late-training SR wander (smaller steps settle into the minimum instead of overshooting).
    lr_schedule = config.get('lr_schedule', 'const')
    dt_final = config.get('lr_final_frac', 0.1) * dt

    head = _resolve_head(hamiltonian, head)
    _ensure_keys(filename, HEAD_KEYS)
    t_log_prev = time.time()

    loop = tqdm(range(n_iter))
    t = t_start
    
    for step in loop:
        step_start = time.time()
        dt_step = (dt_final + 0.5 * (dt - dt_final) * (1.0 + np.cos(np.pi * step / max(n_iter - 1, 1)))
                   if lr_schedule == 'cosine' else dt)

        # --- Per-step wall-clock split (block_until_ready defeats JAX async dispatch
        # so each timer captures real work, not just the launch). This is a
        # controlled variable: identical instrumentation on the CNN and transformer
        # arms. Forcing vstate.samples first caches this step's MC samples, so the
        # subsequent expect_and_grad / QGT reuse them and t_grad excludes sampling.
        t0 = time.time()
        samples = jax.block_until_ready(vstate.samples)
        t_sample = time.time() - t0

        # Compute energy and gradient (reuses the cached samples)
        t0 = time.time()
        E, f = vstate.expect_and_grad(hamiltonian)
        f = jax.block_until_ready(f)
        t_grad = time.time() - t0
        grad_norm = _tree_norm(f)

        # Compute quantum geometric tensor (QGT) and the SR update direction
        t0 = time.time()
        _qgt = (nk.optimizer.qgt.QGTJacobianDense(diag_shift=diag_shift, diag_scale=diag_scale, mode=qgt_mode)
                if qgt_mode else
                nk.optimizer.qgt.QGTJacobianDense(diag_shift=diag_shift, diag_scale=diag_scale))
        S = vstate.quantum_geometric_tensor(_qgt)
        gamma_f = jax.tree.map(lambda x: -1.0 * x, f)
        dtheta, _ = S.solve(
            partial(nk.optimizer.solver.pinv_smooth, rtol=rtol, rtol_smooth=rtol_smooth),
            gamma_f
        )
        dtheta = jax.block_until_ready(dtheta)
        t_sr = time.time() - t0
        dtheta_norm = _tree_norm(dtheta)

        # Update parameters (dt_step = scheduled learning rate)
        vstate.parameters = jax.tree.map(lambda x, y: x + dt_step * y, vstate.parameters, dtheta)

        # Sign-head accounting for this step (host time inside get_conn_padded /
        # pure_callback; covers everything since the previous pop)
        t_head, n_head = _head_stats(head)
        step_wall, t_log_prev = time.time() - t_log_prev, time.time()

        # Save optimization data
        update_data(filename, [
            "iters", "energy", "energy_eom", "energy_var", "tau_corr",
            "Rsplit", "Vscore", "MCMC_accepted", "MCMC_total",
            "t_sample", "t_grad", "t_sr", "grad_norm", "dtheta_norm", "dt_step",
            *HEAD_KEYS
        ], [
            t, E.mean, E.error_of_mean, E.variance, E.tau_corr,
            E.R_hat, config['N'] * E.variance / E.mean**2,
            vstate.sampler_state.n_accepted, vstate.sampler_state.n_steps,
            t_sample, t_grad, t_sr, grad_norm, dtheta_norm, dt_step,
            t_head, n_head, step_wall
        ])

        # Check for NaN values
        if jnp.isnan(E.mean):
            print("Encountered NaN energy, stopping optimization.")
            break

        # Heavier diagnostics (per-block grad norms, QGT conditioning, attention gamma) every K_diag.
        wb_extra = {}   # optional fields folded into this step's single wandb.log (below)
        if step % K_diag == 0:
            try:
                _diag = {
                    "step": step,
                    "block_grad_norms": _block_norms(f),
                    "gammas": _gammas(vstate.parameters),
                }
                # Per-block grad norms as flat scalars => one panel per block on the dashboard.
                wb_extra.update({f"grad_block/{b}": v for b, v in _diag["block_grad_norms"].items()})
                # v3 sign-full: phase usage = circular variance of Im(logPsi) over the samples,
                # 1 - |<e^{i*theta}>|. Sign-FREE => ~0 (phase constant up to a global gauge);
                # sign-FULL => >0 (the complex readout is genuinely encoding a nontrivial phase).
                # Only on the complex-readout path (qgt_mode=='complex') so real runs pay nothing.
                # log_value is a full EXTRA forward pass; doing it every K_diag was ~1000s/run
                # overhead (dominated wall-clock). phase_circ_var is a slowly-varying convergence
                # quantity, so compute it only every K_qgt (5x/run) on a 2048-sample subset.
                # flatten first: vstate.samples is 3D (n_chains, n_per_chain, N); passing 3D feeds a
                # per-chain batch into the CNN's single-config reshape and errors.
                if qgt_mode == 'complex' and step % K_qgt == 0:
                    _flat = samples.reshape(-1, samples.shape[-1])[:2048]
                    _theta = jnp.imag(vstate.log_value(_flat)).reshape(-1)
                    _pcv = float(1.0 - jnp.abs(jnp.mean(jnp.exp(1j * _theta))))
                    _diag["phase_circ_var"] = _pcv
                    wb_extra["phase_circ_var"] = _pcv
                if step % K_qgt == 0:                     # QGT SVD is O(P^3) on host -> coarse cadence
                    _diag["qgt"] = _qgt_cond(S)
                    if "cond" in _diag["qgt"]:
                        wb_extra["qgt_cond"] = _diag["qgt"]["cond"]
                        wb_extra["qgt_rank"] = _diag["qgt"]["rank"]
                with open(filename, 'r') as _fh:
                    _d = json.load(_fh)
                _d["diagnostics"].append(_diag)
                with open(filename, 'w') as _fh:
                    json.dump(_d, _fh)
            except Exception as _e:  # diagnostics must never interrupt training
                print(f"[diag] skipped at step {step}: {_e}")

        # Call any callback functions
        if callbacks is not None and step % 8 == 0:
            for callback in callbacks:
                callback(vstate, step, t, config)
        
        # Update progress bar description (V-score = N·Var/⟨E⟩², matches the JSON)
        vscore = float(jnp.real(config['N'] * E.variance / E.mean**2))
        step_time = time.time() - step_start
        loop.set_description(
            f"E: {E.mean:.6f} ± {E.error_of_mean:.6f} | Vscore: {vscore:.3e} | {step_time:.2f}s/step"
            + (f" | head {t_head:.2f}s" if head is not None else "")
        )

        # W&B: one row per step (no-op unless --wandb). std = sqrt(Var); energy_err is the
        # MC stderr of the mean. E.mean can be complex on the hy!=0 path => take .real.
        n_acc = float(vstate.sampler_state.n_accepted)
        n_tot = float(vstate.sampler_state.n_steps)
        wandb_logger.log_step(step, {
            "energy": float(jnp.real(E.mean)),
            "energy_err": float(E.error_of_mean),
            "energy_std": float(jnp.sqrt(jnp.abs(E.variance))),
            "energy_var": float(jnp.real(E.variance)),
            "Vscore": vscore,
            "tau_corr": float(E.tau_corr),
            "Rsplit": float(E.R_hat),
            "grad_norm": grad_norm,
            "dtheta_norm": dtheta_norm,
            "dt_step": float(dt_step),
            "t_sample": t_sample,
            "t_grad": t_grad,
            "t_sr": t_sr,
            "t_head": t_head,
            "n_head_configs": n_head,
            "step_time": step_time,
            "mcmc_accept_frac": n_acc / max(n_tot, 1.0),
            "sim_t": t,
            **wb_extra,
        })

        # Update time (advance by the scheduled step)
        t = t + dt_step
    
    return vstate


def _alphas(params):
    """Small 'alpha' leaves (the v3 per-head content gates, shape (h,)) per block.
    Skips the v1 factored-attention alpha TABLES (h, n_disp) by the size cutoff."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if _key_str(path[-1]) == "alpha" and np.size(leaf) <= 16:
            out[_block_of(path)] = [round(float(v), 6) for v in np.ravel(np.asarray(leaf))]
    return out


def run_minsr(
    hamiltonian: nk.operator.AbstractOperator,
    vstate: nk.vqs.VariationalState,
    config: Dict[str, Any],
    callbacks: Optional[List[Callable]] = None,
    head=None,
) -> nk.vqs.VariationalState:
    """SR training via NetKet's VMC_SR with the kernel trick (minSR / SRt).

    Same update direction as the hand-rolled dense-QGT TDVP loop when
    n_samples < n_params, but solves the (n_samples x n_samples) NTK system
    instead of the (P x P) QGT one, so cost/memory scale with the SAMPLE budget,
    not the parameter count -- the point of ``--optimizer minsr`` for the ~35k-param
    Variant-3 transformer. Logging (per-step JSON append, tqdm, W&B, callback
    cadence) mirrors ``run_tdvp``; the t_sample/t_grad/t_sr split is not separable
    inside the driver, so those fields are 0 and ``step_time`` carries the total.
    ``head``: as in run_tdvp (per-step ``t_head`` / ``n_head_configs``).
    """
    import optax

    dt = config['dt']
    lr = config.get('lr', 0.0) or dt                         # 0 => reuse dt as the lr
    n_iter = int(config.get('n_steps', 0) or round(config['sim_time'] / dt))
    diag_shift = config['diag_shift']
    filename = config['filename']
    K_diag = 8

    # lr schedule mirrors run_tdvp: const, or cosine decay to lr_final_frac * lr.
    if config.get('lr_schedule', 'const') == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=max(n_iter - 1, 1),
            alpha=config.get('lr_final_frac', 0.1))
    else:
        schedule = lr

    # netket 3.16.x ships minSR as VMC_SRt (the SRt/kernel-trick driver, N_s x N_s
    # solve by construction); later netkets renamed it VMC_SR(use_ntk=True). Try
    # the pinned-env name first.
    opt = optax.sgd(schedule)
    # --minsr_mode: optional jacobian_mode override ('' => netket auto-detects;
    # 'complex' on a real arm tightens Phase-4 A/B trajectory comparability).
    jac_mode = config.get('minsr_mode', '') or None
    try:
        from netket.experimental.driver import VMC_SRt
        driver = VMC_SRt(hamiltonian, opt, diag_shift=diag_shift,
                         jacobian_mode=jac_mode,
                         variational_state=vstate)
    except ImportError:                                      # netket >= 3.17
        try:
            from netket.driver import VMC_SR
        except ImportError:
            from netket.experimental.driver import VMC_SR
        driver = VMC_SR(hamiltonian, opt, diag_shift=diag_shift,
                        variational_state=vstate, use_ntk=True)

    head = _resolve_head(hamiltonian, head)
    _ensure_keys(filename, HEAD_KEYS
                 + (["mix"] if ('mix' in vstate.parameters or 'log_mix' in vstate.parameters) else []))
    t_log_prev = time.time()

    loop = tqdm(range(n_iter))
    t = 0.0
    for step in loop:
        step_start = time.time()
        lr_step = float(schedule(step)) if callable(schedule) else lr

        p_before = vstate.parameters
        driver.advance(1)
        stats = driver._loss_stats
        E = jax.block_until_ready(stats)
        step_time = time.time() - step_start

        # ||dtheta|| recovered from the applied update (theta' = theta - lr * dtheta)
        delta = jax.tree.map(lambda a, b: a - b, vstate.parameters, p_before)
        dtheta_norm = _tree_norm(delta) / max(lr_step, 1e-300)

        t_head, n_head = _head_stats(head)      # host head time inside driver.advance
        step_wall, t_log_prev = time.time() - t_log_prev, time.time()
        if not np.isfinite(dtheta_norm):
            print(f"!!! run_minsr step {step}: NON-FINITE parameter update "
                  f"(||dtheta|| = {dtheta_norm}) -- energy {E.mean}", flush=True)
        # two-branch arm T: the signed mix scalar a ('mix'), per step (cheap)
        mix = vstate.parameters.get('mix') if hasattr(vstate.parameters, 'get') else None
        if mix is None and hasattr(vstate.parameters, 'get') and 'log_mix' in vstate.parameters:
            mix = np.exp(np.asarray(vstate.parameters['log_mix']))     # T+: report a = exp(c)
        mix = [float(v) for v in np.asarray(mix).reshape(-1)] if mix is not None else None

        update_data(filename, [
            "iters", "energy", "energy_eom", "energy_var", "tau_corr",
            "Rsplit", "Vscore", "MCMC_accepted", "MCMC_total",
            "t_sample", "t_grad", "t_sr", "grad_norm", "dtheta_norm", "dt_step",
            *HEAD_KEYS, *(["mix"] if mix is not None else [])
        ], [
            t, E.mean, E.error_of_mean, E.variance, E.tau_corr,
            E.R_hat, config['N'] * E.variance / E.mean**2,
            vstate.sampler_state.n_accepted, vstate.sampler_state.n_steps,
            0.0, 0.0, 0.0, 0.0, dtheta_norm, lr_step,
            t_head, n_head, step_wall, *([mix] if mix is not None else [])
        ])

        if jnp.isnan(E.mean):
            print("Encountered NaN energy, stopping optimization.")
            break

        wb_extra = {}
        if step % K_diag == 0:
            try:
                _diag = {"step": step, "alphas": _alphas(vstate.parameters)}
                for blk, vals in _diag["alphas"].items():
                    for i, v in enumerate(vals):
                        wb_extra[f"alpha/{blk}_h{i}"] = v
                with open(filename, 'r') as _fh:
                    _d = json.load(_fh)
                _d["diagnostics"].append(_diag)
                with open(filename, 'w') as _fh:
                    json.dump(_d, _fh)
            except Exception as _e:  # diagnostics must never interrupt training
                print(f"[diag] skipped at step {step}: {_e}")

        if callbacks is not None and step % 8 == 0:
            for callback in callbacks:
                callback(vstate, step, t, config)

        vscore = float(jnp.real(config['N'] * E.variance / E.mean**2))
        loop.set_description(
            f"E: {E.mean:.6f} ± {E.error_of_mean:.6f} | Vscore: {vscore:.3e} | {step_time:.2f}s/step"
            + (f" | head {t_head:.2f}s" if head is not None else "")
        )

        n_acc = float(vstate.sampler_state.n_accepted)
        n_tot = float(vstate.sampler_state.n_steps)
        wandb_logger.log_step(step, {
            "energy": float(jnp.real(E.mean)),
            "energy_err": float(E.error_of_mean),
            "energy_std": float(jnp.sqrt(jnp.abs(E.variance))),
            "energy_var": float(jnp.real(E.variance)),
            "Vscore": vscore,
            "tau_corr": float(E.tau_corr),
            "Rsplit": float(E.R_hat),
            "dtheta_norm": dtheta_norm,
            "dt_step": lr_step,
            "t_head": t_head,
            "n_head_configs": n_head,
            "step_time": step_time,
            "mcmc_accept_frac": n_acc / max(n_tot, 1.0),
            "sim_t": t,
            **({"mix": mix} if mix is not None else {}),
            **wb_extra,
        })

        t += lr_step

    return vstate


def create_final_callback(calculation_callbacks: List[Callable]) -> Callable:
    """
    Create a final callback function that runs all provided calculation callbacks.
    
    Args:
        calculation_callbacks: List of calculation callbacks to run
        
    Returns:
        Callback function that runs all provided callbacks
    """
    def final_callback(vstate: nk.vqs.VariationalState, config: Dict[str, Any]) -> None:
        print("Running final calculations...")
        for callback in calculation_callbacks:
            callback(vstate, -1, -1, config)
    
    return final_callback 