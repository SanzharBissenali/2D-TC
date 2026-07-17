"""
Module for time-dependent variational principle (TDVP) optimization.
"""

import time
import jax
import jax.numpy as jnp
import netket as nk
from functools import partial
from tqdm import tqdm
from typing import Dict, Any, Callable, Optional, List, Tuple

from utils.config import update_data

def run_tdvp(
    hamiltonian: nk.operator.AbstractOperator,
    vstate: nk.vqs.VariationalState,
    config: Dict[str, Any],
    callbacks: Optional[List[Callable]] = None
) -> nk.vqs.VariationalState:
    """
    Run the time-dependent variational principle optimization.
    
    Args:
        hamiltonian: Hamiltonian operator
        vstate: Variational state
        config: Configuration dictionary
        callbacks: List of callback functions to call after each optimization step
        
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
    
    loop = tqdm(range(n_iter))
    t = t_start
    
    for step in loop:
        step_start = time.time()

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

        # Compute quantum geometric tensor (QGT) and the SR update direction
        t0 = time.time()
        S = vstate.quantum_geometric_tensor(
            nk.optimizer.qgt.QGTJacobianDense(diag_shift=diag_shift, diag_scale=diag_scale)
        )
        gamma_f = jax.tree.map(lambda x: -1.0 * x, f)
        dtheta, _ = S.solve(
            partial(nk.optimizer.solver.pinv_smooth, rtol=rtol, rtol_smooth=rtol_smooth),
            gamma_f
        )
        dtheta = jax.block_until_ready(dtheta)
        t_sr = time.time() - t0

        # Update parameters
        vstate.parameters = jax.tree.map(lambda x, y: x + dt * y, vstate.parameters, dtheta)

        # Save optimization data
        update_data(filename, [
            "iters", "energy", "energy_eom", "energy_var", "tau_corr",
            "Rsplit", "Vscore", "MCMC_accepted", "MCMC_total",
            "t_sample", "t_grad", "t_sr"
        ], [
            t, E.mean, E.error_of_mean, E.variance, E.tau_corr,
            E.R_hat, config['N'] * E.variance / E.mean**2,
            vstate.sampler_state.n_accepted, vstate.sampler_state.n_steps,
            t_sample, t_grad, t_sr
        ])
        
        # Check for NaN values
        if jnp.isnan(E.mean):
            print("Encountered NaN energy, stopping optimization.")
            break
        
        # Call any callback functions
        if callbacks is not None and step % 8 == 0:
            for callback in callbacks:
                callback(vstate, step, t, config)
        
        # Update progress bar description (V-score = N·Var/⟨E⟩², matches the JSON)
        vscore = float(jnp.real(config['N'] * E.variance / E.mean**2))
        step_time = time.time() - step_start
        loop.set_description(
            f"E: {E.mean:.6f} ± {E.error_of_mean:.6f} | Vscore: {vscore:.3e} | {step_time:.2f}s/step"
        )
        
        # Update time
        t = t + dt
    
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