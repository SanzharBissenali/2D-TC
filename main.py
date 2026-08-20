"""
Main entry point for the toric code simulation.

This script sets up and runs a variational Monte Carlo simulation of a toric code
model with various perturbations, using neural network quantum states as the variational
ansatz. It supports different architectural choices, and optimization
parameters. OBC are well tested, but PBC might require some additional work.

Author: D. Kufel
Last modified: March 24th, 2025
"""

import time
import uuid
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
import os
import sys
import json
from netket.utils import struct

from model.geometry import ToricCodeGeometry
from model.hamiltonian import create_hamiltonian
from model.networks import KernelManager, create_model
from simulation.optimizer import run_tdvp, run_minsr, create_final_callback
from simulation.observables import (
    create_wilson_loop_callback, create_magnetization_callback,
    create_renyi_callback, create_2point_callback, create_conditional_callbacks,
    create_plaquette_stabilizer_callback, create_vertex_stabilizer_callback,
    create_Se_callback, create_dressed_star_callback,
    check_Av_invariance, check_Bp_invariance, dump_attention
)
from utils.config import setup_environment, parse_arguments, create_data_dict, save_data
from utils.io import save_model, log_runtime, record_experiment_info
from utils import wandb_logger

# Import custom sampler if needed
from simulation.custom_sampler import create_custom_sampler

def main():
    # Start timing
    start_time = time.time()
    
    # Setup environment
    gpu_assigned, node_assigned, n_chains = setup_environment()
    
    # Parse command line arguments
    config = parse_arguments()

    # Honeycomb NQS wiring is Phase 2 -- without this guard, --lattice honeycomb
    # would silently train the SQUARE toric code while the run JSON records
    # lattice=honeycomb and config['N'] (honeycomb formula) skews the V-score.
    # Placed before any file is written so no stray run JSON is created.
    assert config.get('lattice', 'square') == 'square', \
        "main.py: --lattice honeycomb is ED-only for now (NQS wiring is Phase 2)"

    # Override n_chains with device-specific value
    config['n_chains'] = n_chains
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Start optional W&B logging (no-op unless --wandb). Init here so the full,
    # finalized config (incl. device-detected n_chains) is captured as run config.
    wandb_logger.init_run(config)

    # Create data dictionary
    data = create_data_dict(config, gpu_assigned, node_assigned)
    
    # Save initial data
    save_data(config['filename'], data)
    
    # Set up the geometry
    geometry = ToricCodeGeometry(config['Lx'], config['Ly'], config['bc'])
    
    # Create the Hilbert space
    hi = nk.hilbert.Spin(s=1/2, N=geometry.N)
    
    # Create the Hamiltonian
    H = create_hamiltonian(
        hi=hi,
        vertex_all=geometry.vertex_all,
        plaq_all=geometry.plaq_all,
        bonds=geometry.bonds,
        hx=config['hx'],
        hy=config['hy'],
        hz=config['hz'],
        J=config.get('J', 1.0),
        Jy_v=config.get('Jy_v', 0.0),
        Jy_p=config.get('Jy_p', 0.0),
        Jbond=config.get('Jbond', 0.0),
        h_f=config.get('h_f', 0.0),
        fermion_pairs=geometry.fermion_pairs,
        dual_basis=config.get('dual_basis', False),
        ftc=config.get('ftc', False),
        dressed_stars=geometry.dressed_stars,
        dtype=config['dtype']
    )
    
    # Create the kernel manager
    kernel_manager = KernelManager(
        Lx=config['Lx'],
        Ly=config['Ly'],
        bc=config['bc'],
        kernel_size=config['kernel_size'],
        kernel_size_inv=config['kernel_size_inv'],
        arr_coord=geometry.arr_coord,
        dg_p=geometry.dg_p,
        N=geometry.N,
        dg_v=geometry.dg_v,
        vertex_all=geometry.vertex_all,
        dual=config.get('dual_basis', False)
    )
    
    # Create the neural network model
    model = create_model(config, geometry.plaq_all, kernel_manager)
    print(model)
    
    # Create a sampler based on configuration
    if config.get('use_custom_sampler', False):
        # Use custom sampler with vertex updates
        sa = create_custom_sampler(geometry, hi, config)
        print("Using custom sampler with vertex updates")
    else:
        # Use standard sampler with local rule
        rule = nk.sampler.rules.LocalRule()
        sa = nk.sampler.MetropolisSampler(
            hi, 
            rule=rule, 
            n_chains=config['n_chains'],
            n_sweeps=config['n_sweeps'],
            dtype=jnp.int8
        )
        print("Using standard sampler with local updates")
    
    # Create the variational state (explicit seed => deterministic paired CNN/transformer runs)
    vs = nk.vqs.MCState(
        sa,
        model,
        n_samples=config['n_samples'],
        n_discard_per_chain=config['n_discard'],
        chunk_size=config['chunk_size'],
        seed=config['seed']
    )
    
    # Warm start (curriculum phase B): restore a previous run's serialized MCState.
    # Requires an IDENTICAL model/sampler structure (e.g. phase A with --freeze_chi:
    # stop_gradient changes no param shapes, so its .mpack loads directly).
    if config.get('init_params'):
        import flax as _flax
        with open(config['init_params'], 'rb') as f:
            vs = _flax.serialization.from_bytes(vs, f.read())
        print(f"Warm-started from {config['init_params']}")

    # Update number of parameters in the data dictionary
    with open(config['filename'], 'r') as f:
        data = json.load(f)
    data["sim_params"]["n_params"] = [vs.n_parameters]
    with open(config['filename'], 'w') as f:
        json.dump(data, f)

    # Correctness gate: exact A_v (vertex/gauge) symmetry must hold to machine precision.
    # full_transformer: at init only (odd embedding + identity Block-1 + Wilson fusion).
    # plaquette_transformer (Variant 3): at ANY parameters (B_p tokens are a change of
    # variables), so the same check is a full architecture-correctness gate.
    # Warm-started runs: informational only -- a trained chi is LEGITIMATELY not identity
    # (its A_v deviation is the learned dressing), so we print the number without asserting.
    if config.get('symmetric_block') in ('full_transformer', 'plaquette_transformer', 'variant1'):
        dev = check_Av_invariance(model, vs.parameters, geometry)
        print(f"[A_v init-invariance] max |Delta log psi| = {dev:.2e}")
        if not config.get('init_params'):
            assert dev < 1e-6, (
                f"A_v symmetry BROKEN at init (max dev {dev:.2e}) -- check odd embedding / "
                f"zero-init chi sublayers / channelwise Wilson product (v2/variant1), or the "
                f"B_p tokenization (Variant 3)"
            )

    # Dual-basis Combo CNN: the exactly-embedded-at-init symmetry is B_p (plaquette
    # flips preserve every star product; Block-1's scaled sigmoid maps +-1 -> +-1
    # exactly at identity init). Init-only gate -- training legitimately breaks it
    # via Block-1, exactly as the primal CNN breaks A_v.
    if config.get('dual_basis', False):
        dev = check_Bp_invariance(model, vs.parameters, geometry)
        print(f"[B_p init-invariance] max |Delta log psi| = {dev:.2e}")
        if not config.get('init_params'):
            assert dev < 1e-6, (
                f"B_p symmetry BROKEN at init (max dev {dev:.2e}) -- check the star-Wilson "
                f"masking (-1 sentinels) / identity init of Block-1 / the vertex-grid "
                f"invariant-CNN kernel table"
            )

    # Setup callbacks for observables
    callbacks = create_conditional_callbacks(geometry)

    # Print information before starting optimization
    print(f"Number of qubits: {geometry.N}")
    print(f"Number of model parameters: {vs.n_parameters}")
    print(f"Starting optimization...")
    
    # Run the optimization ('minsr' = NetKet VMC_SR kernel-trick, N_samples-bound;
    # 'tdvp' = the hand-rolled dense P x P QGT baseline)
    run_optimizer = run_minsr if config.get('optimizer', 'tdvp') == 'minsr' else run_tdvp
    print(f"Optimizer: {config.get('optimizer', 'tdvp')}")
    vs = run_optimizer(
        hamiltonian=H,
        vstate=vs,
        config=config,
        callbacks=callbacks
    )
    
    # Save the final model
    save_model(vs, config['filename_base'])
    
    # Calculate observables
    print("Calculating final observables...")
    
    # For Lx >= 6, calculate the Wilson-loop observables at the end (expensive)
    if geometry.Lx >= 6:
        if config.get('dual_basis', False):
            print("WARNING: calculate_wilson_loops labels X/Z in the SIMULATION basis; "
                  "under --dual_basis the physical meanings are swapped (not remapped here).")
        # Calculate Wilson loops
        callback = create_wilson_loop_callback(geometry)
        callback(vs, -1, -1, config)

        # # Calculate two-point correlation functions
        # callback = create_2point_callback(geometry) #doesn't work yet
        # callback(vs, -1, -1, config)

    # Renyi-2 entropy at the end for ALL sizes (calculate_renyi_entropy now falls back to
    # a single central placement at small L, so L=4 no longer crashes on an empty grid).
    callback = create_renyi_callback(geometry)
    callback(vs, -1, -1, config)

    # Always calculate magnetizations at the end
    callback = create_magnetization_callback(geometry)
    callback(vs, -1, -1, config)

    # Final plaquette-stabilizer <B_p> (m-flux / contamination diagnostic)
    callback = create_plaquette_stabilizer_callback(geometry)
    callback(vs, -1, -1, config)

    # Final vertex-stabilizer <A_v> (e-charge diagnostic; complements <B_p>)
    callback = create_vertex_stabilizer_callback(geometry)
    callback(vs, -1, -1, config)

    # Final fermionic (dyon) order parameter <S_e> = <X_a.Z_b>
    callback = create_Se_callback(geometry)
    callback(vs, -1, -1, config)

    # Fermionic TC: final dressed-star <A'_v> = <A_v * B_NE(v)> (the model's actual
    # vertex-sector stabilizers; ~1 at h=0 alongside <A_v> and <B_p>)
    if config.get('ftc', False):
        callback = create_dressed_star_callback(geometry)
        callback(vs, -1, -1, config)

    # Attention interpretability dump: gamma ranges + alpha tables (full_transformer),
    # or the per-(layer, head) content-gate alpha_h (plaquette_transformer / variant1).
    if config.get('symmetric_block') in ('full_transformer', 'plaquette_transformer', 'variant1'):
        dump_attention(vs, config)

    # Log runtime
    log_runtime(config, start_time)
    
    # Record experiment information with a unique ID
    run_id = str(uuid.uuid4())[:8]
    record_experiment_info(
        config=config,
        run_id=run_id,
        description="Toric code simulation with neural network quantum states",
        extra_info={
            "n_params": vs.n_parameters,
            "final_energy": vs.expect(H).mean,
            "runtime": time.time() - start_time
        }
    )
    
    # W&B run-level summary: final energy + runtime + the last value of each order
    # parameter (read back from the JSON the end-of-run callbacks just wrote).
    try:
        with open(config['filename'], 'r') as f:
            _final = json.load(f)
        _summary = {
            "final_energy": float(np.real(np.complex128(_final["energy"][-1]))) if _final["energy"] else None,
            "final_Vscore": float(_final["Vscore"][-1]) if _final["Vscore"] else None,
            "runtime_s": time.time() - start_time,
            "n_params": int(vs.n_parameters),
        }
        for k, v in _final.get("order_params", {}).items():
            if v:  # keep the last recorded value of each observable
                _summary[f"final_{k}"] = v[-1]
        wandb_logger.log_summary(_summary)
    except Exception as e:
        print(f"[wandb] summary skipped: {e}")
    wandb_logger.finish()

    print("Simulation complete.")

if __name__ == "__main__":
    main() 