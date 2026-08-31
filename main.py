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
    check_Av_invariance, check_Bp_invariance, dump_attention,
    _check_flip_invariance
)
from utils.config import setup_environment, parse_arguments, create_data_dict, save_data
from utils.io import save_model, log_runtime, record_experiment_info
from utils import wandb_logger

# Import custom sampler if needed
from simulation.custom_sampler import create_custom_sampler


def _honeycomb_final_observables(vs, geometry, config, sign_frame_head=None):
    """End-of-run <Q_v> and <plaquette term> for the honeycomb models.

    Mirrors observables.calculate_* (per-stabilizer op -> vstate.expect -> JSON
    order_params) but honeycomb-native: Q_v = prod sigma_z over the vertex's
    existing links (2-body at the boundary); plaquette term = X_hex for tc,
    X_hex*D_legs*P_p for ds. Built with PauliStrings algebra like
    create_honeycomb_hamiltonian (a 12-site LocalOperator product OOMs), and
    cast to exactly real weights the same way.

    sign_frame_head (Phase 4, sign_impl 'operator' only): the vstate then holds
    the POSITIVE A while the physical state is S*A, so OFF-DIAGONAL observables
    must be conjugated to S*op*S like the Hamiltonian; diagonal Q_v commutes
    with S and stays bare. With sign_impl 'model' the state itself carries the
    signs -- pass None and both impls must report identical numbers.
    """
    hi = vs.hilbert

    def _ps(f, j):
        return f(hi, int(j), dtype="complex").to_pauli_strings()

    def _prod(ops):
        out = None
        for o in ops:
            out = o if out is None else out @ o
        return out

    def _real(op):
        w = np.asarray(op.weights)
        assert np.abs(w.imag).max() < 1e-9, "honeycomb observable must be exactly real"
        return nk.operator.PauliStrings(hi, [str(s) for s in op.operators],
                                        np.real(w).astype(np.float64))

    def _q(v):
        return _prod(_ps(nk.operator.spin.sigmaz, j)
                     for j in geometry.vertex_all[int(v)] if j != -1)

    qv = [float(np.real(vs.expect(_real(_q(v))).mean))
          for v in range(geometry.n_vertices)]
    pl = []
    for p in range(geometry.n_plaqs):
        op = _prod(_ps(nk.operator.spin.sigmax, int(j)) for j in geometry.plaq_all[p])
        if config.get('model', 'tc') == 'ds':
            for l in geometry.legs_all[p]:
                if l != -1:
                    op = op @ (0.5 * (1 + 1j)
                               + 0.5 * (1 - 1j) * _ps(nk.operator.spin.sigmaz, int(l)))
            for v in geometry.plaq_vertices[p]:
                op = op @ (0.5 + 0.5 * _q(v))
        op = _real(op)
        if sign_frame_head is not None:
            from model.sign_frame import SignFramedOperator
            op = SignFramedOperator(op, sign_frame_head)
        pl.append(float(np.real(vs.expect(op).mean)))

    with open(config['filename'], 'r') as f:
        data = json.load(f)
    op_dict = data.setdefault("order_params", {})
    for key, val in (("Qv_mean", float(np.mean(qv))), ("Qv_min", float(np.min(qv))),
                     ("plaq_term_mean", float(np.mean(pl))),
                     ("plaq_term_std", float(np.std(pl)))):
        op_dict.setdefault(key, []).append(val)
    with open(config['filename'], 'w') as f:
        json.dump(data, f)
    print(f"<Q_v> mean={np.mean(qv):.6f} min={np.min(qv):.6f} | "
          f"<plaq term> mean={np.mean(pl):.6f} std={np.std(pl):.6f}")


def main():
    # Start timing
    start_time = time.time()
    
    # Setup environment
    gpu_assigned, node_assigned, n_chains = setup_environment()
    
    # Parse command line arguments
    config = parse_arguments()

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
    
    is_honeycomb = config.get('lattice', 'square') == 'honeycomb'
    if is_honeycomb:
        # Honeycomb (Levin-Gu TC / doubled semion), Phase 2 wiring. Imports are
        # lazy so the square path never depends on the honeycomb modules.
        from model.honeycomb_geometry import HoneycombGeometry
        from model.hamiltonian import create_honeycomb_hamiltonian
        from model.honeycomb_networks import create_honeycomb_model

        geometry = HoneycombGeometry(config['Lx'], config['Ly'], config['bc'])
        hi = nk.hilbert.Spin(s=1/2, N=geometry.N)
        H = create_honeycomb_hamiltonian(
            hi, geometry, config.get('model', 'tc'),
            J=config.get('J', 1.0), hx=config['hx'], hz=config['hz'],
            hy=config.get('hy', 0.0),
        )
        model = create_honeycomb_model(config, geometry)
        base_model = model            # pre-sign-head network (gates run on this)
        sign_frame_head = None        # set iff impl 'operator' (observables re-frame)
        if config.get('sign_head', 'none') == 'qec':
            # Phase-4 QEC sign head: psi = (-1)^{s(sigma)} A_theta(sigma) with
            # s = MWPM-recovered loop parity (model/sign_head.py). Two exactly
            # equivalent realizations (see model/sign_frame.py):
            #   operator (production): train positive A on H~ = SHS -- sampler,
            #     network, dtype all byte-identical to the sign-free arm;
            #   model (equivalence witness): log psi += 1j*pi*s via host callback.
            try:
                import pymatching  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "--sign_head qec needs pymatching (pip install pymatching "
                    "on a login node, like wandb)") from e
            from model.sign_head import QECSignHead
            head = QECSignHead(geometry, decoder=config.get('decoder', 'mwpm'))
            if config.get('sign_impl', 'operator') == 'operator':
                from model.sign_frame import SignFramedOperator
                H = SignFramedOperator(H, head)
                sign_frame_head = head
                print(f"[sign head] qec/operator: training on H~ = SHS "
                      f"(loop table 2^{head.F}, decoder "
                      f"{config.get('decoder', 'mwpm')})")
            elif config.get('sign_impl') == 'residual':
                from model.honeycomb_networks import ResidualSignedModel
                model = ResidualSignedModel(base_model, head.features,
                                            head.n_features,
                                            hidden=config.get('res_hidden', 16))
                print(f"[sign head] qec/residual: log psi += i*(pi*s + phi_chi), "
                      f"tie-gated MLP hidden={config.get('res_hidden', 16)} over "
                      f"K={head.n_features} decoder features (zero-init phi)")
            else:
                from model.honeycomb_networks import SignedModel
                model = SignedModel(base_model, head.s01)
                print(f"[sign head] qec/model: log psi += i*pi*s(sigma) via "
                      f"pure_callback (loop table 2^{head.F}, decoder-A MWPM)")
        print(model)
    else:
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

    # Honeycomb Combo: hexagon-flip invariance is exact AT IDENTITY INIT (tokens
    # == the raw Q_v values, and each hexagon shares 0 or 2 links with each
    # vertex). It is INIT-ONLY: the tokens are products of Block-1-DRESSED link
    # features, so generic Block-1 parameters legitimately break it -- that IS
    # the "approximate" half of the architecture, exactly like the square
    # Combo's A_v / the dual-basis B_p gates (empirically: perturbing params by
    # 0.05 gives |Delta log psi| ~ 3e-3, the symmetry-breaking scale, which we
    # print as information rather than assert on). PlainCNN: no symmetry at all.
    if is_honeycomb:
        if config.get('architecture', 'Combo') == 'Combo':
            # The gate always runs on the BASE network: the sign head flips
            # log psi by i*pi under hexagon flips BY DESIGN (dressed-operator
            # covariance), so the wrapped model would trivially fail it.
            gate_model, gate_params = model, vs.parameters
            if config.get('sign_head', 'none') != 'none' \
                    and config.get('sign_impl', 'operator') in ('model', 'residual'):
                gate_model, gate_params = base_model, vs.parameters['base']
            clusters = [list(map(int, p)) for p in geometry.plaq_all]
            dev0 = _check_flip_invariance(gate_model, gate_params, geometry.N,
                                          clusters, n_configs=32)
            _rng = np.random.default_rng(1)
            pert = jax.tree_util.tree_map(
                lambda x: x + jnp.asarray(
                    0.05 * _rng.standard_normal(np.shape(x)), dtype=x.dtype),
                gate_params,
            )
            dev1 = _check_flip_invariance(gate_model, pert, geometry.N,
                                          clusters, n_configs=32)
            print(f"[hexflip invariance] max |Delta log psi|: init {dev0:.2e}, "
                  f"perturbed params {dev1:.2e} (init-only gate; the perturbed "
                  f"value is the approximate-symmetry-breaking scale)")
            if not config.get('init_params'):
                assert dev0 < 1e-8, (
                    f"hexagon-flip symmetry BROKEN at init (max dev {dev0:.2e}) -- "
                    "check the vertex Wilson masking (-1 sentinels) / Block-1 "
                    "identity init / Block-3 tables"
                )
        else:
            print("[hexflip invariance] skipped: PlainCNN carries no architectural symmetry")

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

    # Setup callbacks for observables (square-specific: the magnetization /
    # B_p callbacks assume the square geometry's coordinate helpers; honeycomb
    # gets its observables in one end-of-run pass instead)
    callbacks = [] if is_honeycomb else create_conditional_callbacks(geometry)

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
    
    if is_honeycomb:
        # Honeycomb-native pass (<Q_v>, <plaquette term>); the square callbacks
        # below assume ToricCodeGeometry's coordinate helpers and don't apply.
        _honeycomb_final_observables(vs, geometry, config,
                                     sign_frame_head=sign_frame_head)

    # For Lx >= 6, calculate the Wilson-loop observables at the end (expensive)
    if not is_honeycomb and geometry.Lx >= 6:
        if config.get('dual_basis', False):
            print("WARNING: calculate_wilson_loops labels X/Z in the SIMULATION basis; "
                  "under --dual_basis the physical meanings are swapped (not remapped here).")
        # Calculate Wilson loops
        callback = create_wilson_loop_callback(geometry)
        callback(vs, -1, -1, config)

        # # Calculate two-point correlation functions
        # callback = create_2point_callback(geometry) #doesn't work yet
        # callback(vs, -1, -1, config)

    if not is_honeycomb:
        # Renyi-2 entropy at the end for ALL sizes (calculate_renyi_entropy now falls
        # back to a single central placement at small L, so L=4 no longer crashes).
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

        # Fermionic TC: final dressed-star <A'_v> = <A_v * B_NE(v)> (the model's
        # actual vertex-sector stabilizers; ~1 at h=0 alongside <A_v> and <B_p>)
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