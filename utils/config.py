"""
Configuration utilities for handling command line arguments and simulation parameters.
"""

import argparse
import json
import os
from typing import Dict, Any, List, Union, Optional
import time
import subprocess
import sys
import platform
import jax

def setup_environment():
    """Configure environment variables and print hardware information."""
    # Check for GPU availability using jax.devices()
    devices = jax.devices()
    has_gpu = any('cuda' in str(device).lower() for device in devices)
    
    if has_gpu:
        # GPU mode
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        gpu_assigned = "NVIDIA GPU"
        n_chains = 2**10  # 1024 chains for GPU
        
        # Try to get GPU information
        try:
            command = 'nvidia-smi --query-gpu=gpu_name --format=csv,noheader'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            gpu_name, error = process.communicate()
            gpu_assigned = str(gpu_name)
        except FileNotFoundError:
            pass
    else:
        # CPU mode
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        gpu_assigned = "CPU mode"
        n_chains = 2**4  # 16 chains for CPU
    
    # Get hostname
    try:
        command2 = 'hostname'
        process2 = subprocess.Popen(command2.split(), stdout=subprocess.PIPE)
        node_assigned, error = process2.communicate()
        node_assigned = str(node_assigned)
    except FileNotFoundError:
        node_assigned = platform.node()

    print("NODE:", node_assigned)
    print("ASSIGNED DEVICE:", gpu_assigned)
    print("NUMBER OF CHAINS:", n_chains)
    print("AVAILABLE DEVICES:", devices)
    
    return gpu_assigned, node_assigned, n_chains

def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments and return as a dictionary.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description="Quantum Many-Body Simulation with JAX and NetKet")
    
    # Output file parameters
    parser.add_argument('--outindex', required=True, help='Output index for filenames')
    parser.add_argument('--jobid', required=True, help='Job ID for filenames')
    parser.add_argument('--annotation', type=str, default="cluster_16x16_run_hy", 
                       help='Annotation for the output files')
    
    # Geometry parameters
    parser.add_argument('--Lx', type=int, default=2, help='Number of vertices in x direction')
    parser.add_argument('--bc', type=str, choices=['OBC', 'PBC'], default='OBC', help='Boundary conditions')
    
    # Hamiltonian parameters
    parser.add_argument('--hx', type=float, required=True, help='X magnetic field strength')
    parser.add_argument('--hy', type=float, required=True, help='Y magnetic field strength')
    parser.add_argument('--hz', type=float, required=True, help='Z magnetic field strength')
    parser.add_argument('--J', type=float, default=1.0, help='Coupling strength')
    parser.add_argument('--Jy_p', type=float, default=0.0, help='Y plaquette coupling')
    parser.add_argument('--Jy_v', type=float, default=0.0, help='Y vertex coupling')
    parser.add_argument('--Jbond', type=float, default=0.0, help='Bond coupling')
    parser.add_argument('--h_f', type=float, default=0.0,
                        help='Fermionic (dyon) X.Z field strength (real => keeps dtype float64)')
    
    # Optimization parameters
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--diag_shift', type=float, required=True, help='Diagonal shift')
    parser.add_argument('--sim_time', type=float, default=3.5, help='Simulation time')
    parser.add_argument('--lr_schedule', choices=['const', 'cosine'], default='const',
                        help='dt (learning-rate) schedule: const, or cosine-decay to lr_final_frac*dt')
    parser.add_argument('--lr_final_frac', type=float, default=0.1,
                        help='cosine schedule: final dt as a fraction of the initial dt (default 0.1)')
    
    # Neural network parameters
    parser.add_argument('--architecture', type=str, choices=['Combo', 'RPP'], default='Combo', 
                        help='Architecture type')
    parser.add_argument('--channels_noninv', type=str, required=True, 
                        help='Comma-separated integers for non-invariant channels')
    parser.add_argument('--channels_inv', type=str, required=True,
                        help='Comma-separated integers for invariant channels')
    parser.add_argument('--kernel_size', type=int, required=True, help='Kernel size for non-invariant CNN')
    parser.add_argument('--rescale', type=float, default=1.0, help='Rescale factor')

    # Symmetric-block (Block 3) selector + factored-attention transformer hyperparameters.
    # 'cnn' (default) = original global-kernel invariant CNN; 'transformer' = replace Block 3
    # with a factored-attention encoder stack (Block 1 + Wilson nonlinearity unchanged).
    parser.add_argument('--symmetric_block', choices=['cnn', 'transformer', 'full_transformer'], default='cnn',
                        help="Block-3 arch: 'cnn' | 'transformer' (v1) | 'full_transformer' (v2, whole pipeline)")
    parser.add_argument('--tf_layers', type=int, default=2,
                        help='Transformer: number of encoder blocks (Block 2 in full_transformer)')
    parser.add_argument('--tf1_layers', type=int, default=1,
                        help='full_transformer: number of Block-1 spatial-attention encoder blocks')
    parser.add_argument('--tf_gamma_init', type=float, default=7.862,
                        help='full_transformer: local gamma init for the exp(-gamma d) distance kernel')
    parser.add_argument('--tf_dmodel', type=int, default=8, help='Transformer: embedding dimension d')
    parser.add_argument('--tf_heads', type=int, default=2, help='Transformer: number of attention heads (must divide d)')
    parser.add_argument('--tf_ffn_mult', type=int, default=2, help='Transformer: FFN hidden = tf_ffn_mult * d')
    parser.add_argument('--tf_activation', choices=['relu', 'gelu'], default='relu',
                        help='Transformer: FFN activation')
    parser.add_argument('--tf_readout_K', type=int, default=0,
                        help='Transformer: log-cosh readout hidden units K (0 => use d_model)')
    parser.add_argument('--tf_complex_output', action='store_true',
                        help='Transformer: complex (real+1j*imag) readout for sign-full runs')
    parser.add_argument('--seed', type=int, default=0,
                        help='PRNG seed for the variational state (deterministic paired runs)')

    # Weights & Biases logging (opt-in; off by default so baselines stay byte-identical).
    # Offline-by-default on NERSC (compute nodes have no internet) — `wandb sync` later.
    parser.add_argument('--wandb', action='store_true',
                        help='Log E / Vscore / std / timings to Weights & Biases (offline by default)')
    parser.add_argument('--wandb_project', type=str, default='2d-tc', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (user/team); None => account default')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B run group (e.g. per-L campaign)')
    
    # MCMC sampling parameters
    parser.add_argument('--n_samples', type=int, default=2**13, help='Total number of samples')
    parser.add_argument('--n_chains', type=int, help='Number of MCMC chains (will be overridden by device detection)')
    parser.add_argument('--n_discard', type=int, default=2**3, help='Number of burn-in steps per chain')
    parser.add_argument('--chunk_size', type=int, default=2**11, help='Max number of samples to process in parallel')
    parser.add_argument('--n_sweeps', type=int, help='Number of subsampling steps (defaults to N/2)')
    parser.add_argument('--n_samples_fin', type=int, required=True, help='Final number of samples')
    parser.add_argument('--use_custom_sampler', action='store_true', help='Use custom sampler with vertex updates')
    
    # Parse arguments
    if len(sys.argv) <= 12:  # Check if using old positional arguments format
        # For backward compatibility, read from sys.argv directly
        args = {
            'outindex': sys.argv[1],
            'jobid': sys.argv[2],
            'hx': float(eval(sys.argv[3])),
            'hy': float(eval(sys.argv[4])),
            'hz': float(eval(sys.argv[5])),
            'dt': float(eval(sys.argv[6])),
            'diag_shift': float(eval(sys.argv[7])),
            'channels_noninv': [int(el) for el in sys.argv[8].split(',')],
            'channels_inv': [int(el) for el in sys.argv[9].split(',')],
            'kernel_size': int(eval(sys.argv[10])),
            'n_samples_fin': int(eval(sys.argv[11])),
            'architecture': 'Combo',  # Default values for backwards compatibility
            'bc': 'OBC',
            'J': 1.0,
            'Jy_p': 0.0,
            'Jy_v': 0.0,
            'Jbond': 0.0,
            'h_f': 0.0,
            'n_samples': 2**13,
            'n_chains': 2**10,  # Will be overridden by device detection
            'n_discard': 2**3,
            'chunk_size': 2**11,
            'n_sweeps': 2**10 // 2,  # Will be overridden by N/2 if not provided
            'sim_time': 3.5,
            'lr_schedule': 'const',
            'lr_final_frac': 0.1,
            'rescale': 1.0,
            'annotation': "cluster_16x16_run_hy",
            # Symmetric-block / transformer defaults (legacy positional path)
            'symmetric_block': 'cnn',
            'tf_layers': 2,
            'tf1_layers': 1,
            'tf_gamma_init': 7.862,
            'tf_dmodel': 8,
            'tf_heads': 2,
            'tf_ffn_mult': 2,
            'tf_activation': 'relu',
            'tf_readout_K': 0,
            'tf_complex_output': False,
            'seed': 0,
            'wandb': False,
            'wandb_project': '2d-tc',
            'wandb_entity': None,
            'wandb_group': None
        }
    else:
        args = vars(parser.parse_args())
        # Convert channel strings to lists of integers
        args['channels_noninv'] = [int(el) for el in args['channels_noninv'].split(',')]
        args['channels_inv'] = [int(el) for el in args['channels_inv'].split(',')]
    
    # Set Ly equal to Lx
    args['Ly'] = args['Lx']
    
    # Determine N based on boundary conditions
    if args['bc'] == "OBC":
        args['N'] = 2 * args['Lx'] * (args['Lx'] - 1)
    else:
        args['N'] = 2 * args['Lx'] * args['Ly']
    
    # Set default n_sweeps to N/2 if not provided
    if args.get('n_sweeps') is None:
        args['n_sweeps'] = args['N'] // 2
    
    # Calculate kernel_size_inv based on Lx
    args['kernel_size_inv'] = args['Lx'] - 1
    
    # Determine dtype based on parameters.
    # v3 (Viteritti et al., arXiv:2311.16889): with a transformer complex readout the DEEP
    # ENCODER stays REAL (float64) and the complex amplitude/phase are injected only at the
    # shallow readout (re + 1j*im). So --tf_complex_output keeps dtype float64 even for hy!=0,
    # which also arms the non-holomorphic qgt_mode='complex' branch (optimizer.py, gated on
    # tf_complex_output && dtype=='float64'). Without it, hy!=0 falls back to the legacy
    # whole-model-complex CNN path, unchanged.
    # The fermionic (dyon) field h_f is REAL but NOT sign-problem-free: X_a.Z_b has
    # sign-indefinite off-diagonals in the Z basis (verified by small-L ED -- the exact
    # ground state has ~46% negative amplitudes for any h_f>0), which is exactly the
    # fermionic statistics of epsilon = e x m. A float64 Combo produces a strictly
    # POSITIVE amplitude (real log psi) and cannot represent it, so h_f!=0 must use a
    # signful ansatz: either the complex CNN (dtype=complex, like the hy path) or the
    # real-encoder + complex-readout transformer (--tf_complex_output, handled first).
    if args.get('tf_complex_output', False) and args.get('symmetric_block', 'cnn') in ('transformer', 'full_transformer'):
        args['dtype'] = "float64"
    elif args['hy'] != 0.0 or args['Jy_p'] != 0.0 or args['Jy_v'] != 0.0 or args.get('h_f', 0.0) != 0.0:
        args['dtype'] = "complex"
    else:
        args['dtype'] = "float64"
    
    # Set filenames
    args['filename_base'] = f"G-equiv_{args['outindex']}_{args['jobid']}"
    args['filename'] = f"{args['filename_base']}.json"
    args['filename_mpack'] = f"{args['filename_base']}.mpack"
    
    return args

def create_data_dict(config: Dict[str, Any], gpu_assigned: str, node_assigned: str) -> Dict[str, Any]:
    """Create the initial data dictionary for storing simulation results.
    
    Args:
        config: Configuration dictionary
        gpu_assigned: GPU assigned to the job
        node_assigned: Compute node assigned to the job
        
    Returns:
        dict: Data dictionary for storing simulation results
    """
    return {
        "iters": [],
        "energy": [],
        "energy_eom": [],
        "energy_var": [],
        "tau_corr": [],
        "Rsplit": [],
        "Vscore": [],
        "MCMC_accepted": [],
        "MCMC_total": [],
        "equiv_error": [],
        "equiv_error_bulk": [],
        "t_sample": [],
        "t_grad": [],
        "t_sr": [],
        "grad_norm": [],
        "dtheta_norm": [],
        "dt_step": [],
        "diagnostics": [],
        "order_params": {
            "magnetization_Xmean": [],
            "magnetization_Xstd": [],
            "magnetization_Ymean": [],
            "magnetization_Ystd": [],
            "magnetization_Zmean": [],
            "magnetization_Zstd": [],
            "2pointCorrelators": [],
            "WilsonBFFM": [],
            "renyi2_entropy": [],
            "Bp_mean": [],
            "Bp_std": [],
            "Av_mean": [],
            "Av_std": [],
            "Se_mean": [],
            "Se_std": []
        },
        "sim_params": {
            "kind": ["G-NonInv"],
            "architecture_type": [config["architecture"]],
            "Lx": [config["Lx"]],
            "hx": [config["hx"]],
            "hy": [config["hy"]],
            "hz": [config["hz"]],
            "Jy_p": [config["Jy_p"]],
            "Jy_v": [config["Jy_v"]],
            "Jbond": [config["Jbond"]],
            "h_f": [config.get("h_f", 0.0)],
            "BC": [config["bc"]],
            "n_chann_inv": config["channels_inv"],
            "n_chann_noninv": config["channels_noninv"],
            "symmetric_block": [config.get("symmetric_block", "cnn")],
            "tf_layers": [config.get("tf_layers", 0)],
            "tf1_layers": [config.get("tf1_layers", 0)],
            "tf_gamma_init": [config.get("tf_gamma_init", 0.0)],
            "tf_dmodel": [config.get("tf_dmodel", 0)],
            "tf_heads": [config.get("tf_heads", 0)],
            "tf_ffn_mult": [config.get("tf_ffn_mult", 0)],
            "tf_activation": [config.get("tf_activation", "none")],
            "tf_readout_K": [config.get("tf_readout_K", 0)],
            "tf_complex_output": [config.get("tf_complex_output", False)],
            "seed": [config.get("seed", 0)],
            "rescale": [config["rescale"]],
            "kernel_size_noninv": [config["kernel_size"]],
            "kernel_size_inv": [config["kernel_size_inv"]],
            "n_params": [0],  # Will be updated later
            "n_samples": [config["n_samples"]],
            "n_samples_fin": [config["n_samples_fin"]],
            "n_sweeps_fin": ["None"],
            "n_chains": [config["n_chains"]],
            "n_discard_per_chain=": [config["n_discard"]],
            "n_sweeps": [config["n_sweeps"]],
            "chunk_size=": [config["chunk_size"]],
            "dt": [config["dt"]],
            "diag_shift": [config["diag_shift"]],
            "lr_schedule": [config.get("lr_schedule", "const")],
            "lr_final_frac": [config.get("lr_final_frac", 0.1)],
            "diag_shift_init": ["None"],
            "param_dtype": [str(config["dtype"])],
            "runtime": [],
            "annotation": [config["annotation"]],
            "gpu:assigned": [gpu_assigned],
            "node:assigned": [node_assigned]
        }
    }

def save_data(filename: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file.
    
    Args:
        filename: Name of the JSON file
        data: Data dictionary to save
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

def update_data(filename: str, keys: List[str], values: List[Any]) -> None:
    """Update data in a JSON file.
    
    Args:
        filename: Name of the JSON file
        keys: List of keys to update
        values: List of values to add to the corresponding keys
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for i, key in enumerate(keys):
        data[key].append(str(values[i]))
    
    with open(filename, 'w') as f:
        json.dump(data, f) 