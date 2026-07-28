"""
Module for creating the toric code Hamiltonian with various perturbations.
"""

import netket as nk
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple, Optional, Union

def create_hamiltonian(
    hi: nk.hilbert.Spin,
    vertex_all: List[List[int]],
    plaq_all: List[List[int]],
    bonds: List[List[int]],
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    J: float = 1.0,
    Jy_v: float = 0.0,
    Jy_p: float = 0.0,
    Jbond: float = 0.0,
    h_f: float = 0.0,
    fermion_pairs: Optional[List[List[int]]] = None,
    dual_basis: bool = False,
    dtype: Any = complex
) -> nk.operator.AbstractOperator:
    """
    Create the toric code Hamiltonian with perturbations.
    
    Args:
        hi: Hilbert space
        vertex_all: List of vertex operators
        plaq_all: List of plaquette operators
        bonds: List of nearest-neighbor bonds
        hx: X magnetic field strength
        hy: Y magnetic field strength
        hz: Z magnetic field strength
        J: Coupling strength
        Jy_v: Y vertex coupling
        Jy_p: Y plaquette coupling
        Jbond: Bond coupling
        h_f: Fermionic (dyon) X.Z field strength
        fermion_pairs: List of [x_link, z_link] pairs for the S_e = X_a.Z_b field
        dual_basis: Hadamard-conjugate the Hamiltonian (swap sigma_x <-> sigma_z
            everywhere): H_dual = W H W with W = H_2^(x)N. Stars become Z-products
            (diagonal), plaquettes X-products, hx -> sigma_z field, hz -> sigma_x.
            Same spectrum as H (unitary), so ED references are unchanged.
        dtype: Data type for the Hamiltonian

    Returns:
        The toric code Hamiltonian
    """
    H = 0
    N = hi.size

    # Hadamard conjugation is NOT a plain swap for sigma_y (W sy W = -sy) or for the
    # fermionic S_e = X_a.Z_b (maps to Z_a.X_b, a geometrically different operator
    # set), so those perturbations are out of scope in dual mode.
    if dual_basis:
        assert hy == 0 and Jy_v == 0 and Jy_p == 0 and h_f == 0, (
            "dual_basis: sigma_y -> -sigma_y and X_a.Z_b -> Z_a.X_b under Hadamard "
            "conjugation; hy / Jy_v / Jy_p / h_f are not supported in dual mode"
        )
    _sx = nk.operator.spin.sigmaz if dual_basis else nk.operator.spin.sigmax
    _sz = nk.operator.spin.sigmax if dual_basis else nk.operator.spin.sigmaz

    # The Hamiltonian OPERATOR must be complex whenever sigma^y appears (hy or a Y coupling),
    # since sigmay is imaginary. This is independent of the MODEL parameter dtype: for the v3
    # complex-readout path the model stays float64 (real encoder; amplitude+phase injected only
    # at the shallow readout), while the operator still needs to be complex. Decouple here so a
    # float64 model dtype no longer trips the "Y field requires complex Hamiltonian" assertion.
    if hy != 0 or Jy_v != 0 or Jy_p != 0:
        dtype = "complex"

    # Add vertex terms
    for v in range(0, len(vertex_all)):
        # XXXX vertex terms
        op = 1
        for j in range(0, len(vertex_all[v])):
            if vertex_all[v][j] != -1:
                op *= _sx(hi, vertex_all[v][j], dtype=dtype)
        H += -J * op
        
        # YYYY vertex terms
        if Jy_v != 0:
            assert dtype == "complex", "YYYY vertex terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(vertex_all[v])):
                if vertex_all[v][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, vertex_all[v][j], dtype=dtype)
            H += -Jy_v * op
    
    # Add plaquette terms
    for p in range(0, len(plaq_all)):
        # ZZZZ plaquette terms
        op = 1
        for j in range(0, len(plaq_all[p])):
            if plaq_all[p][j] != -1:
                op *= _sz(hi, plaq_all[p][j], dtype=dtype)
        H += -J * op
        
        # YYYY plaquette terms
        if Jy_p != 0:
            assert dtype == "complex", "YYYY plaquette terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(plaq_all[p])):
                if plaq_all[p][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, plaq_all[p][j], dtype=dtype)
            H += -Jy_p * op
    
    # Add magnetic field perturbations
    for j in range(0, N):
        if hz != 0:
            H += -_sz(hi, j, dtype=dtype) * hz
        if hx != 0:
            H += -_sx(hi, j, dtype=dtype) * hx
        if hy != 0:
            assert dtype == "complex", "Y magnetic field requires complex Hamiltonian"
            H += -nk.operator.spin.sigmay(hi, j, dtype=dtype) * hy
    
    # Add 2-qubit perturbations (bonds)
    if Jbond != 0.0:
        for (x, y) in bonds:
            H += -nk.operator.spin.sigmax(hi, x) * nk.operator.spin.sigmax(hi, y) * Jbond
            H += -nk.operator.spin.sigmaz(hi, x) * nk.operator.spin.sigmaz(hi, y) * Jbond
            H += -nk.operator.spin.sigmay(hi, x) * nk.operator.spin.sigmay(hi, y) * Jbond

    # Add fermionic (dyon) perturbation: sum of two-body S_e = X_a . Z_b on the
    # L-shaped edge pairs. Binds an e (Z) to an m (X) so the field proliferates the
    # composite fermion epsilon, NOT independent anyons (unlike an equal X+Z sum).
    # X.Z carries no sigma^y, so this term is REAL and keeps the Hamiltonian float64.
    if h_f != 0.0 and fermion_pairs is not None:
        for (a, b) in fermion_pairs:
            H += -h_f * nk.operator.spin.sigmax(hi, a, dtype=dtype) \
                      * nk.operator.spin.sigmaz(hi, b, dtype=dtype)

    # Convert to Pauli strings for more efficient implementation
    H = H.to_pauli_strings()
    
    return H 