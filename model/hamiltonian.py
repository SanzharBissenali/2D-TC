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
    ftc: bool = False,
    dressed_stars: Optional[List[Tuple[List[int], List[int]]]] = None,
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
        ftc: fermionic toric code (Chen-Kapustin-Radicevic bosonization Gauss law,
            arXiv:1711.00515 Eq. 9). Replaces each vertex star A_v = XXXX by the
            dressed A'_v = A_v * B_NE(v): sigma_x on the star links times sigma_z on
            the NE-plaquette links. On the two shared links (v's up/right) the
            same-site product X.Z = -i*sigma_y makes A'_v = -Y.Y.X.X.Z.Z - a REAL
            but non-stoquastic operator (even Y count), so dtype stays float64.
            Same stabilizer group as the plain TC (A'_v * B_NE = A_v), hence the
            unperturbed ground state and E0 = -(#stars + #plaquettes) are unchanged;
            the fermionic character enters the excitations and perturbed response.
            Boundary rule: vertices without an NE plaquette keep the bare A_v.
        dressed_stars: per-vertex (x_links, z_links) from
            geometry._generate_dressed_stars(); required when ftc=True.
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

    if ftc:
        assert h_f == 0 and Jy_v == 0 and hy == 0 and Jy_p == 0, (
            "ftc: the dressed-star model is defined with the plain plaquette term and "
            "hx/hz field perturbations only; h_f / Jy_v / hy / Jy_p are out of scope "
            "(hy anticommutes with the dressed star same as hx -- unvalidated cut)"
        )
        assert dressed_stars is not None and len(dressed_stars) == len(vertex_all), (
            "ftc=True requires geometry.dressed_stars (one (x_links, z_links) entry "
            "per vertex) - fill in geometry._generate_dressed_stars()"
        )
        assert all(len(entry) == 2 for entry in dressed_stars), (
            "ftc: every geometry.dressed_stars entry must be an (x_links, z_links) pair"
        )

    # Add vertex terms
    for v in range(0, len(vertex_all)):
        if ftc:
            # Dressed star A'_v = A_v * B_NE(v). The x_links/z_links overlap on the
            # vertex's up/right links yields X.Z = -i*sigma_y per shared site, so the
            # -Y.Y.X.X.Z.Z string (and its -1) emerges from plain operator products.
            # Under dual_basis the aliases give (Z-star)(X-plaq) = W A'_v W exactly.
            x_links, z_links = dressed_stars[v]
            op = 1
            for j in x_links:
                if j != -1:
                    op *= _sx(hi, j, dtype=dtype)
            for j in z_links:
                if j != -1:
                    op *= _sz(hi, j, dtype=dtype)
            H += -J * op
            continue

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