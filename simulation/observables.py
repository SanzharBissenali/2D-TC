"""
Module for computing physical observables and quantities of interest.
"""

import json
import numpy as np
import netket as nk
import netket.experimental as nkx
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

def calculate_wilson_loops(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: int = 1,
    plot: bool = False
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate Wilson Loop and BFFM order parameters.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the Wilson loop
        plot: Whether to plot the Wilson loop
        
    Returns:
        Tuple containing:
            - WilsonX_mean: Mean of Wilson X loop
            - WilsonX_std: Standard deviation of Wilson X loop
            - X_BFFMmean: Mean of X BFFM
            - X_BFFMstd: Standard deviation of X BFFM
            - WilsonZ_mean: Mean of Wilson Z loop
            - WilsonZ_std: Standard deviation of Wilson Z loop
            - Z_BFFMmean: Mean of Z BFFM
            - Z_BFFMstd: Standard deviation of Z BFFM
    """
    center = (geometry.Lx - 1) / 2
    shift = geometry.Lx / 2 - radius - 2

    # Grid of loop-center placements to translation-average over. When the system
    # is too small to slide the loop (shift <= 0, e.g. L=6 at radius 1), the arange
    # is empty; fall back to a single central placement so the loop is still measured
    # (mirrors the shift == 0 branch in calculate_renyi_entropy) instead of averaging
    # over zero loops and returning nan.
    placements = np.arange(center - shift, center + shift, 1.0)
    if placements.size == 0:
        placements = np.array([center])

    # Data for X loop calculations (integer radius)
    avWilsonX = []
    Xclosedlength = []
    Xopenlength = []
    X_BFFM = []
    
    # Data for Z loop calculations (non-integer radius)
    avWilsonZ = []
    Zclosedlength = []
    Zopenlength = []
    Z_BFFM = []
    
    # Calculate X Wilson loops with integer radius
    for x in placements:
        for y in tqdm(placements, desc="X Wilson loops"):
            # Calculate Wilson X operators
            closedstringX, openstringX, ClosedWilsonX, OpenWilsonX = wilson_loop_obs_x(
                vstate.hilbert, geometry, [x, y], radius
            )
            
            # Calculate expectation values
            ClosedWilsonX_expect = vstate.expect(ClosedWilsonX).mean
            OpenWilsonX_expect = vstate.expect(OpenWilsonX).mean
            
            # Store results
            avWilsonX.append(ClosedWilsonX_expect)
            X_BFFM.append(OpenWilsonX_expect / np.sqrt(np.abs(ClosedWilsonX_expect)))
            Xclosedlength.append(len(closedstringX))
            Xopenlength.append(len(openstringX))
    
    # Calculate Z Wilson loops with non-integer radius (radius + 0.5)
    z_radius = radius + 0.5
    for x in placements:
        for y in tqdm(placements, desc="Z Wilson loops"):
            try:
                # Calculate Wilson Z operators
                closedstringZ, openstringZ, ClosedWilsonZ, OpenWilsonZ = wilson_loop_obs_z(
                    vstate.hilbert, geometry, [x, y], z_radius
                )
                
                # Calculate expectation values
                ClosedWilsonZ_expect = vstate.expect(ClosedWilsonZ).mean
                OpenWilsonZ_expect = vstate.expect(OpenWilsonZ).mean
                
                # Store results
                avWilsonZ.append(ClosedWilsonZ_expect)
                Z_BFFM.append(OpenWilsonZ_expect / np.sqrt(np.abs(ClosedWilsonZ_expect)))
                Zclosedlength.append(len(closedstringZ))
                Zopenlength.append(len(openstringZ))
            except Exception as e:
                print(f"Error calculating Z Wilson loop at position [{x}, {y}], radius {z_radius}: {e}")
                continue
    
    # Compute and return X and Z statistics
    X_mean = np.mean(avWilsonX) if avWilsonX else np.nan
    X_std = np.std(avWilsonX) if avWilsonX else np.nan
    X_BFFM_mean = np.mean(X_BFFM) if X_BFFM else np.nan
    X_BFFM_std = np.std(X_BFFM) if X_BFFM else np.nan
    
    Z_mean = np.mean(avWilsonZ) if avWilsonZ else np.nan
    Z_std = np.std(avWilsonZ) if avWilsonZ else np.nan
    Z_BFFM_mean = np.mean(Z_BFFM) if Z_BFFM else np.nan
    Z_BFFM_std = np.std(Z_BFFM) if Z_BFFM else np.nan
    
    return (
        X_mean, X_std,
        X_BFFM_mean, X_BFFM_std,
        Z_mean, Z_std,
        Z_BFFM_mean, Z_BFFM_std
    )


def wilson_strings(geometry, pos: Tuple[float, float], radius: float) -> Tuple[np.ndarray, List]:
    """
    Find Wilson strings.
    
    Args:
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius around the position
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
    """
    selectedlocs_small = geometry.select_subset(pos, radius - 1/2)  # Small square ball
    selectedlocs_large = geometry.select_subset(pos, radius)  # Large square ball
    
    # Find the closed loop by subtracting small from large
    closedindices = large_subtract_small(selectedlocs_large, selectedlocs_small)
    
    # Find the open string
    openindices = half_length_wilson(selectedlocs_large, pos, radius)
    
    return closedindices, openindices


def large_subtract_small(largeball: np.ndarray, smallball: np.ndarray) -> np.ndarray:
    """
    Calculate the Wilson loop from large and small balls.
    
    Args:
        largeball: Large ball coordinates
        smallball: Small ball coordinates
        
    Returns:
        Difference between large and small balls
    """
    difference = []
    for a in largeball:
        present = False
        for b in smallball:
            if np.allclose(a, b):
                present = True
                break
        if not present:
            difference.append(a)
    
    return np.array(difference)


def half_length_wilson(largeball: np.ndarray, pos: Tuple[float, float], radius: float) -> List:
    """
    Find half-length Wilson loop string.
    
    Args:
        largeball: Large ball coordinates
        pos: Position (x, y)
        radius: Radius
        
    Returns:
        Half-length Wilson loop string
    """
    dleft = largeball[largeball[:, 0] == pos[0] - radius]
    dleft = dleft[0:int(len(dleft) / 2)]
    
    if not (len(dleft) / 2).is_integer():  # For WilsonZ case open string
        dleft = dleft[0:int(len(dleft) / 2) + 1]
    
    ddown = largeball[largeball[:, 1] == pos[1] - radius]
    dright = largeball[largeball[:, 0] == pos[0] + radius]
    dright = dright[0:int(len(dright) / 2)]
    dtop = largeball[largeball[:, 1] == pos[1] + radius]
    
    return dleft.tolist() + ddown.tolist() + dright.tolist()


def wilson_loop_obs_x(
    hi: nk.hilbert.Spin, 
    geometry, 
    pos: Tuple[float, float], 
    radius: int
) -> Tuple[np.ndarray, List, Any, Any]:
    """
    Calculate Wilson loop X expectation values.
    
    Args:
        hi: Hilbert space
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius of the Wilson loop
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
            - Closed Wilson X operator
            - Open Wilson X operator
    """
    assert float(radius).is_integer() == True, "Valid WilsonX operator only if defined on product of vertices"
    
    def X_Wilson(indices):
        """Create a product of X operators on the given indices."""
        op = 1
        for j in indices:
            op *= nk.operator.spin.sigmax(hi, j)
        return op
    
    closedindices, openindices = wilson_strings(geometry, pos, radius)
    closedstring = geometry.qubit_select(closedindices)
    openstring = geometry.qubit_select(openindices)
    
    return closedindices, openindices, X_Wilson(closedstring), X_Wilson(openstring)


def wilson_loop_obs_z(
    hi: nk.hilbert.Spin, 
    geometry, 
    pos: Tuple[float, float], 
    radius: float
) -> Tuple[np.ndarray, List, Any, Any]:
    """
    Calculate Wilson loop Z expectation values.
    
    Args:
        hi: Hilbert space
        geometry: Geometry object
        pos: Position (x, y)
        radius: Radius of the Wilson loop
        
    Returns:
        Tuple containing:
            - Closed indices: Indices for the closed Wilson loop
            - Open indices: Indices for the open Wilson string
            - Closed Wilson Z operator
            - Open Wilson Z operator
    """
    assert float(radius).is_integer() == False, "Valid WilsonZ operator only if defined on product of plaquettes"
    
    def Z_Wilson(indices):
        """Create a product of Z operators on the given indices."""
        op = 1
        for j in indices:
            op *= nk.operator.spin.sigmaz(hi, j)
        return op
    
    closedindices, openindices = wilson_strings(geometry, pos, radius)
    closedstring = geometry.qubit_select(closedindices)
    openstring = geometry.qubit_select(openindices)
    
    return closedindices, openindices, Z_Wilson(closedstring), Z_Wilson(openstring)


def calculate_renyi_entropy(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.0
) -> Tuple[float, float, int, int]:
    """
    Calculate average Renyi entropy.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the subsystem
        
    Returns:
        Tuple containing:
            - Mean Renyi entropy
            - Standard deviation of Renyi entropy
            - Number of qubits in the subsystem
            - Perimeter of the subsystem
    """
    center = (geometry.Lx - 1) / 2
    shift = geometry.Lx / 2 - radius - 2
    renyi_mean = []

    # Grid of subsystem-center placements to translation-average over. When the system
    # is too small to slide the ball (shift <= 0, e.g. L=4 at radius 1 => shift=-1 and
    # np.arange(2.5, 0.5, 1.0) is empty), fall back to a single central placement so the
    # entropy is still measured instead of leaving renyi_entropy unbound (this is why the
    # callback used to be gated to Lx>=6); mirrors the guard in calculate_wilson_loops.
    placements = np.arange(center - shift, center + shift, 1.0) if shift != 0 else np.array([center])
    if placements.size == 0:
        placements = np.array([center])

    with tqdm(total=len(placements) * len(placements), desc='ProgressBar') as pbar:
        for x in placements:
            for y in placements:
                renyi_entropy = renyi(vstate, geometry, radius, x, y)
                renyi_mean.append(renyi_entropy[0].mean)
                pbar.update(1)

    return np.mean(renyi_mean), np.std(renyi_mean), renyi_entropy[1], 4 * radius


def renyi(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float,
    centerx: float = None,
    centery: float = None
) -> Tuple[Any, int]:
    """
    Calculate Renyi Entropy2 of qubits in a square centered at (centerx, centery) with the given radius.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the subsystem
        centerx: x-coordinate of the center
        centery: y-coordinate of the center
        
    Returns:
        Tuple containing:
            - Renyi entropy
            - Number of qubits in the subsystem
    """
    if centerx is None:
        centerx = (geometry.Lx - 1) / 2
    if centery is None:
        centery = (geometry.Lx - 1) / 2
        
    hi = vstate.hilbert
    
    # Select qubits in the small square ball
    selectedlocs_small = geometry.select_subset([centerx, centery], radius - 1/2)
    qubits = geometry.qubit_select(selectedlocs_small)
    
    # Calculate Renyi entropy
    renyi = nkx.observable.Renyi2EntanglementEntropy(hi, qubits)
    
    return vstate.expect(renyi), len(qubits)


def calculate_magnetizations(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.0,
    dual: bool = False
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate PHYSICAL magnetizations in x, y, and z directions.

    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the region to consider
        dual: Dual (Hadamard-conjugated) simulation basis -- physical sigma_x
            appears as sigma_z (and vice versa), and sigma_y -> -sigma_y, so the
            constructors are swapped and the Y mean is negated to keep the
            returned values physically meaningful.

    Returns:
        Tuple containing:
            - magnetizationsXmean: Mean magnetization in x direction
            - magnetizationsXstd: Standard deviation of magnetization in x direction
            - magnetizationsYmean: Mean magnetization in y direction
            - magnetizationsYstd: Standard deviation of magnetization in y direction
            - magnetizationsZmean: Mean magnetization in z direction
            - magnetizationsZstd: Standard deviation of magnetization in z direction
    """
    center = (geometry.Lx - 1) / 2
    sel_loc = geometry.select_subset([center, center], radius)
    sel_q = geometry.qubit_select(sel_loc)
    
    magnetizationsXmean = []
    magnetizationsXstd = []
    magnetizationsYmean = []
    magnetizationsYstd = []
    magnetizationsZmean = []
    magnetizationsZstd = []
    
    hi = vstate.hilbert
    
    _sx = nk.operator.spin.sigmaz if dual else nk.operator.spin.sigmax
    _sz = nk.operator.spin.sigmax if dual else nk.operator.spin.sigmaz
    _y_sign = -1.0 if dual else 1.0

    loop = tqdm(sel_q)
    for j in loop:
        magnetizationX = vstate.expect(_sx(hi, j))
        magnetizationY = vstate.expect(nk.operator.spin.sigmay(hi, j))
        magnetizationZ = vstate.expect(_sz(hi, j))

        magnetizationsXmean.append(magnetizationX.mean)
        magnetizationsXstd.append(np.sqrt(magnetizationX.error_of_mean))
        magnetizationsYmean.append(_y_sign * magnetizationY.mean)
        magnetizationsYstd.append(np.sqrt(magnetizationY.error_of_mean))
        magnetizationsZmean.append(magnetizationZ.mean)
        magnetizationsZstd.append(np.sqrt(magnetizationZ.error_of_mean))
    
    return (
        magnetizationsXmean,
        magnetizationsXstd,
        magnetizationsYmean,
        magnetizationsYstd,
        magnetizationsZmean,
        magnetizationsZstd
    )


def calculate_2point_correlators(
    vstate: nk.vqs.VariationalState,
    geometry,
    radius: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 2-point XX and ZZ correlators.
    
    Args:
        vstate: Variational state
        geometry: Geometry object
        radius: Radius of the region to consider
        
    Returns:
        Tuple containing:
            - unique_distances: Unique distances between qubits
            - xx_average: Average XX correlator at each distance
            - zz_average: Average ZZ correlator at each distance
            - xx_std: Standard deviation of XX correlator at each distance
            - zz_std: Standard deviation of ZZ correlator at each distance
    """
    xxval = []
    zzval = []
    distval = []
    
    center = (geometry.Lx - 1) / 2
    bulklocs = geometry.select_subset([center, center], radius)
    
    hi = vstate.hilbert
    
    with tqdm(total=len(bulklocs) * (len(bulklocs) - 1), desc='ProgressBar') as pbar:
        for x in bulklocs:
            qubitx = geometry._mapping2Dto1D(geometry.arr_coord, x)[0][0]
            for y in bulklocs:
                dist = distance(x, y)
                if np.round(dist, 3) > 0.0:
                    qubity = geometry._mapping2Dto1D(geometry.arr_coord, y)[0][0]
                    xx = x_connected_2point_correlator(vstate, hi, qubitx, qubity)
                    zz = z_connected_2point_correlator(vstate, hi, qubitx, qubity)
                    xxval.append(xx)
                    zzval.append(zz)
                    distval.append(dist)
                pbar.update(1)
    
    # Calculate statistics
    unique_distances, inverse_indices = np.unique(distval, return_inverse=True)
    xxval = np.array(xxval)
    zzval = np.array(zzval)
    
    xx_average = np.array([xxval[inverse_indices == i].mean() for i in range(len(unique_distances))])
    zz_average = np.array([zzval[inverse_indices == i].mean() for i in range(len(unique_distances))])
    xx_std = np.array([xxval[inverse_indices == i].std() for i in range(len(unique_distances))])
    zz_std = np.array([zzval[inverse_indices == i].std() for i in range(len(unique_distances))])
    
    return unique_distances, xx_average, zz_average, xx_std, zz_std


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def x_connected_2point_correlator(
    vstate: nk.vqs.VariationalState,
    hi: nk.hilbert.Spin,
    qubitA: int,
    qubitB: int
) -> float:
    """
    Calculate connected 2-point correlator in x direction.
    
    Args:
        vstate: Variational state
        hi: Hilbert space
        qubitA: First qubit index
        qubitB: Second qubit index
        
    Returns:
        Connected 2-point correlator
    """
    return (
        vstate.expect(nk.operator.spin.sigmax(hi, qubitA) * nk.operator.spin.sigmax(hi, qubitB)).mean - 
        vstate.expect(nk.operator.spin.sigmax(hi, qubitA)).mean * vstate.expect(nk.operator.spin.sigmax(hi, qubitB)).mean
    )


def z_connected_2point_correlator(
    vstate: nk.vqs.VariationalState,
    hi: nk.hilbert.Spin,
    qubitA: int,
    qubitB: int
) -> float:
    """
    Calculate connected 2-point correlator in z direction.
    
    Args:
        vstate: Variational state
        hi: Hilbert space
        qubitA: First qubit index
        qubitB: Second qubit index
        
    Returns:
        Connected 2-point correlator
    """
    return (
        vstate.expect(nk.operator.spin.sigmaz(hi, qubitA) * nk.operator.spin.sigmaz(hi, qubitB)).mean - 
        vstate.expect(nk.operator.spin.sigmaz(hi, qubitA)).mean * vstate.expect(nk.operator.spin.sigmaz(hi, qubitB)).mean
    )


def calculate_plaquette_stabilizer(
    vstate: nk.vqs.VariationalState,
    geometry,
    dual: bool = False
) -> Tuple[float, float]:
    """Mean and std of the PHYSICAL plaquette-stabilizer expectations <B_p>.

    On the sign-free (hz-only) cut every plaquette operator commutes with H, so
    the ground state has <B_p> = 1 exactly; a deviation from 1 diagnoses that the
    ansatz has leaked out of the B_p = +1 sector (a contaminated state). In the
    primal basis B_p = ZZZZ (diagonal, cheap); in the dual (Hadamard-conjugated)
    simulation basis the SAME physical operator is the X-product, so `dual=True`
    swaps the constructor and the JSON key keeps its physical meaning.

    Returns (Bp_mean, Bp_std) over all plaquettes.
    """
    hi = vstate.hilbert
    _s = nk.operator.spin.sigmax if dual else nk.operator.spin.sigmaz
    vals = []
    for plaq in geometry.plaq_all:
        idx = [j for j in plaq if j != -1]
        op = 1
        for j in idx:
            op = op * _s(hi, j)
        vals.append(vstate.expect(op).mean)
    vals = np.array([float(np.real(v)) for v in vals])
    return float(vals.mean()), float(vals.std())


def create_plaquette_stabilizer_callback(geometry) -> Callable:
    """Callback logging <B_p> mean/std. Fired by the optimizer every 8 steps (and
    optionally at the end); no internal step guard so an explicit final call runs."""
    def plaquette_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        Bp_mean, Bp_std = calculate_plaquette_stabilizer(
            vstate, geometry, dual=bool(config.get('dual_basis', False)))

        with open(config['filename'], 'r') as f:
            data = json.load(f)

        data["order_params"]["Bp_mean"].append(Bp_mean)
        data["order_params"]["Bp_std"].append(Bp_std)

        with open(config['filename'], 'w') as f:
            json.dump(data, f)

    return plaquette_callback


def calculate_Se(
    vstate: nk.vqs.VariationalState,
    geometry,
    dual: bool = False
) -> Tuple[float, float]:
    """Mean and std of the fermionic (dyon) term expectations <S_e> = <X_a . Z_b>.

    S_e = X_a . Z_b is the L-shaped two-body operator that binds an e (Z) to an m (X)
    into the composite fermion. In the pure toric ground state <X_a Z_b> = 0; as the
    h_f field is turned on the state polarizes along S_e so <S_e> grows 0 -> finite,
    and its susceptibility d<S_e>/dh_f peaks at the transition. Pairs come from
    geometry.fermion_pairs ([x_link, z_link]).

    Returns (Se_mean, Se_std) over all fermion pairs.
    """
    hi = vstate.hilbert
    # dual (Hadamard-conjugated) simulation basis: physical X_a.Z_b appears as Z_a.X_b.
    # h_f itself is asserted 0 in dual runs, so this stays a pure diagnostic there.
    if dual:
        _sa, _sb = nk.operator.spin.sigmaz, nk.operator.spin.sigmax
    else:
        _sa, _sb = nk.operator.spin.sigmax, nk.operator.spin.sigmaz
    vals = []
    for (a, b) in geometry.fermion_pairs:
        op = _sa(hi, a) * _sb(hi, b)
        vals.append(vstate.expect(op).mean)
    vals = np.array([float(np.real(v)) for v in vals])
    return float(vals.mean()), float(vals.std())


def create_Se_callback(geometry) -> Callable:
    """Callback logging <S_e> mean/std (the fermionic X.Z order parameter). No internal
    step guard so an explicit end-of-run call always runs."""
    def se_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        Se_mean, Se_std = calculate_Se(
            vstate, geometry, dual=bool(config.get('dual_basis', False)))

        with open(config['filename'], 'r') as f:
            data = json.load(f)

        data["order_params"]["Se_mean"].append(Se_mean)
        data["order_params"]["Se_std"].append(Se_std)

        with open(config['filename'], 'w') as f:
            json.dump(data, f)

    return se_callback


def calculate_vertex_stabilizer(
    vstate: nk.vqs.VariationalState,
    geometry,
    dual: bool = False
) -> Tuple[float, float]:
    """Mean and std of the PHYSICAL vertex/star-stabilizer expectations <A_v>.

    Complement to <B_p>: A_v detects e (charge) excitations, B_p detects m (flux).
    The fermionic field creates both e and m, so BOTH <A_v> and <B_p> should
    degrade from 1 across the transition (A_v tracks e-condensation, B_p tracks m).
    In the primal basis A_v = XXXX (off-diagonal but cheap at L=4); in the dual
    simulation basis the same physical operator is the Z-star (diagonal), so
    `dual=True` swaps the constructor and the JSON key keeps its physical meaning.

    Returns (Av_mean, Av_std) over all vertex stars.
    """
    hi = vstate.hilbert
    _s = nk.operator.spin.sigmaz if dual else nk.operator.spin.sigmax
    vals = []
    for vert in geometry.vertex_all:
        idx = [j for j in vert if j != -1]
        op = 1
        for j in idx:
            op = op * _s(hi, j)
        vals.append(vstate.expect(op).mean)
    vals = np.array([float(np.real(v)) for v in vals])
    return float(vals.mean()), float(vals.std())


def create_vertex_stabilizer_callback(geometry) -> Callable:
    """Callback logging <A_v> mean/std. No internal step guard so an explicit
    end-of-run call always runs."""
    def vertex_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        Av_mean, Av_std = calculate_vertex_stabilizer(
            vstate, geometry, dual=bool(config.get('dual_basis', False)))

        with open(config['filename'], 'r') as f:
            data = json.load(f)

        data["order_params"]["Av_mean"].append(Av_mean)
        data["order_params"]["Av_std"].append(Av_std)

        with open(config['filename'], 'w') as f:
            json.dump(data, f)

    return vertex_callback


def calculate_dressed_star(
    vstate: nk.vqs.VariationalState,
    geometry,
    dual: bool = False
) -> Tuple[float, float]:
    """Mean and std of the dressed-star expectations <A'_v> (fermionic TC, --ftc).

    A'_v = A_v * B_NE(v): sigma_x on the star links times sigma_z on the NE-plaquette
    links; the same-site X.Z overlap on the vertex's up/right links gives the two
    sigma_y factors and the -1 automatically (same construction as hamiltonian.py).
    Vertices without an NE plaquette contribute their bare A_v (z_links == []), so
    the mean runs over the model's actual 16 vertex-sector stabilizers at L=4 and
    must sit at ~1 in the unperturbed ground state, degrading under a field.
    Contamination check alongside <A_v> and <B_p> (all three ~1 at h=0).

    Returns (Avp_mean, Avp_std) over all vertices.
    """
    hi = vstate.hilbert
    _sa = nk.operator.spin.sigmaz if dual else nk.operator.spin.sigmax
    _sb = nk.operator.spin.sigmax if dual else nk.operator.spin.sigmaz
    vals = []
    for (x_links, z_links) in geometry.dressed_stars:
        op = 1
        for j in x_links:
            op = op * _sa(hi, j)
        for j in z_links:
            op = op * _sb(hi, j)
        vals.append(vstate.expect(op).mean)
    vals = np.array([float(np.real(v)) for v in vals])
    return float(vals.mean()), float(vals.std())


def create_dressed_star_callback(geometry) -> Callable:
    """Callback logging <A'_v> mean/std (dressed stars, --ftc). No internal step
    guard so an explicit end-of-run call always runs."""
    def dressed_star_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        Avp_mean, Avp_std = calculate_dressed_star(
            vstate, geometry, dual=bool(config.get('dual_basis', False)))

        with open(config['filename'], 'r') as f:
            data = json.load(f)

        data["order_params"]["Avp_mean"].append(Avp_mean)
        data["order_params"]["Avp_std"].append(Avp_std)

        with open(config['filename'], 'w') as f:
            json.dump(data, f)

    return dressed_star_callback


def _check_flip_invariance(model, params, N, clusters, n_configs: int = 8) -> float:
    """Max |Delta log psi| over configs x clusters when flipping each cluster's spins.

    Shared primitive for the exact-symmetry gates: a symmetry that acts by spin flips
    (X-type stabilizer in the simulation basis) must leave log psi unchanged.
    """
    assert len(clusters) > 0, "no flip clusters to test -- gate would pass vacuously"
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 1.0], size=(n_configs, N))
    base = np.asarray(model.apply({'params': params}, X))
    max_dev = 0.0
    for cluster in clusters:
        Xf = X.copy()
        Xf[:, np.asarray(cluster)] *= -1
        flipped = np.asarray(model.apply({'params': params}, Xf))
        max_dev = max(max_dev, float(np.max(np.abs(flipped - base))))
    return max_dev


def check_Av_invariance(model, params, geometry, n_configs: int = 8) -> float:
    """Exact vertex/gauge (A_v) symmetry check: log psi(sigma) == log psi(A_v sigma).

    At init the full_transformer (odd embedding + identity Block-1 + channelwise Wilson
    fusion) is exactly A_v-invariant; a nonzero deviation means the symmetry-at-init is
    broken (biased embedding, a live sublayer, or a wrong fusion). Uses ALL vertex stars
    (bulk 4-edge + boundary 2-3-edge). Returns max |Delta log psi| over configs x vertices.
    """
    stars = list(geometry.vertex_bulk_hetero) + list(geometry.vertex_edge_hetero)
    return _check_flip_invariance(model, params, geometry.N, stars, n_configs)


def check_Bp_invariance(model, params, geometry, n_configs: int = 8) -> float:
    """Exact plaquette (B_p) symmetry check for DUAL-BASIS runs.

    In the Hadamard-conjugated simulation basis the physical B_p is an X-product, so
    applying it flips the 4 plaquette edges; a star-Wilson network (function of star
    products only, e.g. the dual Combo at identity init) must leave log psi unchanged.
    Returns max |Delta log psi| over configs x plaquettes.
    """
    plaqs = [[e for e in p if e != -1] for p in geometry.plaq_all]
    return _check_flip_invariance(model, params, geometry.N, plaqs, n_configs)


def dump_attention(vstate, config) -> None:
    """Dump Block-1 gamma (attention range) + alpha tables at convergence (interpretability).

    Writes <filename_base>_attn.json with softplus(gamma) and the raw alpha tables per
    (block, head). Only meaningful for the full_transformer arm.
    """
    import jax

    def _key(k):
        for a in ("key", "name", "idx"):
            if hasattr(k, a):
                return str(getattr(k, a))
        return str(k)

    gammas, alphas = {}, {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(vstate.parameters):
        name = _key(path[-1])
        blk = ".".join(_key(k) for k in path[:-1])
        a = np.asarray(leaf)
        if name == "gamma_raw":
            gammas[blk] = (np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))).tolist()
        elif name == "alpha":
            alphas[blk] = a.tolist()
    out = {"gammas": gammas,
           "alpha_shapes": {k: list(np.shape(v)) for k, v in alphas.items()},
           "alpha": alphas}
    with open(f"{config['filename_base']}_attn.json", 'w') as f:
        json.dump(out, f)
    print(f"Wrote attention dump: {config['filename_base']}_attn.json "
          f"({len(gammas)} gamma sets, {len(alphas)} alpha tables)")


def create_wilson_loop_callback(geometry) -> Callable:
    """
    Create a callback to calculate Wilson loops during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating Wilson loops
    """
    def wilson_loop_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating Wilson loops...")
        wilson_stats = calculate_wilson_loops(vstate, geometry, radius=1)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert to real floats
        data["order_params"]["WilsonBFFM"].append([float(x.real) for x in wilson_stats])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return wilson_loop_callback


def create_magnetization_callback(geometry) -> Callable:
    """
    Create a callback to calculate magnetizations during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating magnetizations
    """
    def magnetization_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        # Calculate every 8 steps regardless of Lx
        if step % 8 == 0:
            print(f"Step {step}: Calculating magnetizations...")
            magnetizationsXmean, magnetizationsXstd, magnetizationsYmean, magnetizationsYstd, magnetizationsZmean, magnetizationsZstd = calculate_magnetizations(
                vstate, geometry, dual=bool(config.get('dual_basis', False)))
            
            with open(config['filename'], 'r') as f:
                data = json.load(f)
            
            # Convert to real floats
            data["order_params"]["magnetization_Xmean"].append([float(x.real) for x in magnetizationsXmean])
            data["order_params"]["magnetization_Xstd"].append([float(x.real) for x in magnetizationsXstd])
            data["order_params"]["magnetization_Ymean"].append([float(x.real) for x in magnetizationsYmean])
            data["order_params"]["magnetization_Ystd"].append([float(x.real) for x in magnetizationsYstd])
            data["order_params"]["magnetization_Zmean"].append([float(x.real) for x in magnetizationsZmean])
            data["order_params"]["magnetization_Zstd"].append([float(x.real) for x in magnetizationsZstd])
            
            with open(config['filename'], 'w') as f:
                json.dump(data, f)
    
    return magnetization_callback


def create_renyi_callback(geometry) -> Callable:
    """
    Create a callback to calculate Renyi entropy during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating Renyi entropy
    """
    def renyi_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating Renyi entropy...")
        renyi_mean, renyi_std, qubit_no, perimeter = calculate_renyi_entropy(vstate, geometry)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert NumPy values to Python floats and integers
        data["order_params"]["renyi2_entropy"].append([
            float(renyi_mean),
            float(renyi_std),
            int(qubit_no),
            int(perimeter)
        ])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return renyi_callback


def create_2point_callback(geometry) -> Callable:
    """
    Create a callback to calculate 2-point correlators during optimization.
    
    Args:
        geometry: Geometry object
        
    Returns:
        Callback function for calculating 2-point correlators
    """
    def correlator_callback(vstate: nk.vqs.VariationalState, step: int, time: float, config: Dict[str, Any]) -> None:
        print(f"Step {step}: Calculating 2-point correlators...")
        unique_distances, xx_average, zz_average, xx_std, zz_std = calculate_2point_correlators(vstate, geometry)
        
        with open(config['filename'], 'r') as f:
            data = json.load(f)
        
        # Convert NumPy arrays to lists
        data["order_params"]["2pointCorrelators"].append([
            unique_distances.tolist(),
            xx_average.tolist(),
            zz_average.tolist(),
            xx_std.tolist(),
            zz_std.tolist()
        ])
        
        with open(config['filename'], 'w') as f:
            json.dump(data, f)
    
    return correlator_callback


def create_conditional_callbacks(geometry) -> List[Callable]:
    """
    Create callbacks based on system size.
    
    Args:
        geometry: Geometry object
        
    Returns:
        List of callback functions
    """
    callbacks = []

    # Always add magnetization callback
    callbacks.append(create_magnetization_callback(geometry))

    # Plaquette-stabilizer <B_p> diagnostic (must stay ~1 on the sign-free cut)
    callbacks.append(create_plaquette_stabilizer_callback(geometry))

    # if geometry.Lx > 6:
    #     callbacks.append(create_wilson_loop_callback(geometry))
    #     callbacks.append(create_renyi_callback(geometry))
    #     callbacks.append(create_2point_callback(geometry))
    
    return callbacks 