"""
Module for handling the lattice geometry, constructing stabilizers, and building
the connectivity for the toric code model.
"""

import numpy as np
import netket as nk
from typing import List, Tuple, Dict, Any, Optional, Union

class ToricCodeGeometry:
    """
    Class to handle the geometry and topology of the toric code model.
    
    Attributes:
        Lx (int): Number of vertices in the x direction
        Ly (int): Number of vertices in the y direction
        bc (str): Boundary conditions, either 'OBC' or 'PBC'
        N (int): Total number of qubits in the system
        arr_coord (np.ndarray): Array of qubit coordinates
        dg_v (nk.graph.Graph): Dual lattice for vertex stabilizers
        dg_p (nk.graph.Graph): Dual lattice for plaquette stabilizers
        vertex_all (List): List of all vertex stabilizers
        plaq_all (List): List of all plaquette stabilizers
        bonds (List): List of nearest-neighbor bonds
    """
    
    def __init__(self, Lx: int, Ly: int, bc: str = 'OBC'):
        """
        Initialize the geometry for a toric code model.
        
        Args:
            Lx: Number of vertices in the x direction
            Ly: Number of vertices in the y direction
            bc: Boundary conditions, 'OBC' for open or 'PBC' for periodic
        """
        self.Lx = Lx
        self.Ly = Ly
        self.bc = bc
        
        # Calculate the number of qubits in the system
        if bc == "OBC":
            self.N = 2 * Lx * Ly - Lx - Ly  # OBC
        else:
            self.N = 2 * Lx * Ly  # PBC
            
        # Create the lattice
        self._setup_lattice()
        
        # Generate stabilizers
        self.vertex_all = self._generate_stabilizer_qubits(kind="vertex")
        self.plaq_all = self._generate_stabilizer_qubits(kind="plaq")
        
        # Generate nearest-neighbor bonds
        self.bonds = self._generate_bonds()
        self.Nbonds = len(self.bonds)

        # Generate fermionic (dyon) hopping pairs for the S_e = X_a.Z_b field
        self.fermion_pairs = self._generate_fermion_pairs()
        self.Nfermion = len(self.fermion_pairs)

        # Dressed stars A'_v = A_v * B_NE(v) for the fermionic toric code (--ftc)
        self.dressed_stars = self._generate_dressed_stars()
        
        # Extract non-boundary vertex stabilizers
        self.vertex_bulk_hetero, self.vertex_edge_hetero = self._separate_vertex_stabilizers()
        
    def _setup_lattice(self):
        """Setup the lattice and calculate atomic coordinates."""
        # Lattice basis
        basis = [np.array([1, 0]), np.array([0, 1])]
        # Atom basis for toric code
        basis_atoms = [[1/2, 0], [0, 1/2]]
        
        # Generate lattice coordinates
        lattice_coord = [self._map1Dto2D(j)[0] * basis[0] + 
                         self._map1Dto2D(j)[1] * basis[1] 
                         for j in range(0, self.Lx * self.Ly)]
        
        # Generate atom coordinates
        atom_coord = []
        for j in range(0, len(basis_atoms)):
            atom_coord += (lattice_coord + np.array(basis_atoms[j])).tolist()
        atom_coord = np.array(atom_coord)
        
        # Sort atomic coordinates
        atom_coord = atom_coord[np.lexsort((atom_coord[:, 1], atom_coord[:, 0]))]
        
        # Handle open boundary conditions
        if self.bc == "OBC":
            self.arr_coord = self._select_obc_subset(atom_coord)
            self.dg_v = nk.graph.Lattice(basis_vectors=basis, pbc=False, extent=[self.Lx, self.Ly])
            self.dg_p = nk.graph.Lattice(basis_vectors=basis, pbc=False, 
                                         site_offsets=[1/2, 1/2], extent=[self.Lx-1, self.Ly-1])
        else:
            self.arr_coord = atom_coord
            self.dg_v = nk.graph.Lattice(basis_vectors=basis, pbc=True, extent=[self.Lx, self.Ly])
            self.dg_p = nk.graph.Lattice(basis_vectors=basis, pbc=True, 
                                         site_offsets=[1/2, 1/2], extent=[self.Lx, self.Ly])
    
    def _map1Dto2D(self, n: int) -> Tuple[int, int]:
        """Map a 1D index to 2D coordinates."""
        return int(np.floor(n / self.Ly)), n % self.Ly
    
    def _select_obc_subset(self, arr: np.ndarray) -> np.ndarray:
        """Select OBC qubits by removing those outside the range."""
        return arr[np.logical_and(
            np.logical_and(arr[:, 0] <= self.Lx-1, arr[:, 1] <= self.Ly-1),
            np.logical_and(arr[:, 0] >= 0, arr[:, 1] >= 0))]
    
    def _mapping2Dto1D(self, arr: np.ndarray, entry: np.ndarray) -> np.ndarray:
        """Map 2D coordinates to 1D index."""
        return np.argwhere(np.logical_and(arr[:, 0] == entry[0], arr[:, 1] == entry[1]))
    
    def _generate_stabilizer_qubits(self, kind: str) -> List[List[int]]:
        """
        Generate stabilizer operators.
        
        Args:
            kind: Either "vertex" or "plaq" for vertex or plaquette stabilizers
            
        Returns:
            List of lists, where each inner list contains the qubit indices for a stabilizer
        """
        if kind == "vertex":
            dg = self.dg_v
        else:
            dg = self.dg_p
            
        if self.bc == "OBC":
            neighbors = np.array([
                dg.positions + np.array([1/2, 0]),
                dg.positions + np.array([0, 1/2]),
                dg.positions + np.array([-1/2, 0]),
                dg.positions + np.array([0, -1/2])
            ])  # top-right, top-left, bottom-left, bottom-right neighbor coordinate pairs
        else:
            neighbors = np.array([
                (dg.positions + np.array([1/2, 0])) % self.Lx,
                (dg.positions + np.array([0, 1/2])) % self.Lx,
                (dg.positions + np.array([-1/2, 0])) % self.Lx,
                (dg.positions + np.array([0, -1/2])) % self.Lx
            ])
            
        return [[self._mapping2Dto1D(self.arr_coord, coord)[0][0] 
                 if len(self._mapping2Dto1D(self.arr_coord, coord)) > 0 else -1 
                 for coord in neighbors[:, k, :]] for k in range(0, neighbors.shape[1])]
    
    def _generate_bonds(self) -> List[List[int]]:
        """Generate nearest-neighbor bonds between qubits."""
        bonds = []
        for j in range(0, self.N):
            nn_topright = self._mapping2Dto1D(
                self.arr_coord, (self.arr_coord + np.array([1/2, 1/2]))[j]
            )
            nn_downright = self._mapping2Dto1D(
                self.arr_coord, (self.arr_coord + np.array([1/2, -1/2]))[j]
            )
            
            if len(nn_topright) > 0:
                bonds.append([j, nn_topright[0][0]])
            if len(nn_downright) > 0:
                bonds.append([j, nn_downright[0][0]])
                
        return bonds

    def _generate_fermion_pairs(self) -> List[List[int]]:
        """Fermionic (dyon) hopping pairs [x_link, z_link] for the S_e = X_a.Z_b field.

        S_e binds an e (Z, moved between vertices) to an m (X, moved between plaquettes)
        as the composite fermion epsilon; a term is a product on two perpendicular links
        meeting at a shared vertex (the "L" in the figure). For every link j the partner
        is the link at arr_coord[j] + [1/2, 1/2] (the same top-right offset used in
        _generate_bonds), and the rule is UNIFORM: Z on the source link j, X on the
        partner. This reproduces BOTH figure orientations (families):
          - j horizontal (x+1/2, y): partner is the vertical link above its right
            endpoint => Z on horizontal, X on vertical      (figure left panel).
          - j vertical (x, y+1/2): partner is the horizontal link right-and-above its
            top endpoint => Z on vertical, X on horizontal   (figure right panel).
        Each term is keyed by its unique Z (source) link, so there is no double-counting.
        Pairs whose partner falls off the OBC boundary (empty argwhere) are skipped.
        Returns a list of [x_link, z_link] qubit-index pairs.
        """
        pairs = []
        for j in range(0, self.N):
            partner = self._mapping2Dto1D(
                self.arr_coord, (self.arr_coord + np.array([1/2, 1/2]))[j]
            )
            if len(partner) == 0:
                continue  # partner off the OBC boundary -> skip
            p = partner[0][0]
            x_link, z_link = p, j  # Z on source link j, X on the +[1/2,1/2] partner
            pairs.append([x_link, z_link])
        return pairs

    def _generate_dressed_stars(self) -> List[Tuple[List[int], List[int]]]:
        """Dressed vertex stabilizers A'_v = A_v * B_NE(v) for the fermionic toric code.

        Each entry corresponds positionally to vertex_all / dg_v.positions and is a
        pair (x_links, z_links) consumed by hamiltonian.create_hamiltonian(ftc=True):
          - x_links: the star links of v (vertex_all[v] with the -1 sentinels removed)
          - z_links: the 4 links of the plaquette diagonally up-right (north-east) of
            v, i.e. plaq_all[p] for the plaquette centered at position(v) + [1/2, 1/2];
            [] when that plaquette falls off the OBC boundary (bare star, chosen
            boundary rule: top row / right column vertices stay plain A_v = XX(XX)).

        The vertex's up and right links are SHARED between the star and the NE
        plaquette and must appear in BOTH lists: the Hamiltonian multiplies
        sigma_x * sigma_z on those sites, giving -i*sigma_y each, so the dressed
        operator -Y.Y.X.X.Z.Z (with its (-i)^2 = -1) emerges with no explicit sign
        bookkeeping. Counts at L=4: 9 dressed stars (one per plaquette) + 7 bare;
        the corner vertex (0,0) is fully shared (star = {up, right} only), so its
        dressed operator is a 4-body -Y.Y.Z.Z. Cross-check against the independent
        numpy reference in scratchpad/validate_ftc.py.
        """
        dressed = []
        for v in range(len(self.vertex_all)):
            x_links = [j for j in self.vertex_all[v] if j != -1]
            ne_pos = self.dg_v.positions[v] + np.array([1 / 2, 1 / 2])
            hit = self._mapping2Dto1D(self.dg_p.positions, ne_pos)
            z_links = self.plaq_all[hit[0][0]] if len(hit) > 0 else []
            dressed.append((x_links, z_links))
        return dressed

    def _separate_vertex_stabilizers(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Separate vertex stabilizers into bulk and edge operators."""
        vertex_bulk_hetero = []
        vertex_edge_hetero = []
        
        for v in self.vertex_all:
            lst = [el for el in v if el != -1]
            if len(lst) == 4:
                vertex_bulk_hetero.append(lst)
            else:
                vertex_edge_hetero.append(lst)
                
        return vertex_bulk_hetero, vertex_edge_hetero
    
    def get_vertex_all_hetero(self) -> List[List[int]]:
        """Get all vertex stabilizers with -1 entries removed."""
        return [[el for el in v if el != -1] for v in self.vertex_all]
    
    def construct_Wilson_generators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct generators for the Z2xZ2xZ2x...=(Z2)^(n) group.
        
        Returns:
            Tuple containing:
                - generators_mat: Matrix representation of generators
                - generators_lst: List representation of generators
        """
        generators_lst = []
        generators_mat = []
        vertex_all_hetero = self.get_vertex_all_hetero()
        
        for v in range(0, len(vertex_all_hetero)):
            mat = np.ones(self.N)  # vector with the size of the qubits
            mat[np.array(vertex_all_hetero[v])] = -1  # to act with a vertex operator simply swap 1 to -1
            generators_lst.append(mat)
            generators_mat.append(np.diag(mat))
            
        return np.array(generators_mat), np.array(generators_lst)
    
    def find_generators(self, vertex_lst: List[List[int]]) -> np.ndarray:
        """
        Find generators for a subset of vertex operators.
        
        Args:
            vertex_lst: List of vertex operators
            
        Returns:
            Array of generators
        """
        generators_lst = []
        vertex_all_hetero = self.get_vertex_all_hetero()
        
        for v in range(0, len(vertex_lst)):
            mat = np.ones(self.N)
            mat[np.array(vertex_all_hetero[v])] = -1
            generators_lst.append(np.diag(mat))
            
        return np.array(generators_lst)
    
    def select_subset(self, pos: Tuple[float, float], radius: float) -> np.ndarray:
        """
        Select subset of qubits bounded by pos-radius and pos+radius points.
        
        Args:
            pos: Center position (x, y)
            radius: Radius around the center
            
        Returns:
            Subset of qubit coordinates
        """
        pos_x, pos_y = pos
        xmax = pos_x + radius
        xmin = pos_x - radius
        ymax = pos_y + radius
        ymin = pos_y - radius
        
        return self.arr_coord[np.logical_and(
            np.logical_and(self.arr_coord[:, 0] <= xmax, self.arr_coord[:, 1] <= ymax),
            np.logical_and(self.arr_coord[:, 0] >= xmin, self.arr_coord[:, 1] >= ymin)
        )]
    
    def qubit_select(self, selected_locs: np.ndarray) -> List[int]:
        """
        Map 2D coordinates to 1D qubit indices.
        
        Args:
            selected_locs: Array of 2D coordinates
            
        Returns:
            List of 1D qubit indices
        """
        return [self._mapping2Dto1D(self.arr_coord, en)[0][0] for en in selected_locs]
    
    def select_bulk(self) -> np.ndarray:
        """
        Select only bulk qubits from the set of qubits.
        
        Returns:
            Array of bulk qubit coordinates
        """
        boundary_locs = np.vstack((
            self.arr_coord[(self.arr_coord == np.max(self.arr_coord)).any(axis=1)],
            self.arr_coord[(self.arr_coord == np.min(self.arr_coord)).any(axis=1)]
        ))
        
        set1 = set(map(tuple, self.arr_coord))
        set2 = set(map(tuple, boundary_locs))
        bulk_locs = np.array(list(set1 - set2))
        
        return bulk_locs 