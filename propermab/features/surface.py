# Copyright 2024 Regeneron Pharmaceuticals Inc. All rights reserved.
#
# License for Non-Commercial Use of PROPERMAB code
#
# All files in this repository (“source code”) are licensed under the following terms below:
#
# “You” refers to an academic institution or academically employed full-time personnel only.
#
# “Regeneron” refers to Regeneron Pharmaceuticals, Inc.
#
# Regeneron hereby grants You a right to use, reproduce, modify, or distribute the PROPERMAB source 
# code, in whole or in part, whether in original or modified form, for academic research purposes only. 
# The foregoing right is royalty-free, worldwide (subject to applicable laws of the United States), 
# revocable, non-exclusive, and non-transferable.
#
# Prohibited Uses: The rights granted herein do not include any right to use by commercial entities 
# or commercial use of any kind, including, without limitation, (1) any integration into other code 
# or software that is used for further commercialization, (2) any reproduction, copy, modification 
# or creation of a derivative work that is then incorporated into a commercial product or service or 
# otherwise used for any commercial purpose, (3) distribution of the source code, in whole or in part, 
# or any resulting executables, in any commercial product, or (4) use of the source code, in whole 
# or in part, or any resulting executables, in any commercial online service.
#
# Except as expressly provided for herein, nothing in this License grants to You any right, title or 
# interest in and to the intellectual property of Regeneron (either expressly or by implication or estoppel).  
# Notwithstanding anything else in this License, nothing contained herein shall limit or compromise 
# the rights of Regeneron with respect to its own intellectual property or limit its freedom to practice 
# and to develop its products and product candidates.
#
# If the source code, whole or in part and in original or modified form, is reproduced, shared or 
# distributed in any manner, it must (1) identify Regeneron Pharmaceuticals, Inc. as the original 
# creator, (2) retain any copyright or other proprietary notices of Regeneron, (3) include a copy 
# of the terms of this License.
#
# TO THE GREATEST EXTENT PERMITTED UNDER APPLICABLE LAW, THE SOURCE CODE (AND ANY DOCUMENTATION) IS 
# PROVIDED ON AN “AS-IS” BASIS, AND REGENERON PHARMACEUTICALS, INC. EXPRESSLY DISCLAIMS ALL 
# REPRESENTATIONS, WARRANTIES, AND CONDITIONS WITH RESPECT THERETO OF ANY KIND CONCERNING THE SOURCE 
# CODE, IN WHOLE OR IN PART AND IN ORIGINAL OR MODIFIED FORM, WHETHER EXPRESS, IMPLIED, STATUTORY, OR 
# OTHER REPRESENTATIONS, WARRANTIES AND CONDITIONS, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS, 
# ACCURACY, COMPLETENESS, RIGHT TO QUIET ENJOYMENT, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR 
# NOT KNOWN OR DISCOVERABLE.  REGENERON DOES NOT WARRANT THAT THE SOURCE CODE WILL OPERATE IN AN 
# UNINTERRUPTED FASHION AND DATA MAY BE LOST OR UNRECOVERABLE. IN THE EVENT ANY OF THE PRIOR DISCLAIMERS 
# ARE UNENFORCEABLE UNDER APPLICABLE LAW, THE LICENSES GRANTED HEREIN WILL IMMEDIATELY BE NULL AND 
# VOID AND YOU SHALL IMMEDIATELY RETURN TO REGENERON THE SOURCE CODE OR DESTROY IT.
#
# IN NO CASE SHALL REGENERON BE LIABLE FOR ANY LOSS, CLAIM, DAMAGE, OR EXPENSES, OF ANY KIND, WHICH 
# MAY ARISE FROM OR IN CONNECTION WITH THIS LICENSE OR THE USE OF THE SOURCE CODE. YOU WAIVE AND 
# RELEASE REGENERON FOREVER FROM ANY LIABILITY AND YOU SHALL INDEMNIFY AND HOLD REGENERON, ITS AFFILAITES 
# AND ITS AND THEIR EMPLOYEES AND AGENTS HARMLESS FROM ANY LOSS, CLAIM, DAMAGE, EXPENSES, OR LIABILITY, 
# OF ANY KIND, FROM A THIRD-PARTY WHICH MAY ARISE FROM OR IN CONNECTION WITH THIS LICENSE OR YOUR USE 
# OF THE SOURCE CODE.

# You agree that this License and its terms are governed by the laws of the State of New York, without 
# regard to choice of law rules and the United Nations Convention on the International Sale of Goods 
# shall not apply.
#
# Please reach out to Regeneron Pharmaceuticals Inc./Administrator relating to any non-academic or 
# commercial use of the source code.
from typing import Union, List, Dict

import itertools
import numpy as np
from scipy import spatial
import numpy.typing as npt

from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model

import networkx as nx
from sklearn.cluster import DBSCAN

from .hydrophobicity import HeidenHydrophobicPotential
from .hydrophobicity import CRIPPEN_PARAMS, PDB_TO_CRIPPEN
from ..sequence import numbering


class SurfaceVertex:
    """Place-holding class, might be useful to make the surface.py module more
    object-oriented.
    """
    def __init__(self, vertex_coord, vertex_id, atom=None, residue=None, features=None):
        self.vertex_coord = vertex_coord
        self.vertex_id = vertex_id
        self.atom = atom
        self.residue = residue
        self.features = features


class TriangleFace:
    def __init__(self, coords: npt.ArrayLike, vertices: List[int] = None,
                 face_id: int = None, atom: Atom = None,
                 residue: Residue = None, features: np.ndarray = None):
        """A class representing the triangle face.

        Parameters
        ----------
        coords : npt.ArrayLike
            Cartesian coordinates of the vertices.
        vertices : List[int]
            IDs of the vertices.
        face_id : int
            ID of the face.
        atom : Atom
            The Biopython Atom object to which this face is assigned.
        residue : Residue
            The Biopython Residue object to which this face is assigned.
        features : np.ndarray
            A numerical vector encoding this face.
        """
        assert np.array(coords).shape == (3, 3)
        assert len(vertices) == 3
        self.coords = np.asarray(coords)
        self.vertices = vertices
        self.face_id = face_id
        self.atom = atom
        self.residue = residue
        self.features = features
        self._center = None
        self._area = None

    @property
    def area(self):
        if self._area is None:
            # the cross product of two sides is a normal vector
            normal = np.cross(
                self.coords[1] - self.coords[0],
                self.coords[2] - self.coords[0]
            )
            # the norm of the cross product of two sides is twice the area
            self._area = np.linalg.norm(normal) / 2
        return self._area

    @property
    def center(self):
        if self._center is None:
            self._center = np.mean(self.coords, axis=0)
        return self._center


class SurfacePatch:
    def __init__(self, vertices: npt.ArrayLike, faces: List[TriangleFace],
                 prop: str = None, name: str = None):
        """A class representing surface patches.

        Parameters
        ----------
        vertices : List[int]
            IDs of the vertices of this surface patch.
        faces : List[TriangleFace]
            The triangle faces of this surface patch.
        prop : str
            Biophysical property of this surface patch.
        name : str
            Name of this surface patch.
        """
        self.vertices = vertices
        self.vertex_coords = None
        self.faces = faces
        self.prop = prop
        self.name = name
        self._area = None

    @property
    def area(self):
        if self._area is None:
            self._area = sum([f.area for f in self.faces])
        return self._area

    @staticmethod
    def find_patch_boundary(patch_vertices: list, surf_graph: nx.Graph) -> set:
        """Finds the boundary vertices of a patch in the context of the given graph.

        The boundary vertices of a patch are defined as the subset of vertices of
        the patch that are connected to at least one vertex not in the patch.

        Parameters
        ----------
        patch_vertices : List[int]
            The vertices of the patch.
        surf_graph : nx.Graph
            The graph context for the patch.

        Returns
        -------
        List
            A subset of vertices of the patch.

        """
        boundary_edges = list(nx.edge_boundary(surf_graph, patch_vertices))
        boundary_nodes = set([u for u, _ in boundary_edges])
        return boundary_nodes

    def distance_to_cdr(self, cdr_vertices: dict, surf_graph: nx.Graph) -> dict:
        """Calculates the shortest geodesic distances from this patch to each of the CDR regions.

        The geodesic distance is approximated by the shortest path distance between two vertices
        according to the Dijkstra's algorithm. Edge weight is simply the length of the edge.

        By using only the boundary vertices, the routine speeds up roughly by a factor of ~4.

        Parameters
        ----------
        cdr_vertices : dict
            Vertices of each CDR region.
        surf_graph : nx.Graph
            A networkx Graph object representing the surface triangular mesh of the antibody.

        Returns
        -------
        dict
            A dict where the keys are the CDR names, and the values are the respective
            shortest geodesic distance between this patch and the CDR region.
        """
        largest_component = max(nx.connected_components(surf_graph), key=len)
        patch_boundary_vertices = self.find_patch_boundary(self.vertices, surf_graph)
        patch_to_cdr_distances = {}
        for cdr, vertices in cdr_vertices.items():
            cdr_boundary_vertices = self.find_patch_boundary(vertices, surf_graph)
            shortest_dist_to_cdr = np.inf
            for u in cdr_boundary_vertices:
                if u not in largest_component:
                    continue
                for v in patch_boundary_vertices:
                    if v not in largest_component:
                        continue
                    dist = nx.shortest_path_length(surf_graph, u, v, weight='length')
                    if dist < shortest_dist_to_cdr:
                        shortest_dist_to_cdr = dist
            patch_to_cdr_distances[cdr] = shortest_dist_to_cdr

        return patch_to_cdr_distances

    def is_near_cdr(self, cdr_vertices: dict, surf_graph: nx.Graph, dist_cutoff: float = 5.) -> bool:
        """Determine whether this surface patch is near CDR. If the geodesic distance between
        any pair of vertices from `cdr_vertices` and vertices of this patch is below `dist_cutoff`,
        then this patch is considered near CDR.

        The geodesic distance is approximated by the shortest path distance between two vertices
        according to the Dijkstra's algorithm. Edge weight is simply the length of the edge.

        By using only the boundary vertices, the routine speeds up roughly by a factor of ~4.

        Parameters
        ----------
        cdr_vertices : dict
            Vertices of each CDR region.
        surf_graph : nx.Graph
            A networkx Graph object representing the surface triangular mesh of the antibody.
        dist_cutoff : float

        Returns
        -------
        bool
            True if the patch is within `dist_cutoff` of any CDR region, else False.

        """
        largest_component = max(nx.connected_components(surf_graph), key=len)
        patch_boundary_vertices = self.find_patch_boundary(self.vertices, surf_graph)
        for cdr, vertices in cdr_vertices.items():
            if len(set(vertices).intersection(self.vertices)) != 0:
                return True

            cdr_boundary_vertices = self.find_patch_boundary(vertices, surf_graph)
            for u in cdr_boundary_vertices:
                if u not in largest_component:
                    continue
                for v in patch_boundary_vertices:
                    if v not in largest_component:
                        continue
                    dist = nx.shortest_path_length(surf_graph, u, v, weight='length')
                    if dist <= dist_cutoff:
                        return True
        return False

    def is_near_cdr_kd_tree(self, cdr_vertex_coords: dict, dist_cutoff: float = 5.0) -> bool:
        """KDTree based algorithm for determining if a surface patch is near a CDR region.

        Two KD trees are constructed. One for the surface patch, the other for the CDR region.
        Then a neighbor search of the surface patch tree against the CDR tree is performed.
        The search returns a list of lists. The length of the nested list is N, i.e. the number
        of vertices of the surface patch. Each list i in the nested list has the indices of the
        vertices from the CDR tree that are the neighbors of vertex i of the surface patch. List
        i will be empty if vertex i of the surface patch does not have neighbors from the CDR
        patch.

        Parameters
        ----------
        cdr_vertex_coords : dict
            A mapping from CDR names to CDR vertex coordinates.
        dist_cutoff : float
            Euclidean distance cutoff between two vertices for them to be considered neighbors.

        Returns
        -------
        bool
            True if there is at least one vertex from any CDR regions found to be a neighbor of
            this surface patch vertices, else False.

        """
        if self.vertex_coords is None:
            raise ValueError('No valid vertex coordinates. Set coordinates first!')

        patch_kd_tree = spatial.KDTree(self.vertex_coords)
        for cdr, cdr_coords in cdr_vertex_coords.items():
            cdr_kd_tree = spatial.KDTree(cdr_coords)
            indexes = patch_kd_tree.query_ball_tree(cdr_kd_tree, r=dist_cutoff)
            for neighbor_list in indexes:
                if len(neighbor_list) != 0:
                    return True
        return False


class Surface:
    def __init__(self, vertices: npt.ArrayLike, faces: List[List],
                 vertex_atom_ids: List[int] = None,
                 struct: Model = None, apbs_values: List[float] = None):
        """A class representing the surface of a protein.

        Parameters
        ----------
        vertices : npt.ArrayLike
            The Cartesian coordinates of all vertices.
        faces : List[List]
            The composing vertices of all faces.
            Note that vertices from NanoShaper are numbered starting from 1. Thus, to
            get the coordinates of vertex i of a face, we use `self.vertices[i - 1]`.
        vertex_atom_ids : List[int]
            The atom number that each vertex is assigned to.
        struct : Bio.PDB.Model
            The structure that this surface represents. It must be parsed from the same PDB
            file that was used as input to NanoShaper and APBS.
        apbs_values : List[float]
            The electrostatic potentials at the vertices.
        """
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)
        if vertex_atom_ids is not None:
            assert len(vertices) == len(vertex_atom_ids)
        self.vertex_atom_ids = vertex_atom_ids
        self.struct = struct
        self._residue_faces = None
        if apbs_values is not None:
            assert len(vertices) == len(apbs_values)
        self.apbs_values = apbs_values

        self._triangle_faces = None
        self._total_area = None

    @property
    def triangle_faces(self):
        """Represents each triangle face as a TriangleFace object.

        Returns
        -------
        self._triangle_faces

        """
        if self._triangle_faces is None:
            all_triangle_faces = []
            # faces are numbered starting from 1
            for i, face_vertices in enumerate(self.faces, start=1):
                face_coords = self._get_face_coords(face_vertices)
                all_triangle_faces.append(
                    TriangleFace(coords=face_coords, vertices=face_vertices, face_id=i)
                )
            self._triangle_faces = all_triangle_faces
        return self._triangle_faces

    def find_residue_vertices(self, residue: Residue = None) -> Union[Dict[Residue, List], List]:
        """Finds all the vertices belonging to the requested residue.

        Each line in the NanoShaper .vert file represents a vertex. The line has the Cartesian
        coordinates of the vertex and the ID of the atom the vertex is assigned to by NanoShaper.
        This routine simply extracts the residue the atom belongs to, then assigns the vertex
        to the residue.

        Parameters
        ----------
        residue : Residue
            The residue for which vertices are requested.

        Returns
        -------
        Union[Dict[Residue, List], List]
            All the vertices belonging to the requested residue. If residue is None,
            then return the vertices for all residues as a dict.

        """
        residue_vertices = {}
        atoms = list(self.struct.get_atoms())
        for vertex, vertex_atom_id in zip(self.vertices, self.vertex_atom_ids):
            # vertex_atom_id numbering starts from 1
            vertex_atom = atoms[int(vertex_atom_id) - 1]
            vertex_residue = vertex_atom.parent
            if vertex_residue not in residue_vertices:
                residue_vertices[vertex_residue] = [vertex]
            else:
                residue_vertices[vertex_residue].append(vertex)

        if residue is None:
            return residue_vertices
        elif isinstance(residue, Residue):
            return residue_vertices[residue]
        elif isinstance(residue, int):
            for res, vertices in residue_vertices.items():
                if res.id[1] == residue:
                    return vertices
            raise ValueError(f'{residue} not found!')
        else:
            raise ValueError(f'{residue} is invalid!')

    def vertex_to_residue(self, vertex_index: int = None) -> Union[List[Residue], Residue]:
        """Get the residue to which the given vertex is assigned.

        Parameters
        ----------
        vertex_index : int
            Numbering starts from 1.

        Returns
        -------
        Union[List[Residue], Residue]
            Residue to which the vertex is assigned to. If None, then return residue assignment
            for all vertices.

        """
        vertex_residues = []
        for vertex_ball in self.vertex_atom_ids:
            vertex_atom = list(self.struct.get_atoms())[vertex_ball - 1]
            vertex_residues.append(vertex_atom.parent)

        if vertex_index is not None:
            return vertex_residues[vertex_index - 1]
        else:
            return vertex_residues

    def find_residue_faces(self, residue: Union[Residue, int] = None,
                           return_coords: bool = False) -> Union[Dict[Residue, List], List]:
        """Finds all the triangular faces belonging to the requested residue.

        A triangular face is assigned to a residue if the residue has the maximum number
        of this triangle's vertices. For example, if two vertices of the triangle belongs to
        residue X, and the remaining vertex belongs to residue Y, then the triangle is
        assigned to residue X.

        Parameters
        ----------
        residue : Union[Residue, int]
            The residue for which the triangular faces are requested.
        return_coords : bool
            If True, returns Cartesian coordinates of vertices.

        Returns
        -------
        Union[Dict[Residue, List], List]
            All the triangles belonging to the requested residue. If residue is None,
            then return the triangles for all residues as a dict.

        """
        residue_faces = {}
        # get vertex to residue mapping
        vertex_residues = self.vertex_to_residue()

        # a face is represented as a triplet of vertices
        for face in self.faces:
            current_face_residues = []
            for vertex in face:
                current_face_residues.append(vertex_residues[vertex - 1])

            # in principle, a triangle face could cover three residues at maximum
            max_res = current_face_residues[0]
            max_count = 0
            for res in current_face_residues:
                res_count = current_face_residues.count(res)
                if res_count > max_count:
                    max_count = res_count
                    max_res = res

            if max_res not in residue_faces:
                if return_coords:
                    residue_faces[max_res] = [self._get_face_coords(face)]
                else:
                    residue_faces[max_res] = [face]
            else:
                if return_coords:
                    residue_faces[max_res].append(self._get_face_coords(face))
                else:
                    residue_faces[max_res].append(face)

        if residue is None:
            return residue_faces
        elif isinstance(residue, Residue):
            return residue_faces[residue]
        elif isinstance(residue, int):
            for res, res_faces in residue_faces.items():
                if res.id[1] == residue:
                    return res_faces
            raise ValueError(f'{residue} not found!')
        else:
            raise ValueError(f'{residue} is invalid!')

    def _get_face_coords(self, face: List = None) -> npt.ArrayLike:
        """Get the Cartesian coordinates of the three vertices of the face.

        Parameters
        ----------
        face : List
            A triplet of vertices.

        Returns
        -------
        npt.ArrayLike
            A NumPy 2DArray of (3, 3) shape.

        """
        face_coords = []
        for vertex in face:
            face_coords.append(self.vertices[vertex - 1])
        return np.array(face_coords)

    def compute_face_features(self, prop: str) -> npt.ArrayLike:
        """Compute the request feature for each of the triangular face.

        Parameters
        ----------
        prop : str
            The type of the biophysical property. Choose among "pos", "neg", or "hyd".

        Returns
        -------
        npt.ArrayLike
            A list of feature values for each of the triangular face.

        """
        prop_name = prop.lower()[:3]
        if prop_name == 'hyd':
            prop_values = self.hyd_potential()
        else:
            prop_values = self.apbs_values

        face_features = []
        for face in self.faces:
            face_features.append(np.mean([prop_values[v - 1] for v in face]))
        return np.array(face_features)

    def get_face_centers(self):
        face_centers = []
        for face in self.faces:
            face_coords = self._get_face_coords(face)
            face_centers.append(np.mean(face_coords, axis=0))
        return np.array(face_centers)

    def to_vertex_graph(self):
        edges = []
        for face in self.faces:
            edges.extend(list(itertools.combinations(face, 2)))
        edge_list = set(edges)
        vertex_graph = nx.from_edgelist(edge_list)
        for source, target in edge_list:
            vertex_graph[source][target]['length'] = np.linalg.norm(
                self.vertices[target - 1] - self.vertices[source - 1]
            )
        return vertex_graph

    def to_face_graph(self) -> nx.Graph:
        """Build a graph representation of the triangular mesh where the nodes
        in the graph represent triangles.

        There is an edge between two triangle faces if they share two vertices.
        Note that faces are numbered starting at 1.

        Returns
        -------
        nx.Graph

        """
        edge_list = []
        n_faces = len(self.faces)
        for i in range(1, n_faces + 1):
            for j in range(i + 1, n_faces + 1):
                i_vertices = self.faces[i - 1]
                j_vertices = self.faces[j - 1]
                if len(set(i_vertices).intersection(j_vertices)) == 2:
                    edge_list.append((i, j))
        return nx.from_edgelist(edge_list)

    def hyd_potential(self, r_cutoff: float = 5., alpha: float = 1.5) -> npt.ArrayLike:
        """Computes the hydrophobic potential at each one of the vertices of the surface.

        Parameters
        ----------
        r_cutoff : float, optional
            Twice the value at which the weight is 0.5, by default 5.
        alpha : float, optional
            Controls the steepness of the Fermi function.
            Larger alpha makes the curve steeper, by default 1.5

        Returns
        -------
        npt.ArrayLike
            The hydrophobic potential at each one of the vertices of the surface.

        """
        all_atoms = list(self.struct.get_atoms())
        atom_coords = np.array([atom.coord for atom in all_atoms])
        crippen_atom_types = []
        for atom in all_atoms:
            # C-terminal oxygen atom
            if atom.name == 'OXT':
                crippen_atom_types.append('O12')
            # N-terminal hydrogen atoms
            elif atom.name == 'H2' or atom.name == 'H3':
                crippen_atom_types.append('H3')
            else:
                crippen_atom_types.append(
                    PDB_TO_CRIPPEN[(atom.parent.resname, atom.name)]
                )
        atom_logps = [CRIPPEN_PARAMS[a_type][0] for a_type in crippen_atom_types]
        hyd_pot_evaluator = HeidenHydrophobicPotential(atom_coords, atom_logps, r_cutoff, alpha)
        return hyd_pot_evaluator.evaluate(self.vertices)

    def heiden_score(self, r_cutoff: float = 5., alpha: float = 1.5, hyd_scale: str = 'CRIPPEN'):
        """Implemented according to https://pubmed.ncbi.nlm.nih.gov/36120542/

        Parameters
        ----------
        r_cutoff
        alpha
        hyd_scale

        Returns
        -------

        """
        # compute area for each vertex
        vertex_areas = np.zeros(self.vertices.shape[0])
        for face in self.triangle_faces:
            for vertex_id in face.vertices:
                vertex_areas[vertex_id - 1] += face.area / 3

        return np.sum([
            a * p for a, p in zip(vertex_areas, self.hyd_potential(r_cutoff, alpha)) if p > 0.
        ])

    def find_cdr_vertices(self, numbering_scheme: str = 'IMGT') -> Dict[str, List]:
        """Find the vertices that belong to CDR regions.

        A vertex belongs to CDR regions if the residue it belongs to is within
        the CDR regions.

        Parameters
        ----------
        numbering_scheme : str
            Antibody sequence numbering scheme. Supported schemes include "IMGT", "CHOTHIA",
            and "KABAT".

        Returns
        -------
        Dict[str, List]
            A dict keyed by CDR loop IDs.

        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = numbering.IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = numbering.KABAT_SCHEME
        else:
            cdr_boundaries = numbering.CHOTHIA_SCHEME

        cdr_vertices = {}
        for k in cdr_boundaries.keys():
            cdr_vertices[k] = []

        for vertex_id, residue in enumerate(self.vertex_to_residue(), start=1):
            chain_id = residue.parent.id
            # an insertion residue's number defaults to the integer part, i.e. 112A -> 112
            residue_number = residue.id[1]

            for cdr_name, (cdr_start, cdr_end) in cdr_boundaries.items():
                if chain_id == cdr_name[0] and cdr_start <= residue_number <= cdr_end:
                    cdr_vertices[cdr_name].append(vertex_id)

        return cdr_vertices

    def find_cdr_faces(self, numbering_scheme: str = 'IMGT') -> Dict[str, List]:
        """Find the triangular faces that belong to CDR regions.

        A triangular face belongs to CDR regions if the residue it belongs to is within
        the CDR regions.

        Parameters
        ----------
        numbering_scheme : str
            Antibody sequence numbering scheme. Support schemes include "IMGT", "CHOTHIA",
            and "KABAT".

        Returns
        -------
        Dict[str, List]
            A dict keyed by CDR loop IDs. Each surface patch consists of a list of
            (triangle face, area) pairs.

        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = numbering.IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = numbering.KABAT_SCHEME
        else:
            cdr_boundaries = numbering.CHOTHIA_SCHEME

        cdr_faces = {}
        for k in cdr_boundaries.keys():
            cdr_faces[k] = []

        for residue, face in self.find_residue_faces().items():
            chain_id = residue.parent.id
            # an insertion residue's number defaults to the integer part, i.e. 112A -> 112
            residue_number = residue.id[1]

            for cdr_name, (cdr_start, cdr_end) in cdr_boundaries.items():
                if chain_id == cdr_name[0] and cdr_start <= residue_number <= cdr_end:
                    cdr_faces[cdr_name].extend(face)

        return cdr_faces

    @property
    def total_area(self):
        if self._total_area is None:
            self._total_area = sum([face.area for face in self.triangle_faces])
        return self._total_area

    def total_hyd_area(self):
        total = 0.
        vertex_hyd_potentials = self.hyd_potential()
        for face in self.triangle_faces:
            face_hyd_potential = np.mean([vertex_hyd_potentials[i - 1] for i in face.vertices])
            if face_hyd_potential > 0.:
                total += face.area
        return total

    def find_patches_dbscan(self, prop: str, prop_cutoff: float, eps: float = 2.,
                            min_samples: int = 10, area_cutoff: float = 30.) -> dict:
        """Finds surface patches based on the DBSCAN clustering algorithm.

        Parameters
        ----------
        prop : str
            Choose among {positive, negative}
        eps : float
            DBSCAN parameter
        min_samples : int
            DBSCAN parameter
        prop_cutoff : float
            Electrostatic potential cutoff value for positively charged or negative charged patches
        area_cutoff : float
            Area cutoff value for patches.

        Returns
        -------
        dict
            A dict containing the identified surface patches. Keyed by patch ID, starting from 1.
            Each surface patch consists of a list of (triangle face, area) pairs.

        """
        triangle_faces = []
        for i, face_vertices in enumerate(self.faces, start=1):
            face_coords = self._get_face_coords(face_vertices)
            triangle_faces.append(
                TriangleFace(coords=face_coords, vertices=face_vertices, face_id=i)
            )

        # face_centers = np.array([f.center for f in triangle_faces])
        # face_areas = np.array([f.area for f in triangle_faces])
        face_features = self.compute_face_features(prop)

        prop_name = prop.lower()[:3]
        thresholded_faces = []
        if prop_name == 'pos' or prop_name == 'hyd':
            for face, feature in zip(triangle_faces, face_features):
                if feature >= prop_cutoff:
                    thresholded_faces.append(face)
            if prop_name == 'pos':
                patch_prefix = 'pos_patch'
            else:
                patch_prefix = 'hyd_patch'
        else:
            for face, feature in zip(triangle_faces, face_features):
                if feature <= prop_cutoff:
                    thresholded_faces.append(face)
            patch_prefix = 'neg_patch'

        if len(thresholded_faces) == 0:
            raise ValueError(f'No triangle faces met prop_cutoff = {prop_cutoff}!')

        # clustering
        thresholded_centers = np.array([f.center for f in thresholded_faces])
        face_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        face_dbscan.fit(thresholded_centers)

        # extract the triangles in each patch
        patches = {}
        for face, label in zip(thresholded_faces, face_dbscan.labels_):
            if label == -1:  # skip noise
                continue
            patch_label = f'{patch_prefix}_{label}'
            if patch_label not in patches:
                patches[patch_label] = [(face.vertices, face.area)]
            else:
                patches[patch_label].append((face.vertices, face.area))

        if area_cutoff is not None:
            thresholded_patches = {}
            for patch_label, triangle_faces in patches.items():
                patch_area = sum(face_area for _, face_area in triangle_faces)
                if patch_area < area_cutoff:
                    continue
                else:
                    thresholded_patches[patch_label] = triangle_faces
            return thresholded_patches
        else:
            return patches

    def find_patches_graph(self, prop: str, prop_cutoff: float, area_cutoff: float = 40.) -> dict:
        """Finds surface patches using graph algorithms.

        A set of surface triangles form a patch if they are a
        connected component.

        Parameters
        ----------
        prop : str
            Type of biophysical property. Choose among "pos", "neg", or "hyd".
        prop_cutoff : float
            Cutoff value of the property for a triangle to be included in
            the graph.
        area_cutoff : float
            Cutoff value of the area for a connected set of triangles to
            be considered as a patch.

        Returns
        -------
        dict
            A dict containing the identified surface patches. Keyed by patch ID, starting from 1.
            Each surface patch consists of a list of (triangle face, area) pairs.

        """
        # create a graph where each node represents a surface triangle
        face_graph = self.to_face_graph()
        face_features = self.compute_face_features(prop=prop)

        # exclude triangles (nodes) that do not meet cutoff
        thresholded_nodes = []
        for face_node in face_graph.nodes():
            feature = face_features[face_node - 1]
            if prop.lower()[:3] == 'pos' or prop.lower()[:3] == 'hyd':
                if feature > prop_cutoff:
                    thresholded_nodes.append(face_node)
            else:
                if feature < prop_cutoff:
                    thresholded_nodes.append(face_node)

        # create a new graph from the thresholded nodes
        edge_list = []
        for u, v in face_graph.edges():
            # an edge is created between nodes u, v if they are both in thresholded nodes.
            if u in thresholded_nodes and v in thresholded_nodes:
                edge_list.append((u, v))
        thresholded_graph = nx.from_edgelist(edge_list)

        # extract the patches that meet the area cutoff
        patches = {}
        patch_id = 0
        for component in sorted(
            nx.connected_components(thresholded_graph), key=len, reverse=True
        ):
            patch_area = 0.
            patch_faces = []
            for face_id in component:
                face_vertices = self.triangle_faces[face_id - 1].vertices
                face_area = self.triangle_faces[face_id - 1].area
                patch_area += face_area
                patch_faces.append((face_vertices, face_area))
            if patch_area >= area_cutoff:
                patch_id += 1
                patch_label = f'patch_{patch_id}'
                patches[patch_label] = patch_faces

        return patches
