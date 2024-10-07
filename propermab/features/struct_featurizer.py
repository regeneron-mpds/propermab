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
import numpy as np

import openmm.app as openmm_app

from .hydrophobicity import KD_SCALE
from .hydrophobicity import EISENBERG_SCALE
from ..structure import md
from ..structure import sasa
from ..structure import io as struct_io
from ..sequence import numbering
from .spatial_stats import AverageNearestNeighbor
from .spatial_stats import RipleyK


class StructFeaturizer:
    def __init__(self, pdb_file: str):
        """Encapsulation of functionalities for computing structure-based features.

        Parameters
        ----------
        pdb_file : str
            A PDB file storing the structure.
        """
        self.pdb_file = pdb_file
        self.openmm_pdb = openmm_app.PDBFile(pdb_file)
        self._struct = None
        self._residue_sasa = None
        self._atom_sasa = None
        self._atom_charges = None
        self._atoms = None
        self._residues = None
        self._system_and_topology = None

    @property
    def struct(self):
        if self._struct is None:
            self._struct = struct_io.load_structure(self.pdb_file)
        return self._struct

    @property
    def atoms(self):
        if self._atoms is None:
            self._atoms = list(self.openmm_pdb.topology.atoms())
        return self._atoms

    @property
    def residues(self):
        if self._residues is None:
            self._residues = list(self.openmm_pdb.topology.residues())
        return self._residues

    @property
    def residue_sasa(self):
        """Residue solvent accessible surface area.

        Returns
        -------
        Nested dict
            First level key is chain ID, second level key is residue number.
        """
        if self._residue_sasa is None:
            sasa_results = sasa.apply_sasa(self.pdb_file)
            self._residue_sasa = sasa_results.residueAreas()
        return self._residue_sasa

    @property
    def atom_sasa(self):
        """Atom solvent accessible surface area.

        Returns
        -------

        """
        if self._atom_sasa is None:
            sasa_results = sasa.apply_sasa(self.pdb_file, inc_hydrogen=True)
            atom_sasas = np.array([
                sasa_results.atomArea(i) for i in range(sasa_results.nAtoms())
            ])
            atom_chain_ids = [a.full_id[2] for a in self.struct.get_atoms()]
            atom_sasa_dict = {}
            for chain_id, s in zip(atom_chain_ids, atom_sasas):
                if chain_id not in atom_sasa_dict:
                    atom_sasa_dict[chain_id] = [s]
                else:
                    atom_sasa_dict[chain_id].append(s)
            self._atom_sasa = atom_sasa_dict
        return self._atom_sasa

    @property
    def atom_charges(self):
        if self._atom_charges is None:
            system, topology = self.system_and_topology
            atom_charges = {}
            chain_ids = [chain.id for chain in self.struct.get_chains()]
            for chain_id in chain_ids:
                atom_charges[chain_id] = np.array([
                    c._value for c in md.get_partial_charges_system(system, topology, chains=chain_id)
                ])
            self._atom_charges = atom_charges
        return self._atom_charges

    @property
    def system_and_topology(self):
        if self._system_and_topology is None:
            self._system_and_topology = md.simple_system(self.pdb_file)
        return self._system_and_topology

    def net_charge(self) -> float:
        total_net_charge = 0
        for _, chain_atom_charges in self.atom_charges.items():
            total_net_charge += np.sum(chain_atom_charges)
        return total_net_charge

    def exposed_net_charge(self) -> float:
        """The net charge of atoms at the surface.

        Returns
        -------
        float
            Total charge exposed at the solvent exposed surface.

        """
        total_exposed_charge = 0.
        chain_ids = self.atom_charges.keys()
        for chain_id in chain_ids:
            for c, s in zip(self.atom_charges[chain_id], self.atom_sasa[chain_id]):
                if s > 0:
                    total_exposed_charge += c
        return total_exposed_charge

    def net_charge_cdr(
        self, numbering_scheme: str = 'IMGT', chain_ids: str = None, exposed : bool = False
    ) -> float:
        """Calculates the net charge of CDR regions.

        Parameters
        ----------
        numbering_scheme : str
            The numbering scheme used to define CDR regions. 
        chain_ids : str
            Letter IDs of heavy and light chains.
        exposed : bool
            Whether to do the calculation based on atoms that are solvent exposed.

        Returns
        -------
        float
            Net charge of the CDR regions.

        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = numbering.IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = numbering.KABAT_SCHEME
        else:
            cdr_boundaries = numbering.CHOTHIA_SCHEME

        # get chain IDs
        if chain_ids is None:
            chain_ids = self.atom_charges.keys()

        cdr_net_charge = 0
        for chain_id in chain_ids:
            if exposed:
                # atom sasa values are needed
                for atom, c, s in zip(
                    self.struct[0][chain_id].get_atoms(),
                    self.atom_charges[chain_id],
                    self.atom_sasa[chain_id]
                ):
                    if s == 0:
                        continue
                    else:
                        res_number = atom.get_parent().id[1]
                        for _, (cdr_start, cdr_end) in cdr_boundaries.items():
                            if cdr_start <= res_number <= cdr_end:
                                cdr_net_charge += c
            else:
                # no need for atom sasa values
                for atom, c in zip(
                    self.struct[0][chain_id].get_atoms(),
                    self.atom_charges[chain_id]
                ):
                    res_number = atom.get_parent().id[1]
                    for _, (cdr_start, cdr_end) in cdr_boundaries.items():
                        if cdr_start <= res_number <= cdr_end:
                            cdr_net_charge += c
        return cdr_net_charge

    def dipole_moment(self) -> float:
        """Compute the dipole moment of the given structure in Debye unit.

        The dipole moment is computed as the vector sum of a cloud of point charges,
        i.e. mu = sum(q_i * r_i) where q_i and r_i are the partial charge and the
        position vector of atom i respectively, and i runs over all atoms of the protein.

        References
        https://academic.oup.com/nar/article/35/suppl_2/W512/2922221
        https://www.cell.com/fulltext/S0006-3495(95)80001-9

        Returns
        -------
        float
            The magnitude of the dipole moment.
        """
        # atom_coords = self.openmm_pdb.getPositions(asNumpy=True)
        atom_coords = []
        for atom in self.struct.get_atoms():
            atom_coords.append(atom.coord)
        atom_coords = np.array(atom_coords)
        center_coords = np.mean(atom_coords, axis=0)
        centered_atom_coords = atom_coords - center_coords

        all_atom_charges = np.concatenate(list(self.atom_charges.values()))
        atom_charges = all_atom_charges.reshape((-1, 1))
        dipole_vector = np.sum(centered_atom_coords * atom_charges, axis=0)
        return np.linalg.norm(4.803 * dipole_vector)

    def hyd_moment(self, hyd_scale: str = 'kd') -> float:
        """Computes the first-order hydrophobic moment of the given structure.

        The hydrophobic moment is computed as the vector sum of a cloud of hydrophobicity
        points, i.e. mu = sum(h_j * r_j) where h_j and r_j are the hydrophobicity and the
        position vector of residue j respectively, and j runs over all residues of the protein.

        References
        https://www.pnas.org/doi/10.1073/pnas.081086198

        Returns
        -------
        float
            The magnitude of the hydrophobic moment.
        """
        # first compute center of geometry for each residue
        residue_cog_all = []
        residue_hyd_all = []
        for residue in self.struct.get_residues():
            residue_atom_coords = np.array([
                a.coord for a in residue.get_atoms()
            ])
            residue_cog_all.append(np.mean(residue_atom_coords, axis=0))
            if hyd_scale == 'kd':
                residue_hyd_all.append(
                    KD_SCALE[residue.resname]
                )
            else:
                residue_hyd_all.append(
                    EISENBERG_SCALE[residue.resname]
                )

        residue_cog_all = np.array(residue_cog_all)
        residue_hyd_all = np.array(residue_hyd_all).reshape((-1, 1))
        hyd_vector = np.sum(residue_cog_all * residue_hyd_all, axis=0)
        return np.linalg.norm(hyd_vector)

    def fv_chml(self) -> float:
        """Formal charge of the VH minus the formal charge of the VL domains.
        """
        vh_charge = np.sum(self.atom_charges['H'])
        vl_charge = np.sum(self.atom_charges['L'])
        return vh_charge - vl_charge

    def exposed_fv_chml(self) -> float:
        """Formal charge of the VH minus the formal charge of the VL domains.
        """
        exposed_vh_charge = 0
        for h_c, h_s in zip(self.atom_charges['H'], self.atom_sasa['H']):
            if h_s > 0:
                exposed_vh_charge += h_c
        exposed_vl_charge = 0
        for l_c, l_s in zip(self.atom_charges['L'], self.atom_sasa['L']):
            if l_s > 0:
                exposed_vl_charge += l_c
        return exposed_vh_charge - exposed_vl_charge

    def hyd_asa(self) -> float:
        """Total hydrophobic accessible surface area.
        """
        total_hyd_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                total_hyd_asa += residue_area.apolar
        return total_hyd_asa

    def hph_asa(self) -> float:
        """Total hydrophilic accessible surface area.
        """
        total_hph_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                total_hph_asa += residue_area.polar
        return total_hph_asa

    def aromatic_asa(self) -> float:
        """Count the total number of exposed aromatic residues.

        Parameters
        ----------
        rsa_cutoff : float
            Cutoff for the relative solvent accessible surface area above which a residue
            will considered exposed.

        Returns
        -------
        int
            The total number of aromatic residues considered exposed.
        """
        total_asa_aromatic = 0
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                res_name = residue_area.residueType
                total_asa = residue_area.total
                if res_name in ['PHE', 'TYR', 'TRP']:
                    total_asa_aromatic += total_asa
        return total_asa_aromatic

    def cdr_length(self, cdr: str = 'H3', numbering_scheme: str = 'IMGT') -> int:
        """Count the number of  residues in CDR-H3.

        Parameters
        ----------
        cdr: str
            Name of the CDR region. Choose among [H1, H2, H3, L1, L2, L3].
        numbering_scheme : str
            Numbering scheme for the Fv region.

        Returns
        -------
        int
            Length of CDR-H3, i.e. the number of residues in CDR-H3.
        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = numbering.IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = numbering.KABAT_SCHEME
        else:
            cdr_boundaries = numbering.CHOTHIA_SCHEME

        # look at the correct chain
        chain_id = cdr.upper()[0]
        correct_chain = None
        for chain in self.struct.get_chains():
            if chain.id == chain_id:
                correct_chain = chain

        # count the number of residues in the CDR region
        cdr_start, cdr_end = cdr_boundaries[cdr.upper()]
        length = 0
        for residue in correct_chain.get_residues():
            res_number = residue.id[1]
            if cdr_start <= res_number <= cdr_end:
                length += 1
        return length

    def aromatic_cdr(self, numbering_scheme: str = 'IMGT') -> int:
        """Count the number of aromatic residues in CDR regions.

        Parameters
        ----------
        numbering_scheme : str
            Numbering scheme for the Fv region.

        Returns
        -------
        int
            Number of aromatic residues in CDR regions.
        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = numbering.IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = numbering.KABAT_SCHEME
        else:
            cdr_boundaries = numbering.CHOTHIA_SCHEME

        cdr_aromatic_content = 0
        for residue in self.struct.get_residues():
            res_number = residue.id[1]
            res_name = residue.resname

            for _, (cdr_start, cdr_end) in cdr_boundaries.items():
                if cdr_start <= res_number <= cdr_end and res_name in ['PHE', 'TYR', 'TRP']:
                    cdr_aromatic_content += 1

        return cdr_aromatic_content

    def exposed_aromatic(self, rsa_cutoff: float = 0.05) -> int:
        """Count the total number of exposed aromatic residues.

        Parameters
        ----------
        rsa_cutoff : float
            Cutoff for the relative solvent accessible surface area above which a residue
            will be considered exposed.

        Returns
        -------
        int
            The total number of aromatic residues considered exposed.
        """
        total_exposed_aromatic = 0
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff and res_name in ['PHE', 'TYR', 'TRP']:
                    total_exposed_aromatic += 1
        return total_exposed_aromatic

    def ann_index(
        self, prop: str = 'pos', rsa_cutoff: float = 0.05, n: int = 1000
    ) -> float:
        """Calculates the Average Nearest Neighbor (ANN) statistic for the given property.

        Parameters
        ----------
        prop : str
            Property (amino acid) type for which the ANN statistic is calculated.
            Distance cutoff at/shorter than which two residues are considered neighbors.
        rsa_cutoff : float
            Relative solvent accessibility cutoff above which a residue is considered 
            solvent exposed, i.e. at the surface.
        n : int
            Number of permutation runs for simulating the null distribution.

        Returns
        -------
        float
            The ANN statistic for the given property.

        """
        if prop.lower()[:3] == 'neg':
            prop_res_names = ['ASP', 'GLU']
        elif prop.lower()[:3] == 'aro':
            prop_res_names = ['PHE', 'TYR', 'TRP']
        else:
            prop_res_names = ['ARG', 'LYS', 'HIS']

        prop_ca_coords = []
        all_ca_coords = []
        for chain_id, chain_area in self.residue_sasa.items():
            chain_ca_coords = [
                r['CA'].coord for r in self.struct[0][chain_id].get_residues()
            ]
            for ca_coord, residue_area in zip(chain_ca_coords, chain_area.values()):
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff:
                    all_ca_coords.append(ca_coord)
                    if res_name in prop_res_names:
                        prop_ca_coords.append(ca_coord)
        ann = AverageNearestNeighbor(
            feature_coords=prop_ca_coords, allowed_coords=all_ca_coords, n=n
        )
        return ann.ann_index

    def ripley_k(
        self, prop: str = 'pos', distance: float = 8.0,
        rsa_cutoff: float = 0.05, n: int = 1000
    ) -> float:
        """Calculates a variant of the Ripley's K statistic for the given property.

        Parameters
        ----------
        prop : str
            Property (amino acid) type for which the Ripley's K statistic is calculated.
        distance : float
            Distance cutoff at/shorter than which two residues are considered neighbors.
        rsa_cutoff : float
            Relative solvent accessibility cutoff above which a residue is considered 
            solvent exposed, i.e. at the surface.
        n : int
            Number of permutation runs for simulating the null distribution.

        Returns
        -------
        float
            The Ripley's K statistic for the given property.

        """
        if prop.lower()[:3] == 'neg':
            prop_res_names = ['ASP', 'GLU']
        elif prop.lower()[:3] == 'aro':
            prop_res_names = ['PHE', 'TYR', 'TRP']
        else:
            prop_res_names = ['ARG', 'LYS', 'HIS']

        prop_ca_coords = []
        all_ca_coords = []
        for chain_id, chain_area in self.residue_sasa.items():
            chain_ca_coords = [
                r['CA'].coord for r in self.struct[0][chain_id].get_residues()
            ]
            for ca_coord, residue_area in zip(chain_ca_coords, chain_area.values()):
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff:
                    all_ca_coords.append(ca_coord)
                    if res_name in prop_res_names:
                        prop_ca_coords.append(ca_coord)
        this_ripley_k = RipleyK(
            obs_coords=prop_ca_coords, allowed_coords=all_ca_coords,
            distance=distance, n=n
        )
        return this_ripley_k.ripley_k
