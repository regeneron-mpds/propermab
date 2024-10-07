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
import os
from statistics import geometric_mean
import tempfile
from collections import deque

import Bio.PDB.Entity
import numpy as np
from Bio.PDB import PDBParser, PDBIO

from propermab import defaults
from propermab import utils
from propermab.io import geometry_io
from propermab.utils.nanoshaper import NanoShaper
from propermab.utils.apbs import APBS
from propermab.features import grid


class ToVertices:
    def __init__(self, nanoshaper=None):
        """Callable object that generates a surface vertex representation of
        a protein from its PDB file.

        Parameters
        ----------
        nanoshaper : NanoShaper
            A callable NanoShaper object.
        """
        if nanoshaper is None:
            # use default
            self.nanoshaper = NanoShaper()
        else:
            self.nanoshaper = nanoshaper

    def __call__(self, pdb_file):
        """This method makes a ToVertices object callable.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file of the protein for which a surface vertex representation
            will be generated.

        Returns
        -------
        np.ndarray
            Cartesian coordinates of the surface vertices stored in a NumPy 2DArray.
        """
        return np.array(self.nanoshaper(pdb_file))


class ToVoxelGrid:
    def __init__(
        self, nanoshaper=None, apbs=None, width=70., height=70., depth=70.,
        voxel_size=1., rotate=False, tmp_path='/tmp/'
    ):
        """Callable object that generates a featurized voxel grid representation of
        a protein from its PDB file.

        Parameters
        ----------
        nanoshaper : NanoShaper
            A callable NanoShaper object.
        apbs : APBS
            A callable APBS object.
        width : float
            Width of the voxel grid, in angstroms.
        height : float
            Height of the voxel grid, in angstroms.
        depth : float
            Depth of the voxel grid, in angstroms.
        voxel_size : float
            Size of individual voxels, in angstroms.
        rotate : bool
            Whether to augment the voxel grid by the 24 cube rotation symmetries.
        tmp_path
        """
        # set objects that process PDB files
        if nanoshaper is None:
            self.nanoshaper = NanoShaper()
        else:
            self.nanoshaper = nanoshaper
        if apbs is None:
            self.apbs = APBS()
        else:
            self.apbs = apbs

        # set voxel grid dimensions and voxel size
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size

        self.rotate = rotate

        self.tmp_path = tmp_path

    def __call__(self, pdb_file):
        """This makes a ToVoxelGrid object callable.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file of the protein for which a voxel grid representation
            will be generated.

        Returns
        -------
        np.ndarray
            Featurized voxel grid representation of the protein stored in a numerical
            NumPy 3Darray object.

        """
        pdb_basename = os.path.basename(pdb_file)
        output_prefix = '.'.join(pdb_basename.split('.')[:-1])

        # generate surface vertices
        vertex_file, face_file = self.nanoshaper(pdb_file, tmp_path=self.tmp_path)
        surface_vertices, _, _, _ = geometry_io.read_nanoshaper(
            vertex_file, face_file
        )

        # create surface voxels
        surface_voxel_grid = grid.VoxelGrid.create_from_point_cloud(
            surface_vertices, self.width, self.height, self.depth, self.voxel_size
        )

        tmp_dir = tempfile.mkdtemp(dir=self.tmp_path)
        _, tmp_voxel_file = tempfile.mkstemp(suffix=".csv", dir=tmp_dir)
        surface_voxel_grid.write_voxel_grid_coords(tmp_voxel_file)

        # compute electrostatic potentials
        apbs_output_file = self.apbs(pdb_file, self.tmp_path)

        # call multivalue to extract electrostatic potentials at surface voxels
        feature_values = utils.apbs.run_multivalue(
            defaults.system_config['multivalue_binary_path'],
            tmp_voxel_file, apbs_output_file, output_prefix
        )

        # create featurized voxel grid
        surface_voxel_grid.featurize(feature_values)

        if self.rotate:
            # get the 24 rotations of the given 3D tensor
            return np.array(
                list(grid.rotate_feature_tensor(surface_voxel_grid.feature_tensor))
            )
        else:
            return surface_voxel_grid.feature_tensor


class RotateStructure:
    def __init__(self, euler_angles=None):
        """

        Parameters
        ----------
        euler_angles
        """
        if euler_angles is None:
            self.euler_angles = np.random.uniform(low=0, high=2 * np.pi, size=3)
        else:
            self.euler_angles = euler_angles

    def __call__(self, pdb_file: str) -> str:
        """Rotate the structure specified in the given PDB file.

        Parameters
        ----------
        pdb_file : str
            PDB file of the structure to be rotated.

        Returns
        -------
        str
            Absolute path to the PDB file for the rotated structure.
        """
        psi, theta, phi = self.euler_angles

        rotation_about_x = np.array([
            [1., 0., 0.],
            [0., np.cos(psi), -np.sin(psi)],
            [0., np.sin(psi), np.cos(psi)]
        ])
        rotation_about_y = np.array([
            [np.cos(theta), 0., np.sin(theta)],
            [0., 1., 0.],
            [-np.sin(theta), 0., np.cos(theta)]
        ])
        rotation_about_z = np.array([
            [np.cos(phi), -np.sin(phi), 0.],
            [np.sin(phi), np.cos(phi), 0.],
            [0., 0., 1.]
        ])

        pdb_parser = PDBParser()
        structure = pdb_parser.get_structure(id=None, file=pdb_file)

        # get center of mass of structure
        structure_cog = structure.center_of_mass(geometric=True)

        # move structure such that its center of geometry is at the origin
        structure.transform(np.eye(3), -structure_cog)

        # now rotate structure and move back to its original position
        structure.transform(
            np.dot(np.dot(rotation_about_x, rotation_about_y), rotation_about_z),
            structure_cog
        )

        pdb_basename = os.path.basename(pdb_file)
        output_prefix = '.'.join(pdb_basename.split('.')[:-1])

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(f'{output_prefix}_rotated.pdb')

        return os.path.abspath(f'{output_prefix}_rotated.pdb')
