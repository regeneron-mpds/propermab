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
import pytest

import numpy as np
from Bio import PDB

from propermab.io import geometry_io
from propermab.features import surface


tests_dir = os.path.dirname(__file__)
pdb_file = os.path.join(tests_dir, 'pembrolizumab_ib.pdb')
test_vertex_file = os.path.join(tests_dir, 'pembrolizumab_ib.vert')
test_face_file = os.path.join(tests_dir, 'pembrolizumab_ib.face')
first_residue_vertex_coords = np.loadtxt(
    os.path.join(tests_dir, 'first_residue_vertex_coords.txt')
)
last_residue_vertex_coords = np.loadtxt(
    os.path.join(tests_dir, 'last_residue_vertex_coords.txt')
)
first_residue_faces = np.loadtxt(
    os.path.join(tests_dir, 'first_residue_faces.txt')
)
last_residue_faces = np.loadtxt(
    os.path.join(tests_dir, 'last_residue_faces.txt')
)

vertices, faces, atom_numbers, _ = geometry_io.read_nanoshaper(
    test_vertex_file, test_face_file
)

pdb_parser = PDB.PDBParser()
struct_model = pdb_parser.get_structure(id='pembrolizumab_ib', file=pdb_file)[0]
residues = list(struct_model.get_residues())

test_surface = surface.Surface(
    vertices, faces, atom_numbers, struct_model, apbs_values=None
)


def test_triangle_faces():
    """_summary_
    """
    assert len(test_surface.triangle_faces) == 25896

    first_triangle_face = test_surface.triangle_faces[0]
    
    # first triangle must be numbered 1
    assert first_triangle_face.face_id == 1

    # check its vertices
    assert np.array_equal(first_triangle_face.vertices, np.array([403, 8515, 8689]))
    
    # check coordinates of its vertices
    assert np.array_equal(
        first_triangle_face.coords,
        np.array([
            [3.565, 4.357, -20.517],
            [2.806, 4.807, -20.517],
            [3.565, 4.807, -20.768]
        ])
    )


@pytest.mark.parametrize(
    argnames='residue, expected_vertices',
    argvalues=[
        (residues[0], first_residue_vertex_coords),
        (residues[-1], last_residue_vertex_coords)
    ]
)
def test_find_residue_vertices(residue, expected_vertices):
    residue_vertices = test_surface.find_residue_vertices(residue)
    assert np.array_equal(residue_vertices, expected_vertices)


@pytest.mark.parametrize(
    argnames='vertex_idx, expected_residue',
    argvalues=[
        (1, residues[-1]), # vertex 1 gets assigned to atom 3520, which belongs to the last residue
        (8199, residues[0]) # vertex 8199 gets assigned to atom 11, which belongs to residue 1
    ]
)
def test_vertex_to_residue(vertex_idx, expected_residue):
    residue = test_surface.vertex_to_residue(vertex_idx)
    assert residue == expected_residue


@pytest.mark.parametrize(
    argnames='residue, expected_residue_faces',
    argvalues=[
        (residues[0], first_residue_faces), 
        (residues[-1], last_residue_faces)
    ]
)
def test_find_residue_faces(residue, expected_residue_faces):
    residue_faces = np.array(test_surface.find_residue_faces(residue))
    assert np.array_equal(residue_faces, expected_residue_faces)
