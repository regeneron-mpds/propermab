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

import numpy as np
from Bio import PDB

from propermab import defaults
from propermab.utils import nanoshaper


test_dir = os.path.dirname(__file__)
propermab_dir = os.path.abspath(os.path.join(test_dir, '../'))
pdb_to_xyzr_script = os.path.join(propermab_dir, 'scripts/pdb_to_xyzr.py')
pdb_file = os.path.join( os.path.dirname(__file__), 'pembrolizumab_ib.pdb')
xyzr_file = os.path.join(os.path.dirname(__file__), 'pembrolizumab_ib.xyzr')


defaults.system_config.update_from_json(
    os.path.join(propermab_dir, 'default_config.json')
)
atom_radii_file = defaults.system_config['atom_radii_file']
assert os.path.exists(atom_radii_file)


def test_run_pdb_to_xyzr():
    """If run_pdb_to_xyzr() succeeded, pembrolizumab_ib.xyzr should be available 
    in the tests directory.
    """
    nanoshaper.run_pdb_to_xyzr(
        pdb_file, pdb_to_xyzr_script, atom_radii_file
    )
    assert os.path.exists(xyzr_file)


structure = PDB.PDBParser(PERMISSIVE=True).get_structure(
    id='test_pdb', file=pdb_file
)
all_atoms = list(structure.get_atoms())
with open(xyzr_file, 'rt') as xyzr_handle:
    xyzr_lines = [line.strip() for line in xyzr_handle]


def test_check_number_of_lines():
    # there must be one line for each atom
    assert len(all_atoms) == len(xyzr_lines)


def test_check_atom_coords():
    # coordinates in xyzr file must match coordinates in PDB file
    pdb_atom_coords = np.array([
        atom.coord for atom in all_atoms
    ], dtype=np.float32)
    xyzr_atom_coords = np.array([
        line.split()[:3] for line in xyzr_lines
    ], dtype=np.float32)
    assert np.array_equal(pdb_atom_coords, xyzr_atom_coords)


def test_check_atom_radii():
    # xyzr lines must have the correct radii
    atom_a_radius = float(xyzr_lines[0].split()[-1])
    atom_a_expected_radius = 1.824
    assert atom_a_radius == atom_a_expected_radius

    atom_b_radius = float(xyzr_lines[200].split()[-1])
    atom_b_expected_radius = 1.908
    assert atom_b_radius == atom_b_expected_radius
    
    atom_c_radius = float(xyzr_lines[1998].split()[-1])
    atom_c_expected_radius = 1.721
    assert atom_c_radius == atom_c_expected_radius