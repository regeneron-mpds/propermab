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
#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser

from Bio.PDB import PDBParser


def parse_cmd_args():
    parser = ArgumentParser(
        prog='pdb_to_xyzr.py',
        description='''
        Run this script to generate a .xyzr file, which is needed as an input by NanoShaper.
        '''
    )
    parser.add_argument(
        '--pdb-file', dest='pdb_file', type=str, required=True,
        help='''The PDB file for which a xyzr file is to be generated.'''
    )
    parser.add_argument(
        '--ff-file', dest='ff_file', type=str, required=True,
        help='''Force field file containing atomic radii information.'''
    )
    parser.add_argument(
        '--output-prefix', dest='output_prefix', type=str, required=False,
        help='''Prefix for the output files.'''
    )
    return parser.parse_args()


def parse_pdb_file(pdb_file):
    """Parses the given PDB file and returns all protein atoms.

    Parameters
    ----------
    pdb_file : str
        PDB file for the protein.

    Returns
    -------
    Atoms
        All atoms that is part of the protein.

    """
    pdb_parser = PDBParser(PERMISSIVE=1)
    structure = pdb_parser.get_structure(id=None, file=pdb_file)
    protein_atoms = []
    for residue in structure.get_residues():
        # skip water and non-protein/nucleic residues
        if residue.id[0] != ' ':
            continue

        protein_atoms.extend(list(residue.get_atoms()))
    return protein_atoms


def parse_ff_file(ff_file):
    """Parses the given force field file and returns atomic radius information.

    Parameters
    ----------
    ff_file : str
        Force field file.

    Returns
    -------
    dict
        Python dict of atomic radius information keyed by atom names.

    """
    default_radii = {}
    specific_radii = {}
    with open(ff_file, 'rt') as ff_file_handle:
        line_number = 0
        for line in ff_file_handle:
            line_number += 1
            # skip empty lines
            if not line.strip():
                continue
            # skip comment lines
            if line.strip().startswith('!'):
                continue
            # DelPhi file identifying line
            if line.startswith('atom__res_radius_'):
                print('Found DelPhi header in radii .siz file')
                continue
            # parse atom lines
            fields = line.split()
            if len(fields) == 2:
                default_radii[fields[0]] = float(fields[1])
            elif len(fields) >= 3:
                specific_radii[f'{fields[0]}_{fields[1]}'] = float(fields[2])
            else:
                print(
                    f'Error! Unrecognized record at line {line_number}.'
                )
                sys.exit(1)

    return default_radii, specific_radii


def main():
    cmd_args = parse_cmd_args()

    # get the appropriate prefix for the output files
    if cmd_args.output_prefix is None:
        pdb_basename = os.path.basename(cmd_args.output_prefix)
        if not pdb_basename.lower().endswith('pdb'):
            output_prefix = pdb_basename
        else:
            output_prefix = '.'.join(pdb_basename.split('.')[:-1])
    else:
        output_prefix = cmd_args.output_prefix

    # get all protein atoms and atom radii
    protein_atoms = parse_pdb_file(cmd_args.pdb_file)
    default_radii, specific_radii = parse_ff_file(cmd_args.ff_file)

    # write coordinates and radii info of each atom to file
    with open(output_prefix + '.xyzr', 'wt') as xyzr_handle:
        for atom in protein_atoms:
            atom_name = atom.name
            res_name = atom.parent.resname
            radius_key = f'{atom_name}_{res_name}'

            if radius_key not in specific_radii:
                if atom.name not in default_radii:
                    print(f'Radius for {res_name} {atom_name} is missing, set to 1.0')
                    radius = 1.0
                else:
                    radius = default_radii[atom_name]
            else:
                radius = specific_radii[radius_key]

            coord = atom.coord
            xyzr_handle.write(f'{coord[0]:.3f} {coord[1]:.3f} {coord[2]:.3f} {radius:.3f}\n')

    print(f'Done! {output_prefix}.xyzr was created for PDB file {cmd_args.pdb_file}.')


if __name__ == '__main__':
    main()