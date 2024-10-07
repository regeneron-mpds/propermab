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
# /usr/bin/env python3

import os
import sys
import time
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

import pymol
from ImmuneBuilder import ABodyBuilder2

script_dir = os.path.dirname(__file__)
propermab_dir = os.path.abspath(os.path.join(script_dir, '../'))
sys.path.append(propermab_dir)

from propermab.utils import transforms
from propermab import defaults


defaults.system_config.update_from_json(os.path.join(propermab_dir, 'my_config.json'))


class VhVlPair:
    def __init__(self, vh_id: str = None, vl_id: str = None, vh_seq: str = None,
                 vl_seq: str = None):
        self.vh_id = vh_id
        self.vl_id = vl_id
        self.vh_seq = vh_seq
        self.vl_seq = vl_seq

    def to_dict(self):
        """Converts the VhVlPair object to a dictionary recognizable by ImmuneBuilder.
        """
        return {
            'H': self.vh_seq,
            'L': self.vl_seq
        }

    def make_struct_id(self):
        return self.vh_id + '.' + self.vl_id


def predict_structure(vh_vl_pair, predictor):
    """Calls ImmuneBuilder to generate a structure for the given sequence pair.

    Parameters
    ----------
    vh_vl_pair : VhVlPair
        A VhVlPair object storing a pair of sequences and their IDs.

    predictor
        Any structure predictor object that has a ```predict()``` member function taking
        ```vh_vl_pair``` as an argument.

    Returns
    -------
    str
        Absolute path to where the PDB file is stored.

    """
    pdb_file = os.path.abspath(vh_vl_pair.make_struct_id() + '.pdb')
    if os.path.exists(pdb_file):
        print(f'{pdb_file} already exists, skip structure prediction.')
        return pdb_file

    print(
        f'Predicting structure for VH/VL pair: {vh_vl_pair.vh_id}|{vh_vl_pair.vl_id}'
    )
    pred_start_time = time.time()
    vh_vl_struct = predictor.predict(vh_vl_pair.to_dict())
    vh_vl_struct.save(pdb_file)  # ImmuneBuilder refinement is called in save()
    pred_end_time = time.time()
    print(
        f'Done. Took {(pred_end_time - pred_start_time):.2f} seconds. '
        f'Predicted structure saved to {pdb_file}.'
    )
    return pdb_file


def make_voxel_grids(pdb_files, voxel_size=1.0, num_rotations=0):
    """Makes a voxel grid for each of the given PDB files.

    Parameters
    ----------
    pdb_files : list
        A list of PDB files.
    voxel_size : float
        Side length of individual voxels
    num_rotations : int
        How many times each structure will be randomly rotated.

    Returns
    -------
    np.NDArray
        A tensor of rank 5 where the size of the first dimension is
        the number of PDB files, the second dimension is the number of rotations
        performed on each structure, and the remaining three dimensions
        store the width, height, and depth of the voxel grid.

        Note that when the number of rotations is zero, i.e. no rotations, the
        size of the second dimension of the tensor will also be 1.

    """
    all_voxel_grids = []
    for pdb_file in pdb_files:
        voxel_grids = []
        if num_rotations == 0:
            to_vg = transforms.ToVoxelGrid(voxel_size=voxel_size)
            voxel_grids.append(to_vg(pdb_file))
        else:
            for _ in range(num_rotations):
                to_vg = transforms.ToVoxelGrid(voxel_size=voxel_size)
                struct_rotater = transforms.RotateStructure()
                voxel_grids.append(
                    to_vg(struct_rotater(pdb_file))
                )
        all_voxel_grids.append(voxel_grids)
    voxel_grids_arr = np.array(all_voxel_grids)
    return voxel_grids_arr


def parse_cmd_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', dest='input', type=str, required=True,
        help='''A CSV file where each row is a pair of VH/VL sequences and their IDs:
            first column: heavy chain ID,
            second column: light chain ID,
            third column: heavy chain amino acid sequence,
            fourth column: light chain amino acid sequence.
        '''
    )
    parser.add_argument(
        '-o', '--output-prefix', dest='output_prefix', type=str, required=False, default='./',
        help='''Prefix of the npy file in which the voxel grids will be stored.'''
    )
    parser.add_argument(
        '-n', '--num-rotations', dest='rotations', type=int, required=False, default=0,
        help='''Number of times to rotate each structure.'''
    )
    parser.add_argument(
        '-r', '--ref-struct', dest='ref_struct', type=str, required=False,
        help='''PDB file of the reference structure to which all predicted structure will
        be aligned to before their voxel grid representation is computed.'''
    )

    args = parser.parse_args()
    if args.rotations != 0 and args.ref_struct is not None:
        parser.error(
            'When a reference structure is provided for alignment, no rotation is allowed.'
        )

    return parser.parse_args()


def main():
    cmd_args = parse_cmd_args()

    # parse the input csv file
    vh_vl_pairs = []
    with open(cmd_args.input, 'rt') as ipf:
        for line in ipf:
            vh_id, vl_id, vh_seq, vl_seq = line.split(',')
            vh_vl_pairs.append(
                VhVlPair(vh_id.strip(), vl_id.strip(), vh_seq.strip(), vl_seq.strip())
            )

    # predict structures
    pdb_files = []
    predictor = ABodyBuilder2(
        weights_dir=defaults.system_config['immunebuilder_weights_dir']
    )
    for vh_vl_pair in vh_vl_pairs:
        pdb_files.append(predict_structure(vh_vl_pair, predictor))

    # PDB files to be used for voxel grid computation
    pdb_files_vg = pdb_files

    # align structures to reference
    if cmd_args.ref_struct is not None:
        aligned_pdb_files = []
        for pdb_file in pdb_files:
            pymol.cmd.load(pdb_file, 'mobile')
            pymol.cmd.load(cmd_args.ref_struct, 'ref_struct')

            # align the mobile structure to the target
            results = pymol.cmd.align(mobile='mobile', target='ref_struct')
            print(f'RMSD: {results[0]:.2f}')
            print(f'Number of atoms aligned: {results[1]}')
            print(f'Number of residues aligned: {results[-1]}')

            # save the aligned mobile structure
            aligned_pdb_file = os.path.join(
                os.path.dirname(pdb_file),
                'aligned_' + os.path.basename(pdb_file)
            )
            pymol.cmd.save(aligned_pdb_file, 'mobile')

            # must delete the mobile object before loading the next structure
            # otherwise the resulting aligned PDB file will be incorrect and may
            # also be corrupted
            pymol.cmd.delete('mobile')
            
            aligned_pdb_files.append(aligned_pdb_file)

        # now voxel grid computation will be based on aligned structures
        pdb_files_vg = aligned_pdb_files

    # now generate voxel grids
    grid_start_time = time.time()
    voxel_grids_arr = make_voxel_grids(
        pdb_files_vg, voxel_size=1.0, num_rotations=cmd_args.rotations
    )
    grid_end_time = time.time()
    print(f'Voxel grid generation took {(grid_end_time - grid_start_time):.2f} seconds.')

    np.save(
        file=cmd_args.output_prefix + '.npy',
        arr=voxel_grids_arr, allow_pickle=False
    )


if __name__ == '__main__':
    main()
