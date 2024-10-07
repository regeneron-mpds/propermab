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
import time
import tempfile
import numpy as np
from collections import defaultdict

from ImmuneBuilder import ABodyBuilder2

from Bio import PDB
import propermab as pm
from propermab.utils import nanoshaper
from propermab.utils import apbs
from propermab.io import geometry_io
from propermab import defaults
from propermab.features import surface

from propermab.features import SeqFeaturizer
from propermab.features import StructFeaturizer


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


def get_all_seq_features(
    heavy_seq: str, 
    light_seq: str, 
    is_fv: bool,
    isotype: str, 
    lc_type: str, 
    pH: float=7.4
) -> dict:
    """Calculates all currently implemented sequence-based features for the given antibody.

    Parameters
    ----------
    heavy_seq : str
        Amino acid sequence of the heavy chain.
    light_seq : str
        Amino acid sequence of the light chain.
    is_fv : bool
        Is the given sequence only the Fv region.
    isotype : str
        Isotype of the antibody. Select one from ['igg1', 'igg2', 'igg4'].
    lc_type : str
        Type of the light chain, either kappa or lambda.

    Returns
    -------
    dict
        Sequence features of the given antibody as a dictionary keyed by feature names.
    """
    seq_featurizer = SeqFeaturizer(
        seqs=(heavy_seq, light_seq), is_fv=is_fv, isotype=isotype, 
        lc_type=lc_type, pH=pH
    )

    seq_features = {
        'theoretical_pi': seq_featurizer.theoretical_pi(),
        'n_charged_res': seq_featurizer.n_charged_res(),
        'n_charged_res_fv': seq_featurizer.n_charged_res_fv(),
        'fv_charge': seq_featurizer.fv_charge(),
        'fv_csp': seq_featurizer.fv_csp(),
        'fc_charge': seq_featurizer.fc_charge(),
        'fab_fc_csp': seq_featurizer.fab_fc_csp()
    }

    return seq_features


def calculate_patch_features(pdb_file: str, tmp_dir: str='/tmp/') -> dict:
    """Calculates the set of patch-based features from the given PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file based on which features will be calculated.
    tmp_dir : str, optional
        The temporary directory in which to do NanoShaper and APBS calculations, 
        by default '/tmp/'

    Returns
    -------
    dict
        Patch-based features of the given antibody as a dictionary keyed by feature names.
    """
    # compute surface triangulation
    nanoshaper_runner = nanoshaper.NanoShaper(grid_scale=0.5)
    vert_file, face_file = nanoshaper_runner(pdb_file, tmp_dir)
    vertices, faces, atom_nums, _ = geometry_io.read_nanoshaper(
        vert_file, face_file
    )

    # run APBS to compute electrostatic potential
    apbs_runner = apbs.APBS()
    apbs_output_file = apbs_runner(pdb_file, tmp_dir)

    # needs to store vertex coordinates in CSV format to run multivalue
    file_prefix = os.path.splitext(os.path.basename(pdb_file))[0]
    tmp_vertex_file = f'{file_prefix}_vertices.csv'
    with open(tmp_vertex_file, 'wt') as opf:
        for line in vertices:
            opf.write(
                f'{float(line[0])},{float(line[1])},{float(line[2])}\n'
            )

    # multivalue extracts electrostatic potential from APBS output file for each vertex
    apbs_values = apbs.run_multivalue(
        defaults.system_config['multivalue_binary_path'],
        tmp_vertex_file, apbs_output_file, file_prefix
    )

    # vertex file is no longer needed
    if os.path.isfile(tmp_vertex_file):
        os.remove(tmp_vertex_file)

    # apbs output file no longer needed
    if os.path.isfile(apbs_output_file):
        os.remove(apbs_output_file)

    # patch related features
    pdb_parser = PDB.PDBParser()
    struct_model = pdb_parser.get_structure(
        id='tmp', file=pdb_file
    )[0]
    mab_surface = surface.Surface(
        vertices, faces, atom_nums, struct_model, apbs_values
    )

    # retrieve coordinates for CDR vertices
    # the coordinates are needed for building KD trees        
    cdr_vertices_dict = mab_surface.find_cdr_vertices()
    cdr_coords_dict = {}
    for cdr, cdr_vertices in cdr_vertices_dict.items():
        cdr_coords_dict[cdr] = np.array([
            mab_surface.vertices[u - 1] for u in cdr_vertices
    ])

    hyd_patches = mab_surface.find_patches_dbscan(
        prop='hyd', prop_cutoff=0.04, eps=2.0, min_samples=5,
        area_cutoff=40.
    )
    total_hyd_area = 0.
    total_hyd_area_cdr = 0.
    for _, hyd_patch in hyd_patches.items():
        hyd_patch_area = sum(face_area for _, face_area in hyd_patch)
        total_hyd_area += hyd_patch_area
        patch_faces = [face for face, _ in hyd_patch]
        patch_vertices = set([x for y in patch_faces for x in y])
        # retrieve coordinates for patch vertices
        # the coordinates are needed for building KD trees
        patch_vertex_coords = np.array([
            mab_surface.vertices[v - 1] for v in patch_vertices
        ])

        surface_patch = surface.SurfacePatch(patch_vertices, patch_faces)
        surface_patch.vertex_coords = patch_vertex_coords
        if surface_patch.is_near_cdr_kd_tree(cdr_coords_dict, dist_cutoff=5.0):
            print('Found a patch near CDR with area:', hyd_patch_area)
            total_hyd_area_cdr += hyd_patch_area

    pos_patches = mab_surface.find_patches_dbscan(
        prop='pos', prop_cutoff=8., eps=2.25, min_samples=8,
        area_cutoff=20.
    )
    total_pos_area = 0.
    total_pos_area_cdr = 0.
    for _, pos_patch in pos_patches.items():
        pos_patch_area = sum(face_area for _, face_area in pos_patch)
        total_pos_area += pos_patch_area
        patch_faces = [face for face, _ in pos_patch]
        patch_vertices = set([x for y in patch_faces for x in y])
        # retrieve coordinates for patch vertices
        # the coordinates are needed for building KD trees
        patch_vertex_coords = np.array([
            mab_surface.vertices[v - 1] for v in patch_vertices
        ])
        surface_patch = surface.SurfacePatch(patch_vertices, patch_faces)
        surface_patch.vertex_coords = patch_vertex_coords
        if surface_patch.is_near_cdr_kd_tree(cdr_coords_dict, dist_cutoff=5.0):
            total_pos_area_cdr += pos_patch_area

    neg_patches = mab_surface.find_patches_dbscan(
        prop='neg', prop_cutoff=-6., eps=2.75, min_samples=8,
        area_cutoff=20.
    )
    total_neg_area = 0.
    total_neg_area_cdr = 0.
    for _, neg_patch in neg_patches.items():
        neg_patch_area = sum(face_area for _, face_area in neg_patch)
        total_neg_area += neg_patch_area
        patch_faces = [face for face, _ in neg_patch]
        patch_vertices = set([x for y in patch_faces for x in y])
        # retrieve coordinates for patch vertices
        # the coordinates are needed for building KD trees
        patch_vertex_coords = np.array([
            mab_surface.vertices[v - 1] for v in patch_vertices
        ])
        surface_patch = surface.SurfacePatch(patch_vertices, patch_faces)
        surface_patch.vertex_coords = patch_vertex_coords
        if surface_patch.is_near_cdr_kd_tree(cdr_coords_dict, dist_cutoff=5.0):
            total_neg_area_cdr += neg_patch_area

    # compute the Heiden score
    heiden_score = mab_surface.heiden_score() 

    # create a dataframe to write the data in tabular format    
    patch_features = {
        'hyd_patch_area': total_hyd_area,
        'pos_patch_area': total_pos_area,
        'neg_patch_area': total_neg_area,
        'hyd_patch_area_cdr': total_hyd_area_cdr,
        'pos_patch_area_cdr': total_pos_area_cdr,
        'neg_patch_area_cdr': total_neg_area_cdr,
        'heiden_score': heiden_score,
    }
    return patch_features


def calculate_features_from_pdb(pdb_file: str, tmp_dir: str='/tmp/') -> dict:
    """Delegation function for the get_all_mol_features() function.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file based on which features will be calculated.
    tmp_dir : str, optional
        The temporary directory in which to do NanoShaper and APBS calculations, 
        by default '/tmp/'

    Returns
    -------
    dict
        Molecular features of the given antibody as a dictionary keyed by feature names.
    """
    # non-patch features
    featurizer = StructFeaturizer(pdb_file)
    asa_hyd = featurizer.hyd_asa()
    asa_hph = featurizer.hph_asa()
    net_charge = featurizer.net_charge()
    exposed_net_charge = featurizer.exposed_net_charge()
    cdr_net_charge = featurizer.net_charge_cdr()
    exposed_cdr_net_charge = featurizer.net_charge_cdr(exposed=True)
    Fv_chml = featurizer.fv_chml()
    exposed_fv_chml = featurizer.exposed_fv_chml()
    dipole_moment = featurizer.dipole_moment()
    hyd_moment = featurizer.hyd_moment()
    cdr_h3_length = featurizer.cdr_length(cdr='H3')
    asa_aromatic = featurizer.aromatic_asa()
    cdr_aromatic = featurizer.aromatic_cdr()
    exposed_aromatic = featurizer.exposed_aromatic()

    pos_ann_index = featurizer.ann_index(prop='pos', rsa_cutoff=0.05, n=1000)
    neg_ann_index = featurizer.ann_index(prop='neg', rsa_cutoff=0.05, n=1000)
    aromatic_ann_index = featurizer.ann_index(prop='aro', rsa_cutoff=0.05, n=1000)
    pos_ripley_k = featurizer.ripley_k(prop='pos', distance=6.0, n=1000)
    neg_ripley_k = featurizer.ripley_k(prop='neg', distance=6.0, n=1000)
    aromatic_ripley_k = featurizer.ripley_k(prop='aro', distance=6.0, n=1000)

    # compute the scm score
    _, scm_score = pm.st.scm.raw_pdb_scm_scoring(pdb_file)

    # compute patch features
    patch_features = calculate_patch_features(pdb_file, tmp_dir)
        
    # create a dataframe to write the data in tabular format    
    mol_features = {
        'hyd_asa': asa_hyd,
        'hph_asa': asa_hph,
        'net_charge': net_charge,
        'exposed_net_charge': exposed_net_charge,
        'net_charge_cdr': cdr_net_charge,
        'exposed_net_charge_cdr': exposed_cdr_net_charge,
        'dipole_moment': dipole_moment,
        'Fv_chml': Fv_chml,
        'exposed_Fv_chml': exposed_fv_chml,
        'hyd_moment': hyd_moment,
        'cdr_h3_length': cdr_h3_length,
        'aromatic_asa': asa_aromatic,
        'aromatic_cdr': cdr_aromatic,
        'exposed_aromatic': exposed_aromatic,
        'scm': scm_score,
        'pos_ann_index': pos_ann_index,
        'neg_ann_index': neg_ann_index,
        'aromatic_ann_index': aromatic_ann_index,
        'pos_ripley_k': pos_ripley_k,
        'neg_ripley_k': neg_ripley_k,
        'aromatic_ripley_k': aromatic_ripley_k
    }
    mol_features.update(patch_features)

    return mol_features


def get_all_mol_features(
    heavy_seq: str=None, 
    light_seq: str=None,
    pdb_file: str=None, 
    num_runs: int=1, 
    tmp_dir: str='/tmp/'
):
    """Calculates all currently implemented molecular features for the given antibody.

    Either heavy_seq and light_seq as a pair or PDB file must be passed. 
    It will raise an exception if both sequences and pdb_file are None. As
    long as a PDB file is passed, the PDB file gets used, sequences and num_runs
    will be ignored.

    Note that currently this function assumes that if given a PDB file, the PDB
    file is IMGT numbered.

    Parameters
    ----------
    heavy_seq : str
        Amino acid sequence of the heavy chain.
    light_seq : str
        Amino acid sequence of the light chain.
    pdb_file : str
        Path to the PDB file based on which features will be calculated.
    num_runs : int, optional
        Number of repeated structure prediction runs, by default 1
    tmp_folder : str, optional
        The temporary directory in which to do NanoShaper and APBS calculations, 
        by default '/tmp/'

    Returns
    -------
    dict
        Molecular features of the given antibody as a dictionary keyed by feature names.
    """
    mol_features = []
    if pdb_file is None:
        if heavy_seq is None or light_seq is None:
            raise ValueError('When pdb_file is None, sequences must be given!')
        else:
            vh_vl_pair = VhVlPair(
                'heavy_chain', 
                'light_chain', 
                heavy_seq.strip(), 
                light_seq.strip()
            )

            # antibody structure predictor
            predictor = ABodyBuilder2(
                weights_dir=defaults.system_config['immunebuilder_weights_dir']
            )

            for _ in range(num_runs):
                with tempfile.NamedTemporaryFile(suffix='.pdb') as temp_pdb_file_obj:

                    # temp_pdb_file gets removed after exiting the conext manager
                    temp_pdb_file = temp_pdb_file_obj.name
                    print(
                        f'Predicting structure for VH/VL pair: {vh_vl_pair.vh_id}|{vh_vl_pair.vl_id}'
                    )
                    pred_start_time = time.time()
                    vh_vl_struct = predictor.predict(vh_vl_pair.to_dict())
                    vh_vl_struct.save(temp_pdb_file)  # ImmuneBuilder refinement is called in save()
                    pred_end_time = time.time()
                    print(
                        f'Done. Took {(pred_end_time - pred_start_time):.2f} seconds. '
                        f'Predicted structure saved to {temp_pdb_file}.'
                    )
                    
                    # must be done inside the context manager because temp_pdb_file
                    # won't be accessible once the conext manager is exited
                    mol_features.append(calculate_features_from_pdb(temp_pdb_file, tmp_dir))
    else:
        mol_features.append(calculate_features_from_pdb(pdb_file, tmp_dir))

    # combine the features into a single dictionary
    combined_mol_features = defaultdict(list) 
    for feature_dict in mol_features:
        for key, value in feature_dict.items():
            combined_mol_features[key].append(value)
    
    return combined_mol_features
