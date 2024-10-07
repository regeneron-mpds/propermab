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
from numba import njit, prange
import numpy as np
from . import md,io,sasa

main_chain = 'CA|HA|N|C|O|HN|H'.split('|')

@njit(parallel=True)
def get_scm_numba(a_mainchain,a_sasa,a_coords,charge_values,d_cutoff,sasa_cutoff):
    """ SCM per atom in structure, with precalculated charges in MD forcefield
    """
    scms = np.zeros((len(a_mainchain),1))
    for i in prange(len(a_mainchain)):
        tmp_scm = 0
        for j in np.arange(len(a_mainchain)):
            if not a_mainchain[j] and i!=j:
                d = np.sqrt(np.sum(np.square(a_coords[i,:]-a_coords[j,:])))
                sasa = a_sasa[j]
                if d<d_cutoff and sasa>sasa_cutoff:
                    tmp_scm += charge_values[j]
        scms[i] = tmp_scm
    return scms

def get_scm(structure, charges, d_cutoff=10, sasa_cutoff=10):
    """ SCM per atom in structure, with precalculated charges in MD forcefield
    
    internally uses numba for faster calculation
    """
    a_mainchain = np.array([a.id in main_chain for a in structure.get_atoms()])
    a_sasa = np.array([a.get_bfactor() for a in structure.get_atoms()]).astype('float32')
    a_coords = np.array([a.coord for a in structure.get_atoms()])
    a_charges = np.array([c._value for c in charges])
    
    scms = get_scm_numba(a_mainchain,a_sasa,a_coords,a_charges,d_cutoff,sasa_cutoff)
    return scms

def score_from_scms(scms):
    """ SCM score from scm values per atom"""
    return abs(np.sum(scms[scms<0]))

def scm_score(structure, charges, d_cutoff=10, sasa_cutoff=10):
    """ for a structure and charges per atom, get final SCM score for structure """
    scms = get_scm(structure, charges, d_cutoff=d_cutoff, sasa_cutoff=sasa_cutoff)
    return score_from_scms(scms)

def raw_pdb_scm_scoring(pdb_path):
    """ Requires PDB with Hydrogens already added
    
    Will not perform any MD, will use CHARMM for potential
    Expects chains named H and L for heavy and light chains
    
    parameters
    -----------
    pdb_path: string
        path to pdb file to get SCM for. must have hydrogens already.
        must have heavy and ligh chains named H and L.
        
    returns
    ---------
    SCMs: Tuple of (scms,scm_score)
        scms: scm values per atom, scm_score, scm_score for whole structure
    """
    structure_ = io.load_structure(pdb_path)
    sasa_result = sasa.apply_sasa(pdb_path)
    sasa_structure = sasa.write_sasa_to_structure(structure_,sasa_result)
    
    system,topology = md.simple_system(pdb_path)
    charges = md.get_partial_charges_system(system,topology,chains='HL')
    
    scms = get_scm(sasa_structure, charges)
    scm_score = score_from_scms(scms)
    return scms, scm_score