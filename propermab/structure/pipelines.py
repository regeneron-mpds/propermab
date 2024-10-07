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
from openmm.app import *
from openmm import *
from openmm import unit
from sys import stdout, exit, stderr
import numpy as np
import copy
import os
from Bio.PDB import *

from . import md,io,sasa,scm

def single_pdb_md(pdb_path, outdir, hchain='H', lchain='L', maxiter=None):
    """ function to run through MD process, and get SCM
    
    Note: was written in some haste and for convenience
     - an OO approach with 'steps' so can be easily adjusted would be preferred
    
    paramters
    ----------
    pdb_path: string
        pth to pdb file
        
    outdir: string
        path to dir to save output files to.
        
            
    hchain: string, optional, default='H'
        Name of chain that is the heavy chain
        
    lchain: string, optional, default='L'
        Name of chain that is the light chain
        
    maxiter: int, optional, default=None
        If None, relax until convergence. If integer, run that many timesteps
    
    returns:
    ---------
    
    files out in outdir: 
            - energy_prior.txt: energy prior to relax
            - energy_post.txt: energy after relax
            - charges.txt: charges on atoms immediately after relax
            - relax_output_mab.pdb: strcuture after relax (just atoms, no solvent)
            - scms.txt: scm per atoms (same ordering as pdb atoms)
            - scm.txt: scm score for whole structure
            - sasa.txt: SASA per atom in structure (repeats per atom in same residue)
            - scm_relax.pdb: PDB of relaxed structure with SCM in b-factor
    """
    
    fix_path, max_size = md.clean_structure(
        pdb_path,
        chains_keep=[hchain,lchain],
        remove_water=True,
        add_solvent=False
    )

    modeller,system,simulation = md.setup_minimization_system(fix_path,ionic=0.015)

    st = simulation.context.getState(getPositions=True,getEnergy=True)
    print("Potential energy before minimization is %s" % st.getPotentialEnergy())
    np.savetxt(os.path.join(outdir, 'energy_prior.txt'), np.array([st.getPotentialEnergy()._value]))
    if maxiter is None:
        simulation.minimizeEnergy()
    else:
        simulation.minimizeEnergy(maxIterations=maxiter)
    st = simulation.context.getState(getPositions=True,getEnergy=True)
    print("Potential energy after minimization is %s" % st.getPotentialEnergy())
    np.savetxt(os.path.join(outdir, 'energy_post.txt'), np.array([st.getPotentialEnergy()._value]))

    charges = md.get_partial_charges_system(system,modeller.topology)
    charges_v = [-c._value for c in charges]
    np.savetxt(os.path.join(outdir, 'charges.txt'), charges_v)
    
    # ------- CLEAN THE STRUCTURE FOR JUST THE MAB ------------------------------
    modeller_ab = copy.deepcopy(modeller)
    # which atoms are in the mAb, or are water,solvent etc
    idxs_ab = []
    idxs_nonab = []
    for a in modeller_ab.topology.atoms():
        if a.residue.chain.id in ['A','B']:
            idxs_ab.append(a.index)
        else:
            idxs_nonab.append(a)

    # delete the non mAb atoms
    modeller_ab.delete(idxs_nonab)

    # collect the positions of the mAb atoms
    allpos = simulation.context.getState(
        getPositions=True,
        getEnergy=True
    ).getPositions()
    ab_pos = [allpos[i] for i in idxs_ab]

    # put the mAb atoms into a Qaunitity object
    ab_pos_q = unit.quantity.Quantity(unit=unit.nanometer)
    for p in ab_pos:
        ab_pos_q.append(p)

    # write only the atoms/positions of the mAb atoms
    PDBFile.writeFile(
        modeller_ab.topology, 
        ab_pos_q, 
        open(
            os.path.join(
                outdir, 
                'relax_output_mab.pdb'
            ),
            'w'
        )
    )
    # ------- WRITTEN RELAXED MAB WITH HYDROGENS TO FILE --------------------------

    relax_o_path = os.path.join(outdir, 'relax_output_mab.pdb')
    structure_r = io.load_structure(relax_o_path)
    sasa_result = sasa.apply_sasa(relax_o_path)
    sasa_structure = sasa.write_sasa_to_structure(structure_r,sasa_result)
    scms = scm.get_scm(sasa_structure, charges)
    scm_score = scm.score_from_scms(scms)

    np.savetxt(os.path.join(outdir, 'scms.txt'), scms)
    np.savetxt(os.path.join(outdir, 'scm.txt'), np.array([scm_score]))

    sasas = []
    for i,atom in enumerate(sasa_structure.get_atoms()):
        sasas.append(atom.bfactor)
    np.savetxt(os.path.join(outdir, 'sasa.txt'), sasas)

    # WRITE A SCM version of PDB TO FILE FOR EASE OF PYMOL PLOTTING
    for i,atom in enumerate(sasa_structure.get_atoms()):
        atom.bfactor = scms[i]
    io.save_structure(sasa_structure,os.path.join(outdir,'scm_relax.pdb'))
    return None