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
""" Module for tools around Molecular dynamics, charge calculation """

from openmm.app import *
from openmm import *
from Bio.PDB import *
from pdbfixer import PDBFixer

def get_forcefield(kind='amber'):
    if kind=='amber':
        return ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    else:
        return ForceField('charmm36.xml','charmm36/tip3p-pme-b.xml')

def simple_system(path):
    
    pdb = PDBFile(path)
    forcefield = get_forcefield(kind='charmm')
    system = forcefield.createSystem(
        pdb.topology
    )
    return system, pdb.topology


def clean_structure(
    pdb_path,
    chains_keep=None,
    ph=6.0,
    ionic=0.015,
    remove_water=True,
    add_solvent=False,
    solvent_box=None
):
    fix_name = pdb_path.split('.')[0]+'_fixed_{:.2f}.pdb'.format(ph)
        
    fixer = PDBFixer(filename=pdb_path)
    
    if chains_keep is None:
        chains_keep = ['H','L']
    if isinstance(chains_keep,list):
        chains_keep = set(chains_keep)
        
    chains_remove_dict = {c.id: c.index for c in fixer.topology.chains()}
    chains_remove = list(set([c.id for c in fixer.topology.chains()]) - chains_keep )
    chain_id_remove = [chains_remove_dict[k] for k in chains_remove]
    
    fixer.removeChains(chain_id_remove)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    het_arg = not remove_water
    fixer.removeHeterogens(het_arg) #False removes all hetatm inc. water, can be necessary just because of pdb chain formatting. True remove hetatm, but not water.
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    fix_name_no_h = pdb_path.split('.')[0]+'_fixed_noH.pdb'
    PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name_no_h, 'w'))
    
    fixer.addMissingHydrogens(ph)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name, 'w'))

    maxSize=0*unit.angstrom
    if add_solvent:
        if solvent_box is None:
            fixer.addSolvent(fixer.topology.getUnitCellDimensions(), ionicStrength=ionic*unit.molar)
        else:
            maxSize = max(max((pos[i] for pos in fixer.positions))-min((pos[i] for pos in fixer.positions)) for i in range(3))
            print('maxSize = ', maxSize)
            maxSize += solvent_box*unit.angstrom
            boxSize = maxSize*Vec3(1, 1, 1)
            #fixer.topology.setUnitCellDimensions(boxSize)
            #fixer.addSolvent(fixer.topology.getUnitCellDimensions(), ionicStrength=ionic*unit.molar)
            fixer.addSolvent(boxSize,  ionicStrength=ionic*unit.molar)
            
        fix_name_solv = pdb_path.split('.')[0]+'_fixed_{:.2f}_solvent.pdb'.format(ph)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name_solv, 'w'))
    return fix_name,maxSize

def setup_minimization_system(
    path,
    field='charmm',
    ionic=0.015,
    padding=1.2,
    bond_cutoff=1.2,
    temp=300,
    dt=0.001,
    
):
    pdb = PDBFile(path)

    forcefield = get_forcefield(kind='charmm')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, padding=padding*unit.nanometers, ionicStrength=ionic*unit.molar)
    
    system = forcefield.createSystem(
        modeller.topology, 
        nonbondedMethod=PME, # ideally construct own PME and addd switching distance to that obj, then pass in here
        nonbondedCutoff=bond_cutoff*unit.nanometer, 
        constraints=None 
    )
    
    integrator = LangevinMiddleIntegrator(temp*unit.kelvin, 1/unit.picosecond, dt*unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    return modeller, system, simulation

def get_partial_charges_system(system,topology,chains='AB'):
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atoms = [a for a in topology.atoms()]
    charges = []
    for i in range(system.getNumParticles()):
        if atoms[i].residue.chain.id in chains:
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge)
    return charges

def get_partial_charges(
    pdb_path,
    forcefield,
    chains='AB'
):
    """ Get partial charges on all atoms
    
    Will skip waters if solvent added to structure
    
    Assumes two chain structure with chain_ids A and B in that order. 
     - How pdbfixer will output for two chains selected
    
    """
    pdb = PDBFile(pdb_path)
    system = forcefield.createSystem(pdb.topology)
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atoms = [a for a in pdb.topology.atoms()]
    charges = []
    for i in range(system.getNumParticles()):
        if atoms[i].residue.chain.id in chains:
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge)
    return charges

def minimize(
    pdb_path,
    forcefield,
    nb_cutoff=12,
    temp=300,
    max_iter=200,
):
    pdb = PDBFile(pdb_path)
    system = forcefield.createSystem(
    pdb.topology, 
        nonbondedMethod=PME, 
        nonbondedCutoff=(nb_cutoff/10.)*nanometer, 
        constraints=None 
    )
    integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
