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
from Bio.PDB import *
from Bio import SeqIO
from anarci import anarci
import os

from . import io
from .. import defaults

def generate_ab_dict(fasta_path, split_char=' ', chain_names = {'HC': 'H', 'LC': 'L'}):
    record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    all_abs = dict()
    for k,v in record_dict.items():
        name = k.split(split_char)[0]
        ch = k.split(split_char)[1]
        if name not in all_abs:
            all_abs[name] = dict()
        all_abs[name][chain_names[ch]] = v._seq._data 
        
    return all_abs

class IMGTSelect(Select):
    def accept_residue(self, residue):
        imgt_status = residue.xtra.get('IMGT',False)
        return int(imgt_status)
    
class HydrogenSelect(Select):
    def accept_atom(self, atom):
        status = atom.element!='H'
        return int(status)
    
def removeH(name, loadpath, outpath):
    """ remove hydrogens from a pdb file 
    saves new PDB to now file w/o H
    """
    structure = io.load_structure(
        os.path.join(loadpath, name+'.pdb')
    )
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    out_file = os.path.join(outpath, name+'_Hremoved.pdb')
    pdbio.save(out_file, HydrogenSelect())
    return out_file   
    
def get_imgt_seqs(seq_a=None, seq_b=None, structure=None):
    """
    
    if provide structure, don't previde seq_a and seq_b
    if no strcuture, only provide seq_a and seq_b
    
    """
    
    def build_idx_mapping(numbering,idx_start):
        c_idx = idx_start
        mapping = dict()
        for imgtcode, aa in numbering:
            if aa!='-':
                mapping[imgtcode] = c_idx
                c_idx+=1
        return mapping
    
    def build_idx_imgt_mapping(numbering,idx_start):
        c_idx = idx_start
        mapping = dict()
        for imgtcode, aa in numbering:
            if aa!='-':
                mapping[c_idx] = imgtcode
                c_idx+=1
        return mapping
    
    #print('structure = ',structure)
    if structure is not None:
        seq_a = ''.join([Polypeptide.three_to_one(r.resname) for r in structure[0]['A'].get_residues()])
        seq_b = ''.join([Polypeptide.three_to_one(r.resname) for r in structure[0]['B'].get_residues()])

    results = anarci(
        [
            ('A',seq_a),
            ('B',seq_b)
        ], 
        scheme = "imgt", 
        allowed_species = ['human'], 
        output = False,
        hmmerpath = defaults.system_config.config['hmmer_binary_path'] 
    )
    numbering, alignment_details, hit_tables = results
    
    numbering_a = numbering[0][0][0]
    numbering_b = numbering[1][0][0]

    chain_a = alignment_details[0][0]['chain_type']
    chain_b = alignment_details[1][0]['chain_type']

    imgt_seq_a = ''.join([b for a,b in numbering_a]).replace('-','')
    imgt_seq_b = ''.join([b for a,b in numbering_b]).replace('-','')
    
    idx_start_a = seq_a.find(imgt_seq_a)
    idx_start_b = seq_b.find(imgt_seq_b)
    
    mapping_a = build_idx_imgt_mapping(numbering_a,idx_start_a)
    mapping_b = build_idx_imgt_mapping(numbering_b,idx_start_b)
    
    def chain_imgt_state_write(chain,mapping):
        for i,r in enumerate(chain.get_residues()):
            if i in mapping:
                r.xtra.update({'IMGT':True})
                r.xtra.update({'IMGT_number':mapping[i]})
            else:
                r.xtra.update({'IMGT':False})
                r.xtra.update({'IMGT_number':None})
    
    def replace_k_ch(ch):
        if ch=='K':
            return 'L'
        else:
            return ch
    
    chain_a = replace_k_ch(chain_a)
    chain_b = replace_k_ch(chain_b)
    
    seq_d = {
        chain_a: imgt_seq_a,
        chain_b: imgt_seq_b,
    }
    
    if structure is not None:
        chain_imgt_state_write(structure[0]['A'],mapping_a)
        chain_imgt_state_write(structure[0]['B'],mapping_b)

        structure[0]['A'].id = chain_a
        structure[0]['B'].id = chain_b
    
    return seq_d

