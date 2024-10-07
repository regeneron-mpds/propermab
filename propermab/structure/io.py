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
""" Loading, saving PDB/CIF """

import requests
import os
from Bio.PDB import *
import tempfile
import zipfile

def get_pdb_sabdab(pdb_code,numbering='imgt'):
    """ get a Bio.PDB.Structure for a pdb_code by temporarily downloading it  
    
    The pdb file for the requested code will be retreived from the SabDab 
    via url request, and loaded into a temp file. The pdb will be parsed 
    and returned as a Bio.PDB.structure
    
    parameters:
    -----------
    pdb_code: (string) The pdb code
    
    returns:
    ---------
    structure: (Bio.PDB.Structure) pdb structure file
    """
    if numbering not in ['chothia', 'imgt']:
        raise ValueError("numbering type : ", numbering," , not supported")
    address = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/'+pdb_code+'/?scheme='+numbering
    r = requests.get(address)
    parser = PDBParser()
    with tempfile.NamedTemporaryFile(mode = "w",delete=True) as tmp:
        tmp.write(r.text)
        tmp.flush()
        tmp.seek(0)
        structure = parser.get_structure('bound', tmp.name)
        tmp.close()
    return structure

def pdb_from_file(pdb_code,pdb_file_location=None):
    """ get a Bio.PDB.Structure for a pdb from file  
    
    parameters:
    -----------
    pdb_file_location: (string) The pdb code file location
    clean_non_ca: (bool, optional) remove residues without defined C_alpha atom
    
    returns:
    ---------
    structure: (Bio.PDB.Structure) pdb structure file
    """
    parser = PDBParser()
    structure = parser.get_structure('bound', os.path.join(pdb_file_location,pdb_code))
    return structure

class PDBFromFileLoader:
    
    def __init__(self,dir_load,clean_non_ca,pdb_post_str=None):
        """ Initialize
        
        parameters
        -----------
        dir_load: (str) location to load pdb files from
        clean_non_ca: (bool) whether to remove `residues` without an alpha carbon
        pdb_post_str: (str, optional) string to append to pdb code for filenames to be loaded
        
        """
        self.dir_load = dir_load
        self.clean_non_ca = clean_non_ca
        self.pdb_post_str = pdb_post_str
      
    def __call__(self,pdb_code):
        """ load file for pdb code
        
        parameters
        ------------
        pdb_code: (str) the pdb code to load
        
        returns
        --------
        structure: (BIo.PDB.Structure) Structure containing the PDB info
        
        """
        if self.pdb_post_str is not None:
            pdb_code = pdb_code+self.pdb_post_str+'.pdb'
        return pdb_from_file(pdb_code,
                      pdb_file_location=self.dir_load,
                      clean_non_ca=self.clean_non_ca)
        
class PDBFromWebLoader:
    
    def __init__(self,clean_non_ca,numbering='chothia'):
        self.clean_non_ca = clean_non_ca
        self.numbering=numbering
        
    def __call__(self,pdb_code):
        """ load file for pdb code from Web
        
        parameters
        ------------
        pdb_code: (str) the pdb code to load
        
        returns
        --------
        structure: (BIo.PDB.Structure) Structure containing the PDB info
        
        """
        return get_pdb_sabdab(pdb_code,
                              numbering=self.numbering,
                              clean_non_ca=self.clean_non_ca)
    
class PDBFromZipLoader:
    
    def __init__(self,zip_dir,clean_non_ca,numbering='chothia'):
        self.rxiv = zipfile.ZipFile(zip_dir,'r')
        self.clean_non_ca = clean_non_ca
        self.numbering=numbering
        
    def __call__(self,pdb_code):
        """ load file for pdb code from zip archive

        parameters
        ------------
        pdb_code: (str) the pdb code to load

        returns
        --------
        structure: (BIo.PDB.Structure) Structure containing the PDB info

        """
        try:
            file = self.rxiv.open(self.numbering+'/'+pdb_code+'.pdb','r')
        except KeyError:
            print("WARNING missing PDB code in zip archive: ", pdb_code)
            return None
        f_str = file.read().decode("utf-8")
        parser = PDBParser()
        with tempfile.NamedTemporaryFile(mode = "w",delete=True) as tmp:
            tmp.write(f_str)
            tmp.flush()
            tmp.seek(0)
            structure = parser.get_structure('bound', tmp.name)
            tmp.close()
        return structure

def save_structure(structure,pdb_out_path):
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(pdb_out_path)

def load_structure(path,name='tmp'):
    parser = PDBParser()
    structure = parser.get_structure(name, path)
    return structure