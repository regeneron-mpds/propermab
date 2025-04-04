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
""" Module for plotting commands for protein structures """

from IPython.core.display import display, HTML
import os
import importlib_metadata
import py3Dmol
from Bio.PDB import *
from Bio.SeqUtils import IUPACData
import tempfile
import copy

def plot_protein(pdb_path,color='b',gradient='rwb', byres=False, base='cartoon',surface=True,vmin=0.5,vmax=0.9,system='jupyter'):
    p = py3Dmol.view()
    p.addModel(open(pdb_path,'r').read(),'pdb')
    if color in ['b','q']:
        p.setStyle({'byres': byres},{base: {'colorscheme': {'prop':color,'gradient': gradient,'min':vmin,'max':vmax}}})
        if surface:
            p.addSurface(
                py3Dmol.SAS, 
                {
                    'opacity': 0.9,
                    'colorscheme': {'prop':color,'gradient': gradient,'min':vmin,'max':vmax},
                }
            )
    else:
        p.setStyle({'cartoon': {'colorscheme':{'color':'chain'}}})
    html_dis = p.zoomTo()._make_html()
    html_dis_split=html_dis.split()
    html_dis_split[5] = '1200px;'
    if system=='jupyter':
        display(HTML(html_dis))
    else:
        display_HTML(html_dis)
    
def plot_protein_w_values(structure,atom_values,color='b',base='cartoon',gradient='rwb',byres=False,surface=True,vmin=0.5,vmax=0.9,system='jupyter'):
    p = py3Dmol.view()
    
    struc = copy.deepcopy(structure)
    for i,a in enumerate(struc[0].get_atoms()):
        a.bfactor = atom_values[i]
    
    with tempfile.NamedTemporaryFile(mode = "w",delete=True) as tmp:
        pdbio = PDBIO()
        pdbio.set_structure(struc)
        pdbio.save(tmp.name)
        p.addModel(open(tmp.name,'r').read(),'pdb')
        tmp.close()
        
    if color in ['b','q']:
        p.setStyle({'byres': byres},{base: {'colorscheme': {'prop':color,'gradient': gradient,'min':vmin,'max':vmax}}})
        if surface:
            p.addSurface(
                py3Dmol.SAS, 
                {
                    'opacity': 0.9,
                    'colorscheme': {'prop':color,'gradient': gradient,'min':vmin,'max':vmax},
                }
            )
    else:
        p.setStyle({'cartoon': {'colorscheme':{'color':'chain'}}})
    html_dis = p.zoomTo()._make_html()
    html_dis_split=html_dis.split()
    html_dis_split[5] = '1200px;'
    if system=='jupyter':
        display(HTML(html_dis))
    else:
        display_HTML(html_dis)
