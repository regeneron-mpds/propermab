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
import numpy as np
from scipy import spatial


# normalized hydrophobicity scales
# extracted from https://pubmed.ncbi.nlm.nih.gov/36120542/ Table 2
HYD_SCALES = {
    'BM': {
        'ALA': 0.37,
        'ARG': -1.52,
        'ASN': -0.79,
        'ASP': -1.43,
        'CYS': 0.55,
        'GLN': -0.76,
        'GLU': -1.4,
        'GLY': 0.0,
        'HIS': -1.0,
        'ILE': 1.34,
        'LEU': 1.34,
        'LYS': -0.67,
        'MET': 0.73,
        'PHE': 1.52,
        'PRO': 0.64,
        'SER': -0.43,
        'THR': -0.15,
        'TRP': 1.16,
        'TYR': 1.16,
        'VAL': 1.0
    },
    'BR': {
        'ALA': 0.75,
        'ARG': -0.02,
        'ASN': -0.16,
        'ASP': -0.5,
        'CYS': 2.6,
        'GLN': -0.11,
        'GLU': -0.54,
        'GLY': 0.0,
        'HIS': 0.57,
        'ILE': 2.19,
        'LEU': 1.97,
        'LYS': -0.9,
        'MET': 1.22,
        'PHE': 1.92,
        'PRO': 0.72,
        'SER': 0.11,
        'THR': 0.47,
        'TRP': 1.51,
        'TYR': 1.36,
        'VAL': 1.88
    },
    'EI': {
        'ALA': 0.15,
        'ARG': -3.09,
        'ASN': -1.29,
        'ASP': -1.42,
        'CYS': -0.19,
        'GLN': -1.37,
        'GLU': -1.26,
        'GLY': 0.0,
        'HIS': -0.9,
        'ILE': 0.92,
        'LEU': 0.6,
        'LYS': -2.03,
        'MET': 0.16,
        'PHE': 0.73,
        'PRO': -0.37,
        'SER': -0.68,
        'THR': -0.55,
        'TRP': 0.34,
        'TYR': -0.23,
        'VAL': 0.61
    },
    'JA': {
        'ALA': 0.06,
        'ARG': -0.32,
        'ASN': 0.13,
        'ASP': -0.43,
        'CYS': 0.55,
        'GLN': 0.46,
        'GLU': -0.72,
        'GLY': 0.0,
        'HIS': 0.03,
        'ILE': 1.54,
        'LEU': 1.54,
        'LYS': -1.07,
        'MET': 0.49,
        'PHE': 2.48,
        'PRO': 0.44,
        'SER': 0.07,
        'THR': 0.16,
        'TRP': 2.81,
        'TYR': 1.84,
        'VAL': 0.97
    },
    'KD': {
        'ALA': 0.76,
        'ARG': -1.41,
        'ASN': -1.06,
        'ASP': -1.06,
        'CYS': 1.0,
        'GLN': -1.06,
        'GLU': -1.06,
        'GLY': 0.0,
        'HIS': -0.96,
        'ILE': 1.68,
        'LEU': 1.44,
        'LYS': -1.2,
        'MET': 0.79,
        'PHE': 1.1,
        'PRO': -0.41,
        'SER': -0.14,
        'THR': -0.1,
        'TRP': -0.17,
        'TYR': -0.31,
        'VAL': 1.58
    },
    'ME': {
        'ALA': 0.07,
        'ARG': 0.11,
        'ASN': 0.11,
        'ASP': -1.08,
        'CYS': -0.9,
        'GLN': -0.63,
        'GLU': -2.23,
        'GLY': 0.0,
        'HIS': -0.46,
        'ILE': 1.83,
        'LEU': 1.16,
        'LYS': 0.01,
        'MET': 0.63,
        'PHE': 1.74,
        'PRO': 0.8,
        'SER': 0.16,
        'THR': 0.36,
        'TRP': 1.96,
        'TYR': 0.8,
        'VAL': 0.36
    },
    'MI': {
        'ALA': 0.4,
        'ARG': -0.15,
        'ASN': -0.37,
        'ASP': -0.43,
        'CYS': 1.65,
        'GLN': -0.29,
        'GLU': -0.4,
        'GLY': 0.0,
        'HIS': 0.3,
        'ILE': 2.08,
        'LEU': 1.91,
        'LYS': -0.74,
        'MET': 2.14,
        'PHE': 2.18,
        'PRO': -0.29,
        'SER': -0.19,
        'THR': 0.0,
        'TRP': 1.52,
        'TYR': 0.68,
        'VAL': 1.51
    },
    'RO': {
        'ALA': 0.18,
        'ARG': -0.71,
        'ASN': -0.8,
        'ASP': -0.89,
        'CYS': 1.69,
        'GLN': -0.89,
        'GLU': -0.89,
        'GLY': 0.0,
        'HIS': 0.53,
        'ILE': 1.42,
        'LEU': 1.16,
        'LYS': -1.78,
        'MET': 1.16,
        'PHE': 1.42,
        'PRO': -0.71,
        'SER': -0.53,
        'THR': -0.18,
        'TRP': 1.16,
        'TYR': 0.36,
        'VAL': 1.25
    },
    'WW': {
        'ALA': -0.2,
        'ARG': -0.41,
        'ASN': -0.51,
        'ASP': -1.53,
        'CYS': 0.31,
        'GLN': -0.71,
        'GLU': -2.51,
        'GLY': 0.0,
        'HIS': -0.2,
        'ILE': 0.35,
        'LEU': 0.71,
        'LYS': -0.59,
        'MET': 0.3,
        'PHE': 1.43,
        'PRO': -0.55,
        'SER': -0.15,
        'THR': -0.16,
        'TRP': 2.33,
        'TYR': 1.19,
        'VAL': -0.08
    }
}


# Kyte Doolittle scale
KD_SCALE = {
    'ILE':  4.5,
    'VAL':  4.2,
    'LEU':  3.8,
    'PHE':  2.8,
    'CYS':  2.5,
    'MET':  1.9,
    'ALA':  1.8,
    'GLY': -0.4,
    'THR': -0.7,
    'SER': -0.8,
    'TRP': -0.9,
    'TYR': -1.3,
    'PRO': -1.6,
    'HIS': -3.2,
    'GLU': -3.5,
    'GLN': -3.5,
    'ASP': -3.5,
    'ASN': -3.5,
    'LYS': -3.9,
    'ARG': -4.5
}

EISENBERG_SCALE = {
    'ILE':  1.38,
    'PHE':  1.19,
    'VAL':  1.08,
    'LEU':  1.06,
    'TRP':  0.81,
    'MET':  0.64,
    'ALA':  0.62,
    'GLY':  0.48,
    'CYS':  0.29,
    'TYR':  0.26,
    'PRO':  0.12,
    'THR': -0.05,
    'SER': -0.18,
    'HIS': -0.40,
    'GLE': -0.74,
    'ASN': -0.78,
    'GLN': -0.85,
    'ASP': -0.90,
    'LYS': -1.50,
    'ARG': -2.53
}

# adapted from https://github.com/liedllab/surface_analyses/blob/main/surface_analyses/data.py
# (log_p, molecular refractivity)
CRIPPEN_PARAMS = {
    "Br": (0.8456, 8.927),
    "C1": (0.1441, 2.503),
    "C10": (-0.0516, 2.488),
    "C11": (0.1193, 2.582),
    "C12": (-0.0967, 2.576),
    "C13": (-0.5443, 4.041),
    "C14": (0.0, 3.257),
    "C15": (0.245, 3.564),
    "C16": (0.198, 3.18),
    "C17": (0.0, 3.104),
    "C18": (0.1581, 3.35),
    "C19": (0.2955, 4.346),
    "C2": (0.0, 2.433),
    "C20": (0.2713, 3.904),
    "C21": (0.136, 3.509),
    "C22": (0.4619, 4.067),
    "C23": (0.5437, 3.853),
    "C24": (0.1893, 2.673),
    "C25": (-0.8186, 3.135),
    "C26": (0.264, 4.305),
    "C27": (0.2148, 2.693),
    "C3": (-0.2035, 2.753),
    "C4": (-0.2051, 2.731),
    "C5": (-0.2783, 5.007),
    "C6": (0.1551, 3.513),
    "C7": (0.0017, 3.888),
    "C8": (0.08452, 2.464),
    "C9": (-0.1444, 2.412),
    "CS": (0.08129, 3.243),
    "Cl": (0.6895, 5.853),
    "F": (0.4202, 1.108),
    "H1": (0.123, 1.057),
    "H2": (-0.2677, 1.395),
    "H3": (0.2142, 0.9627),
    "H4": (0.298, 1.805),
    "HS": (0.1125, 1.112),
    "Hal": (-2.996, np.nan),
    "I": (0.8857, 14.02),
    "Me1": (-0.3808, 5.754),
    "Me2": (-0.0025, np.nan),
    "N1": (-1.019, 2.262),
    "N10": (-1.95, np.nan),
    "N11": (-0.3239, 2.202),
    "N12": (-1.119, np.nan),
    "N13": (-0.3396, 0.2604),
    "N14": (0.2887, 3.359),
    "N2": (-0.7096, 2.173),
    "N3": (-1.027, 2.827),
    "N4": (-0.5188, 3.0),
    "N5": (0.08387, 1.757),
    "N6": (0.1836, 2.428),
    "N7": (-0.3187, 1.839),
    "N8": (-0.4458, 2.819),
    "N9": (0.01508, 1.725),
    "NS": (-0.4806, 2.134),
    "O1": (0.1552, 1.08),
    "O10": (0.1129, 0.2215),
    "O11": (0.4833, 0.389),
    "O12": (-1.326, np.nan),
    "O2": (-0.2893, 0.8238),
    "O3": (-0.0684, 1.085),
    "O4": (-0.4195, 1.182),
    "O5": (0.0335, 3.367),
    "O6": (-0.3339, 0.7774),
    "O7": (-1.189, 0.0),
    "O8": (0.1788, 3.135),
    "O9": (-0.1526, 0.0),
    "OS": (-0.1188, 0.6865),
    "P": (0.8612, 6.92),
    "S1": (0.6482, 7.591),
    "S2": (-0.0024, 7.365),
    "S3": (0.6237, 6.691),
}

# adapted from https://github.com/liedllab/surface_analyses/blob/main/surface_analyses/data.py
PDB_TO_CRIPPEN = {
    ('ALA', 'C'): 'C5',
    ('ALA', 'CA'): 'C4',
    ('ALA', 'CB'): 'C1',
    ('ALA', 'H'): 'H3',
    ('ALA', 'HA'): 'H1',
    ('ALA', 'HB1'): 'H1',
    ('ALA', 'HB2'): 'H1',
    ('ALA', 'HB3'): 'H1',
    ('ALA', 'N'): 'N2',
    ('ALA', 'O'): 'O9',
    ('ARG', 'C'): 'C5',
    ('ARG', 'CA'): 'C4',
    ('ARG', 'CB'): 'C1',
    ('ARG', 'CD'): 'C3',
    ('ARG', 'CG'): 'C1',
    ('ARG', 'CZ'): 'C5',
    ('ARG', 'H'): 'H3',
    ('ARG', 'HA'): 'H1',
    ('ARG', 'HB2'): 'H1',
    ('ARG', 'HB3'): 'H1',
    ('ARG', 'HD2'): 'H1',
    ('ARG', 'HD3'): 'H1',
    ('ARG', 'HE'): 'H3',
    ('ARG', 'HG2'): 'H1',
    ('ARG', 'HG3'): 'H1',
    ('ARG', 'HH11'): 'H3',
    ('ARG', 'HH12'): 'H3',
    ('ARG', 'HH21'): 'H3',
    ('ARG', 'HH22'): 'H3',
    ('ARG', 'N'): 'N2',
    ('ARG', 'NE'): 'N2',
    ('ARG', 'NH1'): 'N10',
    ('ARG', 'NH2'): 'N1',
    ('ARG', 'O'): 'O9',
    ('ASN', 'C'): 'C5',
    ('ASN', 'CA'): 'C4',
    ('ASN', 'CB'): 'C1',
    ('ASN', 'CG'): 'C5',
    ('ASN', 'H'): 'H3',
    ('ASN', 'HA'): 'H1',
    ('ASN', 'HB2'): 'H1',
    ('ASN', 'HB3'): 'H1',
    ('ASN', 'HD21'): 'H3',
    ('ASN', 'HD22'): 'H3',
    ('ASN', 'N'): 'N2',
    ('ASN', 'ND2'): 'N1',
    ('ASN', 'O'): 'O9',
    ('ASN', 'OD1'): 'O9',
    ('ASP', 'C'): 'C5',
    ('ASP', 'CA'): 'C4',
    ('ASP', 'CB'): 'C1',
    ('ASP', 'CG'): 'C5',
    ('ASP', 'H'): 'H3',
    ('ASP', 'HA'): 'H1',
    ('ASP', 'HB2'): 'H1',
    ('ASP', 'HB3'): 'H1',
    ('ASP', 'N'): 'N2',
    ('ASP', 'O'): 'O9',
    ('ASP', 'OD1'): 'O9',
    ('ASP', 'OD2'): 'O12',
    ('CYS', 'C'): 'C5',
    ('CYS', 'CA'): 'C4',
    ('CYS', 'CB'): 'C3',
    ('CYS', 'H'): 'H3',
    ('CYS', 'HA'): 'H1',
    ('CYS', 'HB2'): 'H1',
    ('CYS', 'HB3'): 'H1',
    ('CYS', 'HG'): 'H2',
    ('CYS', 'N'): 'N2',
    ('CYS', 'O'): 'O9',
    ('CYS', 'SG'): 'S1',
    ('GLN', 'C'): 'C5',
    ('GLN', 'CA'): 'C4',
    ('GLN', 'CB'): 'C1',
    ('GLN', 'CD'): 'C5',
    ('GLN', 'CG'): 'C1',
    ('GLN', 'H'): 'H3',
    ('GLN', 'HA'): 'H1',
    ('GLN', 'HB2'): 'H1',
    ('GLN', 'HB3'): 'H1',
    ('GLN', 'HE21'): 'H3',
    ('GLN', 'HE22'): 'H3',
    ('GLN', 'HG2'): 'H1',
    ('GLN', 'HG3'): 'H1',
    ('GLN', 'N'): 'N2',
    ('GLN', 'NE2'): 'N1',
    ('GLN', 'O'): 'O9',
    ('GLN', 'OE1'): 'O9',
    ('GLU', 'C'): 'C5',
    ('GLU', 'CA'): 'C4',
    ('GLU', 'CB'): 'C1',
    ('GLU', 'CD'): 'C5',
    ('GLU', 'CG'): 'C1',
    ('GLU', 'H'): 'H3',
    ('GLU', 'HA'): 'H1',
    ('GLU', 'HB2'): 'H1',
    ('GLU', 'HB3'): 'H1',
    ('GLU', 'HG2'): 'H1',
    ('GLU', 'HG3'): 'H1',
    ('GLU', 'N'): 'N2',
    ('GLU', 'O'): 'O9',
    ('GLU', 'OE1'): 'O9',
    ('GLU', 'OE2'): 'O12',
    ('GLY', 'C'): 'C5',
    ('GLY', 'CA'): 'C3',
    ('GLY', 'H'): 'H3',
    ('GLY', 'HA2'): 'H1',
    ('GLY', 'HA3'): 'H1',
    ('GLY', 'N'): 'N2',
    ('GLY', 'O'): 'O9',
    ('HIS', 'C'): 'C5',
    ('HIS', 'CA'): 'C4',
    ('HIS', 'CB'): 'C10',
    ('HIS', 'CD2'): 'C18',
    ('HIS', 'CE1'): 'C18',
    ('HIS', 'CG'): 'C21',
    ('HIS', 'H'): 'H3',
    ('HIS', 'HA'): 'H1',
    ('HIS', 'HB2'): 'H1',
    ('HIS', 'HB3'): 'H1',
    ('HIS', 'HD1'): 'H3',
    ('HIS', 'HD2'): 'H1',
    ('HIS', 'HE1'): 'H1',
    ('HIS', 'HE2'): 'H3',
    ('HIS', 'N'): 'N2',
    ('HIS', 'ND1'): 'N11',
    ('HIS', 'NE2'): 'N11',
    ('HIS', 'O'): 'O9',
    ('ILE', 'C'): 'C5',
    ('ILE', 'CA'): 'C4',
    ('ILE', 'CB'): 'C2',
    ('ILE', 'CD1'): 'C1',
    ('ILE', 'CG1'): 'C1',
    ('ILE', 'CG2'): 'C1',
    ('ILE', 'H'): 'H3',
    ('ILE', 'HA'): 'H1',
    ('ILE', 'HB'): 'H1',
    ('ILE', 'HD11'): 'H1',
    ('ILE', 'HD12'): 'H1',
    ('ILE', 'HD13'): 'H1',
    ('ILE', 'HG12'): 'H1',
    ('ILE', 'HG13'): 'H1',
    ('ILE', 'HG21'): 'H1',
    ('ILE', 'HG22'): 'H1',
    ('ILE', 'HG23'): 'H1',
    ('ILE', 'N'): 'N2',
    ('ILE', 'O'): 'O9',
    ('LEU', 'C'): 'C5',
    ('LEU', 'CA'): 'C4',
    ('LEU', 'CB'): 'C1',
    ('LEU', 'CD1'): 'C1',
    ('LEU', 'CD2'): 'C1',
    ('LEU', 'CG'): 'C2',
    ('LEU', 'H'): 'H3',
    ('LEU', 'HA'): 'H1',
    ('LEU', 'HB2'): 'H1',
    ('LEU', 'HB3'): 'H1',
    ('LEU', 'HD11'): 'H1',
    ('LEU', 'HD12'): 'H1',
    ('LEU', 'HD13'): 'H1',
    ('LEU', 'HD21'): 'H1',
    ('LEU', 'HD22'): 'H1',
    ('LEU', 'HD23'): 'H1',
    ('LEU', 'HG'): 'H1',
    ('LEU', 'N'): 'N2',
    ('LEU', 'O'): 'O9',
    ('LYS', 'C'): 'C5',
    ('LYS', 'CA'): 'C4',
    ('LYS', 'CB'): 'C1',
    ('LYS', 'CD'): 'C1',
    ('LYS', 'CE'): 'C3',
    ('LYS', 'CG'): 'C1',
    ('LYS', 'H'): 'H3',
    ('LYS', 'HA'): 'H1',
    ('LYS', 'HB2'): 'H1',
    ('LYS', 'HB3'): 'H1',
    ('LYS', 'HD2'): 'H1',
    ('LYS', 'HD3'): 'H1',
    ('LYS', 'HE2'): 'H1',
    ('LYS', 'HE3'): 'H1',
    ('LYS', 'HG2'): 'H1',
    ('LYS', 'HG3'): 'H1',
    ('LYS', 'HZ1'): 'H3',
    ('LYS', 'HZ2'): 'H3',
    ('LYS', 'HZ3'): 'H3',
    ('LYS', 'N'): 'N2',
    ('LYS', 'NZ'): 'N10',
    ('LYS', 'O'): 'O9',
    ('MET', 'C'): 'C5',
    ('MET', 'CA'): 'C4',
    ('MET', 'CB'): 'C1',
    ('MET', 'CE'): 'C3',
    ('MET', 'CG'): 'C3',
    ('MET', 'H'): 'H3',
    ('MET', 'HA'): 'H1',
    ('MET', 'HB2'): 'H1',
    ('MET', 'HB3'): 'H1',
    ('MET', 'HE1'): 'H1',
    ('MET', 'HE2'): 'H1',
    ('MET', 'HE3'): 'H1',
    ('MET', 'HG2'): 'H1',
    ('MET', 'HG3'): 'H1',
    ('MET', 'N'): 'N2',
    ('MET', 'O'): 'O9',
    ('MET', 'SD'): 'S1',
    ('PHE', 'C'): 'C5',
    ('PHE', 'CA'): 'C4',
    ('PHE', 'CB'): 'C10',
    ('PHE', 'CD1'): 'C18',
    ('PHE', 'CD2'): 'C18',
    ('PHE', 'CE1'): 'C18',
    ('PHE', 'CE2'): 'C18',
    ('PHE', 'CG'): 'C21',
    ('PHE', 'CZ'): 'C18',
    ('PHE', 'H'): 'H3',
    ('PHE', 'HA'): 'H1',
    ('PHE', 'HB2'): 'H1',
    ('PHE', 'HB3'): 'H1',
    ('PHE', 'HD1'): 'H1',
    ('PHE', 'HD2'): 'H1',
    ('PHE', 'HE1'): 'H1',
    ('PHE', 'HE2'): 'H1',
    ('PHE', 'HZ'): 'H1',
    ('PHE', 'N'): 'N2',
    ('PHE', 'O'): 'O9',
    ('PRO', 'C'): 'C5',
    ('PRO', 'CA'): 'C4',
    ('PRO', 'CB'): 'C1',
    ('PRO', 'CD'): 'C3',
    ('PRO', 'CG'): 'C1',
    ('PRO', 'HA'): 'H1',
    ('PRO', 'HB2'): 'H1',
    ('PRO', 'HB3'): 'H1',
    ('PRO', 'HD2'): 'H1',
    ('PRO', 'HD3'): 'H1',
    ('PRO', 'HG2'): 'H1',
    ('PRO', 'HG3'): 'H1',
    ('PRO', 'N'): 'N7',
    ('PRO', 'O'): 'O9',
    ('SER', 'C'): 'C5',
    ('SER', 'CA'): 'C4',
    ('SER', 'CB'): 'C3',
    ('SER', 'H'): 'H3',
    ('SER', 'HA'): 'H1',
    ('SER', 'HB2'): 'H1',
    ('SER', 'HB3'): 'H1',
    ('SER', 'HG'): 'H2',
    ('SER', 'N'): 'N2',
    ('SER', 'O'): 'O9',
    ('SER', 'OG'): 'O2',
    ('THR', 'C'): 'C5',
    ('THR', 'CA'): 'C4',
    ('THR', 'CB'): 'C4',
    ('THR', 'CG2'): 'C1',
    ('THR', 'H'): 'H3',
    ('THR', 'HA'): 'H1',
    ('THR', 'HB'): 'H1',
    ('THR', 'HG1'): 'H2',
    ('THR', 'HG21'): 'H1',
    ('THR', 'HG22'): 'H1',
    ('THR', 'HG23'): 'H1',
    ('THR', 'N'): 'N2',
    ('THR', 'O'): 'O9',
    ('THR', 'OG1'): 'O2',
    ('TRP', 'C'): 'C5',
    ('TRP', 'CA'): 'C4',
    ('TRP', 'CB'): 'C10',
    ('TRP', 'CD1'): 'C18',
    ('TRP', 'CD2'): 'C19',
    ('TRP', 'CE2'): 'C19',
    ('TRP', 'CE3'): 'C18',
    ('TRP', 'CG'): 'C21',
    ('TRP', 'CH2'): 'C18',
    ('TRP', 'CZ2'): 'C18',
    ('TRP', 'CZ3'): 'C18',
    ('TRP', 'H'): 'H3',
    ('TRP', 'HA'): 'H1',
    ('TRP', 'HB2'): 'H1',
    ('TRP', 'HB3'): 'H1',
    ('TRP', 'HD1'): 'H1',
    ('TRP', 'HE1'): 'H3',
    ('TRP', 'HE3'): 'H1',
    ('TRP', 'HH2'): 'H1',
    ('TRP', 'HZ2'): 'H1',
    ('TRP', 'HZ3'): 'H1',
    ('TRP', 'N'): 'N2',
    ('TRP', 'NE1'): 'N11',
    ('TRP', 'O'): 'O9',
    ('TYR', 'C'): 'C5',
    ('TYR', 'CA'): 'C4',
    ('TYR', 'CB'): 'C10',
    ('TYR', 'CD1'): 'C18',
    ('TYR', 'CD2'): 'C18',
    ('TYR', 'CE1'): 'C18',
    ('TYR', 'CE2'): 'C18',
    ('TYR', 'CG'): 'C21',
    ('TYR', 'CZ'): 'C23',
    ('TYR', 'H'): 'H3',
    ('TYR', 'HA'): 'H1',
    ('TYR', 'HB2'): 'H1',
    ('TYR', 'HB3'): 'H1',
    ('TYR', 'HD1'): 'H1',
    ('TYR', 'HD2'): 'H1',
    ('TYR', 'HE1'): 'H1',
    ('TYR', 'HE2'): 'H1',
    ('TYR', 'HH'): 'H2',
    ('TYR', 'N'): 'N2',
    ('TYR', 'O'): 'O9',
    ('TYR', 'OH'): 'O2',
    ('VAL', 'C'): 'C5',
    ('VAL', 'CA'): 'C4',
    ('VAL', 'CB'): 'C2',
    ('VAL', 'CG1'): 'C1',
    ('VAL', 'CG2'): 'C1',
    ('VAL', 'H'): 'H3',
    ('VAL', 'HA'): 'H1',
    ('VAL', 'HB'): 'H1',
    ('VAL', 'HG11'): 'H1',
    ('VAL', 'HG12'): 'H1',
    ('VAL', 'HG13'): 'H1',
    ('VAL', 'HG21'): 'H1',
    ('VAL', 'HG22'): 'H1',
    ('VAL', 'HG23'): 'H1',
    ('VAL', 'N'): 'N2',
    ('VAL', 'O'): 'O9'
}


class HeidenHydrophobicPotential:
    def __init__(self, centers, log_p, r_cutoff=5., alpha=1.5):
        """Abstraction of objects for evaluating the hydrophobic potential
        at surface vertices.

        References
        https://link.springer.com/article/10.1007/BF00124359

        Parameters
        ----------
        centers : list or np.ndarray
            Usually Cartesian coordinates of atoms.
        log_p : list or np.ndarray
            The log_p value of the atoms.
        r_cutoff : float
            Twice the value at which the weight is 0.5
        alpha : float
            Controls the steepness of the Fermi function.
            Larger alpha makes the curve steeper.

        """
        self.centers = np.asarray(centers)
        self.log_p = np.asarray(log_p)
        assert len(self.centers) == len(self.log_p)
        self.r_cutoff = r_cutoff
        self.alpha = alpha

    def evaluate(self, points: np.ndarray):
        """Evaluates the hydrophobic potential at each of the given points.

        Parameters
        ----------
        points : np.ndarray
            Points for which hydrophobic potential are to be evaluated.

        Returns
        -------
        np.ndarray
            The hydrophobic potential at each of the given points stored in a
            1D array.

        """
        points = np.asarray(points)
        kd_tree = spatial.KDTree(self.centers)
        values = []
        for point, neighbor_idx in zip(points, kd_tree.query_ball_point(points, self.r_cutoff)):
            dist_to_neighbors = np.linalg.norm(point - self.centers[neighbor_idx], axis=1)
            neighbor_logp = self.log_p[neighbor_idx]
            neighbor_weights = self.heiden_weight(dist_to_neighbors)
            # neighbor_weights = self.heiden_weight(dist_to_neighbors) / np.sum(self.heiden_weight(dist_to_neighbors))
            hyd_potential = np.average(neighbor_logp, weights=neighbor_weights)
            values.append(hyd_potential)
        return np.array(values)

    def heiden_weight(self, r):
        """Weighting function by Heiden et al. 1993

        References
        https://link.springer.com/article/10.1007/BF00124359

        This is a Fermi function with a scaling factor such that g(0) is
        always 1, and g(r_cut/2) is 0.5. Alpha controls how hard the cutoff
        is (the higher the steeper)

        Parameters:
            r: np.ndarray
                Distances to compute weights for.
        """
        numerator = np.exp(-self.alpha * self.r_cutoff / 2) + 1
        denominator = np.exp(self.alpha * (r - self.r_cutoff / 2)) + 1
        return numerator / denominator
        # return 1 / denominator
