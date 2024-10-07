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
import anarci

# numbering schemes were based on details given on this page:
# http://www.bioinf.org.uk/abs/info.html
# The Kabat definition is based on sequence variability and is the most commonly used
KABAT_SCHEME = {
    'L1': (24, 34),
    'L2': (50, 56),
    'L3': (89, 97),
    'H1': (31, 35),
    'H2': (50, 65),
    'H3': (95, 102)
}

# The Chothia definition is based on the location of the structural loop regions
CHOTHIA_SCHEME = {
    'L1': (24, 34),
    'L2': (50, 56),
    'L3': (89, 97),
    'H1': (26, 32),
    'H2': (52, 56),
    'H3': (95, 102)
}

# https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
IMGT_SCHEME = {
    'L1': (27, 38),
    'L2': (56, 65),
    'L3': (105, 117),
    'H1': (27, 38),
    'H2': (56, 65),
    'H3': (105, 117)
}


class NumberScheme:
    def __init__(self, scheme):
        self.scheme = scheme.lower()
        assert self.scheme in ['kabat', 'chothia', 'imgt']

    def get_range(self, domain=None):
        valid_domains = ['L1', 'L2', 'L3', 'H1', 'H2', 'H3']
        if domain not in valid_domains:
            raise ValueError(f'Given domain must be one of {valid_domains}')
        if self.scheme == 'kabat':
            return KABAT_SCHEME[domain]
        if self.scheme == 'chothia':
            return CHOTHIA_SCHEME[domain]
        if self.scheme == 'imgt':
            return IMGT_SCHEME[domain]


def number_sequence(input_seq: str, scheme: str = None) -> tuple:
    numbering, chain_type = anarci.number(sequence=input_seq, scheme=scheme)
    numbered_seq_dict = {
        ''.join(str(x) for x in t).strip(): aa for t, aa in numbering
    }
    return numbered_seq_dict, chain_type


class SeqAnnotation:
    def __init__(self, seq: str, scheme: str = 'imgt'):
        self.seq = seq
        self.number_scheme = NumberScheme(scheme)
        numbered_seq_dict, chain_type = number_sequence(seq, scheme)
        self.numbered_seq_dict = numbered_seq_dict
        self.chain_type = chain_type

    def get_cdr_seq(self, cdr: str) -> str:
        start, end = self.number_scheme.get_range(domain=cdr.upper())
        return ''.join(
            self.numbered_seq_dict[str(i)] for i in range(start, end + 1)
        )
