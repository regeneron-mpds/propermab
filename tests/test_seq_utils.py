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
from typing import Union, List
import pytest

import numpy as np
from Bio import SeqIO

from propermab.sequence import seq_utils


tests_dir = os.path.dirname(os.path.realpath(__file__))


TEST_VH_SEQ = str(
    SeqIO.read(
        os.path.join(tests_dir, 'pembrolizumab_vh.fasta'), 
        format='fasta'
    ).seq
)
TEST_VL_SEQ = str(
    SeqIO.read(
        os.path.join(tests_dir, 'pembrolizumab_vl.fasta'), 
        format='fasta'
    ).seq
)


def test_get_uniprot_seq():
    seq_url = 'https://rest.uniprot.org/uniprotkb/E0CX11.fasta'
    retrieved_seq = seq_utils.get_uniprot_seq(seq_url)
    expected_seq = 'MLQFLLGFTLGNVVGMYLAQNYDIPNLAKKLEEIKKDLDAKKKPPSA'
    assert retrieved_seq == expected_seq


@pytest.mark.parametrize(
    argnames='sequence, expected_anarci_numbers',
    argvalues=[
        (
            TEST_VH_SEQ,
            [
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', 
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                '29', '30', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', 
                '59', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '74', 
                '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', 
                '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', 
                '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', 
                '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', 
                '123', '124', '125', '126', '127', '128'
            ]
        ),
        (
            TEST_VL_SEQ,
            [
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                '30', '31', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '65', '66', 
                '67', '68', '69', '70', '71', '72', '74', '75', '76', '77', '78', '79', '80', '83', 
                '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', 
                '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', 
                '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', 
                '126', '127'
            ]
        )
    ]
)

def test_get_anarci_numbers(sequence: str, expected_anarci_numbers: List[str]):
    """Test of the get_anarci_numbers() utility function.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the antibody chain.
    expected_anarci_numbers : List[str]
        Expected residues numbers as a list of strings.
    """
    anarci_numbers = seq_utils.get_anarci_numbers(sequence)
    assert anarci_numbers == expected_anarci_numbers


@pytest.mark.parametrize(
    argnames='sequence, expected_gapped_sequence',
    argvalues=[
        (
            TEST_VH_SEQ,
            'QVQLVQSGV-EVKKPGASVKVSCKASGYTF----TNYYMYWVRQAPGQGLEWMGGINPS--NGGTNFNEKFK-NRVTLTTDSSTTT'
            'AYMELKSLQFDDTAVYYCARRDYRF--------------------------DMGFDYWGQGTTVTVSS'
        ),
        (
            TEST_VL_SEQ,
            'EIVLTQSPATLSLSPGERATLSCRASKGVST--SGYSYLHWYQQKPGQAPRLLIYLA-------SYLESGVP-ARFSGSG--SGTD'
            'FTLTISSLEPEDFAVYYCQHSRD------------------------------LPLTFGGGTKVEIK-'
        )
    ]
)

def test_seq_to_gapped_seq_imgt(sequence: str, expected_gapped_sequence: str):
    """Test of the seq_to_gapped_seq_imgt() utility function.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the antibody chain.
    expected_gapped_sequence : str
        Expected gapped sequence of the antibody chain.
    """
    imgt_numbers = seq_utils.get_anarci_numbers(sequence, scheme='imgt')
    gapped_sequence = seq_utils.seq_to_gapped_seq_imgt(sequence, imgt_numbers)
    assert gapped_sequence == expected_gapped_sequence


@pytest.mark.parametrize(
    argnames='sequence, expected_encoding',
    argvalues=[
        (
            'A',
            np.array([
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ])
        ),
        (
            'C-',
            np.array([
                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ])
        )
    ]
)

def test_onehot_encode(sequence: str, expected_encoding: np.ndarray):
    """Test of the onehot_encode() utility function.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the antibody chain.
    expected_encoding : np.ndarray
        Expected onehot encoding.
    """
    encoding = seq_utils.onehot_encode(sequence)
    assert np.array_equal(encoding, expected_encoding)


@pytest.mark.parametrize(
    argnames='sequence, expected_pi',
    argvalues=[
        (
            'E',
            3.85
        ),
        (
            'APKHAY', 
            9.30
        ),
        (
            ['QVQLV', 'EIVLT'],
            4.09
        ),
    ]
)

def test_calculate_pi(sequence: Union[str, List[str]], expected_pi: float):
    """Test of the calculate_pi() utility function.

    Parameters
    ----------
    sequence : Union[str, List[str]]
        A single sequence for monomers, or a list of sequences for multimers.
    expected_pi : float
        Expected isoelectric point of the protein.
    """
    calculated_pi = seq_utils.calculate_pi(sequence)
    assert np.isclose(calculated_pi, expected_pi, rtol=0.01)


def test_calculate_seq_charge():
    """Test of the calculate_seq_charge() utility function.
    """
    sequence = 'HKRDEAC'
    charge_a = seq_utils.calculate_seq_charge(sequence)
    assert charge_a == 0

    charge_b = seq_utils.calculate_seq_charge(sequence, pH=5.0)
    assert charge_b == 1