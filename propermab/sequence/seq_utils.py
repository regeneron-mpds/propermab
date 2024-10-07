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
import io
from typing import Union, List
from urllib.request import urlopen

import numpy as np
from scipy import optimize

from anarci import anarci
from Bio import SeqIO


def get_uniprot_seq(uniprot_url: str) -> str:
    """Extract sequence from UniProt based on the given URL.

    Parameters
    ----------
    uniprot_url : str
        URL to the FASTA file of the sequence.

    Returns
    -------
    str
        Sequence extracted as a str.
    """
    with io.StringIO(urlopen(uniprot_url).read().decode()) as handle:
        seq_record = SeqIO.read(handle, format='fasta')
    return str(seq_record.seq)


def get_anarci_numbers(seq: str, scheme: str='imgt') -> list:
    """Number the given sequence using ANARCI and return the numbers in a list.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    scheme : str, optional
        Immunoprotein sequence numbering scheme, by default 'imgt'

    Returns
    -------
    list
        Residue numbers as numbered by ANARCI, in str type.
    """
    numbering, _, _ = anarci(
        [('input_seq', seq)], scheme=scheme, output=False
    )
    # sequence, domain, domain numbering
    domain_numbering = numbering[0][0][0]
    seq_numbers = []
    for res_num, res in domain_numbering:
        if res != '-':
            res_num_str = ''.join([str(x) for x in res_num]).strip()
            seq_numbers.append(res_num_str)
    return seq_numbers


def seq_to_gapped_seq_imgt(seq: str, imgt_numbers: list) -> str:
    """Insert gaps into the given amino acid sequence.
    A gap is inserted wherever there is no amino acid for an IMGT residue number.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    imgt_numbers : list
        Residue numbers of the given sequence in the IMGT numbering scheme.

    Returns
    -------
    str
        Amino acid sequence with gaps inserted.

    Raises
    ------
    ValueError
        Raised when the lengths of seq and imgt_numbers do not match.
    """
    if len(seq) != len(imgt_numbers):
        raise ValueError(
            'Inputs must be of the same length! ' 
            f'seq length: {len(seq)}, imgt_numbers length: {len(imgt_numbers)}'
        )
        
    ordered_allowed_imgt = [
        # residues 1 through 111, inclusive
        '{:d}'.format(i) for i in range(1, 112) 
    ] + [
        # residues 111A through 111M, inclusive
        '111' + chr(65 + i) for i in range(13)
    ] + [
        # residues 112M through 112A inclusive
        '112' + chr(65 + i) for i in range(13)
    ][::-1] + [
        # residues 112 through 128 inclusive
        '{:d}'.format(i) for i in range(112, 129)
    ]

    imgt_dict = {name: idx for idx, name in enumerate(ordered_allowed_imgt)}
    out_seq = ['-'] * len(imgt_dict)
    for aa, imgt_n in zip(seq, imgt_numbers):
        out_seq[imgt_dict[imgt_n]] = aa

    return ''.join(out_seq)


def onehot_encode(gapped_seq: str, flatten: bool=False) -> np.ndarray:
    """Onehot-encode the given sequence.

    Parameters
    ----------
    gapped_seq : str
        Amino acid sequence.
    flatten : bool, optional
        Whether to flatten the one-hot matrix into a vector, by default False.

    Returns
    -------
    np.ndarray
        A vector if flatten is True, otherwise a matrix.
    """
    aa_to_idx = {
        aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY' + '-')
    }
    
    one_hot = np.zeros(shape=(len(gapped_seq), 21))
    for i, aa in enumerate(gapped_seq):
        one_hot[i, aa_to_idx[aa]] = 1.
    
    if flatten:
        # by default the order is 'C', i.e. row-major
        return one_hot.flatten()
    return one_hot



# Based on EMBOSS entry from this link: http://isoelectric.org/theory.html
pKa_dict = {
    'N_term': (8.6, 1),
    'C_term': (3.6, -1),
    'D': (3.9, -1),
    'E': (4.1, -1),
    'H': (6.5, 1),
    'Y': (10.1, -1),
    'K': (10.8, 1),
    'R': (12.5, 1)
}


def calculate_pi(seq: Union[str, List[str]]) -> float:
    """Calculates the theoretical pI based on the given sequence.

    Based on formulas from this link: http://isoelectric.org/theory.html

    Based on `scipy.optimize.minimize_scalar`. It is more robust than a grid search for
    the pH value that results in a net charge within a certain tolerance around 0. In testing,
    this implmentation fixed all four cases where the grid search-based method failed.

    Note that for antibodies you can either pass sequences of the four subunits as a list
    of length four, or concatenate the four sequences into a single string and pass the
    concatenated string.

    Parameters
    ----------
    seq : Union[str, List[str]]
        Amino acid sequence of the protein. Can be a single string for monomers or a list
        of sequences for multimers.

    Returns
    -------
    float
        Theoretical isoelectric point (pI) of the antibody, calucated based on the sequence.
        Returns np.nan if scipy.optimize.minimize_scalar() failed.
    """
    if isinstance(seq, str):
        N_term_count = 1
        C_term_count = 1
        full_seq = seq
    else:
        N_term_count = len(seq)
        C_term_count = len(seq)
        full_seq = ''.join(seq)

    aa_counts = {
        'N_term': N_term_count,
        'C_term': C_term_count,
        'D': 0,
        'E': 0,
        'H': 0,
        'Y': 0,
        'K': 0,
        'R': 0
    }

    for aa in full_seq:
        if aa in aa_counts:
            aa_counts[aa] += 1

    def net_charge(x):
        neg_charge = 0
        pos_charge = 0
        for aa in pKa_dict.keys():
            aa_pKa, aa_charge = pKa_dict[aa]
            aa_count = aa_counts[aa]
            if aa_charge < 0:
                neg_charge += aa_count * (-1 / (1 + np.power(10, aa_pKa - x)))
            else:
                pos_charge += aa_count * (1 / (1 + np.power(10, x - aa_pKa)))
        net_charge = neg_charge + pos_charge
        return net_charge ** 2

    # pH, especially protein pI, rarely falls outside of the (0, 14) range
    opt_results = optimize.minimize_scalar(net_charge, bounds=(0, 14))

    if opt_results.success:
        return opt_results.x
    else:
        return np.nan


def extract_fv_seq(seq: str, scheme: str='imgt') -> str:
    """Uses ANARCI to extract sequence of the Fv domain.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    scheme : str, optional
        Sequence numbering scheme, by default 'imgt'

    Returns
    -------
    str
        Extracted sequence of the Fv domain.
    """
    numberings, _, _ = anarci(
        [('tmp', seq)], scheme=scheme, output=False
    )
    numbering = numberings[0]
    fv_seq = ''.join([a for _, a in numbering[0][0] if a != '-'])
    return fv_seq


def calculate_seq_charge(seq: str, pH: float=7.4) -> float:
    """Calculates the charge at the given pH based on sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    pH : float, optional
        pH condition, by default 7.4

    Returns
    -------
    float
        Charge at the given pH.

    """
    if pH < pKa_dict['H'][0]:
        return seq.count('H') + seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
    else:
        return seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
