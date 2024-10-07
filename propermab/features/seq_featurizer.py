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

from propermab.sequence import seq_utils


# get_uniprot_seq() gets called only when HC_SEQS is first created
HC_SEQS = {
    # UniProt P01857-2 is the canonical one
    'IGHG1': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P01857.fasta'),
    # UniProt P01859-2 is the canonical one
    'IGHG2': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P01859.fasta'),
    # UniProt P01860-2 is the canonical one
    'IGHG3': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P01860.fasta'),
    # UniProt P01861-2 is the canonical one
    'IGHG4': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P01861.fasta')
}

LC_SEQS = {
    'IGKC': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P01834.fasta'),
    'IGLC1': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P0CG04.fasta'),
    'IGLC2': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P0DOY2.fasta'),
    'IGLC3': seq_utils.get_uniprot_seq('https://rest.uniprot.org/uniprotkb/P0DOY3.fasta')
}

HINGE_REGIONS = {
    # https://www.uniprot.org/uniprotkb/P01857/entry#family_and_domains
    'IGHG1': (99, 110),
    # https://www.uniprot.org/uniprotkb/P01859/entry#family_and_domains
    'IGHG2': (99, 110),
    # https://www.uniprot.org/uniprotkb/P01860/entry#family_and_domains
    'IGHG3': (99, 160),
    # https://www.uniprot.org/uniprotkb/P01861/entry#family_and_domains
    'IGHG4': (99, 110)
}


class SeqFeaturizer:
    """This class exposes the API for calculating sequence-derived features.
    """
    def __init__(self, seqs: tuple, is_fv: bool, isotype: str, lc_type: str, pH: float=7.4) -> None:
        """Constructor of SeqFeaturizer, which exposes the API for calculating sequence-derived 
        features.

        Parameters
        ----------
        seqs : tuple
            A pair of heavy and light chain sequences.
        is_fv : bool
            Whether the given pair of sequences are Fv domain only.
        isotype : str
            Isotype of the heavy chain.
        lc_type : str
            Type of the light chain.
        pH : float, optional
            The pH condiation at which to calculate charge related features, by default 7.4

        """
        self.seqs = seqs
        self.isotype = isotype
        self.lc_type = lc_type
        self.pH = pH

        if is_fv:
            self.vh_seq, self.vl_seq = seqs

            # create full sequence for the heavy chain
            if self.isotype in HC_SEQS:
                self.h_full_seq = self.vh_seq + HC_SEQS[self.isotype]
            elif self.isotype.lower() == 'igg1':
                self.h_full_seq = self.vh_seq + HC_SEQS['IGHG1']
            elif self.isotype.lower() == 'igg2':
                self.h_full_seq = self.vh_seq + HC_SEQS['IGHG2']
            elif self.isotype.lower() == 'igg4':
                self.h_full_seq =  self.vh_seq + HC_SEQS['IGHG4']
            else:
                raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')

            # create full sequence for the light chain
            if self.lc_type in LC_SEQS:
                self.l_full_seq = self.vl_seq + LC_SEQS[self.lc_type]
            elif self.lc_type.lower() == 'kappa':
                self.l_full_seq = self.vl_seq + LC_SEQS['IGKC']
            elif self.lc_type.lower() == 'lambda':
                self.l_full_seq = self.vl_seq + LC_SEQS['IGLC2']
            else:
                raise ValueError(f'Unknown light chain type {self.lc_type}!')
        else:
            self.h_full_seq, self.l_full_seq = seqs
            self.vh_seq = seq_utils.extract_fv_seq(self.h_full_seq)
            self.vl_seq = seq_utils.extract_fv_seq(self.l_full_seq)

        # pH aware charged residues
        if self.pH < 4.0 or self.pH > 10.0:
            raise ValueError(
                f'Invalid pH value {self.pH}!'
                'Please choose a pH in [4.0, 10.0]. Values outside this range may lead to' 
                'incorrect features, especially those related to charges.'
            )
        if self.pH < seq_utils.pKa_dict['H'][0]:
            self.charged_aas = 'DEHKR'
        else:
            self.charged_aas = 'DEKR'
 
    def n_charged_res(self) -> int:
        """Counts the number of charged residues for one pair of heavy and light chains.

        Returns
        -------
        int
            Number of charged residues.
        """
        return np.sum([
            aa in self.charged_aas for aa in self.h_full_seq + self.l_full_seq
        ])

    def n_charged_res_fv(self) -> int:
        """Counts the number of charged residues in the Fv domain (only one arm).

        Returns
        -------
        int
            Number of charged residues in the Fv domain.
        """
        return np.sum([
            aa in self.charged_aas for aa in self.vh_seq + self.vl_seq
        ])

    def vh_charge(self) -> float:
        """Calculates the charge of the VH domain.

        Returns
        -------
        float
            The charge of the VH domain
        """
        return seq_utils.calculate_seq_charge(self.vh_seq, self.pH)

    def vl_charge(self) -> float:
        """Calculates the charge of the VL domain.

        Returns
        -------
        float
            The charge of the VL domain.
        """
        return seq_utils.calculate_seq_charge(self.vl_seq, self.pH)

    def fv_charge(self) -> float:
        """Calcualtes the charge of the Fv domain.

        Returns
        -------
        float
            The charge of the Fv domain.
        """
        return self.vh_charge() + self.vl_charge()

    def fv_csp(self) -> float:
        """Calculates the charge separation of the Fv domain.

        Returns
        -------
        float
            The charge separation of the Fv domain.
        """
        return self.vh_charge() * self.vl_charge()

    def theoretical_pi(self) -> float:
        """Calculates the theoretical pI of the antibody (full sequence, both arms)

        Returns
        -------
        float
            The theoretical pI of the antibody.
        """
        return seq_utils.calculate_pi([self.h_full_seq] * 2 + [self.l_full_seq] * 2)

    def fab_charge(self) -> float:
        """Calculates the charge of the Fab domain.

        Returns
        -------
        float
            The charge of the Fab domain.
        """
        ch_seq = self.h_full_seq[len(self.vh_seq):]

        # get the sequence for the Fab domain
        if self.isotype in HINGE_REGIONS:
            fab_end = HINGE_REGIONS[self.isotype][0] - 1
        elif self.isotype.lower() == 'igg1':
            fab_end = HINGE_REGIONS['IGHG1'][0] - 1
        elif self.isotype.lower() == 'igg2':
            fab_end = HINGE_REGIONS['IGHG2'][0] - 1
        elif self.isotype.lower() == 'igg3':
            fab_end = HINGE_REGIONS['IGHG3'][0] - 1
        elif self.isotype.lower() == 'igg4':
            fab_end = HINGE_REGIONS['IGHG4'][0] - 1
        else:
            raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')
        fab_seq = self.vh_seq + ch_seq[:fab_end] + self.l_full_seq
        
        return seq_utils.calculate_seq_charge(fab_seq)

    def fc_charge(self) -> float:
        """Calculates the charge of the Fc domain.

        Returns
        -------
        float
            The charge of the Fc domain.
        """
        ch_seq = self.h_full_seq[len(self.vh_seq):]

        # get the sequence for the Fc domain
        if self.isotype in HINGE_REGIONS:
            fc_start = HINGE_REGIONS[self.isotype][1]
        elif self.isotype.lower() == 'igg1':
            fc_start = HINGE_REGIONS['IGHG1'][1]
        elif self.isotype.lower() == 'igg2':
            fc_start = HINGE_REGIONS['IGHG2'][1]
        elif self.isotype.lower() == 'igg3':
            fc_start = HINGE_REGIONS['IGHG3'][1]
        elif self.isotype.lower() == 'igg4':
            fc_start = HINGE_REGIONS['IGHG4'][1]
        else:
            raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')
        fc_seq = ch_seq[fc_start:]

        return seq_utils.calculate_seq_charge(fc_seq)

    def fab_fc_csp(self) -> float:
        """Calculates the charge separation parameter (CSP) between the Fab domain and Fc domain.

        Returns
        -------
        float
            The charge separation parameter between the Fab domain and the Fc domain.
        """
        return self.fab_charge() * self.fc_charge()
