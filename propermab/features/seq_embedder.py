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
import torch
import esm
from antiberty import AntiBERTyRunner


def extract_esm_embeddings(heavy_seq: str, light_seq: str):
    """Helper function for calculating ESM embeddings.

    Parameters
    ----------
    heavy_seq : str
        Amino acid sequence of the heavy chain.
    light_seq : str
        Amino acid sequence of the ligth chain.

    Returns
    -------
    tuple
        A pair of NumPy 2D array. 
        First dim: number of residues.
        Second dim: embedding size.
    """
    # compute embedddings using the ESM1b pretrained model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    # switch to evaluation mode, disables dropout for deterministic results
    model.eval()

    # create input, must be a list of tuples
    data = [
        ('heavy', heavy_seq),
        ('light', light_seq)
    ]

    # get the amino acid tokens in terms of indices
    _, _, batch_tokens = batch_converter(data)
    batch_lengths = (batch_tokens != alphabet.padding_idx).sum(axis=1)

    # extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results['representations'][33]

    # generate per-sequence representation
    heavy_emb = token_representations[0, 1 : batch_lengths[0] - 1].numpy()
    light_emb = token_representations[1, 1 : batch_lengths[1] - 1].numpy()

    return heavy_emb, light_emb


def extract_abt_embeddings(heavy_seq: str, light_seq: str):
    """Helper function for calculating ESM embeddings.

    Parameters
    ----------
    heavy_seq : str
        Amino acid sequence of the heavy chain.
    light_seq : str
        Amino acid sequence of the ligth chain.

    Returns
    -------
    tuple
        A pair of NumPy 2D array. 
        First dim: number of residues.
        Second dim: embedding size.
    """
    # compute embedddings using the AntiBERTy pretrained model
    abt_runner = AntiBERTyRunner()
    abt_embed = abt_runner.embed([heavy_seq, light_seq], return_attention=False) 

    # generate per-sequence representation
    heavy_emb = abt_embed[0].numpy()
    light_emb = abt_embed[1].numpy()

    return heavy_emb, light_emb


class SeqEmbedder:
    """This class exposes the API for sequence embedding.
    """
    def __init__(
        self, 
        heavy_seq: str, 
        light_seq: str, 
        plm: str='ESM', 
        agg_strategy: str=None
    ) -> None:
        """Constructor of SeqEmbedder.

        Parameters
        ----------
        seqs : tuple
            A pair of heavy and light chain sequences.
        plm : str, optional
            The protein language model to use, by default 'ESM'.
        agg_strategy : str, optional
            Aggregation strategy, by default None
        """
        self.heavy_seq = heavy_seq
        self.light_seq = light_seq

        if plm.lower() not in ['esm', 'antiberty']:
           raise ValueError(f'Unknown protein language model {plm}!') 
        self.plm = plm

        if (agg_strategy is not None) and \
        (agg_strategy.lower() not in ['mean', 'max', 'concat']):
           raise ValueError(f'Unknown aggregation strategy {agg_strategy}') 
        self.agg_strategy = agg_strategy

    def embed(self):
        """Calcuates the embedding for the sequence.

        Returns
        -------
        tuple
            If agg_strategy is None, returns a pair of NumPy 2D arrays. 
            First dim: number of residues.
            Second dim: embedding size.
            If agg_strategy is 'mean' or 'max', then returns a pair of 1D arrays, both of the
            shape (embedding_size,).
            If agg_strategy is 'concat', then returns a pair of 1D arrays of the shapes
            (heavy_num_res * embedding_size, ), (light_num_res * embedding_size, ).
        """
        if self.plm.lower() == 'esm':
            # compute embeddings using ESM-1b model
            heavy_emb, light_emb = extract_esm_embeddings(self.heavy_seq, self.light_seq)        
        elif self.plm.lower() == 'antiberty':
            # compute embedddings using the AntiBERTy pretrained model
            heavy_emb, light_emb = extract_abt_embeddings(self.heavy_seq, self.light_seq)

        if self.agg_strategy is None:
            return heavy_emb, light_emb
        if self.agg_strategy.lower() == 'mean':
            return heavy_emb.mean(axis=0), light_emb.mean(axis=0)
        if self.agg_strategy.lower() == 'max':
            return heavy_emb.max(axis=0), light_emb.max(axis=0)
        if self.agg_strategy.lower() == 'concat':
            return heavy_emb.flatten(order='C'), light_emb.flatten(order='C')
