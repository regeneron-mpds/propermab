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
import tempfile
import subprocess
import shlex

from argparse import ArgumentParser

from propermab import defaults


def run_pdb2pqr(
        pdb_file, pdb2pqr='pdb2pqr', force_field='CHARMM', output_prefix=None,
        other_options=None
    ):
    """Runs pdb2pqr to generate a PQR file from the given PDB file. The PQR file will be
    used as input to APBS.

    Parameters
    ----------
    pdb_file : str
        PDB file of the protein.
    pdb2pqr : str
        Path to pdb2pqr or pdb2pqr30.
    force_field : str
        Name of force field, choose between AMBER, CHARMM, or PARSE
    output_prefix : str
        Prefix of the output PQR file.
    other_options : str
        Other command-line options to run pdb2pqr.
        Run ```pdb2pqr -h``` on the command line to see all available options.

    Returns
    -------
        Generate the following files:
        ```
        f'{output_prefix}.pqr'
        f'{output_prefix}_apbs.in'
        ```
    """
    # run pdb2pqr
    pdb2pqr_command = [
        pdb2pqr, f'--ff={force_field}', '--apbs-input', f'{output_prefix}_apbs.in',
        pdb_file, f'{output_prefix}.pqr'
    ]
    if other_options is not None:
        pdb2pqr_command += shlex.split(other_options)
    print(' '.join(pdb2pqr_command))
    output_info = subprocess.run(
        pdb2pqr_command, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(output_info.stderr)


def run_apbs(apbs_bin, apbs_input, other_options=None):
    """Runs APBS to compute electrostatics given an input configuration file.

    Parameters
    ----------
    apbs_bin : str
        Path to the APBS binary executable.
    apbs_input : str
        Input configuration file to APBS.
    other_options : str
        Other command-line options to run APBS.
        Run ```apbs --help``` on the command line to see all available options.

    Returns
    -------
    Generates a .dx file in which electrostatic potentials are stored.

    """
    apbs_command = [apbs_bin, apbs_input]
    if other_options is not None:
        apbs_command += shlex.split(other_options)
    print(' '.join(apbs_command))
    output_info = subprocess.run(
        apbs_command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(output_info.stderr)

    apbs_output_file = None
    with open(apbs_input, 'rt') as in_file:
        for line in in_file:
            if line.strip().startswith('write'):
                write_line_fields = line.strip().split()
                apbs_output_file = f'{write_line_fields[-1]}.{write_line_fields[-2]}'

    if apbs_output_file is None:
        raise FileNotFoundError('No APBS output file found.')

    # return path to the .dx file
    return os.path.abspath(apbs_output_file)


def run_multivalue(multivalue_bin, vertex_file, apbs_output, output_prefix):
    """Runs the multivalue program to extract electrostatic potentials
    for a given set of vertices.

    Parameters
    ----------
    multivalue_bin : str
        Path to the multivalue binary executable.
    vertex_file : str
        CSV file in which each line has the Cartesian coordinates of a vertex.
    apbs_output : str
        Output file from running APBS. Usually, this is the file with the .dx suffix.
    output_prefix : str
        Prefix of the output file of running multivalue.

    Returns
    -------
    Generates a CSV file in which each line has the Cartesian coordinates
    and the electrostatic potential of a vertex.

    """
    multivalue_command = [
        multivalue_bin, vertex_file, apbs_output, f'{output_prefix}_apbs.csv'
    ]
    print(' '.join(multivalue_command))
    output_info = subprocess.run(
        multivalue_command, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # print error messages if any
    print(output_info.stderr)

    # get electrostatic potential values
    with open(f'{output_prefix}_apbs.csv', 'rt') as in_handle:
        esp_values = [float(line.split(',')[-1]) for line in in_handle]
    return esp_values


class APBS:
    """
    This is a wrapper class for running the APBS electrostatics calculation program.
    """
    def __init__(self, apbs_bin=None, pdb2pqr=None, ld_lib_paths=None):
        """Constructor.

        Parameters
        ----------
        apbs_bin : str
            Path to the APBS binary executable.
        pdb2pqr : str
            Path to the pdb2pqr program.
        ld_lib_paths : list
            Paths to runtime libraries needed by APBS.
        """
        # use system default if arguments not set explicitly
        if apbs_bin is None:
            self.apbs_bin = defaults.system_config['apbs_binary_path']
        else:
            self.apbs_bin = apbs_bin
        if pdb2pqr is None:
            self.pdb2pqr = defaults.system_config['pdb2pqr_path']
        else:
            self.pdb2pqr = pdb2pqr
        if ld_lib_paths is None:
            default_ld_paths = defaults.system_config['apbs_ld_library_paths']
            # in case paths are specified as a single string, separated by colon
            if isinstance(default_ld_paths, str):
                self.ld_lib_paths = default_ld_paths.split(':')
            else:
                self.ld_lib_paths = default_ld_paths
        else:
            self.ld_lib_paths = ld_lib_paths

    def __call__(self, pdb_file, tmp_path='/tmp/', **kwargs):
        """This function makes APBS objects callable.

        Parameters
        ----------
        pdb_file : str
            PDB file of the protein for which electrostatic potentials are to be computed.
        tmp_path : str
            Temporary directory in which the calculations will be done and
            where files will be stored.
        kwargs
            Other keyword arguments.

        Returns
        -------
        See the returns of ```run_pdb2pqr()``` and ```run_apbs()```.

        """
        # export required libraries file for running APBS
        os.environ['LD_LIBRARY_PATH'] = ':'.join(
            ['${LD_LIBRARY_PATH}'] + self.ld_lib_paths
        )

        # get the absolute path of the PDB file to avoid copying the PDB file to temp dir
        pdb_file = os.path.abspath(pdb_file)

        tmp_dir = tempfile.mkdtemp(dir=tmp_path)
        old_dir = os.getcwd()
        os.chdir(tmp_dir)

        print(f'Doing APBS calculations in a temporary directory: {tmp_dir}')

        pdb_basename = os.path.basename(pdb_file)
        output_prefix = '.'.join(pdb_basename.split('.')[:-1])

        # first run pdb2pqr
        if 'force_field' in kwargs:
            run_pdb2pqr(pdb_file, self.pdb2pqr, kwargs['force_field'], output_prefix)
        else:
            run_pdb2pqr(pdb_file, self.pdb2pqr, 'CHARMM', output_prefix)

        # now run apbs
        apbs_output_file = run_apbs(self.apbs_bin, apbs_input=f'{output_prefix}_apbs.in')

        if os.path.exists(f'{output_prefix}.pqr'):
            os.remove(f'{output_prefix}.pqr')

        os.chdir(old_dir)

        return apbs_output_file

