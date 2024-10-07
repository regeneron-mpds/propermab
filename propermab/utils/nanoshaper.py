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
import shutil
import subprocess
import textwrap

import propermab
from propermab import defaults


def create_nanoshaper_config_file(nanoshaper_config_file, grid_scale=1.0, output_prefix='PREFIX'):
    # end first line with \ to avoid the leading empty line at the top of the output file!
    nanoshaper_config = f"""\
        Compute_Vertex_Normals = true
        Save_Mesh_MSMS_Format = true
        Vertex_Atom_Info = true
        Surface = ses
        Probe_Radius = 1.4
        Accurate_Triangulation = true
        Triangulation = true
        Grid_scale = {grid_scale}
        XYZR_FileName = {output_prefix}.xyzr
    """
    with open(nanoshaper_config_file, 'wt') as f:
        f.write(textwrap.dedent(nanoshaper_config))


def run_pdb_to_xyzr(
        pdb_file, pdb_to_xyzr='pdb_to_xyzr.py', ff_file='amber'
    ):
    """Run pdb_to_xyzr.py to generate a .xyzr file from the given PDB file.

    Parameters
    ----------
    pdb_file : str
        PDB file of the protein.
    pdb_to_xyzr : str
        Path to the pdb_to_xyzr.py script.
    ff_file : str
        Path to the force field file containing atom radii information.
    """
    pdb_basename = os.path.basename(pdb_file)
    output_prefix = '.'.join(pdb_basename.split('.')[:-1])

    # run pdb_to_xyzr.py
    # some platforms may not recognize pdb_to_xyzr.py as an executable
    # thus, calling python to execute the script
    command = [
        'python', pdb_to_xyzr, '--pdb-file', pdb_file, '--ff-file', ff_file,
        '--output-prefix', output_prefix
    ]
    print(' '.join(command))
    output_info = subprocess.run(
        command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # print error messages if any
    print(output_info.stderr)


def run_nanoshaper(
        nanoshaper='NanoShaper', config_file=None
    ):
    """A simple function that runs NanoShaper in a subprocess.

    Parameters
    ----------
    nanoshaper : str
        Path to the binary executable of NanoShaper.
    config_file : str
        NanoShaper configuration file in which parameters for running the program
        are configured.

    Returns
    -------
        triangulatedSurf.vert: vertices in MSMS format (https://ccsb.scripps.edu/msms/documentation/)
        triangulatedSurf.face: triangles in MSMS format
        triangleAreas.txt: areas of the triangles
        exposed.xyz: info about atoms that are solvent exposed
        exposedIndices.txt: indices of the vertices that are solvent exposed.

    """
    # some platforms may not recognize NanoShaper bash script as an executable
    # thus, calling bash to execute the script
    if defaults.system_config['require_bash']:
        command = ['bash', nanoshaper, config_file]
    else:
        command = [nanoshaper, config_file]
    print(' '.join(command))
    output_info = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print error messages if any
    print(output_info.stderr)


class NanoShaper:

    def __init__(
            self, nanoshaper_bin=None, nanoshaper_conf=None, pdb_to_xyzr=None,
            atom_radii_file=None, grid_scale=1.0
    ) :
        """This is a wrapper class for running the NanoShaper protein surface
        triangulation program.

        Parameters
        ----------
        nanoshaper_bin : str
            Path to the binary executable of NanoShaper.
        nanoshaper_conf : str
            NanoShaper configuration file in which parameters for running the program
            are configured.
        pdb_to_xyzr : str
            Path to the pdb_to_xyzr.py script. This scripts parses a PDB file to generate
            a .xyzr file, which is required by NanoShaper.
        atom_radii_file : str
            Force field file containing atom radii information.
        grid_scale : float
            Parameter to adjust the density of the mesh.
        """
        # use system default if arguments not set explicitly
        if nanoshaper_bin is None:
            self.nanoshaper_bin = defaults.system_config['nanoshaper_binary_path']
        else:
            self.nanoshaper_bin = nanoshaper_bin
        self.nanoshaper_conf = nanoshaper_conf
        if pdb_to_xyzr is None:
            module_path = propermab.__file__
            propermab_dir = os.path.abspath(os.path.join(module_path, '../../'))
            self.pdb_to_xyzr = os.path.join(propermab_dir, 'scripts/pdb_to_xyzr.py')
        else:
            self.pdb_to_xyzr = pdb_to_xyzr
        if atom_radii_file is None:
            self.atom_radii_file = defaults.system_config['atom_radii_file']
        else:
            self.atom_radii_file = atom_radii_file
        self.grid_scale = grid_scale

    def __call__(self, pdb_file, tmp_path='/tmp/'):
        """This function makes NanoShaper object callable.

        Parameters
        ----------
        pdb_file : str
            PDB file of the protein on which NanoShaper is to be called.
        tmp_path : str
            Temporary directory in which the calculations will be done and
            where files will be stored.

        Returns
        -------
        list
            A list in which each element is the Cartesian coordinates of a vertex.

        As part of the results, calling this object will also generate the following files.
            .xyzr: atom coodinates and radius
            .vert: vertices in MSMS format (https://ccsb.scripps.edu/msms/documentation/)
            .face: triangles in MSMS format
            _areas.txt: areas of the triangles
            _exposed_atoms.xyz: info about atoms that are solvent exposed
            _exposed_indices.txt: indices of the vertices that are solvent exposed.

        """
        # get absolute path to the PDB file
        pdb_file = os.path.abspath(pdb_file)

        # create a temporary direcory and do calculations there
        tmp_dir = tempfile.mkdtemp(dir=tmp_path)
        old_dir = os.getcwd()
        os.chdir(tmp_dir)

        print(f'Doing calculations in a temporary directory: {tmp_dir}')

        # run pdb_to_xyzr.py
        run_pdb_to_xyzr(pdb_file, self.pdb_to_xyzr, self.atom_radii_file)

        pdb_basename = os.path.basename(pdb_file)
        output_prefix = '.'.join(pdb_basename.split('.')[:-1])

        # create a temporary nanoshaper configuration file
        if self.nanoshaper_conf is None:
            _, tmp_conf_file = tempfile.mkstemp(suffix='.prm', dir=tmp_dir)
            create_nanoshaper_config_file(tmp_conf_file, self.grid_scale, output_prefix)
            self.nanoshaper_conf = tmp_conf_file

        # run nanoshaper
        run_nanoshaper(
            self.nanoshaper_bin, self.nanoshaper_conf
        )
        shutil.move('triangulatedSurf.vert', f'{output_prefix}.vert')
        shutil.move('triangulatedSurf.face', f'{output_prefix}.face')
        vertex_file_path = os.path.abspath(f'{output_prefix}.vert')
        face_file_path = os.path.abspath(f'{output_prefix}.face')

        os.chdir(old_dir)

        return vertex_file_path, face_file_path

