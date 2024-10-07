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

import numpy as np
import numpy.typing as npt
import plyfile
import meshio

import matplotlib as mpl

from propermab.features import surface


def read_nanoshaper(vertex_file: str, face_file: str=None):
    """Parses NanoShaper results into Python data structures.

    Parameters
    ----------
    vertex_file : str
        File containing surface vertices in NanoShaper output format.
    face_file : str, optional
        File containing triangles in NanoShaper output format, by default None

    Returns
    -------
    tuple
        Cartesian coordinates of vertices,
        vertex id for each face,
        number ID of the atom the vertex is assigned to,
        vertex normals.
    """
    # read the surface from the NanoShaper output
    # NanoShaper outputs two files: {file_root}.vert and {file_root}.face
    with open(vertex_file, 'rt') as ipf:
        vertex_data = [line.strip() for line in ipf]

    vertices = []
    atom_numbers = []
    normalv = []
    num_vertices = int(vertex_data[2])
    for vertex_line in vertex_data[3:]:
        vertex_fields = vertex_line.split()
        vertices.append([float(x) for x in vertex_fields[:3]])
        normalv.append([float(y) for y in vertex_fields[3:6]])
        atom_numbers.append(int(vertex_fields[7]))
        num_vertices -= 1
    assert num_vertices == 0
    
    # parse faces
    if face_file is None:
        base_name = '.'.join(os.path.basename(vertex_file).split('.')[:-1])
        face_file = os.path.join(os.path.dirname(vertex_file), base_name + '.face')
    
    with open(face_file, 'rt') as face_ipf:
        face_data = [line.strip() for line in face_ipf]

    faces = []
    num_faces = int(face_data[2])
    for face_line in face_data[3:]:
        face_fields = face_line.split()
        faces.append([int(x) for x in face_fields[:3]])
        num_faces -= 1
    assert num_faces == 0

    return vertices, faces, atom_numbers, normalv


def write_to_ply(
        filename: str, 
        vertices: npt.ArrayLike, 
        faces: npt.ArrayLike, 
        vertex_features: npt.ArrayLike,
        threshold=20., 
        cmap=mpl.cm.bwr_r
    ):
    """Write reduced surface (with numerical features) in PLY format for visualization.

    Feature values less than -threshold will be mapped to the left end of the spectrum, 
    values larger than threshold will be mapped to the right end of the spectrum.

    Parameters
    ----------
    filename : str
        Name of the file to which the PLY format of will be written to.
    vertices : npt.ArrayLike
        Vertices of the surface.
    faces : npt.ArrayLike
        Triangle faces of the surface.
    vertex_features : npt.ArrayLike
        Numerical feature of each of the vertices of the surface.
    threshold : _type_, optional
        Threshold for color mapping, by default 20.
    cmap : _type_, optional
        Color map used, by default mpl.cm.bwr_r
    """
    norm = mpl.colors.Normalize(vmin=-threshold, vmax=threshold, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    vertices_with_color = []
    for vertex, esp in zip(vertices, vertex_features):
        vertex_rgb = mapper.to_rgba(esp, bytes=True)[:3]
        vertices_with_color.append(tuple(vertex) + vertex_rgb)

    vertices = np.array(
        vertices_with_color,
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )

    # make vertex index to start at 1
    faces_start_at_one = (np.array(faces) - 1).tolist()
    faces = np.array(
        [(x,) for x in faces_start_at_one],
        dtype=[('vertex_indices', 'i4', (3,))]
    )

    vertex_elements = plyfile.PlyElement.describe(
        data=vertices, name='vertex'
    )
    face_elements = plyfile.PlyElement.describe(
        data=faces, name='face'
    )
    plyfile.PlyData([vertex_elements, face_elements], text=True).write(
        os.path.abspath(filename)
    )


def write_patch(
    output_file: str, 
    surface_patch: surface.SurfacePatch, 
    file_format, 
    point_data=None, 
    cell_data=None
):
    """A wrapper function around the meshio package to write the surface patch in selected
    file format for visualization.

    Parameters
    ----------
    output_file : str
        Filename.
    surface_patch : surface.SurfacePatch
        A SurfacePatch object.
    file_format : _type_
        File format, see the meshio package for options.
    point_data : _type_, optional
        Specific to the meshio package, see the meshio package for more information, by default None
    cell_data : _type_, optional
        See the meshio package for more information, by default None
    """
    # sort the coords and vertices
    coords_sorted = [
        coord for _, coord in sorted(
            zip(surface_patch.vertices, surface_patch.vertex_coords),
            key=lambda pair:pair[0]
        )
    ]
    vertices_sorted = sorted(surface_patch.vertices)

    vertex_idx_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(vertices_sorted)
    }
    reindexed_faces = [
        [
            vertex_idx_map[face[0]], vertex_idx_map[face[1]], vertex_idx_map[face[2]]
        ] for face in surface_patch.faces
    ]

    points = coords_sorted
    cells = [
        ('triangle', reindexed_faces)
    ]

    # create a mesh and write to file for visualization
    mesh = meshio.Mesh(
        points=points, cells=cells, point_data=point_data, cell_data=cell_data
    )
    mesh.write(path_or_buf=output_file, file_format=file_format)
