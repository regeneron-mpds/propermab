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
import open3d as o3d

from propermab.io import struct_io


class Voxel:
    """Class implementing the voxel object.

    A voxel has the following attributes:
        size: length of each side of the cube
        index: A voxel should be part of a voxel grid, thus it should have an index
        center coordinates: Cartesian coordinates of the center of the voxel
        feature vector: similar to a pixel that has color values, a voxel should have
        feature values. The default is None.

    """
    def __init__(
            self,
            size: float,
            index: tuple = (0, 0, 0),
            center_coords: np.ndarray = np.array([0., 0., 0.]),
            feature_vector: np.ndarray = None
    ) -> None:
        """Constructor.

        Parameters
        ----------
        size : float
            Length of each side of the cube.
        index : tuple, optional
            A voxel should be part of a voxel grid, thus it should have an index.
            By default (0, 0, 0), this makes sense only when a voxel is created in isolate.
        center_coords : np.ndarray, optional
            Cartesian coordinates of the center of the voxel.
            By default ```np.array([0., 0., 0.])```.
        feature_vector : np.ndarray, optional
            Feature vector encoding the voxel, by default None
        """
        self.size = size
        self.index = index
        self.center_coords = center_coords
        self._feature_vector = feature_vector

    @property
    def feature_vector(self):
        """feature_vector property, intended for more complex feature vector computations.

        Returns
        -------
        np.ndarray
            Feature vector encoding this voxel.
        """
        if self._feature_vector is None:
            # compute feature vector
            self._feature_vector = 0.0
        return self._feature_vector

    @feature_vector.setter
    def feature_vector(self, feature_values):
        """Assign values to the feature vector.

        Parameters
        ----------
        feature_values : list
            A list like object containing floats.
        """
        self._feature_vector = np.array(feature_values)

    def __repr__(self):
        """Returns a string representation of the voxel object.

        Returns
        -------
        str
            A str representation of the voxel that can be passed as an argument to eval() to
            re-create the voxel object.
        """
        class_name = type(self).__name__
        return f'{class_name}' \
               f'({self.size}, {repr(self.index)}, {repr(self.center_coords)}, ' \
               f'{repr(self._feature_vector)})'

    def __str__(self):
        """Returns a string representation of the voxel for ```print()```.

        Returns
        -------
        str
            String representation as output of ```print()```.
        """
        return f'({self.size}, {self.index}, {self.center_coords}, {self._feature_vector})'


class VoxelGrid:
    """Class implementing a regular grid of voxels.

    A voxel grid should have the following attributes:
        origin: the point from which width, height, and depth are extended
        width: width of the voxel grid
        height: height of the voxel grid
        depth: depth of the voxel grid
        voxel_size: size of the voxel
        voxels: voxels contained in this voxel grid
    """

    def __init__(self, origin, width, height, depth, voxel_size, voxels=None) -> None:
        """Default constructor.

        Parameters
        ----------
        origin
        width
        height
        depth
        voxel_size
        voxels
        """
        self.origin = origin
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.voxels = voxels
        self._feature_tensor = None

    @classmethod
    def create_from_point_cloud(cls, point_cloud, width, height, depth, voxel_size=1.0):
        """Alternative constructor. Creates a sparse voxel grid from a point cloud.

        Parameters
        ----------
        point_cloud : list
            _description_
        width : float
            _description_
        height : float
            _description_
        depth : float
            _description_
        voxel_size : float
            _description_

        Returns
        -------
        _type_
            _description_
        """
        o3d_point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(point_cloud)
        )
        pcd_center = o3d_point_cloud.get_center()

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            o3d_point_cloud, voxel_size,
            min_bound=np.array([-width, -height, -depth]) / 2. + pcd_center,
            max_bound=np.array([width, height, depth]) / 2. + pcd_center
        )

        new_voxel_grid = []
        for voxel in voxel_grid.get_voxels():
            new_voxel_grid.append(
                Voxel(
                    voxel_size, voxel.grid_index,
                    voxel_grid.get_voxel_center_coordinate(voxel.grid_index),
                    feature_vector=None
                )
            )
        return cls(voxel_grid.origin, width, height, depth, voxel_size, new_voxel_grid)

    @classmethod
    def create_from_origin(cls, origin, width, height, depth, voxel_size=1.0):
        """Alternative constructor. Creates a dense voxel grid from given origin.

        Parameters
        ----------
        origin : list
            _description_
        width : float
            _description_
        height : float
            _description_
        depth : float
            _description_
        voxel_size : float, optional
            _description_, by default 1.0
        """
        voxel_grid = o3d.geometry.VoxelGrid.create_dense(
            origin, color=[0.0, 0.0, 0.0],
            voxel_size=voxel_size, width=width, height=height, depth=depth
        )

        new_voxel_grid = []
        for voxel in voxel_grid.get_voxels():
            new_voxel_grid.append(
                Voxel(
                    voxel_size, voxel.grid_index,
                    voxel_grid.get_voxel_center_coordinate(voxel.grid_index),
                    feature_vector=None
                )
            )
        return cls(voxel_grid.origin, width, height, depth, voxel_size, new_voxel_grid)

    def featurize(self, feature_values=None):
        """Assigns values to the feature_vector attribute of each voxel.

        Parameters
        ----------
        feature_values : list
            A list of feature values.
        """
        if feature_values is None:
            # compute feature values in situ
            pass

        # make sure the length of feature values is equal to the number of voxels
        if len(feature_values) != len(self.voxels):
            raise ValueError(
                'The number of feature values must be equal to the number of voxels.'
            )

        self._feature_tensor = np.zeros(self.get_shape())
        for voxel, feat_value in zip(self.voxels, feature_values):
            voxel.feature_vector = feat_value
            self._feature_tensor[tuple(voxel.index)] = feat_value

    def get_shape(self) -> tuple:
        """Computes and returns the shape of the voxel grid.

        Returns
        -------
        tuple
            The shape of the voxel grid.
        """
        return (
            int(np.round(self.width / self.voxel_size)),
            int(np.round(self.height / self.voxel_size)),
            int(np.round(self.depth / self.voxel_size))
        )

    @property
    def feature_tensor(self) -> np.ndarray:
        """feature_tensor property.

        Returns
        -------
        np.ndarray
            The voxel grid encoded into a 3D feature tensor.
        """
        if self._feature_tensor is None:
            self._feature_tensor = np.zeros(self.get_shape())
            for voxel in self.voxels:
                self._feature_tensor[tuple(voxel.index)] = voxel.feature_vector
        return self._feature_tensor

    @feature_tensor.setter
    def feature_tensor(self, new_tensor=None):
        """Assigns a new feature tensor to encode the voxel grid.

        Parameters
        ----------
        new_tensor : np.ndarray
            New feature tensor.
        """
        new_tensor_arr = np.array(new_tensor)
        if new_tensor_arr.shape != self.get_shape():
            raise ValueError(
                'The shape of new tensor must be the same as the shape of the voxel grid.'
            )
        self._feature_tensor = new_tensor_arr

    def get_centers(self) -> np.ndarray:
        """Gets the Cartesian coordinates of each voxel in the voxel grid.

        Returns
        -------
        np.ndarray
            The Cartesian coordinates of each voxel in the voxel grid.
        """
        return np.array([
            voxel.center_coords for voxel in self.voxels
        ])

    def write_voxel_grid_coords(self, csv_file: str) -> None:
        """Writes the Cartesian coordinates of each voxel to a CSV file.

        Parameters
        ----------
        csv_file : str
            The CSV file to which voxel coordinates are to be written.
        """
        voxel_grid_coords = self.get_centers()
        np.savetxt(csv_file, voxel_grid_coords, fmt='%.2f', delimiter=',')

    def to_pdb(self, filename, voxel_values=None, threshold=None):
        """Write the voxel grid to a dummy PDB file for visualization with PyMOL.

        Parameters
        ----------
        filename : str
        voxel_values : np.ndarray
        threshold : float

        Returns
        -------

        """
        voxel_grid_coords = self.get_centers()

        if voxel_values is None:
            struct_io.write_to_pdb(filename, voxel_grid_coords)
        else:
            if voxel_values.shape != self.get_shape():
                raise ValueError(
                    f'Invalid shape of voxel values {voxel_values.shape}, expected {self.get_shape()}'
                )
            if threshold is not None:
                thresholded_coords = []
                thresholded_values = []
                for coord, value in zip(voxel_grid_coords, voxel_values.flatten()):
                    if value >= threshold:
                        thresholded_coords.append(coord)
                        thresholded_values.append(value)
                struct_io.write_to_pdb(filename, thresholded_coords, thresholded_values)
            else:
                struct_io.write_to_pdb(filename, voxel_grid_coords, voxel_values.flatten())


def rotate_feature_tensor(feature_tensor):
    """List all 24 rotations of the given 3D tensor.

    Parameters
    ----------
    feature_tensor : np.ndarray
`       The feature tensor that encodes the voxel grid that contains it.

    Returns
    -------
    generator
        A generator object that yields all 24 rotations of the tensor.
    """

    def rotate_about_axes(tensor_3d, axes):
        for i in range(4):
            yield np.rot90(tensor_3d, i, axes)

    # imagine shape is pointing in axis 0 (up)
    # 4 rotations about axis 0
    yield from rotate_about_axes(feature_tensor, (1, 2))

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotate_about_axes(np.rot90(feature_tensor, 2, axes=(0, 2)), (1, 2))

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotate_about_axes(np.rot90(feature_tensor, axes=(0, 2)), (0, 1))
    yield from rotate_about_axes(np.rot90(feature_tensor, -1, axes=(0, 2)), (0, 1))

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotate_about_axes(np.rot90(feature_tensor, axes=(0, 1)), (0, 2))
    yield from rotate_about_axes(np.rot90(feature_tensor, -1, axes=(0, 1)), (0, 2))


def save_feature_tensor(feature_tensor, filename, mode='b'):
    """Saves feature tensor to disk file.

    Parameters
    ----------
    feature_tensor : np.ndarray
        3D feature tensor to be saved in a disk file.
    filename : str
        Name of disk file to which the feature tensor will be written.
    mode : str
        If 'b', then save the tensor in binary format, else save in text format.
        Text format may not work for tensors with rank > 3.
    """
    # Write the array to disk
    if mode == 'b':
        # setting allow_pickle to False to reduce security risk and improve portability
        np.save(filename, feature_tensor, allow_pickle=False)
    else:
        with open(filename, 'wt') as out_file:
            # Any line starting with "#" will be ignored by numpy.loadtxt
            out_file.write('# Array shape: {0}\n'.format(feature_tensor.shape))

            # iterating through a n dimensional array produces slices along
            # the first axis. This is equivalent to data[i,:,:] in this case
            for data_slice in feature_tensor:

                # the formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(out_file, data_slice, fmt='%-5.2f')

                # write out a break to indicate different slices...
                out_file.write('# New slice\n')
