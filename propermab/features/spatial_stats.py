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
import numpy.typing as npt

from scipy import spatial


class RipleyK:
    def __init__(
        self, obs_coords: npt.ArrayLike, allowed_coords: npt.ArrayLike,
        distance: float = 8., p: int = 2, n: int = 1000
    ):
        """

        Parameters
        ----------
        obs_coords
        distance
        """
        self.obs_coords = np.asarray(obs_coords)
        self.allowed_coords = np.asarray(allowed_coords)
        self.distance = distance
        self.p = p
        self.n = n
        self._ripley_k = None

    @property
    def ripley_k(self) -> float:
        """

        Returns
        -------

        """
        if self._ripley_k is None:
            feature_size = self.obs_coords.shape[0]
            denominator = feature_size * (feature_size - 1)
            k_o = self.get_number_of_pairs(self.obs_coords, self.distance) / denominator
            rng = np.random.default_rng()
            k_e_null = []
            for _ in range(self.n):
                new_locations = rng.choice(self.allowed_coords.shape[0], size=feature_size)
                new_coords = self.allowed_coords[new_locations]
                k_e_null.append(
                    self.get_number_of_pairs(new_coords, self.distance) / denominator
                )
            k_e = np.mean(k_e_null)

            self._ripley_k = k_o / k_e
        return self._ripley_k

    @staticmethod
    def get_number_of_pairs(coords: npt.ArrayLike, distance):
        """Computes the number of neighboring pairs.

        Parameters
        ----------
        coords : npt.ArrayLike
            Cartesian coordinates of the features.

        Returns
        -------
        float
            The the number of neighboring pairs.

        """
        kd_tree = spatial.KDTree(coords)
        neighbor_pairs = kd_tree.query_pairs(r=distance)
        return len(neighbor_pairs)


class AverageNearestNeighbor:
    def __init__(
        self, feature_coords: npt.ArrayLike, allowed_coords: npt.ArrayLike,
        p: int = 2, n: int = 1000
    ):
        """

        Parameters
        ----------
        coords
        p : int
            Which Minkowski p-norm to use.

        n : int
            Number of permutations to do in deriving the expected mean distance.
        """
        self.feature_coords = np.asarray(feature_coords)
        self.allowed_coords = np.asarray(allowed_coords)
        self.p = p
        self.n = n
        self._ann_index = None

    @property
    def ann_index(self):
        """Computes the Average Nearest Neighbor index feature.
        The Average Nearest Neighbor index feature is defined as the ratio of observed mean distance
        to the expected mean distance.

        Returns
        -------
        float
            The Average Nearest Neighbor inex feature.

        """
        if self._ann_index is None:
            # observed mean distance each point and its nearest neighbor
            d_o = self.compute_nn_mean_distance(self.feature_coords)

            # expected mean distance for the features given a "random" pattern
            feature_size = self.feature_coords.shape[0]
            rng = np.random.default_rng()
            d_e_null = []
            for _ in range(self.n):
                new_locations = rng.choice(self.allowed_coords.shape[0], size=feature_size)
                new_coords = self.allowed_coords[new_locations]
                d_e_null.append(
                    self.compute_nn_mean_distance(new_coords)
                )
            d_e = np.mean(d_e_null)
            self._ann_index = d_o / d_e
        return self._ann_index

    @staticmethod
    def compute_nn_mean_distance(coords: npt.ArrayLike) -> float:
        """Computes the mean distance of nearest neighbors.

        Parameters
        ----------
        coords : npt.ArrayLike
            Cartesian coordinates of the features.

        Returns
        -------
        float
            The mean distance of nearest neighbors.

        """
        nn_distances = []
        kd_tree = spatial.KDTree(coords)
        for point in coords:
            # point is contained in the tree, so its nearest neighbor
            # should exclude itself, hence k=2
            dists, _ = kd_tree.query(point, k=2)
            nn_distances.append(dists[1])
        return np.mean(nn_distances)
