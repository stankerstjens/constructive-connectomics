#  MIT License
#
#  Copyright (c) 2022. Stan Kerstjens
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
"""

Volume
======

Container class for a volume of voxels

.. seealso:: :ref:`volume`

"""
from typing import Tuple, List, Dict, Optional, Union, Iterable

import numpy as np

from abianalysis.anatomy import Anatomy
from .load import load_volume, default_volume_file_name
from .preprocess import match_genes, shuffle_expression, \
    impute_missing_expression, \
    normalize_expression, valid_voxels, valid_genes

#: The available ages in the Allen Brain Institute developing brain atlas data
AGES = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P28', 'P56']

#: The sizes of the voxels per age
_VOXEL_SIZES: Dict[str, int] = {
    'E11.5': 80, 'E13.5': 100, 'E15.5': 120, 'E18.5': 140,
    'P4': 160, 'P14': 200, 'P28': 200, 'P56': 200
}


class VoxelSizesSingleton:
    """Provides the default behaviour that if the voxel size is not known,
    it returns 1 instead of throwing an error """

    def __init__(self, voxel_sizes):
        self.voxel_sizes = voxel_sizes

    def __getitem__(self, key):
        return self.voxel_sizes.get(key, 1)


VOXEL_SIZES = VoxelSizesSingleton(_VOXEL_SIZES)


class Volume:
    """A volume of expression data at a particular age.

    The ages provided by the Allen Institute for Brain Science (and
    supported by this package) are E11.5, E13.5, E15.5, E18.5, P4, P14,
    P28, and P56.

    The volume can be used to access the expression, position, and
    anatomy matrices, and to access the individual voxels.

    :param expression:
        an n x m expression matrix, where n is the number of (voxels) and m the
        number of genes.
    :param voxel_indices:
        an n x 3 position matrix, where  is the number of samples and the
        columns are x, y, z coordinates.
    :param anatomy: List of anatomy objects, one per voxel in the volume.
    :param genes: List of gene names, one per gene in the volume.
    :param age: The age of the mice the volume is based on.


    """

    @classmethod
    def load(cls, age: str, file_name: str = None) -> 'Volume':
        """ See :py:func:`.load_volume` """
        return cls(**load_volume(age, file_name))

    def __init__(self, age: str, expression: np.ndarray,
                 voxel_indices: np.ndarray,
                 genes: List[str] = None, anatomy: List[Anatomy] = None):
        self.expression = np.asarray(expression)
        self.genes = (np.array([f'g{i}' for i in range(expression.shape[1])])
                      if genes is None else np.asarray(genes))
        self.voxel_indices = np.asarray(voxel_indices)
        self.anatomy = None if anatomy is None else np.asarray(anatomy)
        self.age = age

        self._check_consistency()

    @property
    def n_voxels(self) -> int:
        """The total number of voxels in the volume."""
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        """The number of genes in the volume."""
        return self.expression.shape[1]

    @property
    def voxel_positions(self) -> np.array:
        """The positions of all voxels (in µm)"""
        return self.voxel_indices * self.voxel_size

    @property
    def voxel_size(self) -> int:
        """The voxel size corresponding to the age of the volume.

        Will raise a KeyError if the age is invalid.

        .. seealso:: :py:const:`.VOXEL_SIZES`

        """
        return VOXEL_SIZES[self.age]

    def match_genes(self, *volume: 'Volume') -> None:
        """See :py:func:`.match_genes`"""
        match_genes(self, *volume)

    def save(self, file_name: Optional[str] = None) -> None:
        """Save volume to a npz archive.

        :param file_name:
            The file name to save to.  If not specified, the default path is
            used, as in :py:func:`.default_volume_file_name`.

        """
        if file_name is None:
            file_name = default_volume_file_name(self.age)
        np.savez_compressed(file_name,
                            expression=self.expression,
                            positions=self.voxel_indices,
                            anatomy=np.array([a.id_ for a in self.anatomy]),
                            genes=self.genes)

    def filter_anatomy(self, anatomy: Union[Anatomy, str, int]):
        """Filter voxels to only contain this anatomy

        :param anatomy:
            Constrain the volume to this anatomy (and its descendants).
            This can be anything accepted by :py:meth:`.Anatomy.get`.

        """
        ids = Anatomy.get(anatomy).id_set
        self.filter_voxels([anatomy.id_ in ids for anatomy in self.anatomy])

    def filter_missing(self, threshold: float = .2):
        """Filter voxels and genes with too many missing values.

        Voxels are filtered *before* genes, because many voxels have a
        large number of invalid values, which skews the number of invalid
        genes.

        :param threshold:
            All voxels with more than this ratio of invalid values are filtered
            out.

        .. seealso:: :py:func:`.valid_voxels`, :py:func:`.valid_genes`

        """
        self.filter_voxels(valid_voxels(self.expression, threshold))
        self.filter_genes(valid_genes(self.expression, threshold))

    def preprocess(self, threshold: float = .2,
                   anatomy: Optional[Union[str, Anatomy]] = 'NP') -> None:
        """Remove missing values, the mean, and rescale to unit standard
        deviation per gene.

        :param threshold:
            All voxels with more than this ratio of invalid values are filtered
            out.
        :param anatomy:
            Constrain the voxels to those in the specified anatomy.  The
            anatomy can be specified as anything accepted by
            :py:func:`.Anatomy.get`.

        .. seealso::
            :py:meth:`.Volume.filter_anatomy`,
            :py:meth:`.Volume.filter_missing`,
            :py:func:`.impute_missing_expression`,
            :py:func:`.normalize_expression`

        """
        if anatomy is not None and self.anatomy is not None:
            self.filter_anatomy(anatomy)
        self.filter_missing(threshold)
        self.expression = impute_missing_expression(self.expression)
        self.expression = normalize_expression(self.expression)

    def randomize(self) -> None:
        """Replaces the expression with values drawn from a normal
        distribution (zero mean, unit variance)."""
        self.expression = np.random.randn(*self.expression.shape)

    def shuffle(self) -> None:
        """See :py:func:`.shuffle_expression`"""
        self.expression = shuffle_expression(self.expression)

    def shuffle_positions(self) -> None:
        """Randomly permute the positions of the voxels."""
        np.random.shuffle(self.voxel_indices)

    def filter_voxels(self, voxels_idx) -> None:
        """Keep only the voxels selected by `voxels_idx`.

        :param voxels_idx:
            Something that can be used to index a numpy array (like a list
            of integers or boolean values).

        """
        self.expression = self.expression[voxels_idx]
        self.voxel_indices = self.voxel_indices[voxels_idx]
        if self.anatomy is not None:
            self.anatomy = self.anatomy[voxels_idx]

    def select_random_genes(self, n: int):
        """Filter out all but a randomly selected set of genes.

        :param n: the number of genes to keep

        """
        sel = np.random.choice(self.n_genes, size=n, replace=False)
        self.filter_genes(sel)

    def select_genes(self, genes: Iterable[str]):
        """Filter out all but a specific set of named genes.

        :param genes: An iterable of names of genes to keep.

        """
        genes = np.array(list(genes))
        self.filter_genes(np.isin(self.genes, genes))

    def filter_genes(self, genes_idx) -> None:
        """Keep only the genes selected by `genes_idx`.

        :param genes_idx:
            Something that can be used to index a numpy array (like a list
            of integers or boolean values).

        """
        self.expression = self.expression[:, genes_idx]
        self.genes = self.genes[genes_idx]

    def spatial_extent(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the minimum and maximum positions for each dimension"""
        pos = self.voxel_positions
        return np.min(pos, 0), np.max(pos, 0)

    def _check_consistency(self) -> None:
        """Checks the consistency of the matrix shapes of the volume"""

        if len(self.voxel_indices) != self.n_voxels:
            raise ValueError('The `expression` and `position` matrices must '
                             'have an equal number of samples')

        if self.anatomy is not None and len(self.anatomy) != self.n_voxels:
            raise ValueError('The `expression` and `structures` must have an '
                             'equal number of samples')

        if len(self.genes) != self.n_genes:
            raise ValueError('The number of genes in `expression` and `genes`'
                             'do not match.')
