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

Volume preprocessing
====================

Functions to pre-process data stored in :py:class:`.Volume`

.. seealso:: :ref:`volume`


"""
from typing import List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np


def match_genes(*volumes) -> List[str]:
    """Remove all genes from all volumes that do not appear in every volume.

    :param volumes:  The volumes whose genes to match.
    :returns: a list of genes present in all volumes after preprocessing

    """
    volume_iterator = iter(volumes)
    genes = set(next(volume_iterator).genes)
    for volume in volume_iterator:
        genes &= set(volume.genes)
    genes = np.asarray(list(genes))
    for volume in volumes:
        volume.filter_genes(np.isin(volume.genes, genes))

    return list(genes)


def _valid_mask(expression: np.ndarray,
                threshold: float,
                axis: int) -> np.ndarray:
    invalid_ratio = (expression == -1).mean(axis)
    return invalid_ratio <= threshold


def valid_voxels(expression: np.ndarray, threshold: float) -> np.ndarray:
    """Return a mask selecting all voxels with sub-threshold invalid values.

    :param expression:  Expression matrix with rows as voxels
    :param threshold:  Threshold below which to filter.

    """
    return _valid_mask(expression, threshold, axis=1)


def valid_genes(expression: np.ndarray, threshold: float) -> np.ndarray:
    """Return a mask selecting all genes with sub-threshold invalid values.

    :param expression:  Expression matrix with rows as voxels
    :param threshold:  Threshold below which to filter.

    """
    return _valid_mask(expression, threshold, axis=0)


def impute_missing_expression(expression: np.ndarray) -> np.ndarray:
    """Replace missing expression values with the mean over the gene. The
    passed matrix will be modified in-place if possible.

    :param expression:
        An expression matrix with voxels as rows and genes as columns.

    """
    return SimpleImputer(missing_values=-1, copy=False) \
        .fit_transform(expression)


def normalize_expression(expression: np.ndarray) -> np.ndarray:
    """Scale expression to unit variance and zero mean.

    :param expression:
        An expression matrix with voxels as rows and genes as columns.

    :return: Normalized matrix

    """
    scaler = StandardScaler(copy=False)
    return scaler.fit_transform(expression)


def shuffle_expression(expression: np.ndarray):
    """Shuffle expression values to random voxels representing a random
    gene.

    :param expression:
        An expression matrix with voxels as rows and genes as columns.

    """
    shape = expression.shape
    flat = expression.flatten()
    np.random.shuffle(flat)
    return flat.reshape(shape)
