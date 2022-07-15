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

Voxel Graph Factories
=====================

Provides methods to produce a spatial graph from voxels.

.. seealso:: :py:class:`.VoxelGraph`

"""
from typing import Callable, Tuple, Iterable, List

import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay, cKDTree

from abianalysis.spatial.graph import VoxelGraph
from abianalysis.volume import Volume

EdgeGenerator = Callable[[np.ndarray], Iterable[Tuple[int, int]]]


def gabriel_edges(points: np.array):
    """Return the edges of the Gabriel graph among the points.

    The Gabriel graph is a connected sub-graph of the Delaunay
    triangulation

    :param points:
        (n_points x n_dimensions) matrix of points in Euclidean space.

    """
    tri = Delaunay(points)
    lil = lil_matrix((len(points), len(points)))
    indices, indptr = tri.vertex_neighbor_vertices
    for k in range(len(points)):
        lil.rows[k] = list(indptr[indices[k]:indices[k + 1]])
        lil.data[k] = list(np.ones_like(lil.rows[k]))
    coo = lil.tocoo()
    conns = np.vstack((coo.row, coo.col)).T
    delaunay_conns = np.sort(conns, axis=1)

    c = tri.points[delaunay_conns]
    m = (c[:, 0, :] + c[:, 1, :]) / 2
    r = np.sqrt(np.sum((c[:, 0, :] - c[:, 1, :]) ** 2, axis=1)) / 2

    # noinspection PyArgumentList
    tree = cKDTree(points)
    # noinspection PyUnresolvedReferences
    n = tree.query(x=m, k=1)[0]
    g = n >= r * 0.999
    return np.unique(delaunay_conns[g], axis=0)


def delaunay_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    """Iterate over edges corresponding to a Delaunay graph among the
    points.

    The Delaunay graph connects points so that the circle through every
    triangle of connected points contains no other points.

    :param points:
        (n_points x n_dimensions) matrix of points in Euclidean space.

    """
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    edges = set()
    for i, _ in enumerate(points):
        for neighbor in indices[indptr[i]:indptr[i + 1]]:
            if i != neighbor:
                edge = (i, neighbor) if i < neighbor else (neighbor, i)
                edges.add(edge)
    return list(edges)


def voxel_graph_from_volume(volume: Volume,
                            edges: EdgeGenerator = gabriel_edges
                            ) -> VoxelGraph:
    """Make a VoxelGraph from a volume and an edge generator.

    :param volume: The volume whose voxels are to be connected
    :param edges:
        A callable taking an array of points (e.g. see
        :py:func:`.gabriel_edges`) and iterating over tuples of voxel
        indices that are adjacent in the graph.

    """
    return VoxelGraph(volume.n_voxels, edges(volume.voxel_indices))
