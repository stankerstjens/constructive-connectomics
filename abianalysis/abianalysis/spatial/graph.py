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

Spatial Graphs
==============

"""

import igraph
import numpy as np


class VoxelGraph:
    """An undirected graph on the voxels of a :py:class:`.Volume`.  Used to
    represent spatial neighbors of voxels.

    :param n_voxels: the number of voxel in the graph.
    :param edges: An iterator over edges (tuples of two voxel indices).

    """

    def __init__(self, n_voxels, edges):
        self._graph = igraph.Graph(directed=False)
        self._graph.add_vertices(n_voxels)
        self._graph.add_edges(edges)
        self._graph.to_directed(mode='mutual')

    @property
    def n_edges(self) -> int:
        """The number of edges in the graph."""
        return len(self._graph.es)

    @property
    def n_vertices(self) -> int:
        """The number of vertices (voxels) in the graph."""
        return len(self._graph.vs)

    @property
    def edges(self) -> np.ndarray:
        """Array of source-target pairs of voxel indices."""
        return np.array([(e.source, e.target) for e in self._graph.es])

    @property
    def vertices(self) -> np.ndarray:
        """The vertices are voxel indices, which simply range from 0 to the
        total number of voxels."""
        return np.arange(len(self._graph.vs))

    def get_neighbors(self, voxel: int):
        return self._graph.neighbors(voxel)

    def get_gradient(self, vertex_signal: np.ndarray) -> np.ndarray:
        """Calculate the gradient (difference between connected voxels) of a
        vertex signal.

        :param vertex_signal:
            An array of values of shape (n_voxels, n_signals), one for each
            voxel (in the same order as the voxels are listed in the volume).

        :return:
            Array of gradient values, in the order of
            :py:meth:`.VoxelGraph.edges`, with shape (n_edges, n_signals).

        """
        src, tar = self.edges.T
        return vertex_signal[tar] - vertex_signal[src]
