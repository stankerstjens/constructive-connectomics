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

from functools import cached_property
from typing import List, Union, Iterable, Optional

import igraph

from abianalysis.guidance.axon import Axon
from abianalysis.guidance.factory import make_guidance_graphs
from abianalysis.guidance.find_axon import find_axon
from abianalysis.hierarchy import Hierarchy
from abianalysis.spatial.graph import VoxelGraph
from pylineage.node import TreeNode


def just_not_leaves(root: TreeNode) -> Iterable[TreeNode]:
    for n in root.descendants():
        if any(c.is_leaf for c in n.children):
            yield n


class GuidanceGraph:
    """

    A vertex is the combination of a hierarchy node and voxel.  (There is a
    vertex for each voxel for each hierarchy node.)

    """

    @classmethod
    def create(cls, hierarchy: Hierarchy, voxel_graph: VoxelGraph,
               **kwargs) -> 'GuidanceGraph':
        """Create a guidance graph from a hierarchy, and a voxel graph.

        :param hierarchy:  The hierarchy for the guidance graph
        :param voxel_graph:  The spatial voxel graph.
        :param kwargs:
            Any other parameter that can be passed to
            :py:func:`.make_guidance_graphs`

        """
        graph = GuidanceGraph()
        graph._up_graph, graph._down_graph = make_guidance_graphs(
            hierarchy, voxel_graph, **kwargs)
        return graph

    def __init__(self):
        self._up_graph = igraph.Graph()
        self._down_graph = igraph.Graph()

    def get_vertex_from_axon(self, axon: Axon, axon_vertex):
        """Convert a vertex inside an axon to a vertex of the guidance
        graph."""
        return self.get_vertex(axon.get_hierarchy(axon_vertex),
                               axon.get_voxel_index(axon_vertex))

    def get_voxel_index(self, vertex: int):
        """Convert a vertex indices to the voxel indices.

        :param vertex: A single vertex, or a list of vertices.

        """
        return self._up_graph.vs['voxel'][vertex]

    @property
    def sources(self):
        """A list of all nodes with zero in-degree (in the up-graph)"""
        return [v.index for v in self._up_graph.vs.select(
            hierarchy_in=set(just_not_leaves(self.hierarchy))
        )]

    @cached_property
    def hierarchy(self):
        """Get the hierarchy this graph was based on.

        The hierarchy is derived as the root of the hierarchy of the first
        vertex.  It is assumed that all hierarchies in the axon are branches
        of a single hierarchy.

        """
        return self._up_graph.vs['hierarchy'][0].root()

    def get_leaf_vertex(self, voxels: Union[int, List[int]]
                        ) -> Optional[List[int]]:
        """Given a list of voxels, return all vertices that correspond to a
        leaf of the hierarchy."""

        try:
            voxels = list(voxels)
        # not iterable
        except TypeError:
            voxels = [voxels]

        result = [s.index for s in self._up_graph.vs.select(
            voxel_in=voxels,
            hierarchy_in=set(just_not_leaves(self.hierarchy)),
        )]
        if len(result) == 1:
            result = result[0]
        return result

    def get_source_vertex(self, voxel: int):
        """Get the source vertex corresponding to the provided voxel."""
        selection = self._up_graph.vs.select(_indegree=0, voxel=voxel)
        assert len(selection) == 1
        return selection[0].index

    def get_hierarchy(self, vertex: int) -> Hierarchy:
        """Get the hierarchy connected to a vertex id."""
        return self._up_graph.vs['hierarchy'][vertex]

    def get_region_vertices(self, hierarchy: Hierarchy):
        """Get the vertices corresponding to a hierarchy node."""
        return self._up_graph.vs.select(hierarchy=hierarchy)

    def get_landscape(self, hierarchy: Hierarchy):
        """Get the landscape values for the vertices of the given hierarchy."""
        return self.get_region_vertices(hierarchy)['landscape']

    def get_region_voxels(self, hierarchy: Hierarchy):
        """Get the voxel indices"""
        return self.get_region_vertices(hierarchy)['voxel']

    def get_vertex(self, hierarchy: Hierarchy, voxel_index: int) -> int:
        """Convert a voxel index to a vertex."""
        selection = self._up_graph.vs.select(
            hierarchy=hierarchy, voxel=voxel_index)
        assert len(selection) == 1, f'{len(selection)} is not 1'
        return selection[0].index

    def find_axon(self, source_vertex: int) -> Axon:
        """Get the predicted axonal tree starting from a vertex.

        :param source_vertex:
            The source vertex to start the tree from.  This should be a
            vertex corresponding to a just-not-leaf hierarchy node.

        """
        return find_axon(self._up_graph, self._down_graph, source_vertex)
