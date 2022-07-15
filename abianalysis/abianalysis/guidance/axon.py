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
from typing import List, Optional

import igraph
import numpy as np
from scipy.spatial.distance import cdist

from abianalysis.hierarchy import Hierarchy
from abianalysis.volume import Volume


def _remove_consecutive_duplicates(ids: List[int]) -> np.ndarray:
    """

    >>> _remove_consecutive_duplicates([1, 2, 3, 3, 2, 4, 4, 4, 1])
    np.array([1, 2, 3, 2, 4, 1])

    """
    ids = np.asarray(ids)
    diff = np.array([True, *np.diff(ids)])
    return ids[diff != 0]


class Axon:
    """An axon is a tree-shaped graph starting at a node of the guidance
    graph, and extending to one or more targets.
    """

    @classmethod
    def from_igraph_tree(cls, tree):
        """Create an axon from an igraph Graph object."""
        axon = Axon()
        axon._tree = tree
        return axon

    def __init__(self):
        self._tree = igraph.Graph()

    def get_voxel_edges(self) -> np.ndarray:
        """Get all edges between voxel indices."""
        edges = np.array([[e.source, e.target] for e in self._tree.es],
                         dtype=int)
        idx = self.get_voxel_index(edges.flatten()).reshape(-1, 2)
        return np.unique(idx, axis=0)

    @property
    def voxel_edges(self):
        """Get all edges between voxel indices."""
        return self.get_voxel_edges()

    def get_hierarchy(self, vertex: int):
        """Get hierarchy from vertex.

        :param vertex: Either a single vertex, or a list of vertices.

        """
        return self._tree.vs['hierarchy'][vertex]

    def get_voxel_index(self, vertex):
        """Get voxel indices from vertex indices.

        :param vertex: Either a single vertex, or a list of vertices.

        """
        return np.array(self._tree.vs['voxel'])[vertex]

    @property
    def source(self) -> int:
        """The source vertex of the axon"""
        selection = self._tree.vs.select(_indegree=0)
        assert len(selection) == 1, 'There should only be a single source.'
        return selection[0].index

    @property
    def source_voxel(self) -> int:
        """The source voxel of the axon"""
        return self.get_voxel_index(self.source)

    @property
    def targets(self) -> List[int]:
        """The target vertices."""
        just_not_leaves = set(
            filter(lambda n: any(child.is_leaf for child in n.children),
                   self.hierarchy.descendants()))
        selection = self._tree.vs.select(hierarchy_in=just_not_leaves)
        return [v.index for v in selection]

    @property
    def reached_voxels(self) -> List[int]:
        """Get the set of all reached voxels"""
        # return list(set(self._tree.vs['voxel']))
        return list(set(self._tree.vs['voxel']))

    @property
    def _voxel_tree(self):
        """Axonal tree ignoring state"""
        tree = self._tree.copy()
        unique_voxels = np.unique(tree.vs['voxel'])
        idx = {v: i for i, v in enumerate(unique_voxels)}
        tree.contract_vertices([idx[v] for v in tree.vs['voxel']])
        tree.simplify()
        tree.vs['voxel'] = unique_voxels
        return tree

    @property
    def tips(self):
        """All the tip voxels of the axon"""
        return self._voxel_tree.vs.select(_outdegree=0)['voxel']

    @property
    def hierarchy(self) -> Optional[Hierarchy]:
        """Get the hierarchy this graph was based on.

        The hierarchy is derived as the root of the hierarchy of the first
        vertex.  It is assumed that all hierarchies in the axon are branches
        of a single hierarchy.

        """
        if 'hierarchy' in self._tree.vs.attributes():
            return self._tree.vs['hierarchy'][0].root()

    @property
    def volume(self) -> Optional[Volume]:
        """Get the volume this axon lives in."""
        if self.hierarchy is not None:
            return self.hierarchy.volume

    @property
    def voxel_paths(self):
        """Get all paths in terms of voxels, omitting state transitions in
        the same voxel."""
        return [self._voxel_tree.vs[p]['voxel'] for p in
                self._voxel_tree.get_shortest_paths(
                    self._voxel_tree.vs.select(voxel=self.source_voxel)[0],
                    to=self._voxel_tree.vs.select(voxel_in=self.tips),
                    mode='out',
                )]

    @property
    def branch_paths(self) -> List[List[int]]:
        """Get all paths, but each path starts at a terminal and ends
        in a voxel of another path, instead of from root to terminal.

        The branches are ordered in descrending length

        """
        visited_voxels = set()
        branches = []
        for path in sorted(self.voxel_paths, key=lambda p: -len(p)):
            branch = []
            for p in path[::-1]:
                branch.append(int(p))
                if p in visited_voxels:
                    break
            visited_voxels.update(branch)
            branches.append(branch)
        return branches

    @cached_property
    def paths(self) -> List[List[int]]:
        """Get a list of all paths from the source to each target, in terms
        of vertex sequences."""
        return self._tree.get_shortest_paths(
            self.source,
            to=self.targets,
            mode='out',
            weights='weight',
        )

    @property
    def edge_paths(self):
        """Get a list of all paths from source to target, in terms of edge
        sequences"""
        return self._tree.get_shortest_paths(
            self.source,
            to=self.targets,
            mode='out',
            weights='weight',
            output="epath"
        )

    def overlap(self, other_axon: 'Axon'):
        s1 = set(self.reached_voxels)
        s2 = set(other_axon.reached_voxels)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def average_minimum_distance(self, other_axon: 'Axon'):
        """Calculates the average minimum distance between the voxels
        reached by two axons"""
        this_pos = self.volume.voxel_indices[self.reached_voxels]
        that_pos = self.volume.voxel_indices[other_axon.reached_voxels]
        dist = cdist(this_pos, that_pos, metric='euclidean')
        return (dist.min(axis=0).mean() + dist.min(axis=1).mean()) / 2

    def average_minimum_distance_tips(self, other_axon: 'Axon'):
        """Calculates the average minimum distance between the voxels
        reached by two axons"""
        this_pos = self.volume.voxel_indices[self.tips]
        that_pos = self.volume.voxel_indices[other_axon.tips]
        dist = cdist(this_pos, that_pos, metric='euclidean')
        return (dist.min(axis=0).mean() + dist.min(axis=1).mean()) / 2
