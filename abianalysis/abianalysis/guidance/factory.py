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

from functools import partial
from typing import Callable

import igraph
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from abianalysis import Hierarchy
from abianalysis.spatial.graph import VoxelGraph


def normalized_weight(gradient: np.ndarray) -> np.ndarray:
    """Normalized gradients to a range of [0, 1], where the largest gradient
    is mapped to 0 and the smallest to 1"""
    return 1 - minmax_scale(gradient)


def correlation_landscape(hierarchy: Hierarchy, threshold: float):
    """Returns the correlation of all voxels with each of the nodes in the
    hierarchy.

    :returns:
        a matrix with one row per voxel and one column per hierarchy node.
    """
    interior = list(hierarchy.interior())
    corr = 1 - cdist(
        hierarchy.volume.expression,
        np.stack([h.expression for h in interior]),
        metric='correlation')
    corr[corr < threshold] = 0
    return dict(zip(interior, corr.T))


def threshold_edge_mask(gradient: np.ndarray, threshold: float):
    return gradient > threshold


_default_landscape = partial(correlation_landscape, threshold=.1)
_default_edge_mask = partial(threshold_edge_mask, threshold=.1)


def make_guidance_graphs(
        hierarchy: Hierarchy,
        voxel_graph: VoxelGraph,
        hierarchy_to_landscape: Callable = _default_landscape,
        gradient_to_weight: Callable = normalized_weight,
        edge_mask: Callable = _default_edge_mask):
    """Creates a pair of up and down guidance graphs.

    :param hierarchy: Hierarchy providing the states of the graph.
    :param voxel_graph:
        A VoxelGraph providing the spatial template among the voxels.
    :param hierarchy_to_landscape:
        A function taking a hierarchy, and converting it to a dictionary of
        from hierarchy to landscapes.  (Each landscape is a signal over all
        voxels).
    :param gradient_to_weight:
        A function from gradient to graph weight
    :param edge_mask:
        A function producing an edge mask from gradients.

    """

    hierarchies, landscapes = zip(*hierarchy_to_landscape(hierarchy).items())
    gradients = voxel_graph.get_gradient(np.vstack(landscapes).T)
    weights = gradient_to_weight(gradients)

    branching = None

    n_voxels = hierarchy.volume.n_voxels
    up_graph = igraph.Graph(directed=True)
    for h, w, g, ls in tqdm(zip(hierarchies, weights.T, gradients.T,
                                landscapes), total=len(weights.T),
                            desc='Making guidance graph'):
        mask = edge_mask(g)
        edges = voxel_graph.edges[mask]

        graph = igraph.Graph(n_voxels, directed=True)
        graph.vs['voxel'] = np.arange(n_voxels)
        graph.vs['landscape'] = ls
        graph.add_edges(edges, attributes={'weight': w[mask]})

        assert len(edges) == len(w[mask])

        if branching is not None:
            to_delete = []
            for vertex in graph.vs.select(_outdegree_gt=1):
                out = graph.incident(vertex, mode='out')
                weight = graph.es[out]['weight']
                sort = np.argsort(weight)
                to_delete.extend([out[i] for i in sort[branching:]])
            graph.delete_edges(to_delete)

        region = graph.induced_subgraph(h.voxels)
        if len(region.es) == 0:
            region.es['weight'] = []

        n = len(up_graph.vs)
        up_graph.add_vertices(len(region.vs), attributes={
            'voxel': region.vs['voxel'],
            'landscape': region.vs['landscape'],
            'hierarchy': h
        })

        up_graph.add_edges(
            [(n + e.source, n + e.target) for e in region.es],
            attributes={'weight': region.es['weight']}
        )

    down_graph = up_graph.copy()
    for h in hierarchies:
        if h.parent:
            _add_transition_edges(up_graph, h, h.parent)
            _add_transition_edges(down_graph, h.parent, h)

    up_graph.vs['name'] = [v.index for v in up_graph.vs]
    down_graph.vs['name'] = [v.index for v in down_graph.vs]

    return up_graph, down_graph


def _add_transition_edges(graph: igraph.Graph,
                          from_hierarchy: Hierarchy,
                          to_hierarchy: Hierarchy):
    overlap = set(from_hierarchy.voxels).intersection(to_hierarchy.voxels)
    srcs = graph.vs.select(hierarchy=from_hierarchy, voxel_in=overlap)
    tars = graph.vs.select(hierarchy=to_hierarchy, voxel_in=overlap)
    assert all(s['voxel'] == t['voxel'] for s, t in zip(srcs, tars))
    graph.add_edges(zip(srcs, tars),
                    attributes={'weight': [100] * len(overlap)})
