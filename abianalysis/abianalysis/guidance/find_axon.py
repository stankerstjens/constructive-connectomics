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

from itertools import repeat
from typing import List

import igraph

from abianalysis.guidance.axon import Axon


def get_reachable_single_source(graph: igraph.Graph, source: int):
    """Find all nodes reachable from a single source by following arcs."""
    return graph.subcomponent(source, mode='out')


def get_reachable_multiple_sources(graph: igraph.Graph, sources: List[int]):
    """Find all nodes reachable from any of multiple sources by following
    arcs."""
    dummy_down_source: igraph.Vertex = graph.add_vertex()
    graph.add_edges(
        zip(repeat(dummy_down_source), sources),
        attributes={'weight': 100}
    )
    down_reachable = graph.subcomponent(
        dummy_down_source, mode='out')
    graph.delete_vertices(dummy_down_source.index)
    down_reachable.remove(dummy_down_source.index)
    return down_reachable


def find_axon(up_graph: igraph.Graph, down_graph: igraph.Graph, source: int):
    """Get the predicted axonal tree starting from a vertex.

    :param up_graph:
        A graph containing all arcs that move up the state hierarchy.
    :param down_graph:
        A graph containing all arcs moving down the state hierarchy.
    :param source: The source vertex to start exploration from.

    .. seealso:: :py:class:`.GuidanceGraph.find_axon`

    """
    up_reachable = get_reachable_single_source(up_graph, source)
    up_axon = up_graph.induced_subgraph(up_reachable)

    down_reachable = get_reachable_multiple_sources(down_graph, up_reachable)
    down_axon = down_graph.induced_subgraph(down_reachable)

    axon_graph: igraph.Graph = igraph.union([up_axon, down_axon])
    source_selection = axon_graph.vs.select(name=source)
    assert len(source_selection) == 1
    axon_branches = axon_graph.get_shortest_paths(
        source_selection[0],
        to=axon_graph.vs.select(_outdegree=0),
        mode='out',
        weights='weight',
        output='epath'
    )

    edges = list(set(e for branch in axon_branches for e in branch))

    axon_tree: igraph.Graph = axon_graph.subgraph_edges(edges)
    if len(axon_tree.vs) == 0:
        axon_tree = axon_graph.induced_subgraph(source_selection[0])

    assert len(axon_tree.vs.select(_indegree=0)) == 1, \
        "There should only be a single source"

    return Axon.from_igraph_tree(axon_tree)
