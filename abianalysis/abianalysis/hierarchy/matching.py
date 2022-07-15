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

Matching nodes between Hierarchies
==================================

Hierarchies established separately, even from the same volume, are not
automatically matched, so that when they are simultaneously traversed
corresponding nodes are reached at the same time.

This module also provides functionality to compare hierarchies, and compare
projections of volumes onto hierarchies.

"""

import collections
from typing import List, Optional, Dict, Tuple

import numpy as np

from abianalysis import Hierarchy


def _common_gene_selectors(g1, g2):
    mask1 = np.isin(g1, g2)
    mask2 = np.isin(g2, g1)
    return mask1, mask2


def rotate_to_match(hierarchy1: Hierarchy,
                    hierarchy2: Hierarchy) -> Dict[Hierarchy, Hierarchy]:
    """Rotate the children of the passed hierarchy to match the
    current hierarchy.

    Rotation is based on the component direction: if the correlation
    between the component of the node in this hierarchy and the
    component in the corresponding node of the other hierarchy is
    negative, then the children of the node of the other hierarchy
    are put in reverse order (i.e. rotated).

    Only genes the hierarchies (or more accurately their volumes) have in
    common are considered.

    :param hierarchy1:
        The first of two hierarchies to match.  This hierarchy will not be
        modified.

    :param hierarchy2:
        The second of two hierarchies to match.  This hierarchy will be
        modified, so that the order of children, and the sign of the
        components, matches `hierarchy1`.

    :return:
        A dictionary from `hierarchy1` to `hierarchy2`, making pairs of matched
        hierarchy nodes.

    """

    match_map = {}

    mask1, mask2 = _common_gene_selectors(hierarchy1.volume.genes,
                                          hierarchy2.volume.genes)

    def _rotate_condition(_node, _other_node):
        return np.dot(_node.component[mask1], _other_node.component[mask2]) < 0

    queue = [(hierarchy1, hierarchy2)]
    while queue:
        node1, node2 = queue.pop()
        match_map[node1] = node2

        if len(node1.children) == 2 and len(node2.children) == 2:
            if (node1.component is not None and node2.component is not None
                    and _rotate_condition(node1, node2)):
                node2._children = node2.children[::-1]
                node2.component = -node2.component
            queue.extend(zip(node1.children, node2.children))

    return match_map


def match_components(node_map: Dict[Hierarchy, Hierarchy]):
    """Calculate the average correlation of components between two
    mapped hierarchies, weighted by the depth of the nodes.

    A deeper node contributes a smaller score.

    :param node_map:
        A dictionary mapping nodes between two hierarchies.  See
        :py:func:`.rotate_to_match`

    """

    # These n1, n2 are just used to get the gene names and masks.
    n1, n2 = next(iter(node_map.items()))
    mask1, mask2 = _common_gene_selectors(n1.volume.genes, n2.volume.genes)

    def _score(_n1: Hierarchy, _n2: Hierarchy):
        return np.corrcoef(_n1.component[mask1], _n2.component[mask2])[0, 1]

    total_score = 0
    normalization_factor = 0
    for n1, n2 in node_map.items():
        assert n1.depth == n2.depth, \
            'Two mapped nodes will, by construction, have the same depth'
        if n1.component is not None and n2.component is not None:
            weight = 1 / 2 ** n1.depth
            total_score += _score(n1, n2) * weight
            normalization_factor += weight

    return total_score / normalization_factor


def score_voxel_to_hierarchy_match(voxels1: List[Hierarchy],
                                   voxels2: List[Hierarchy],
                                   node_map: Optional[
                                       Dict[Hierarchy, Hierarchy]] = None,
                                   depth=None):
    """Calculate the extent to which two lists of hierarchies.
    `match_1` and `match_2` are lists of hierarchy nodes, but from different
    hierarchies.  Therefore, to check agreement one must first determine
    which node in one hierarchy corresponds to which node in the other
    hierarchy.

    :param depth:
    :param voxels1: A list of Hierarchy nodes, one for each voxel
    :param voxels2: Same as match 1
    :param node_map:
        A map from nodes between two hierarchies.  If None, the hierarchies
        are assumed to be identical.

    """

    assert len(voxels1) == len(voxels2), \
        "The number of voxels must be equal"

    if node_map:
        assert voxels1[0].root() in node_map, \
            "The root of the hierarchy should always be in the node_map"

        def _map_nodes(_h1, _h2):
            _h1 = next(filter(node_map.__contains__, _h1.ancestors()), None)
            return node_map[_h1], _h2

    else:
        if voxels1[0].root() != voxels2[0].root():
            raise ValueError('Voxel lists must be matched to the same '
                             'hierarchy')

        def _map_nodes(_h1, _h2):
            return _h1, _h2

    if depth is not None:
        voxels1 = [v.ancestor_at_depth(depth) for v in voxels1]
        voxels2 = [v.ancestor_at_depth(depth) for v in voxels2]

        def score(h1, h2):
            return 1 if h1 == h2 else 0

    else:
        def score(h1, h2):
            common_ancestor = h1.least_common_ancestor(h2)
            return 1 - 1 / 2 ** common_ancestor.depth

    i = 0
    for h1, h2 in (_map_nodes(h1, h2) for h1, h2 in zip(voxels1, voxels2)):
        i += score(h1, h2)
    n = len(voxels1)

    return i / n


def match_voxels_to_hierarchy_nodes(hierarchy: Hierarchy,
                                    expression: np.ndarray,
                                    depth=None) -> List[Hierarchy]:
    """Match each voxel (row of the expression matrix) to a node in
    the hierarchy.

    :param depth:
    :param hierarchy:
        The root node of the hierarchy the voxels should be sorted
        into.

    :param expression:
        An expression matrix where each row is a voxel and each column a gene.

    :return: A list of hierarchy nodes where the index in the list
        corresponds to the row index of the expression/position
        matrices.

    """
    n_voxels, n_genes = expression.shape
    result = np.zeros(n_voxels, dtype=object)

    queue = collections.deque([(hierarchy, np.arange(n_voxels))])

    while queue:
        node, voxel_indices = queue.pop()

        if ((depth is not None and node.depth > depth)
                or node.component is None):
            result[voxel_indices] = node

        else:
            exp = expression[voxel_indices]
            pc = node.project_component(exp - exp.mean(0))
            go_left = pc < pc.mean()

            left, right = node.children
            if any(go_left):
                queue.append((left, voxel_indices[go_left]))
            if not all(go_left):
                queue.append((right, voxel_indices[~go_left]))

    assert not any(r is None for r in result), \
        "There should be no remaining None values in the result " \
        "list"

    return list(result)
