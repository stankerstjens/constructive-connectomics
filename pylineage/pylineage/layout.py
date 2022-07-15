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

Layout
======

Functions for spatial layouts of Trees.

"""
import math
from collections import deque
from typing import Dict, Tuple, Optional

from pylineage.node import TreeNode

Coordinate = Tuple[float, float]
Layout = Dict[TreeNode, Coordinate]


def tree_layout(root: TreeNode, depth: Optional[int] = None) -> Layout:
    """Tree layout that can be used for top-down and polar trees."""

    pos = {}
    all_leaves = [n for n in root.depth_first() if n.is_leaf]
    n_leaves = len(all_leaves)

    if depth is None:
        nodes = list(root.breadth_first())
        leaves = all_leaves
    else:
        nodes = [n for n in root.breadth_first()
                 if depth is None or n.depth <= depth]
        leaves = [n for n in root.depth_first()
                  if n.depth == depth or (n.depth < depth and n.is_leaf)]

    for idx, leaf in enumerate(filter(lambda x: x.is_leaf, root.depth_first())):
        # The leaves are evenly distributed over the x-axis at y=1.
        pos[leaf] = (idx / n_leaves, 1.)

    def _leaf_x(n, i):
        while n.children:
            n = n.children[i]
        return pos[n][0]

    max_depths = dict()
    for n in leaves:
        d = n.depth
        for a in n.ancestors():
            if a not in max_depths or max_depths[a] < d:
                max_depths[a] = d

    for node in reversed(nodes):
        if node in pos:
            continue
        x = (_leaf_x(node, 0) + _leaf_x(node, -1)) / 2
        y = node.depth / max_depths[node]
        pos[node] = (x, y)

    return pos


def radial_tree_layout(node: TreeNode, depth=None) -> Dict[
        TreeNode, Tuple[float]]:
    return {n: (y * math.cos(2 * x * math.pi), y * math.sin(2 * x * math.pi))
            for n, (x, y) in tree_layout(node, depth=depth).items()}


def h_tree_layout(root: TreeNode) -> Dict[TreeNode, Tuple[float]]:
    """Layout the nodes of a tree along an H-tree.

    If a node has exactly two children, they are put on the end-points
    of a line-segment orthogonal to the previous line segment, with
    either the same length, or half the length. If a node has more than
    two children, the children are linearly dispersed over the next line
    segment.

    :param root:  The root TreeNode
    :return:  A dictionary giving an x, y coordinate for each node in
        the tree.  All coordinate asymptotically reach to a range
        [-1, 1] as the tree grows in number of generations.

    """
    pos = {root: (0, 0)}
    queue = deque([(root, 1, True)])
    while queue:
        node, size, hor = queue.popleft()
        new_size = size / 2 if hor else size
        x, y = pos[node]
        nc = len(node.children)
        if nc == 2:
            left, right = node.children
            pos[left] = (x + new_size, y) if hor else (x, y + new_size)
            pos[right] = (x - new_size, y) if hor else (x, y - new_size)
            queue.append((left, new_size, not hor))
            queue.append((right, new_size, not hor))
        elif nc > 2:
            for i, child in enumerate(node.children):
                s = (i / nc - .5) * new_size
                pos[child] = (x + s, y) if hor else (x, y + s)
        else:
            for child in node.children:
                pos[child] = pos[node]
    return pos
