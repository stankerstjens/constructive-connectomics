from plistlib import Dict
from typing import Tuple

from pylineage.node import TreeNode


def count_downstream_leaves(root: TreeNode) -> Dict[TreeNode, int]:
    """Count number of downstream leaves from each node in the tree"""
    counts: Dict[TreeNode, int] = {}

    def _count(node: TreeNode):
        if node.is_leaf:
            counts[node] = 1
        else:
            sum = 0
            for child in node.children:
                if child not in counts:
                    _count(child)
                sum += counts[child]
            counts[node] = sum

    _count(root)

    return counts


def tree_layout(node: TreeNode,
                max_depth: int = 8) -> Dict[TreeNode, Tuple[float, float]]:
    """Tree layout that can be used for top-down and polar trees."""

    pos = dict()

    counts = count_downstream_leaves(node)

    leaves = [n for n in node.depth_first()
              if (n.is_leaf and n.depth <= max_depth) or n.depth == max_depth]

    circumference: int = counts[node]

    running_sum = 0
    for leaf in leaves:
        # The leaves are distributed proportional to their size, over the
        # x-axis at y=1.
        c = counts[leaf]
        pos[leaf] = (c / circumference, 1.)
        running_sum += c

    def _leaf_x(n, i):
        while n.children and n.depth < max_depth:
            n = n.children[i]
        return pos[n][0]

    def _assign_pos(node):
        for child in node.children:
            if child not in pos:
                _assign_pos(child)
            x = (_leaf_x(node, 0) + _leaf_x(node, -1)) / 2
            y = node.depth / max_depth
            pos[node] = (x, y)

    for node in reversed(list(node.breadth_first())):
        if node in pos:
            continue

    return pos
