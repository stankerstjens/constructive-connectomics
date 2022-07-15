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

Hierarchical Decomposition
==========================

Perform decomposition of a :py:class:`.Volume` into a :py:class:`.Hierarchy`.

"""

from functools import partial
from typing import Callable, Optional, Iterable

import numpy as np
from sklearn.decomposition import PCA

from abianalysis.hierarchy import Hierarchy
from abianalysis.volume import Volume

#: A function that takes a hierarchy node and groups its children into
# two new hierarchy nodes.
SplitFn = Callable[['Hierarchy'], Iterable[bool]]


def _init_hierarchy(volume: Volume):
    root = Hierarchy(volume=volume)
    for i in range(volume.n_voxels):
        Hierarchy(voxel_index=i, parent=root)
    return root


def split_node_children(node: Hierarchy, go_left_fn: SplitFn):
    """Split children of a node into two new nodes based on a boolean list.

    This function is used as a utility for node split functions like
    :py:func:`.pca_split` and :py:func:`.random_split`, and is encouraged
    to be used in user-defined functions.

    If this function is called on a node that has zero or one children, it does
    nothing.

    If this function is called on a node with a single child, that child is
    connected to the parent of the node, and the node's parent is set to
    None. Effectively, this 'cuts out the middle node' and makes the
    former grand-child a direct child of its former grand-parent (now its
    parent).

    """
    n_children = len(node.children)
    if n_children == 0:
        return

    if n_children == 1:
        node.parent.replace_child(node, node.children[0])
        return

    go_left = go_left_fn(node)

    if all(go_left) or not any(go_left):
        print(node.leaves_expression)
        raise ValueError(
            "A good split method does not split everything into one group")

    children = list(node.children)
    node.clear_children()

    left = Hierarchy(parent=node)
    right = Hierarchy(parent=node)

    # noinspection PyTypeChecker
    for child, go_left in zip(children, go_left):
        # Assigning the parent automatically adds the child to
        # the parent's list of children
        child.parent = left if go_left else right


def pca_split(node: Hierarchy) -> Iterable[bool]:
    """Split the children of a hierarchy node into two new hierarchy nodes
    based on the projection of their individual expression on the first
    principal component across their expression.

    .. seealso::
        :py:func:`.make_hierarchy`,
        :py:func:`.make_balanced_hierarchy`

    """
    pc = _project_pca(node)
    if len(pc) == 2 and pc[0] == pc[1]:
        return [False, True]
    return pc < pc.mean()


def random_split(node: Hierarchy) -> Iterable[bool]:
    """Split the children of a hierarchy node randomly into two new hierarchy
    nodes.

    If the number of children is 2 or larger, it is guaranteed that there is at
    least one child sorted into each branch.

    .. seealso::
        :py:func:`.make_hierarchy`,
        :py:func:`.make_balanced_hierarchy`

    """
    n_children = len(node.children)

    if n_children < 2:
        return np.ones(n_children, dtype=bool)

    first_left, first_right = np.random.choice(n_children,
                                               size=2,
                                               replace=False)
    draw = np.random.rand(n_children) < .5
    draw[first_left] = True
    draw[first_right] = False
    return draw


def random_projection_split(node: Hierarchy) -> Iterable[bool]:
    """Split the children of a hierarchy node by projecting their gene
    expression onto a random vector.

    .. seealso::
        :py:func:`.make_hierarchy`,
        :py:func:`.make_balanced_hierarchy`

    """
    n_children = len(node.children)

    if n_children < 2:
        return np.ones(n_children, dtype=bool)

    node.component = np.random.randn(node.volume.n_genes)
    proj = node.project_component()
    return proj < proj.mean()


def _balanced_loop(root: Hierarchy, n_iterations: int, body: Callable) -> None:
    """Execute body for descendants in order of decreasing number of
    children.

    At each iteration the largest node is chosen and expanded.

    :param root: The starting node
    :param n_iterations:
        The total number of iterations to be performed, resulting in
        `n_iterations` just-not-leaf nodes (creating the root is counted as
        the first iteration).
    :param body: The function to be performed on each node

    """
    queue = [root]
    for _ in range(n_iterations - 1):
        queue.sort(key=lambda n: len(n.children))
        node = queue.pop()
        body(node)
        queue.extend(node.children)


def _generations_loop(root: Hierarchy, depth: int, body: Callable) -> None:
    """Execute body function for descendants until the fixed depth is
    reached."""
    queue = [root]

    while queue:
        node = queue.pop(0)
        body(node)
        if node.depth < depth - 1:
            queue.extend(node.children)


def make_hierarchy(volume: 'Volume',
                   depth: Optional[int] = None,
                   partition_children: SplitFn = pca_split):
    """Estimate a hierarchy based on the voxels in the volume.

    We start with a single root that contains all children.  Then we
    start traversing the tree starting at the root.  At each iteration
    we split the child voxels in two groups, and assign all (leaf)
    voxels in each group to a one of two new intermediate nodes.  This
    process repeats until the tree is binary, or the maximum depth is
    reached.  If the maximum depth is reached before the tree is fully
    decomposed the last intermediate node has more than two children.

    :param Volume volume:
        The input volume.

    :param Optional[int] depth:
        The number of generations to resolve. By default (by passing
        None) all generations are resolved.  If not all generations are
        resolved the pre-leaves will generally have more than 2
        children.

    :param Optional[str] partition_children:
        The projection method. Default is `pca`. The other option is
        `random`, which projects onto a random direction.


    """

    root = _init_hierarchy(volume)

    if depth is None:
        # A very large value we should never reach, but which prevents
        # a truly infinite loop.
        depth = 1000

    elif depth <= 0:  # Then we are done already
        return root

    _generations_loop(root, depth,
                      partial(split_node_children,
                              go_left_fn=partition_children))

    return root


def make_balanced_hierarchy(volume: Volume,
                            n_iterations: Optional[int] = None,
                            partition_children: SplitFn = pca_split):
    """Make a hierarchy from a volume, resolving a fixed number of iterations.

    The result is balanced, in the sense that at every iteration the largest
    hierarchy node is split. So, the resulting regions will have roughly the
    same sizes (which is not the case for :py:func:`.make_hierarchy`).

    :param volume:  The volume to decompose.
    :param n_iterations:
        The total number of node splits that should be performed.
    :param partition_children:
        A function splitting a node by grouping its current children to two new
        hierarchy nodes.

    .. seealso:: :py:func:`.make_hierarchy:

    """
    root = _init_hierarchy(volume)

    if n_iterations is None:
        n_iterations = 2 * len(root.children)

    _balanced_loop(root, n_iterations,
                   partial(split_node_children, go_left_fn=partition_children))

    return root


def _project_pca(n: Hierarchy) -> np.ndarray:
    """Project leaves_expression to their first principal component"""
    exp = n.leaves_expression
    assert len(exp) >= 2, "PCA on fewer than two points is not likely to " \
                          "be the right thing to do."
    pca = PCA(n_components=1, copy=True)
    pc_coef = pca.fit_transform(exp)[:, 0]
    n.component = pca.components_[0]
    return pc_coef
