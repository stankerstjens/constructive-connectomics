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
"""TreeNode implementing a double linked tree with a single parent per node"""
from collections import deque
from itertools import islice
from typing import Optional, TypeVar, Generic, List, Iterable

from pylineage.node import Node

# The type of any node inheriting from TreeNode (e.g. Hierarchy).
SubNode = TypeVar('SubNode', bound='TreeNode')

# The type of the item contained in the node.
T = TypeVar('T')


class TreeNode(Generic[SubNode, T], Node[SubNode, T]):
    """Each TreeNode has a single parent (or None), and any number of children.

    The best way to connect two nodes is by setting the parent.  This will
    automatically add the child to the children list of the (new) parent.
    """

    def __init__(self, parent: Optional[SubNode] = None,
                 item: Optional[T] = None, *args,
                 **kwargs) -> None:
        super(TreeNode, self).__init__(item=item)
        self._parent: Optional['TreeNode'] = None
        self._children: List['TreeNode'] = []
        self.depth: int = 0

        # calls the parent setter, which also updates the depths.
        self.parent = parent

    @property
    def parent(self) -> Optional[SubNode]:
        """The only (optional) parent."""
        return self._parent

    @parent.setter
    def parent(self, parent: Optional[SubNode]):
        """Setting the parent creates a bidirectional connection."""
        if self._parent is not None:
            self._parent._children.remove(self)
        self._parent = parent
        if self._parent is not None:
            self._parent._children.append(self)
        self._update_depths()

    @property
    def children(self) -> List[SubNode]:
        """A list of the children nodes"""
        return self._children

    @property
    def neighbors(self) -> List[SubNode]:
        """The parent plus the three children"""
        if self.is_root:
            return self.children
        return [self.parent, *self.children]

    def replace_child(self, child: SubNode, new_child: SubNode) -> None:
        """Replace a child with a new child."""
        try:
            # will raise a ValueError if the child is not in the children list
            i = self.children.index(child)
        except ValueError:
            raise ValueError("Child not in children")
        self._children[i] = new_child
        new_child._parent = self
        child._parent = None
        child._update_depths()

    def ancestor_at_depth(self, depth: int) -> SubNode:
        """Returns the ancestor at the specified depth.  If the depth is larger
        than the depth of this node, the node itself is returned."""
        up_steps = self.depth - depth
        if up_steps < 0:
            return self
        return next(islice(self.ancestors(), up_steps, up_steps + 1))

    def clear_children(self) -> None:
        """Remove all children from this node."""
        for child in self._children:
            child._parent = None
            child._update_depths()
        self._children = []

    def ancestors(self) -> Iterable[SubNode]:
        """Traverse through all ancestors starting at the current node."""
        cell = self
        while cell is not None:
            yield cell
            cell = cell.parent

    def interior(self) -> Iterable[SubNode]:
        """Traverse through all descendants that are not leaves, in the same
        order as `descendants`"""
        yield from filter(lambda n: not n.is_leaf, self.descendants())

    def descendants(self) -> Iterable[SubNode]:
        """Traverse through all descendants in a breadth-first order"""
        yield from self.breadth_first()

    def breadth_first(self) -> Iterable[SubNode]:
        """Traverse through all descendants in a breadth-first order"""
        queue = deque()
        queue.append(self)
        while queue:
            cell = queue.popleft()
            yield cell
            queue.extend(cell.children)

    def depth_first(self) -> Iterable[SubNode]:
        """Traverse through all descendants in a depth-first order"""
        stack = [self]
        while stack:
            cell = stack.pop()
            yield cell
            stack.extend(cell.children)

    def leaves(self) -> Iterable[SubNode]:
        """Iterate over all leaves downstream of this node.  Uses the same
        traversal order as `descendants`."""
        for d in self.descendants():
            if d.is_leaf:
                yield d

    def root(self) -> SubNode:
        """Get the root of the tree, i.e. the first ancestor that does not
        itself have a parent."""
        root = self
        for root in self.ancestors():
            pass
        return root

    @property
    def is_root(self) -> bool:
        """A TreeNode is a root if it has no parent."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """A TreeNode is a leaf if it has no children."""
        return len(self.children) == 0

    def least_common_ancestor(self, other: SubNode) -> Optional[SubNode]:
        """Return the most recent common ancestor between this and the other
        node, if it exists."""
        my_ancestors = set(self.ancestors())
        for ancestor in other.ancestors():
            if ancestor in my_ancestors:
                return ancestor

    def distance(self, other: 'TreeNode') -> Optional[int]:
        """Return the distance through the tree between two nodes; or None if
        they are not connected.  The directed-ness of the tree is ignored for
        this method."""
        lca = self.least_common_ancestor(other)
        if lca is not None:
            return self.depth + other.depth - 2 * lca.depth

    def _update_depths(self):
        """Update the internal state of depths.  This is called at every parent
        assignment."""
        for cell in self.depth_first():
            cell.depth = 0 if not cell.parent else cell.parent.depth + 1
