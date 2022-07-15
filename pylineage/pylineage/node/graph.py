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
"""GraphNode implementing a doubly linked graph."""
from typing import Generic, TypeVar, Optional, List

from pylineage.node import Node

SubNode = TypeVar('SubNode', bound='GraphNode')
T = TypeVar('T')


class GraphNode(Generic[SubNode, T], Node[SubNode, T]):
    """Each GraphNode can have any number of neighbors.

    The neighbors are undirected.
    """

    def __init__(self, item: Optional[T] = None):
        super().__init__(item=item)
        self._neighbors = []

    @property
    def neighbors(self) -> List[SubNode]:
        """A list of all neighboring nodes"""
        return self._neighbors

    def connect(self, node: SubNode):
        """Make the passed node and this node neighbors"""
        self._neighbors.append(node)
        node._neighbors.append(self)

    def disconnect(self, node: SubNode):
        """Make a currently neighboring node not a neighbor anymore"""
        try:
            node._neighbors.remove(self)
            self._neighbors.remove(node)
        except ValueError:
            raise ValueError("node must be a neighbor")
