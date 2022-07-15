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
"""Node interface."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Iterable

SubNode = TypeVar('SubNode', bound='Node')
T = TypeVar('T')


class Node(Generic[SubNode, T], ABC):
    """An abstract object with neighbors and an optional item.

    This class is implemented by GraphNode (for undirected graphs of any
    topology) and TreeNode (for directed graphs where each node has a single
    incoming parent-neighbor, and any number of outgoing child neighbors).

    Any subclass should override the neighbors abstract property.
    """

    def __init__(self, item=None):
        self.item: Optional[T] = item

    @property
    @abstractmethod
    def neighbors(self) -> Iterable[SubNode]:
        """Iterate over neighbors"""
