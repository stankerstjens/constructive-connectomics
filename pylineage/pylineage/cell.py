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
from typing import Iterable, Optional

from pylineage.node import TreeNode, items, GraphNode, Node
from pylineage.state import State


class Cell:
    def __init__(self, state: Optional[State] = None):
        self.state: Optional[State] = state
        self._lineage: Optional[TreeNode[Cell]] = None
        self._position: Optional[Node[Cell]] = None

    @property
    def lineage(self):
        return self._lineage

    @lineage.setter
    def lineage(self, lineage: 'TreeNode[TreeNode, Cell]'):
        self._lineage = lineage
        lineage.item = self

    @property
    def n_genes(self):
        return len(self.expression)

    @property
    def expression(self):
        return self.state.expression

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position: GraphNode):
        self._position = position
        position.item = self

    @property
    def parent(self) -> 'Cell':
        return self.lineage.parent.item

    @parent.setter
    def parent(self, parent: 'Cell') -> None:
        if self.lineage is None:
            self.lineage = TreeNode()
        self.lineage.parent = parent.lineage

    @property
    def children(self) -> Iterable['Cell']:
        return items(self.lineage.children)

    @property
    def neighbors(self) -> Iterable['Cell']:
        return items(self.position.neighbors)

    @property
    def is_leaf(self) -> bool:
        return self.lineage.is_leaf

    def descendants(self):
        return items(self.lineage.descendants())

    def leaves(self):
        return items(self.lineage.leaves())

    def root(self):
        return self.lineage.root().item

    @property
    def is_root(self) -> bool:
        return self.lineage.is_root

    def create_child(self, state: State = None) -> 'Cell':
        if state is None:
            state = State()
        cell = Cell(state=state)
        cell.parent = self
        return cell
