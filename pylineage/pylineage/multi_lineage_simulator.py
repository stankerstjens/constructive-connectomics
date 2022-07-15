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
import numpy as np

from pylineage.cell import Cell
from pylineage.divider import Divider, make_asymmetric_child
from pylineage.grid.cell_positioner import CellPositioner
from pylineage.grid.grid import Grid
from pylineage.node import TreeNode
from pylineage.simulator import Simulator
from pylineage.state import State


class MultiLineageSimulator(Simulator):

    def __init__(self, n_dims: int, n_divisions: int, n_roots: int,
                 n_genes: int, symmetric_prob: float):
        super().__init__()
        self.n_roots = n_roots
        self.n_divisions = n_divisions
        self.n_genes = n_genes

        self.root = None

        self.grid = Grid(n_dims)
        self.positioner = CellPositioner(self.grid)
        self.divider = Divider(symmetric_prob)

        self.division_count = 0

        self._setup()

    @property
    def symmetric_prob(self):
        return self.divider.symmetric_prob

    def schedule_division(self, cell: Cell):
        if self.division_count < self.n_divisions:
            self.schedule(np.random.exponential(),
                          lambda c=cell: self.divide(c))
            self.division_count += 1

    def divide(self, cell: Cell):
        self.divider(cell)
        self.positioner.replace_with_children(cell)
        for child in cell.children:
            self.schedule_division(child)

    @property
    def n_dims(self):
        return self.grid.n_dims

    def _setup(self):
        self.root = Cell(state=State(expression=np.zeros(self.n_genes)))
        self.root.lineage = TreeNode()
        self.root.position = self.grid.make_position(index=self.grid.origin,
                                                     item=self.root)

        for _ in range(self.n_roots):
            make_asymmetric_child(self.root)
        self.positioner.replace_with_children(self.root)

        for child in self.root.children:
            self.schedule_division(child)
