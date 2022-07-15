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
"""Positions and move cells in the Grid"""
cimport numpy as np
import numpy as np
from libc.math cimport sqrt

from pylineage.cell import Cell
from pylineage.grid.ray import ray

from pylineage.grid.grid import Idx, Tuple, Grid


class CellPositioner:
    """Positions and moves cell in th grid. Most importantly, can replace a
    cell in the grid with its children"""
    def __init__(self, grid: Grid) -> None:
        self.grid: Grid = grid

    @property
    def n_dims(self) -> int:
        """The number of dimensions in the grid"""
        return self.grid.n_dims

    def replace_with_children(self, cell: Cell):
        """Replace a cell in the grid with its children, by moving other cells
        in a random direction."""
        children = list(cell.children)
        # All except the last child are positioned by moving other cells around.
        for child in children[:-1]:
            idx = cell.position.index
            self.make_space(idx)
            child.position = self.grid.make_position(idx)
        # The last cell is positioned in the previous location of the mother
        # cell
        idx = cell.position.index
        children[-1].position = self.grid.make_position(idx)

    def make_space(self, idx: Idx) -> None:
        """Shift Slots in a random direction"""
        n = len(self.grid)
        self.shift(idx, random_direction(self.n_dims))

    def shift(self, idx: Idx, direction: Tuple) -> None:
        """Shift GridSlots in a specified direction."""
        r = ray(idx, direction)
        stack = [idx]
        for i in r:
            i = tuple(i)
            if i in self.grid:
                stack.append(i)
            else:
                break

        while stack:
            free = i
            i = stack.pop()
            self.grid[free] = self.grid[i]


def random_direction(n) -> np.ndarray:
    """Return a random direction drawn from an n-dimensional hypersphere"""
    if n == 3:
        return random_direction3()
    x = np.random.normal(size=n)
    return x / np.linalg.norm(x)

cdef np.ndarray random_direction3():
    """Return a random 3d direction"""
    cdef double[3] x = np.random.normal(size=3)
    cdef double norm = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    for i in range(3):
        x[i] /= norm
    return np.array(x)
