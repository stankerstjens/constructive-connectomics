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
"""A Grid containing GridSlots (that contain items)."""
from typing import Tuple, Dict, Any, Iterable, Optional

from pylineage.node import Node

Idx = Tuple[int, ...]


class GridSlot(Node):
    """A type of node that has an index. The neighbors of the slot are those
    with adjacent indices in the grid.

    Only a GridSlot that is positioned in a grid will have an index. "Orphaned"
    GridSlots will have an index of None.

    """
    def __init__(self, grid: 'Grid' = None, index: Optional[Idx] = None,
                 item = None):
        super().__init__(item)
        self.grid: 'Grid' = grid
        self.index: Optional[Idx] = None
        if grid is not None and index is not None:
            grid[index] = self
        self.index = index

    @property
    def n_dims(self):
        """The number of dimensions in the grid.  This is also the number of
        entries in each index."""
        return self.grid.n_dims

    @property
    def neighbors(self) -> Iterable['GridSlot']:
        """The neighboring orthogonal positions in the grid that contain a
        slot"""
        if self.grid is None:
            raise ValueError("This slot has no grid.")
        if self.index is None:
            raise ValueError("This slot has no index.")
        for i in range(self.n_dims):
            for s in [-1, +1]:
                new_index = list(self.index)
                new_index[i] += s
                if new_index in self.grid:
                    yield self.grid[new_index]


class Grid:
    """The grid is a dictionary from indices of n_dims dimensions to objects
    stored in GridPosition's"""
    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        # The content of the grid is stored as a dictionary of tuples to
        # slots. Each slot knows its own location.
        self._content: Dict[Idx, GridSlot] = dict()

    @property
    def origin(self) -> Idx:
        """The origin of the grid (at 0,0,...,0)"""
        return tuple(0 for _ in range(self.n_dims))

    def make_position(self, index: Idx, item: Any = None) -> GridSlot:
        """Create a new GridSlot in this grid"""
        return GridSlot(grid=self, index=index, item=item)

    # ---- Dictionary interface

    def __len__(self) -> int:
        """The number of entries in the grid"""
        return len(self._content)

    def __delitem__(self, idx: Idx) -> None:
        "Remove a slot from the grid"
        self._content[idx].index = None
        del self._content[idx]

    def __getitem__(self, idx: Idx):
        return self._content[idx]

    def __setitem__(self, idx: Idx, slot: GridSlot) -> None:
        """Put an item into the grid.  This will also update the slot's index,
        and remove any previously existing slot from the specified location.
        If the slot was already in the grid, it will be removed from its
        previous location."""
        if len(idx) != self.n_dims:
            raise ValueError(f"Index must have the right dimension "
                             f"({self.n_dims})")
        # If slot had a previous index, remove it from the previous index
        if slot.index in self._content:
            del self._content[slot.index]
        # If the new index has a slot, remove that slot from the grid.
        if idx in self._content:
            self._content[idx].index = None
        # set slot to its new index
        slot.index = idx
        self._content[idx] = slot

    def __contains__(self, idx: Idx) -> bool:
        return idx in self._content

    def items(self) -> Iterable[Tuple[Idx, GridSlot]]:
        """Iterate over all index, slot pairs"""
        return self._content.items()

    def values(self) -> Iterable[GridSlot]:
        return self._content.values()

    def keys(self) -> Iterable[GridSlot]:
        return self._content.keys()
