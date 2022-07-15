from typing import List, Dict

import numpy as np
from attrs import define, field
from matplotlib.collections import LineCollection

from abianalysis.hierarchy import Hierarchy
from abianalysis.plot.line import Line
from abianalysis.plot.point import Point
from abianalysis.plot.side import Side
from abianalysis.utils import max_key
from pylineage.color import Color


def dict_histogram(keys, values):
    counts = {}
    for key, value in zip(keys, values):
        c = counts.setdefault(key, {value: 0})
        v = c.setdefault(value, 0)
        c[value] = v + 1
    return counts


@define
class MaxProjection:
    """Calculates a max projection from a hierarchical decomposition."""

    # A set of 2D slots
    slots: Dict[Point, Hierarchy] = field(factory=dict)

    @classmethod
    def from_hierarchy(cls, root: Hierarchy, depth: int, side: Side):
        leaves: List[Hierarchy] = list(root.leaves())
        return MaxProjection.from_leaves(leaves, depth, side, pos=None)

    @classmethod
    def from_leaves(cls, leaves, depth: int, side: Side, pos=None):
        if pos is None:
            pos = [node.volume_index for node in leaves]
        else:
            assert len(pos) == len(leaves)
        points = side.project(pos)
        ancestors = [node.ancestor_at_depth(depth) for node in leaves]

        counts = dict_histogram(points, ancestors)
        return MaxProjection({p: max_key(c)
                              for p, c in counts.items()})

    def to_image(self, color_dict: Dict[Hierarchy, Color]):
        """

        :param Dict[Hierarchy, Color] color_dict:
        :return: image np.array
        """
        xmax, ymax = np.max([[p.x, p.y] for p in self.slots], axis=0)
        arr = np.zeros((xmax + 1, ymax + 1, 4))
        for p, v in self.slots.items():
            arr[p.x, p.y] = [*color_dict[v], 1.]
        return arr

    def __getitem__(self, point: Point):
        return self.slots[point]

    def trim(self) -> None:
        """Shift all positions so the smallest position becomes (0, 0)"""
        minx = min(p.x for p in self.slots)
        miny = min(p.y for p in self.slots)
        self.slots = {Point(p.x - minx, p.y - miny): v
                      for p, v in self.slots.items()}

    def neighbor_pairs(self):
        """return a set of all pairs of neighboring points"""
        positions = set(self.slots)
        return set(tuple(sorted((pos, neighbor)))
                   for pos in positions
                   for neighbor in pos.around()
                   if neighbor in positions)

    def get_outlines(self, color='black', lw=1, alpha=.2, rasterized=True,
                     **kwargs):
        """
        :param rasterized:
        :param alpha:
        :param lw:
        :param color:
        :param kwargs: other parameters passed to LineCollection
        """
        # pairs of neighboring points with different contents
        diff_pairs = ((fr, to) for fr, to in self.neighbor_pairs()
                      if self.slots[fr] != self.slots[to])
        lines = [Line.from_orth_points(fr, to) for fr, to in diff_pairs]
        return LineCollection([l.to_tuple() for l in lines],
                              color=color, lw=lw, alpha=alpha,
                              rasterized=rasterized, **kwargs)
