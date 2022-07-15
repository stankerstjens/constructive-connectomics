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

Color
=====

Color utilities

"""
from typing import Dict, Tuple, Union

import numpy as np
from attr.validators import ge, le, instance_of
from attrs import define, field
from hsluv import hsluv_to_rgb

from pylineage.layout import tree_layout
from pylineage.node import TreeNode


def convert_color(v):
    """Convert any byte larger than 1 to a float between 0 and 1."""
    if isinstance(v, int) and v > 1:
        return v / 255.0
    return v


def color_field():
    return field(default=0,
                 validator=[ge(0), le(1), instance_of(float)],
                 converter=convert_color)


@define
class Color:
    r = color_field()
    g = color_field()
    b = color_field()

    def rgb(self) -> Tuple[float, float, float]:
        return self.r, self.g, self.b

    def rgba(self) -> Tuple[float, float, float, float]:
        return self.r, self.g, self.b, 1


@define
class Range:
    minimum = field()
    maximum = field()

    @classmethod
    def from_tuple_or_value(cls, value: Union[Tuple[float, float], float]
                            ) -> 'Range':
        try:
            if len(value) == 2:
                return Range(*value)
        except TypeError:
            return cls.empty_at_value(value)

    @classmethod
    def empty_at_value(cls, value):
        return cls(minimum=value, maximum=value)

    def interpolate(self, x):
        """0 is mapped to minimum, 1 is mapped to maximum"""
        return x * self.range + self.minimum

    @property
    def range(self):
        return self.maximum - self.minimum


def range_between(minimum, maximum):
    def _validator(instance, attribute, value: Range):
        if (value.minimum < minimum or value.minimum > maximum
                or value.maximum < minimum or value.maximum > maximum):
            raise ValueError(f"Range must be between {minimum} and {maximum}")
        if value.maximum < value.minimum:
            raise ValueError(f"maximum cannot be smaller than the minimum")

    return _validator


@define
class TreeColorMap:
    hue_range: Range = field(converter=Range.from_tuple_or_value,
                             validator=range_between(0., 360.),
                             default=(0., 360.))
    saturation_range: Range = field(converter=Range.from_tuple_or_value,
                                    validator=range_between(0., 100.),
                                    default=(50., 100.))
    lightness_range: Range = field(converter=Range.from_tuple_or_value,
                                   validator=range_between(0., 360.),
                                   default=70.)

    def get_color_map(self, root: TreeNode, depth=None) -> Dict[TreeNode, Color]:
        layout = tree_layout(root, depth=depth)

        x, y = np.array([layout[n] for n in layout]).T

        hue = self.hue_range.interpolate(x)
        sat = self.saturation_range.interpolate(y)
        lig = self.lightness_range.interpolate(y)

        return {
            node: hsluv_to_rgb([h, 0 if node.is_root else s, l])
            for node, h, s, l in zip(layout, hue, sat, lig)
        }