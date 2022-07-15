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
"""Plot a hierarchy as a stacked projection."""
from functools import cached_property
from typing import Tuple, Dict, Optional

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from sklearn.decomposition import PCA

from abianalysis import Hierarchy
from abianalysis.plot.max_projection import MaxProjection
from abianalysis.plot.side import Side
from pylineage.color import TreeColorMap
from pylineage.layout import radial_tree_layout
from pylineage.node import TreeNode


def plot_tree(ax: Axes, root: TreeNode, pos: Optional[Dict] = None,
              s: float = 0., lw: float = .1, color: Optional[Dict] = None,
              depth: int = None):
    """Helper function to plot a tree on an ax

    :param lw: line width
    :param s: The size of the drawn nodes.
    :param pos: An optional dictionary from nodes to 2d positions of nodes. By
        default, this will be a radial layout of the tree.
    :param root: The root of the tree.
    :param ax: The Axes on which to draw
    :param Dict[TreeNode, color] color:  A dictionary from tree nodes to colors.
        By default, creates a hierarchical color map for the whole tree up to the
        specified depth .
    :param int depth: The depth until which to draw the tree.

    """
    if pos is None:
        pos = radial_tree_layout(root, depth=depth)

    if color is None:
        color = TreeColorMap().get_color_map(root, depth=depth)

    nodes = list(root.descendants())
    if depth is not None:
        nodes = [n for n in nodes if n.depth < depth]

    for node in nodes:
        if not node.is_root:
            p1, p2 = pos[node], pos[node.parent]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color[node], lw=lw)

    if s > 0:
        x, y = zip(*(pos[n] for n in nodes))
        ax.scatter(x, y, s=s, c='k')


class HierarchyPlotter:

    def __init__(self, hierarchy: Hierarchy) -> None:
        self.hierarchy = hierarchy
        self.color_map = TreeColorMap(lightness_range=65).get_color_map(
            self.hierarchy)

    def get_stack(self, depth, side):
        return MaxProjection.from_hierarchy(self.hierarchy, depth, side)

    def get_image(self, depth, side):
        return self.get_stack(depth, side).to_image(self.color_map)

    @cached_property
    def expression_2d(self):
        exp = self.hierarchy.volume.expression
        return PCA(n_components=2).fit_transform(exp)

    @property
    def sorted_leaves(self):
        return sorted(self.hierarchy.leaves(), key=lambda h: h.voxel_index)

    def get_voxel_colors(self, depth):
        c = self.color_map
        return [c[h.ancestor_at_depth(depth)]
                for h in self.sorted_leaves]

    def get_outlines(self, depth, side):
        return self.get_stack(depth, side).get_outlines()

    @property
    def image_size(self):
        v = self.hierarchy.volume.voxel_indices
        return v.max(0) - v.min(0) + 1

    def get_sub_gridspec(self, subplot_spec):
        s = self.image_size
        return GridSpecFromSubplotSpec(
            nrows=2, ncols=2,
            subplot_spec=subplot_spec,
            width_ratios=s[[2, 0]],
            height_ratios=s[[0, 1]])

    def plot_expression(self, ax, depth, show_labels=True, s=1, alpha=1,
                        rasterized=True, **kwargs):
        ax.scatter(*self.expression_2d.T,
                   c=self.get_voxel_colors(depth),
                   s=s, alpha=alpha,
                   rasterized=rasterized,
                   **kwargs)
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(c='k')
        ax.set_frame_on(False)
        _add_zero_label(ax)
        if show_labels:
            _add_pc_labels(ax)

    def plot_image(self, ax, depth, side):
        ax.imshow(self.get_image(depth, side), interpolation='antialiased',
                  rasterized=True)
        ax.set_frame_on(False)
        ax.add_collection(self.get_outlines(depth, side))
        ax.axis(False)

    def plot_views(self, fig, subplot_spec, depth, show_labels=True):
        gs = self.get_sub_gridspec(subplot_spec)

        ax1 = fig.add_subplot(gs[1, 0])
        self.plot_image(ax1, depth, Side.SAGITTAL)
        ax2 = fig.add_subplot(gs[0, 0])
        self.plot_image(ax2, depth, Side.HORIZONTAL)
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_image(ax3, depth, Side.FRONTAL)

        if show_labels:
            ax4 = fig.add_subplot(gs[0, 1])
            ax4.text(0, 0, self.hierarchy.volume.age, fontsize=12)
            ax4.axis(False)
            _add_sagittal_label(ax1)
            _add_horizontal_label(ax2)
            _add_coronal_label(ax3)


def _add_zero_label(ax):
    trans = ScaledTranslation(-1, -1, ax.transData)
    ax.text(0, 0, '0', transform=trans, ha='right', va='top')


def _add_pc_labels(ax):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.xaxis.set_label_coords(0, 0, transform=trans)
    ax.set_ylabel('pc2', va='bottom', ha='left')

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.yaxis.set_label_coords(0, 0, transform=trans)
    ax.set_xlabel('pc1', ha='left', va='bottom')


def _add_sagittal_label(ax):
    ax.text(0, 0, 'sagittal', transform=ax.transAxes, ha='left', va='top')


def _add_horizontal_label(ax):
    ax.text(0, 1, 'horizontal', transform=ax.transAxes, ha='left', va='bottom')


def _add_coronal_label(ax):
    ax.text(1, 0, 'coronal', transform=ax.transAxes, ha='left', va='bottom',
            rotation=90)
