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

Hierarchy
=========

A voxel a node in a hierarchical tree The leaves are measured voxels, the
others are inferred.

"""
from functools import cached_property
from typing import Optional, List, Dict, Tuple, TypeVar

import numpy as np

from pylineage.node import TreeNode

from abianalysis.volume import Volume
from abianalysis.json_dict import to_json_dict

T = TypeVar('T')

class Hierarchy(TreeNode['Hierarchy', T]):
    """A node in a hierarchy of which the leaves are voxels of the
    volume.

    The canonical way to set up the hierarchy is using the
    `make_hierarchy` function.

    Positions and expressions are lazily calculated.

    :param volume: The volume this hierarchy is associated with.
    :param parent: The parent hierarchy node.

    :param voxel_index:
        The index of the voxel in the volume this node represents, if it is
        a leaf.  If it is not a leaf, this should remain None (which is the
        default).  See :ref:`hierarchy`.

    :param component:
        The direction in gene expression space over which the node is
        was split (if this was the method used).

    """

    def __init__(self,
                 volume: Optional[Volume] = None,
                 parent: Optional['Hierarchy'] = None,
                 voxel_index: Optional[int] = None,
                 component: Optional[np.ndarray] = None) -> None:
        super().__init__(parent=parent)
        self.volume = (volume if volume is not None
                       else self._volume_from_parent())
        self.voxel_index = voxel_index
        self.component = (np.asarray(component)
                          if component is not None else None)

    def _volume_from_parent(self):
        return self.parent.volume

    def to_json_dict(self,
                     include_leaves=False,
                     include_only=('id', 'children', 'voxel_index',
                                   'component'),
                     **kwargs):
        """Stores all interior nodes of the hierarchy into a json file.

        The leaf nodes can be inferred as the children of their parent.

        .. seealso::
            :py:func:`.Hierarchy.from_json_dict`, :py:func:`.to_json_dict`

        """
        data = to_json_dict(self,
                            include_only=include_only,
                            **kwargs)

        if not include_leaves:
            def _filter_leaves(datum):
                for child in list(datum['children']):
                    if len(child['children']) == 0:
                        datum['children'].remove(child)
                for child in datum['children']:
                    _filter_leaves(child)

            _filter_leaves(data)

        return data

    @classmethod
    def from_json_dict(cls, volume: Volume, data: Dict) -> 'Hierarchy':
        """Load a Hierarchy from a dictionary.

        The node IDs are not maintained, but the nesting of voxels and
        components (if any) are.

        :param volume:
            The volume from which the hierarchy was originally derived.
        :param data:
            The data dictionary containing the hierarchy, like what
            is returned by json.load.

        .. seealso::
            :py:func:`.Hierarchy.to_json_dict`, :py:func:`.to_json_dict`
        """

        def _recursive_call(node, datum):
            node.voxel_index = datum.get('voxel_index', None)
            node.component = datum.get('component', None)
            for child_datum in datum['children']:
                _recursive_call(cls(volume=volume, parent=node), child_datum)
            return node

        return _recursive_call(cls(volume=volume), data)

    @property
    def n_progenitors(self):
        """The number of interior nodes among the descendants"""
        return sum(1 for _ in self.interior())

    @property
    def asymmetry(self) -> np.array:
        """The difference in expression between the left and right
        daughters.

        This value is only defined if the node has exactly two children,
        which should be true for all but the last layer of internal nodes (
        and the leaves).

        """
        if len(self.children) != 2:
            raise ValueError('Asymmetry is only defined for nodes with '
                             'exactly two children')
        left, right = self.children
        return left.expression - right.expression

    def project_component(self, expression: np.ndarray = None) -> np.ndarray:
        """Project the expression of the passed node's voxels onto the
        component of this node.

        Similar to :py:meth`.Hierarchy.project_asymmetry`, except the stored
        component rather than the difference between daughter expressions is
        used.  In some theoretical models these two are identical.

        :param expression:
            An expression matrix (voxels as rows, genes as columns).  If
            None, this node's :py:attr:`.Hierarchy.leaves_expression` is used.

        """
        if expression is None:
            expression = self.leaves_expression
        return expression @ self.component

    def project_asymmetry(self, expression: np.ndarray = None) -> np.array:
        """Project the expression of the passed node's voxels onto the
        asymmetry of this node.

        :param expression:
            An expression matrix (voxels as rows, genes as columns).  If
            None, this node's :py:attr:`.Hierarchy.leaves_expression` is used.

        """
        if expression is None:
            expression = self.leaves_expression
        return expression @ self.asymmetry

    def spatial_spread(self):
        """The average Euclidean distance of the voxels contained in this
        node to the center of those voxels (i.e. the spatial variance)."""
        pos = self.volume.voxel_positions[self.voxels] - self.position
        return np.mean(np.linalg.norm(pos, axis=1))

    @cached_property
    def expression(self) -> np.ndarray:
        """the expression of this voxel.

        This is either the voxel expression (for leaves) or the average
        expression over all voxels in this node.

        The expression is cached in the hierarchy.

        """
        return (np.mean(self.leaves_expression, axis=0)
                if self.voxel_index is None
                else self.volume.expression[self.voxel_index])

    @property
    def leaves_expression(self) -> np.array:
        """The expression matrix over all leaves rooted in this node."""
        return self.volume.expression[self.voxels]

    @cached_property
    def position(self) -> np.array:
        """The position of the voxel.

        This is either the measured position (for leaves), or inferred
        position (average of leaf positions).
        """
        return (np.mean(self.leaves_positions, axis=0)
                if self.voxel_index is None
                else self.volume.voxel_positions[self.voxel_index])

    @property
    def volume_index(self) -> Optional[np.ndarray]:
        """The voxel_index of this node (if it is a leaf)."""
        if self.voxel_index is not None:
            return self.volume.voxel_indices[self.voxel_index]

    @property
    def leaves_positions(self) -> np.array:
        """A matrix of all positions downstream of this node."""
        return self.volume.voxel_positions[self.voxels]

    @property
    def voxels(self) -> List[int]:
        """A list of indices of voxels contained in this node, corresponding
        to rows in the :py:class`.Volume` matrices"""
        return [leaf.voxel_index for leaf in self.leaves()]

    def get_leaf(self, voxel: int) -> 'Hierarchy':
        """Get the leaf with the provided voxel index."""
        for leaf in self.leaves():
            if voxel == leaf.voxel_index:
                return leaf
