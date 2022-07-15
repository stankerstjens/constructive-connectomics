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

Anatomy
=======

Tree of anatomical regions.

"""
import functools
import json
from pathlib import Path
from typing import Union

from pylineage.node import TreeNode


class Anatomy(TreeNode):
    """A tree of anatomical regions, each with a name and id"""

    file_name = Path(__file__).parent / 'structures.json'
    root = None

    @staticmethod
    def load() -> 'Anatomy':
        """Load the tree from file."""
        if Anatomy.root:
            return Anatomy.root

        with open(Anatomy.file_name, 'r') as file_handle:
            data = json.load(file_handle)
        Anatomy.root = _build_anatomy(data, parent=None)
        return Anatomy.root

    def __init__(self, id_, acronym, name, parent=None):
        super().__init__(parent=parent)
        self.id_ = id_
        self.acronym = acronym
        self.name = name

    def __contains__(self, other):
        """Check if the other is part of the subregion"""
        if isinstance(other, Anatomy):
            return self in other.ancestors()

        try:
            self.find(other)
        except KeyError:
            return False
        else:
            return True

    def __eq__(self, other):
        return isinstance(other, Anatomy) and self.id_ == other.id_

    def __hash__(self):
        return hash(self.id_)

    @property
    def id_set(self):
        """Get the set of ids downstream of this anatomy"""
        return set(node.id_ for node in self.descendants())

    @functools.lru_cache(maxsize=None)
    def find(self, key: Union[int, str, 'Anatomy']) -> 'Anatomy':
        """Find a sub-anatomy by id, acronym, or name."""
        if isinstance(key, Anatomy):
            return key
        for anatomy in self.descendants():
            if key in (anatomy, anatomy.id_, anatomy.acronym, anatomy.name):
                return anatomy
        raise KeyError('key not found')

    @classmethod
    def get(cls, anatomy: Union[int, str, 'Anatomy']):
        """Load an anatomy from a string or id."""
        return Anatomy.load().find(anatomy)


def _build_anatomy(datum, parent):
    child = Anatomy(id_=datum['id'], acronym=datum['acronym'],
                    name=datum['name'], parent=parent)

    for child_datum in datum['children']:
        _build_anatomy(child_datum, child)

    return child
