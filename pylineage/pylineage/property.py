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
"""

from typing import Dict, Generic, TypeVar, Any

V = TypeVar('V')


class Property(Generic[V]):
    id_counter = 0

    @classmethod
    def next_id(cls) -> int:
        next_id = cls.id_counter
        cls.id_counter += 1
        return next_id

    def __init__(self) -> None:
        self.id = self.next_id()

    def __eq__(self, other) -> bool:
        return isinstance(other, Property) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def __getitem__(self, item: 'PropertyContainer') -> V:
        return item[self]

    def __setitem__(self, key: 'PropertyContainer', value: V):
        key[self] = value

    def __delitem__(self, key: 'PropertyContainer') -> None:
        del key[self]

    def __contains__(self, item):
        return self in item


class PropertyContainer:
    def __init__(self):
        self._values: Dict[Property, Any] = dict()

    # These methods should not be used by the end-user!
    def __getitem__(self, item: Property):
        return self._values[item]

    def __setitem__(self, key: Property, value):
        self._values[key] = value

    def __delitem__(self, key: Property):
        del self._values[key]

    def __contains__(self, key):
        return key in self._values
