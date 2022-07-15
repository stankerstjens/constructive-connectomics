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
from pylineage.property import Property


def make_asymmetric_child(cell: Cell):
    exp = cell.expression
    delta = np.random.normal(size=exp.shape)
    cell.create_child().state.expression = exp + delta


def make_symmetric_child(cell: Cell):
    cell.create_child().state.expression = cell.expression.copy()


class Divider:
    def __init__(self, symmetric_prob):
        self.symmetric_prob: float = symmetric_prob

    def __call__(self, cell: Cell):
        self.divide(cell)

    def divide(self, cell: Cell):
        if np.random.random() < self.symmetric_prob:
            make_symmetric_child(cell)
            make_symmetric_child(cell)
        else:
            make_asymmetric_child(cell)
            make_asymmetric_child(cell)
