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
"""Iterate through a sequence of grid indices in a fixed direction"""
import numpy as np
cimport numpy as np

def ray(index, slope):
    """Create a ray in a specified direction"""
    if len(index) == 2:
        return ray2(index, slope)
    elif len(index) == 3:
        return ray3(index, slope)
    else:
        raise ValueError("Index length not supported (only 2 or 3)")

def ray3(index, slope):
    """Create a ray in a specified 3D direction."""
    cdef int[3] idx = index
    cdef int[3] slope_sgn = np.sign(slope)
    cdef double[3] slope_abs = np.abs(slope)
    cdef double[3] intercept = np.zeros(3)
    cdef int min_dim
    cdef double min_step

    while True:
        min_dim = -1
        min_step = 10e10
        for i in range(3):
            step_size = (1 - intercept[i]) / slope_abs[i]
            if step_size < min_step:
                min_step = step_size
                min_dim = i
        for i in range(3):
            intercept[i] += min_step * slope_abs[i]
        intercept[min_dim] = 0
        idx[min_dim] += slope_sgn[min_dim]
        yield idx

def ray2(index, slope):
    """Create a ray in a specified 2D direction."""
    cdef int[2] idx = index
    cdef int[2] slope_sgn = np.sign(slope)
    cdef double[2] slope_abs = np.abs(slope)
    cdef double[2] intercept = np.zeros(2)
    cdef int min_dim
    cdef double min_step

    while True:
        min_dim = -1
        min_step = 10e10
        for i in range(2):
            step_size = (1 - intercept[i]) / slope_abs[i]
            if step_size < min_step:
                min_step = step_size
                min_dim = i
        for i in range(2):
            intercept[i] += min_step * slope_abs[i]
        intercept[min_dim] = 0
        idx[min_dim] += slope_sgn[min_dim]
        yield idx
