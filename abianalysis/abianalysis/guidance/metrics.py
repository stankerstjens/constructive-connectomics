"""Some useful metrics on axons"""
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

from typing import Iterable, List

import numpy as np

from abianalysis.volume import Volume
from .axon import Axon


def get_path_positions(volume: Volume,
                       voxel_path: Iterable[int]) -> np.ndarray:
    """Get the positions of all the steps in a voxel path"""
    return volume.voxel_positions[voxel_path]


def get_euclidean_path_length(volume: Volume, voxel_path: List[int]):
    """Get the integrated Euclidean path length."""
    pos = get_path_positions(volume, voxel_path)
    return np.sum(np.sqrt(np.sum((pos[:-1] - pos[1:]) ** 2, 1)))


def get_euclidean_distance(volume: Volume, voxel_path: List[int]):
    """Get the euclidean distance between start and end point of a voxel
    path"""
    start = voxel_path[0]
    end = voxel_path[-1]
    start_pos, end_pos = get_path_positions(volume, [start, end])
    return np.sqrt(np.sum((start_pos - end_pos) ** 2))
