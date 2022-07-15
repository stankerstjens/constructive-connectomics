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

Volume IO
=========

Load and write Volumes parameters from and to numpy archive files.

"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np

from abianalysis.anatomy import Anatomy

#: The default folder that volume data is stored in.
DATA_DIR = Path(__file__).parent / 'data'


def default_volume_file_name(age: str) -> Path:
    """The default name"""
    return DATA_DIR / f'{age}.npz'


def volume_from_numpy_archive(age, file_name) -> Dict:
    """Read volume from numpy archive.

    :param age:  Age of the volume.
    :param file_name:  File name of the archive.


    """
    archive = np.load(file_name)
    anatomy_root = Anatomy.load()
    return dict(expression=archive['expression'],
                voxel_indices=archive['positions'],
                anatomy=list(map(anatomy_root.find, archive['anatomy'])),
                genes=archive['genes'],
                age=age)


def load_volume(age: str, file_name: Optional[str] = None) -> Dict:
    """Load a volume from a numpy archive file.

    :param age:  The age of the Volume.
    :param file_name:
        The file name of the numpy archive.  This is usually omitted,
        and is then derived from the age and the location of the data folder.

    """
    if file_name is None:
        file_name = default_volume_file_name(age)
    return volume_from_numpy_archive(age, file_name)
