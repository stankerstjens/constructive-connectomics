from enum import Enum

import numpy as np

from abianalysis.plot.point import Point


class Side(Enum):
    FRONTAL = [1, 0]
    SAGITTAL = [1, 2]
    HORIZONTAL = [0, 2]

    def project(self, array):
        return [Point(x, y) for x, y in np.array(array).T[self.value].T]
