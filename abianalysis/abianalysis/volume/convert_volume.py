import numpy as np

from abianalysis import Volume
from abianalysis.spatial.factory import gabriel_edges
from abianalysis.hierarchy.region import make_hierarchy
from pylineage.cell import Cell
from pylineage.node import GraphNode, TreeNode
from pylineage.state import State


class VolumeCell(Cell):
    def __init__(self, voxel: int, volume: Volume, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voxel = voxel
        self.volume = volume


class VolumePosition(GraphNode):
    def __init__(self, cell: VolumeCell, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = cell

    @property
    def index(self):
        return self.cell.volume.voxel_indices[self.cell.voxel]


class VolumeState(State):
    def __init__(self, cell, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell = cell

    @property
    def expression(self):
        return self.cell.volume.expression[self.cell.voxel]


class DecompositionCell(Cell):
    def __init__(self, region, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.region = region

    @property
    def component(self):
        return self.region.component

    @property
    def coefficients(self):
        return self.region.coefficients

    @property
    def voxels(self):
        return self.region.voxels


def convert_volume(volume, exclude_spatial=False, exclude_lineage=False):
    print('Creating cells...')
    cells = []
    for voxel in range(volume.n_voxels):
        cell = VolumeCell(volume=volume, voxel=voxel)
        cell.state = VolumeState(cell=cell)
        cell.position = VolumePosition(cell=cell)
        cell.lineage = TreeNode()
        cells.append(cell)

    print('Creating graph...')
    if not exclude_spatial:
        for fr, to in gabriel_edges(volume.voxel_indices):
            cells[fr].position.connect(cells[to].position)

    print('Creating lineage...')
    if not exclude_lineage:

        root_region = make_hierarchy(volume)
        region_to_cell = dict()

        for region in root_region.descendants():
            state = State(expression=np.mean(region.expression, axis=0))
            if len(region.voxels) == 1:
                cell = cells[region.voxels[0]]
            else:
                cell = DecompositionCell(state=state, region=region)
                cell.lineage = TreeNode()
            region_to_cell[region] = cell
            if region.parent:
                cell.parent = region_to_cell[region.parent]

        for cell in cells:
            if cell.parent:
                cell.state.connect(cell.parent.state)

    return cells
