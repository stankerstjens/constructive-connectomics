import json
from pathlib import Path
from typing import Any, Optional, Dict

import numpy as np

from pylineage.cell import Cell
from pylineage.serialize.expression_serializer import MatrixSerializer
from pylineage.serialize.grid_position_serializer import GridPositionSerializer
from pylineage.serialize.machine_serializer import MachineSerializer
from pylineage.serialize.serializer import Serializer
from pylineage.serialize.tree_serializer import TreeSerializer


def load_lineage(json_file_name: str):
    with Path(json_file_name).open('r') as json_file:
        data = json.load(json_file)

    if 'expression_file_name' in data:
        expr_arch = np.load(data['expression_file_name'])
        include_cell_expression = 'cell_expression' in expr_arch
        include_state_expression = 'state_expression' in expr_arch
    else:
        include_cell_expression = include_state_expression = False

    serializer = LineageSerializer(
        include_cell_expression=include_cell_expression,
        include_state_expression=include_state_expression,
    )

    if include_state_expression:
        serializer.state_expression_serializer.matrix = expr_arch[
            'state_expression']
    if include_cell_expression:
        serializer.cell_expression_serializer.matrix = expr_arch[
            'cell_expression']

    return serializer.from_json(data)


def save_lineage(root_cell, json_file_name,
                 expression_file_name=None,
                 include_cell_expression=False,
                 include_state_expression=True,
                 meta: Optional[Dict[str, Any]]=None):
    serializer = LineageSerializer(
        include_cell_expression=include_cell_expression,
        include_state_expression=include_state_expression,
    )

    data = serializer.to_json(root_cell)

    if expression_file_name is not None:
        expression = {}
        if include_cell_expression:
            expression['cell_expression'] = serializer.expression
        if include_state_expression:
            expression[
                'state_expression'] = serializer.machine_serializer.expression

        np.savez_compressed(expression_file_name, **expression)
        data['expression_file_name'] = expression_file_name

        if meta is not None:
            for key, value in meta.items():
                data[key] = value

    with Path(json_file_name).open('w') as json_file:
        json.dump(data, json_file)


class LineageSerializer(Serializer):
    def __init__(self,
                 include_cell_expression=False,
                 include_state_expression=True,
                 machine_serializer=None,
                 position_serializer=None,
                 expression_serializer=None
                 ):

        if machine_serializer is None:
            machine_serializer = MachineSerializer(
                include_expression=include_state_expression
            )

        if position_serializer is None:
            position_serializer = GridPositionSerializer()

        if expression_serializer is None:
            expression_serializer = MatrixSerializer(lambda c: c.expression)

        self.tree_serializer = TreeSerializer(
            item_name="cell",
            item_serializer=CellSerializer(
                include_expression=include_cell_expression,
                expression_serializer=expression_serializer,
                machine_serializer=machine_serializer,
                position_serializer=position_serializer,
            )
        )

    def to_json(self, root_cell: Cell, **kwargs):
        machine_root = root_cell.state

        machine_data = self.machine_serializer.to_json(root_state=machine_root)
        if self.include_expression:
            self.cell_expression_serializer.set_items(root_cell.descendants())
        lineage_data = self.tree_serializer.to_json(root_cell.lineage)

        lineage_data['machine'] = machine_data

        return lineage_data

    def from_json(self, data: dict, **kwargs) -> Cell:
        if 'machine' in data:
            self.machine_serializer.from_json(data['machine'])

        lineage_root = self.tree_serializer.from_json(data)
        return lineage_root.item

    @property
    def expression(self):
        return self.cell_expression_serializer.matrix

    @property
    def cell_expression_serializer(self):
        return self.cell_serializer.expression_serializer

    @property
    def state_expression_serializer(self):
        return self.machine_serializer.expression_serializer

    @property
    def cell_serializer(self) -> 'CellSerializer':
        return self.tree_serializer.item_serializer

    @property
    def machine_serializer(self) -> MachineSerializer:
        return self.cell_serializer.machine_serializer

    @property
    def position_serializer(self) -> GridPositionSerializer:
        return self.cell_serializer.position_serializer

    @property
    def include_expression(self):
        return self.cell_serializer.include_expression


class CellSerializer(Serializer):
    def __init__(self,
                 include_expression: bool = False,
                 expression_serializer: MatrixSerializer = None,
                 position_serializer: GridPositionSerializer = None,
                 machine_serializer: MachineSerializer = None,
                 ):
        self.include_expression = include_expression
        self.expression_serializer = expression_serializer
        self.position_serializer = position_serializer
        self.machine_serializer = machine_serializer

    def from_json(self, data, tree_node=None, **kwargs):
        if self.include_expression:
            idx = data['expression_idx']
            expression = self.expression_serializer.get_array(idx)
        else:
            expression = None

        if 'position' in data:
            position = self.position_serializer.from_json(data['position'])
        else:
            position = None

        if self.machine_serializer:
            state = self.machine_serializer.get_state(data['state_idx'])
        else:
            state = None

        # on initialization, Cell will automatically set itself as the
        # position's item (if not None)
        cell = Cell(
            position=position,
            state=state,
            expression=expression,
        )

        if tree_node is not None:
            cell.lineage = tree_node

        return cell

    def to_json(self, cell: Cell, **kwargs):
        data = {}
        if cell.state is not None:
            data['state_idx'] = self.machine_serializer.get_idx(cell.state)
        if cell.position is not None:
            data['position'] = self.position_serializer.to_json(cell.position)
        if cell.expression is not None and self.include_expression:
            data['expression_idx'] = self.expression_serializer.get_idx(cell)
        return data
