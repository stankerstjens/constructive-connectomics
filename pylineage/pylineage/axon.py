from typing import Iterable, Optional, List

import numpy as np

from pylineage.cell import Cell
from pylineage.node import TreeNode, items
from pylineage.state import State


class GrowthCone:
    correlation_threshold = .1
    gradient_threshold = .0

    def __init__(self, axon_segment: 'AxonSegment', state: State = None):
        self.axon_segment = axon_segment
        self.state = state
        self.local_affinity = 0.

    @property
    def position(self) -> Cell:
        return self.axon_segment.position

    def affinity(self, cell: Cell) -> float:
        return np.corrcoef(self.state.expression, cell.expression)[0, 1]

    def viable_moves(self) -> Iterable[Cell]:
        for neighbor in self.position.neighbors:
            neighbor_affinity = self.affinity(neighbor)
            gradient = neighbor_affinity - self.local_affinity
            if (neighbor_affinity > self.correlation_threshold
                    and gradient > self.gradient_threshold):
                yield neighbor

    def viable_transitions(self) -> Iterable[State]:
        neighbor_states = set(self.state.neighbors)
        prev_state = self.axon_segment.previous_state()
        if prev_state is not None:
            neighbor_states.remove(prev_state)
        return neighbor_states

    def branch(self, axon_segment=None, state=None):
        if axon_segment is None:
            axon_segment = self.axon_segment
        if state is None:
            state = self.state
        return GrowthCone(axon_segment=axon_segment, state=state)

    def extend_axon(self, cell: Cell):
        segment = AxonSegment(cell, self.state)
        segment.parent_segment = self.axon_segment
        self.axon_segment = segment
        return segment

    def run(self):
        cone_fringe = [self]
        time = 0

        while cone_fringe:
            cone = cone_fringe.pop(0)
            cones = [cone, *(cone.branch(state=transition)
                             for transition in cone.viable_transitions())]

            for cone in cones:
                for move in cone.viable_moves():
                    new_cone = self.branch()
                    new_cone.extend_axon(move)
                    new_cone.axon_segment.time = time
                    time += 1
                    cone_fringe.append(new_cone)


class AxonSegment:

    @classmethod
    def from_cell(cls, cell):
        return cls(position=cell, state=cell.state)

    def __init__(self, position: Cell, state: State):
        self.tree = TreeNode(item=self)
        self.state = state
        self.position = position
        self.time = 0

    @property
    def state_transitions(self) -> Iterable[State]:
        return self.state.neighbors

    def previous_state(self) -> Optional[State]:
        for segment in self.upstream_segments():
            if segment.state != self.state:
                return segment.state

    def make_cone(self, state=None):
        if state is None:
            state = self.state
        return GrowthCone(axon_segment=self, state=state)

    @property
    def spatial_neighbors(self) -> Iterable[Cell]:
        return self.position.neighbors

    @property
    def child_segments(self) -> 'List[AxonSegment]':
        return list(items(self.tree.children))

    def upstream_segments(self) -> 'List[AxonSegment]':
        return list(items(self.tree.ancestors()))

    @property
    def parent_segment(self) -> 'AxonSegment':
        return self.tree.parent.item

    @parent_segment.setter
    def parent_segment(self, parent: 'AxonSegment'):
        self.tree.parent = parent.tree

    def leaf_segments(self) -> 'List[AxonSegment]':
        return list(items(self.tree.leaves()))

    def root_segment(self) -> 'AxonSegment':
        return self.tree.root().item
