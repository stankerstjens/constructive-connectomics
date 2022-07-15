from typing import Iterable, Optional

import numpy as np

from pylineage.node import GraphNode, items


class State:
    def __init__(self, expression: Optional[np.ndarray] = None):
        self.graph = GraphNode()
        self._expression: np.ndarray = expression

    @property
    def expression(self):
        return self._expression

    def connect(self, state: 'State'):
        self.graph.connect(state.graph)

    def disconnect(self, state: 'State'):
        self.graph.disconnect(state.graph)

    @expression.setter
    def expression(self, expression):
        self._expression = expression

    @property
    def neighbors(self) -> Iterable['State']:
        return items(self.graph.neighbors)
