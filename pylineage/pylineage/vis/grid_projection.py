from typing import Tuple, Dict, List, Iterable

from abianalysis import Hierarchy
from pylineage.node import TreeNode

Color = Tuple[float, float, float]
Position = Tuple[int, int]
Stack = Dict[Position, TreeNode]
Side = List[int]
FRONTAL = [1, 0]
SAGITTAL = [1, 2]
HORIZONTAL = [0, 2]


def trim_sides(pos_dict):
    """Returns copy dictionary where all positions are shifted so that
    the smallest x and y value are now zero.

    :param pos_dict: Dictionary where all keys are 2D positions.

    """
    minx = min(x for x, _ in pos_dict)
    miny = min(y for _, y in pos_dict)
    return {(x - minx, y - miny): node for (x, y), node in pos_dict.items()}


def view_stack_fn(depth: int, side: Side, trim: bool = True):
    def _get_ancestor(node: TreeNode):
        try:
            return node.ancestor_at_depth(depth)
        except ValueError:
            return node

    def _get_pos(node: Hierarchy):
        return tuple(map(int, node.position / node.volume.voxel_size))

    def _project_pos(pos):
        return tuple(pos[s] for s in side)

    def _get_placed_ancestors(nodes: Iterable[Hierarchy]):
        return {_get_pos(node): _get_ancestor(node) for node in nodes}

    def _count_stack(placed_nodes):
        counts_per_pos = {}
        for pos, node in placed_nodes.items():
            proj_pos = _project_pos(pos)
            counts = counts_per_pos.setdefault(proj_pos, {node: 0})
            counts[node] = counts.get(node, 0) + 1
        return {p: max(counts, key=counts.__getitem__)
                for p, counts in counts_per_pos.items()}

    def _get_stack(nodes):
        placed_ancestors = _get_placed_ancestors(nodes)
        stack = _count_stack(placed_ancestors)

        if trim:
            stack = trim_sides(stack)

        return stack

    return _get_stack


def view_stack(nodes, depth: int, side: Side, trim=True) -> Stack:
    """Create a stacked view of a hierarchy at a specific depth, from
    one side.

    :param nodes:  A map from positions (in 3d) to leaf hierarchy nodes.
        Positions without a node are ignored.
    :param depth:  The depth of the hierarchy to assign to the stack
    :param side:  list of indices deciding the view. For example,
        [1, 0] is a front view, and [1,2] is a side view.

    :return: A dictionary from projected positions to the most common
        Hierarchy node in the collapsed direction
    """
    return view_stack_fn(depth, side, trim)(nodes)


def stack_to_image(stack: Stack,
                   color_dict: Dict[TreeNode, Color]):
    """

    :param Stack stack:
    :param Dict[Hierarchy, Color] color_dict:
    :return:
    """
    x, y = np.array(list(stack), dtype=int).T
    mix, miy = x.min(), y.min()
    arr = np.zeros((x.max() - mix + 1, y.max() - miy + 1, 4))
    for x, y in zip(x, y):
        arr[x - mix, y - miy] = [*color_dict[stack[x, y]], 1]
    return arr
