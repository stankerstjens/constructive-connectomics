import numpy as np

from abianalysis.tree import TreeNode
from pylineage.property import Property


def place_leaves_in_array(root: TreeNode, pos_prop: Property) -> np.ndarray:
    leaves = list(root.leaves())
    positions = np.vstack([pos_prop[leaf] for leaf in leaves])
    mi = np.min(positions, 0)
    ma = np.max(positions, 0)
    array = np.zeros(tuple(ma - mi + 1), dtype=object)
    array[tuple((positions - mi).T)] = leaves
    return array


def pool_nodes(root: TreeNode, pos_prop: Property, block_size):
    array = place_leaves_in_array(root, pos_prop)
    shape = array.shape
    pools = np.zeros((tuple(np.array(shape) // block_size + 1)), dtype=object)
    for x in np.arange(0, shape[0], block_size):
        for y in np.arange(0, shape[1], block_size):
            for z in np.arange(0, shape[2], block_size):
                xe, ye, ze = map(lambda s: s + block_size, (x, y, z))
                cell_block = array[x:xe, y:ye, z:ze]
                bx, by, bz = map(lambda s: s // block_size, (x, y, z))
                pools[bx, by, bz] = [c for c in cell_block.flatten() if c != 0]
    return pools


def pool_expression(node_pools, exp_prop):
    first_cell = next(cell for pool in node_pools.flatten() for cell in pool)
    n_genes = len(exp_prop[first_cell])

    return np.vstack([np.mean([exp_prop[c] for c in pool], axis=0)
                      if pool != 0 and len(pool) > 0 else np.zeros(n_genes)
                      for pool in node_pools.flatten()]
                     ).reshape((*node_pools.shape, n_genes))
