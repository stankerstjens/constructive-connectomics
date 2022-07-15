from pylineage.node import TreeNode
from pylineage.serialize.serializer import Serializer, EmptySerializer


class TreeSerializer(Serializer):
    def __init__(self, item_serializer=None, item_name='item'):
        if item_serializer is None:
            item_serializer = EmptySerializer()
        self.item_serializer = item_serializer
        self.item_name = item_name

    def to_json(self, node: TreeNode, **kwargs):
        return {
            self.item_name: self.item_serializer.to_json(node.item),
            'children': [self.to_json(child) for child in node.children]
        }

    def from_json(self, data: dict, **kwargs):
        node = TreeNode()
        if self.item_name in data:
            node.item = self.item_serializer.from_json(data[self.item_name],
                                                       tree_node=node)

        if 'children' in data:
            for child_data in data['children']:
                self.from_json(child_data).parent = node

        return node
