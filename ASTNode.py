from __future__ import annotations
import abc
from typing import List, Tuple
import random


class ASTNode:
    def __init__(self, *args):
        self.children: List[ASTNode] = []
        self.type = 0
        self.value = []
        self.t_size = 1
        self._build(*args)

    def _build(self, *args):
        ...

    def print_all(self, *args):
        ...

    def pick_node(self, index: int) -> ASTNode:
        if index == 0:
            return self

        cum_weights = 1
        for child in self.children:
            if cum_weights <= index < cum_weights + child.t_size:
                return child.pick_node(index - cum_weights)
            cum_weights += child.t_size

        raise Exception(f'Need index < tree size, but {index=} and {self.t_size=}')

    def pick_random_node(self) -> Tuple[ASTNode, int]:
        index = random.randrange(self.t_size)
        return self.pick_node(index), index

    def vectorize(self):
        all_nodes = [0] * self.t_size
        tree_structure = [0] * self.t_size
        values = [[]] * self.t_size
        def dfs(node, i):
            all_nodes[i] = node.type
            tree_structure[i] = i + node.t_size
            values[i] = node.value
            i += 1
            for child in node.children:
                i = dfs(child, i)
            return i

        k = dfs(self, 0)
        assert k == self.t_size
        return all_nodes, tree_structure, values

    @classmethod
    def build_from_vector(cls, all_nodes, tree_structure, values, i=0):
        node = cls.__new__(cls)
        node.type = all_nodes[i]
        end = tree_structure[i]
        node.t_size = end - i
        node.value = values[i]
        node.children = []
        i += 1
        while i < end:
            child, i = cls.build_from_vector(all_nodes, tree_structure, values, i)
            node.children.append(child)
        return node, i
