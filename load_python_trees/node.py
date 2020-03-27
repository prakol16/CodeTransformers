import random
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import multiprocessing
import pickle
from . import node_types
from .tokenize_identifiers import load_ind2token, get_token2ind, tokenize_node_by_type
from typing import List, Dict, DefaultDict, Optional, Tuple
from ASTNode import ASTNode
from itertools import islice

Py150k_AST = List[Dict]

cpu_count = multiprocessing.cpu_count()


class Node(ASTNode):
    def _build(self, ast: Py150k_AST, i: int, token2ind: DefaultDict[str, int]):
        node = ast[i]
        tp = node['type']
        self.type = node_types.get_index_from_type(tp)
        if 'children' in node:
            for child_index in node['children']:
                n = Node(ast, child_index, token2ind)
                self.children.append(n)
                self.t_size += n.t_size
        if 'value' in node:
            tokens = tokenize_node_by_type(tp, node['value'])
            self.value = [token2ind[token] for token in tokens]

    def print_all(self, ind2token, indent=0):
        print("  " * indent, node_types.all_types[self.type - 1], ' ' if self.value else '',
              '|'.join(ind2token[ind] for ind in self.value), sep='')
        for child in self.children:
            child.print_all(ind2token, indent + 1)


def preprocess(data_file, out_file, subtokens_file, num_lines, max_num_nodes, chunk_size=5000,
               validation: Optional[Tuple[str, float]]=None):
    ind2token = load_ind2token(subtokens_file)
    token2ind = get_token2ind(ind2token)

    def process_line(ast_index, line):
        allowed_head_types = {'FunctionDef', 'ClassDef'}
        parsed: Py150k_AST = json.loads(line)
        result = []
        for i, node in enumerate(parsed):
            if node['type'] in allowed_head_types:
                parsed_node = Node(parsed, i, token2ind)
                if parsed_node.t_size <= max_num_nodes:
                    result.append((parsed_node.vectorize(), ast_index, i))
        return result

    is_validating = validation is not None
    num_examples_1 = num_examples_2 = 0
    if is_validating:
        val_file, keep_prob = validation
        out_val = open(val_file, 'wb')
    with open(data_file) as f, open(out_file, 'wb') as out:
        f_enum = enumerate(tqdm(f, total=num_lines))
        while True:
            chunk_lines = islice(f_enum, chunk_size)
            output = Parallel(n_jobs=cpu_count)(delayed(process_line)(ast_index, line) for ast_index, line in chunk_lines)
            for py_file in output:
                for method in py_file:
                    place_train = not is_validating or random.random() < keep_prob
                    pickle.dump(method, out if place_train else out_val)
                    if place_train:
                        num_examples_1 += 1
                    else:
                        num_examples_2 += 1
            if len(output) < chunk_size:
                break
    if is_validating:
        out_val.close()

    return num_examples_1, num_examples_2


