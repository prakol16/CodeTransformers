from itertools import product
from typing import Dict


def product_join(str_x, str_y):
    for (x, y) in product(str_x, str_y):
        yield x + y


cmp_operators = ['NotEq', 'Eq', 'LtE', 'Lt', 'GtE', 'Gt', 'IsNot', 'Is', 'NotIn', 'In']
len_compare = len('Compare')


all_types = ['MASK'] + ['identifier', 'Module', 'FunctionDef', 'ClassDef', 'Return',
             'Delete', 'Assign', 'Print', 'For', 'While', 'If', 'With', 'Raise', 'TryExcept',
             'TryFinally', 'Assert', 'Import', 'ImportFrom', 'Exec', 'Global', 'Expr', 'Pass', 'Break',
             'Continue', 'BoolOpAnd', 'BoolOpOr', 'UnaryOpInvert', 'UnaryOpNot', 'UnaryOpUAdd', 'UnaryOpUSub',
             'Lambda', 'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp', 'Yield',
             'Call', 'Repr', 'Num', 'Str', 'comprehension', 'ExceptHandler', 'arguments', 'keyword', 'alias'] + \
            list(product_join(['Attribute', 'Subscript', 'Name', 'List', 'Tuple'], ['Load', 'Store', 'Del', 'Param'])) + \
            ['Ellipsis', 'Slice', 'ExtSlice', 'Index'] + \
            list(product_join(['AugAssign', 'BinOp'], ['Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'FloorDiv'])) + \
            ['Compare' + cmpop for cmpop in cmp_operators] + \
            ['body', 'orelse', 'handlers', 'finalbody', 'args', 'defaults', 'vararg', 'kwarg', 'type', 'name', 'bases', 'decorator_list', 'attr']


def cvt_cmp_op(node_type: str) -> str:
    if not node_type.startswith('Compare'):
        return node_type
    node_type = node_type[len_compare:]
    for p in cmp_operators:
        if node_type.startswith(p):
            return 'Compare' + p
    raise Exception(f'Unkown comparison operator {node_type}')


all_types_to_ind: Dict[str, int] = {node_type: i+1 for i, node_type in enumerate(all_types)}


def get_index_from_type(node_type: str) -> int:
    """Given a node type like FunctionDef or NameLoad, return the index.
    The only reason this isn't a straight dictionary lookup is because of the comparison operators"""
    return all_types_to_ind[cvt_cmp_op(node_type)]
