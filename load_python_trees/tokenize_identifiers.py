import re
from tqdm import tqdm
from typing import List, DefaultDict
from collections import Counter, defaultdict
import json


PAD_TOKEN = 0
UNK_TOKEN = 1

Token2Ind = DefaultDict[str, int]

variable_regex = [
    # Put all strings of digits in their own bin
    (re.compile(r'([0-9]+)'), r'|\1|'),
    # Camel case
    (re.compile(r'([a-z])([A-Z])'), r'\1|\2'),
    (re.compile(r'([A-Z])([A-Z][a-z])'), r'\1|\2'),
    # Underscores
    (re.compile(r'_'), r'|'),
    # Periods
    (re.compile(r'\.'), r'|')
]


def get_subtoken_vocab(filename, vocab_size=50000, file_num_lines=None) -> List[str]:
    vocab = Counter()
    with open(filename) as f:
        for i, line in enumerate(tqdm(f, total=file_num_lines)):
            parsed = json.loads(line)
            for node in parsed:
                try:
                    v = node['value']
                    if v:
                        tokens = tokenize_node_by_type(node['type'], v)
                        vocab.update(tokens)
                except KeyError:
                    pass
            if file_num_lines is not None and i >= file_num_lines:
                break
        most_common = vocab.most_common(vocab_size)
        ind2token = ['<PAD>', '<UNK>'] + [word for word, freq in most_common]
        return ind2token


def save_ind2token(filename, ind2token: List[str]):
    with open(filename, 'w') as f:
        for word in ind2token:
            print(word, file=f)


def load_ind2token(filename) -> List[str]:
    with open(filename) as f:
        ind2token = [word.rstrip("\n") for word in f]
        return ind2token


def get_token2ind(ind2token: List[str]) -> Token2Ind:
    return defaultdict(lambda: UNK_TOKEN, ((word, i) for i, word in enumerate(ind2token)))


def tokenize(identifier):
    for regex, repl in variable_regex:
        identifier = re.sub(regex, repl, identifier)
    # Eliminate empty strings and lowercase everything
    return [x.lower() for x in identifier.split('|') if x]


def tokenize_node_by_type(node_type, node_val):
    if node_type == 'Num':
        return ['<NUM>']
    elif node_type == 'Str':
        return ['<STR>']
    else:
        return tokenize(node_val)


