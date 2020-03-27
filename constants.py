import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type_embed_dim = 256
pos_embed_dim = 256
embed_dim = type_embed_dim + pos_embed_dim
n_heads = 8

num_tokens = 50002