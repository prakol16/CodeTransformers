import torch
from torch import nn
import math
from multi_headed_attention_copy import TransformerEncoderLayer


class ASTEmbeddings(nn.Module):
    SCALE_CONST = 4  # We keep the positional embeddings smaller than the node types by a factor of SCALE_CONST^2

    def __init__(self, num_types, num_tokens, type_embed_dim=256, pos_embed_dim=256):
        super().__init__()
        self.type_embed_dim = type_embed_dim
        self.pos_embed_dim = pos_embed_dim

        self.num_types = num_types
        self.num_tokens = num_tokens

        self.gru = nn.GRUCell(type_embed_dim, pos_embed_dim)
        self.type_embeddings = nn.Embedding(num_types, type_embed_dim, padding_idx=0)
        # self.token_embeddings = nn.Embedding(num_tokens, type_embed_dim, padding_idx=0)
        self.token_embeddings = nn.EmbeddingBag(num_tokens, type_embed_dim, mode='mean')

        self.positional_embeddings = nn.Embedding.from_pretrained(ASTEmbeddings.get_positional_embeddings(type_embed_dim))
        self.num_positional_embeddings = self.positional_embeddings.num_embeddings

    @staticmethod
    def get_positional_embeddings(d_model, max_len=1000):
        pe = torch.empty(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def ensure_positional_embeddings_fit(self, max_child_index):
        if max_child_index >= self.num_positional_embeddings:
            print(f"WARNING: Found a child with index {max_child_index}. Growing positional embeddings size")
            self.num_positional_embeddings = max_child_index * 2
            pe = ASTEmbeddings.get_positional_embeddings(self.type_embed_dim, self.num_positional_embeddings * 2)
            self.positional_embeddings = nn.Embedding.from_pretrained(pe)

    def get_node_embeddings_from_type(self, node_types, child_positions):
        # (batch_size, num_nodes) --> (batch_size, num_nodes, type_embed_dim)
        # Unnecessary since we filtered the training data to only include nodes w/ < 1000 children
        # self.ensure_positional_embeddings_fit(child_positions.max().item())
        return (self.type_embeddings(node_types) * ASTEmbeddings.SCALE_CONST
                + self.positional_embeddings(child_positions) / ASTEmbeddings.SCALE_CONST)

    def forward(self, node_types, node_vals, node_val_offsets, last_parent_index, child_positions):
        # node_types: (batch_size, num_nodes)
        # node_vals: ((sum of all the of lengths of the tokens in all the nodes of batch))
        # node_val_offsets: (batch_size, num_nodes)
        # last_parent_index: (batch_size, num_nodes)  <--- indicates index the parent of each node
        # child_positions: (batch_size, num_nodes)   <--- indicates which index of the parent each child is

        # (batch_size, num_nodes, type_embed_dim)
        node_type_embed = self.get_node_embeddings_from_type(node_types, child_positions)
        batch_size, num_nodes = node_types.shape
        hidden_states = torch.empty((num_nodes, batch_size, self.pos_embed_dim), device=self.gru.weight_hh.device)
        for i in range(num_nodes):
            # (batch_size, 1, pos_embed_dim) -> (batch_size, pos_embed_dim)
            prev_hidden_states = (hidden_states[(last_parent_index[:, i], torch.arange(batch_size), None)].squeeze(1)
                                  if i > 0 else torch.zeros((batch_size, self.pos_embed_dim), device=hidden_states.device))
            inputs = node_type_embed[:, i, :]  # (batch_size, type_embed_dim)
            hidden_states[i, :, :] = self.gru(inputs, prev_hidden_states)
        token_embed = self.token_embeddings(node_vals, node_val_offsets.view(-1))
        node_type_embed_final = node_type_embed + \
                                token_embed.view(batch_size, num_nodes, self.type_embed_dim) * ASTEmbeddings.SCALE_CONST
        return torch.cat((node_type_embed_final.transpose(0, 1), hidden_states), dim=-1)



class CodeEncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, dim_feedforward=2048):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=dim_feedforward)
        self.attn_parent = nn.Linear(in_features=embed_dim, out_features=n_heads)
        self.attn_child = nn.Linear(in_features=embed_dim, out_features=n_heads)
        self.embed_dim = embed_dim
        self.n_heads = n_heads

    def forward(self, node_inputs: torch.Tensor, hidden: torch.Tensor, parent_mask: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        # node_inputs: (num_nodes, batch_size, embed_dim),
        # parent_mask: (batch_size, num_nodes, num_nodes); True at (i, j) if i is a parent of j
        # hidden: True at (i, j) if i can't see j b/c of mask; (batch_size, num_nodes, num_nodes)
        # pad_mask: True at i if part of padding; (batch_size, num_nodes)

        # attn_mask: (batch_size, n_heads, num_nodes, num_nodes)
        batch_size = parent_mask.size(0)
        num_nodes = node_inputs.size(0)

        # (num_nodes, batch_size, n_heads) -> # (batch_size, n_heads, num_nodes)
        attn_parent_bias = self.attn_parent(node_inputs).permute((1, 2, 0))
        # (batch_size, n_heads, num_nodes, num_nodes) * (batch_size, n_heads, num_nodes, 1)
        attn_mask = (parent_mask.transpose(1, 2).unsqueeze(1).expand((-1, self.n_heads, -1, -1))
                     * attn_parent_bias.unsqueeze(-1))

        attn_child_bias = self.attn_child(node_inputs).permute((1, 2, 0))
        attn_mask += parent_mask.unsqueeze(1).expand((-1, self.n_heads, -1, -1)) * attn_child_bias.unsqueeze(-1)
        # True at (i, j) if node i can't see node j; (batch_size, num_nodes, num_nodes)
        # I.e. i can't see j if j is in node_mask_indices and j isn't a parent of i, nor i itself
        # hidden = (node_mask_indices.unsqueeze(1).expand((-1, num_nodes, -1)) &
        #           ~parent_mask.transpose(1, 2) &
        #           ~torch.eye(num_nodes, dtype=torch.bool, device=node_mask_indices.device).unsqueeze(0))
        attn_mask.masked_fill_(hidden.unsqueeze(1), float('-inf'))
        return self.encoder_layer(node_inputs, src_mask=attn_mask.view(batch_size * self.n_heads, num_nodes, num_nodes),
                                  src_key_padding_mask=pad_mask)


class CodeEncoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=512, n_heads=8, dim_feedforward=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.layers = nn.ModuleList([CodeEncoderLayer(embed_dim, n_heads, dim_feedforward) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, node_inputs, hidden, parent_mask, pad_mask):
        output = node_inputs
        for layer in self.layers:
            output = layer(output, hidden, parent_mask, pad_mask)
        return output


class CodeMaskPrediction(nn.Module):
    def __init__(self, ast_embed_model, code_encoder):
        super().__init__()
        self.ast_embed_model = ast_embed_model
        self.code_encoder = code_encoder
        self.embed_to_node_type = nn.Linear(in_features=self.code_encoder.embed_dim,
                                            out_features=ast_embed_model.num_types)
        self.embed_to_token = nn.Linear(in_features=self.code_encoder.embed_dim,
                                             out_features=ast_embed_model.num_tokens)

    def forward(self, tree_data, pad_mask):
        # ['node_types', 'parent_mask', 'last_parent_index', 'child_index',
        #  'node_vals', 'node_val_offsets', 'hidden', 'mask_top_index',
        #  'target_node_types', 'target_vals']
        embeddings = self.ast_embed_model(tree_data.node_types, tree_data.node_vals, tree_data.node_val_offsets,
                                          tree_data.last_parent_index, tree_data.child_index)
        encoder_output = self.code_encoder(embeddings, tree_data.hidden, tree_data.parent_mask, pad_mask)
        node_type_predictions = self.embed_to_node_type(encoder_output)
        node_token_predictions = self.embed_to_token(encoder_output)

        return node_type_predictions, node_token_predictions



def test():
    model = CodeEncoderLayer(12, 3, 48).to('cpu')
    model.eval()
    # Focus only on children, for all attention heads
    e_leaf = torch.randn(12) * 5
    model.attn_child.weight[:,:] = e_leaf.repeat(3).view(3, 12)
    model.attn_child.bias = nn.Parameter(torch.tensor([10., 20., 40.]))

    # nodes in tree: 5; tree structure: c0->(c1->(c2 c3) c4)
    node_inputs = torch.randn((5, 1, 12))
    node_inputs[1, 0, :] = e_leaf
    node_mask_indices = torch.zeros(5).bool()
    pad_mask_indices = torch.zeros(5).bool()
    tree_structure = torch.tensor([5, 4, 3, 4, 5]).long()
    ar = torch.arange(5).unsqueeze(0).expand(5, -1)
    parent_mask = (ar.T < ar) & (ar < tree_structure.unsqueeze(1))
    last_parent_index = (parent_mask.T * (ar + 1)).argmax(dim=1)
    last_parent_index[0] = 5 - 1
    direct_parent_mask = torch.zeros((5, 5), dtype=torch.bool)
    direct_parent_mask[(last_parent_index, torch.arange(5))] = True
    child_index = (direct_parent_mask.cumsum(1) * direct_parent_mask).sum(dim=0) - 1
    # parent_mask = torch.zeros((5, 5), dtype=torch.bool)
    # for i in range(5):
    #     parent_mask[i, i+1:tree_structure[i]] = True
    node_mask_indices[1:5] = True
    print(model(node_inputs, node_mask_indices.unsqueeze(0), parent_mask.unsqueeze(0), pad_mask_indices.unsqueeze(0)))


if __name__ == "__main__":
    test()
