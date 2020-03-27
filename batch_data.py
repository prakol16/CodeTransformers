import torch
from collections import namedtuple
from constants import device
import random

MASK_INDEX = 1

TreeData = namedtuple('TreeData', ['node_types', 'parent_mask', 'last_parent_index', 'child_index',
                                  'node_vals', 'node_val_offsets', 'hidden', 'mask_top_index',
                                   'target_node_types', 'target_vals'])


def produce_tree_data(all_nodes, tree_structure_, values) -> TreeData:
    num_nodes = len(all_nodes)
    assert num_nodes == len(tree_structure_) == len(values)
    tree_structure = torch.tensor(tree_structure_, dtype=torch.long, device=device)
    ar = torch.arange(1, num_nodes + 1, device=device).unsqueeze(0).expand(num_nodes, -1)
    parent_mask = (torch.triu(torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=device), diagonal=1)
                   & (ar <= tree_structure.unsqueeze(1)))
    last_parent_index = (parent_mask.T * ar).argmax(dim=1)
    # Technically, there is no parent for the root node, since it is the root, so argmax could return anything
    # since all of the values along that column will be the same (0)
    # However, for the purposes of calculating child_index, we set the root node to be a child of the last node
    # The num_nodes-1 th node by DFS can't possibly have any other children, so this will correctly
    # set child_index[0] to be 0 without messing anything else up
    last_parent_index[0] = num_nodes - 1
    direct_parent_mask = torch.zeros_like(parent_mask)
    direct_parent_mask[(last_parent_index, torch.arange(num_nodes, device=device))] = True
    child_index = (direct_parent_mask.cumsum(1) * direct_parent_mask).sum(dim=0) - 1

    mask_index = random.randrange(num_nodes)
    mask_index_end = tree_structure_[mask_index]
    node_types = torch.tensor(all_nodes, dtype=torch.long, device=device)
    target_node_types = torch.zeros_like(node_types)
    target_node_types[mask_index:mask_index_end] = node_types[mask_index:mask_index_end]
    node_types[mask_index:mask_index_end] = MASK_INDEX

    hidden = (parent_mask[mask_index].unsqueeze(0).expand((num_nodes, -1))
              & ~parent_mask.T & ~torch.eye(num_nodes, dtype=torch.bool, device=device))

    node_vals = torch.tensor([v for i, value_list in enumerate(values)
                              if not (mask_index <= i < mask_index_end) for v in value_list], device=device)
    node_val_offsets = torch.tensor([(len(value_list) if not (mask_index <= i < mask_index_end) else 0)
                                     for i, value_list in enumerate(values)], device=device).cumsum(dim=0)
    target_vals = torch.zeros_like(node_val_offsets)
    target_vals[mask_index:mask_index_end] = torch.tensor(
        [random.choice(value_list) if value_list else 0 for value_list in values[mask_index:mask_index_end]],
        device=device
    )
    return TreeData(node_types, parent_mask, last_parent_index, child_index,
                    node_vals, node_val_offsets, hidden, mask_index, target_node_types, target_vals)


def collate_batch(all_tree_data):
    batch_size = len(all_tree_data)
    max_num_nodes = max(tree_data.node_types.size(0) for tree_data in all_tree_data)

    node_types_out = torch.zeros((batch_size, max_num_nodes), dtype=torch.long, device=device)
    parent_mask_out = torch.zeros((batch_size, max_num_nodes, max_num_nodes), dtype=torch.bool, device=device)
    hidden_out = torch.zeros_like(parent_mask_out)
    last_parent_index_out = torch.zeros_like(node_types_out)
    child_index_out = torch.zeros_like(node_types_out)
    node_val_offsets = torch.empty_like(node_types_out)
    mask_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    target_node_types_out = torch.zeros_like(node_types_out)
    target_vals_out = torch.zeros_like(node_val_offsets)
    pad_mask = torch.zeros((batch_size, max_num_nodes), dtype=torch.bool, device=device)

    node_vals_len = sum(tree_data.node_vals.size(0) for tree_data in all_tree_data)
    node_vals = torch.empty(node_vals_len, dtype=torch.long, device=device)
    val_offset = 0

    for i, tree_data in enumerate(all_tree_data):
        num_nodes = tree_data.node_types.size(0)
        node_types_out[i, :num_nodes] = tree_data.node_types
        last_parent_index_out[i, :num_nodes] = tree_data.last_parent_index
        child_index_out[i, :num_nodes] = tree_data.child_index
        parent_mask_out[i, :num_nodes, :num_nodes] = tree_data.parent_mask
        hidden_out[i, :num_nodes, :num_nodes] = tree_data.hidden
        mask_indices[i] = tree_data.mask_top_index
        target_node_types_out[i, :num_nodes] = tree_data.target_node_types
        target_vals_out[i, :num_nodes] = tree_data.target_vals
        pad_mask[i, num_nodes:] = True

        total_num_vals = tree_data.node_val_offsets[-1].item()
        node_vals[val_offset:(val_offset + total_num_vals)] = tree_data.node_vals
        node_val_offsets[i, 0] = val_offset
        node_val_offsets[i, 1:num_nodes] = tree_data.node_val_offsets[:-1] + val_offset
        node_val_offsets[i, num_nodes:] = total_num_vals + val_offset
        val_offset += total_num_vals

    return TreeData(node_types_out, parent_mask_out, last_parent_index_out, child_index_out, node_vals,
                    node_val_offsets, hidden_out, mask_indices, target_node_types_out, target_vals_out), pad_mask

