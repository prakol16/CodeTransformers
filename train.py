from ipaddress import AddressValueError

import torch
import torch.nn.functional as F
from torch.optim import Adam
import constants
import model
import random
from batch_data import collate_batch, produce_tree_data
import pickle
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm


from load_python_trees.node_types import all_types


parser = argparse.ArgumentParser(description="Trains a CodeTransformer model")
parser.add_argument('--train-src', help="The preprocessed training data")
parser.add_argument('--val-src', help="The preprocessed validation data")
parser.add_argument('--num-epochs', type=int, help="The number of epochs to train for", default=12)
parser.add_argument('--batch-size', type=int, help="Batch size", default=64)
parser.add_argument('--save-every', type=int, help="Save every n epochs", default=3)
parser.add_argument('--model-out', help="The directory to save the models in")


class ChunkedData(IterableDataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        return iter(produce_tree_data(*node) for node in self.data)


def load_all_data(f, batch_size, chunk_size):
    eof = False
    assert chunk_size % batch_size == 0, "Chunk size must be a multiple of batch size"
    while not eof:
        data = []
        for _ in range(chunk_size):
            try:
                data.append(pickle.load(f)[0])
            except EOFError:
                eof = True
                break
        random.shuffle(data)

        yield from DataLoader(ChunkedData(data), batch_size=batch_size, collate_fn=collate_batch)


def get_num_samples():
    from ASTNode import ASTNode

    def get_max_child_index(node: ASTNode):
        return max(len(node.children), max((get_max_child_index(child) for child in node.children), default=0))

    with open("python_150k_data_preprocessed/nodes.train.pkl", "rb") as f:
        num = 0
        max_child_index = 0
        max_num_nodes = 0
        for num in tqdm(range(854224)):
            try:
                node, num_nodes = ASTNode.build_from_vector(*pickle.load(f)[0])
                m = get_max_child_index(node)
                if m > max_child_index:
                    print(f"\nNew max child index {m} at {num=}")
                    max_child_index = m
                if num_nodes > max_num_nodes:
                    print(f"\nNew max num nodes {num_nodes} at {num=}")
                    max_num_nodes = num_nodes
            except EOFError:
                print(f"\nEnd of file encountered {num=}")
                break
            except Exception as e:
                print(f"\nUnkown other error {num=}", e)
                break
        print("\nFinal results", f"{num=} {max_child_index=} {max_num_nodes=}")


def train(train_file, val_file, num_tokens, batch_size, num_epochs, model_out_path, save_every):
    ast_embed_model = model.ASTEmbeddings(len(all_types) + 1, num_tokens)
    code_encoder = model.CodeEncoder()
    mask_prediction_model = model.CodeMaskPrediction(ast_embed_model, code_encoder).to(constants.device)

    mask_prediction_model.train()

    optim = Adam(mask_prediction_model.parameters(), lr=0.001)
    num_iter = 0

    best_val = float('inf')

    for epoch in range(num_epochs):
        train_file.seek(0)
        for tree_data, pad_mask in load_all_data(train_file, batch_size, batch_size * 100):
            optim.zero_grad()

            node_type_predictions, node_token_predictions = mask_prediction_model(tree_data, pad_mask)
            loss_node_type = F.cross_entropy(node_type_predictions.view(-1, ast_embed_model.num_types),
                                             tree_data.target_node_types.T.flatten(), ignore_index=0)
            loss_node_vals = F.cross_entropy(node_token_predictions.view(-1, ast_embed_model.num_tokens),
                                             tree_data.target_vals.T.flatten(), ignore_index=0)
            loss = loss_node_type + loss_node_vals
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mask_prediction_model.parameters(), 5)
            optim.step()

            num_iter += 1
            if num_iter % 10 == 0:
                print("Iter", num_iter, "Training loss:", loss.item())

        print("Finished epoch", epoch)
        print("Validating...")
        evaluation = validate(mask_prediction_model, ast_embed_model.num_types, ast_embed_model.num_tokens,
                              val_file, batch_size)
        print("Result:", evaluation, "Previous best:", best_val)
        if evaluation < best_val:
            print("New best! Saving model...")
            torch.save(mask_prediction_model.state_dict(),  f"{model_out_path}_best_weights.bin")
            torch.save(optim.state_dict(), f"{model_out_path}_best_optim.bin")
            best_val = evaluation
        if (epoch + 1) % save_every == 0:
            print("Saving model...")
            torch.save(mask_prediction_model.state_dict(),  f"{model_out_path}_weights_epoch_{epoch}.bin")
            torch.save(optim.state_dict(), f"{model_out_path}_optim_epoch_{epoch}.bin")


def validate(mask_prediction_model: torch.nn.Module, num_types, num_tokens, val_file, batch_size):
    val_file.seek(0)
    mask_prediction_model.eval()
    total_loss = 0
    num_vals = 0
    with torch.no_grad():
        for tree_data, pad_mask in load_all_data(val_file, batch_size, batch_size * 100):
            node_type_predictions, node_token_predictions = mask_prediction_model(tree_data, pad_mask)
            loss_node_type = F.cross_entropy(node_type_predictions.view(-1, num_types),
                                             tree_data.target_node_types.T.flatten(), ignore_index=0)
            loss_node_vals = F.cross_entropy(node_token_predictions.view(-1, num_tokens),
                                             tree_data.target_vals.T.flatten(), ignore_index=0)
            loss = loss_node_type + loss_node_vals
            total_loss += loss.item()
            num_vals += 1

    return total_loss / num_vals

if __name__ == "__main__":
    # Final results (train): num=1076863 max_child_index=7222, 4100, ....
    # Final results (eval): num=530254 max_child_index=1081, 1031, ...

    # train: 426293  580593
    # val: 107045    145300
    # test: 525781   357943
    args = parser.parse_args()
    with open(args.train_src, "rb") as train_file, open(args.val_src, "rb") as val_file:
        train(train_file, val_file, constants.num_tokens, args.batch_size,
              args.num_epochs, args.model_out, args.save_every)



# Steps:
# 1.