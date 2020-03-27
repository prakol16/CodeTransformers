import argparse

parser = argparse.ArgumentParser(description="""Preprocess data, creating the subtokens.txt file from
subtokens, train_nodes from python100k_train.json, or test_nodes from python50k_eval.json
""")
parser.add_argument('--generate-subtokens', action="store_true", help="Generate subtokens file from infile-train")
parser.add_argument('--generate-node-data', action="store_true", help="Generate preprocessed data")
parser.add_argument('--infile-train', help="The input file for training such as python100k_train.json")
parser.add_argument('--infile-test', nargs='?', help="The input file for testing such as python50k_eval.json (not "
                                                     "needed for generating subtokens.txt)")
parser.add_argument('--subtokens-file', '-s', help="The file for the subtoken vocabulary")
parser.add_argument('--vocab-size', '-v', help="The number of subtokens to keep for the vocabulary size", type=int, default=50000)
parser.add_argument('--nodes-out', '-nout', help="The output file for the pickled nodes. Will be suffixed with _train,"
                                                 "_val, and _test")
parser.add_argument('--chunk-size', '-c', help="The size of the chunks for parsing the nodes", type=int, default=5000)

parser.add_argument('--max-fn-size', '-max', help="The maximum size (in number of nodes) to keep. Functions/Classes "
                                                  "with more nodes will be discarded", type=int, default=1000)
parser.add_argument('--train-val-split', type=float, help="The fraction of infile-train that should go into train"
                                                          " rather than val", default=0.8)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.generate_subtokens:
        from load_python_trees.tokenize_identifiers import get_subtoken_vocab, save_ind2token
        print("Getting subtokens from dataset...")
        ind2token = get_subtoken_vocab(args.infile, vocab_size=args.vocab_size, file_num_lines=args.num_lines)
        save_ind2token(args.subtokens_file, ind2token)
    if args.generate_node_data:
        from load_python_trees.node import preprocess
        print("Saving pickled nodes... (train)")
        num_train, num_val = preprocess(args.infile_train, args.nodes_out + '.train.pkl', args.subtokens_file, 100000,
                   args.max_fn_size, args.chunk_size, validation=(args.nodes_out + '.val.pkl', args.train_val_split))
        print("Num train examples", num_train)
        print("Num val examples", num_val)
        if args.infile_test is not None:
            print("Saving pickled nodes... (test)")
            num_test, _ = preprocess(args.infile_test, args.nodes_out + '.test.pkl', args.subtokens_file, 50000,
                       args.max_fn_size, args.chunk_size)
            assert _ == 0
            print("Num test examples", num_test)

