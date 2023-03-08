import argparse
from net.lstm import BiLSTM
from train_test import train
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="./data/raw_data.txt")
parser.add_argument("--test_path", type=str, default="./data/test.txt")
parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl")
parser.add_argument("--save_path", type=str, default="./models/model.bin")
parser.add_argument("--checkpoint_dir", type=str, default="")
parser.add_argument("--do_train", type=bool, default=True)
parser.add_argument("--do_eval", type=bool, default=True)
parser.add_argument("--bert_config_dir", type=str, default="")
parser.add_argument("--bert_model_dir", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_vocab", type=int, default=100)
parser.add_argument("--n_embed", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--num_train_steps", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--num_warmup_steps", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--require_improvement", type=int, default=1000)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()

if __name__ == "__main__":
    logger.info(args)
    vocab, train_data, test_data = build_dataset(args, use_word=False)
    train_iter = build_iterator(train_data, args)
    test_iter = build_iterator(test_data, args)
    args.n_vocab = len(vocab)
    model = BiLSTM(args).to(args.device)
    train(args, model, train_iter, test_iter)
