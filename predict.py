import os
import torch
import argparse
import pickle as pkl
from logger import logger
from net.lstm import BiLSTM
from train_test import test, evaluate
from utils import  *
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


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


def predict(text, use_word, vocab_path, pad_size, args):
    if not text or len(text) == 0: return 0
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
    args.n_vocab = len(vocab)
    token = tokenizer(text)
    words_line = []
    token = token[:pad_size]
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    x = torch.LongTensor([words_line]).to(args.device)
    seq_len = torch.LongTensor([seq_len]).to(args.device)
    model = BiLSTM(args)
    model.load_state_dict(torch.load(args.save_path))
    out = model((x, seq_len))
    predict = torch.max(out.data, 1)[1].cpu().numpy()
    return predict


if __name__ == "__main__":
    total_count = 0
    accur_count = 0
    totol_time = 0
    file = open(args.test_path, 'r', encoding='utf-8')
    for line in file.readlines():
        true = line[:10]
        sent = line[11:]
        sent = sent.strip().strip('\n')
        if not sent or len(sent) == 0: continue
        total_count += 1
        start_time = time.time()
        pred = predict(sent, use_word=False, vocab_path=args.vocab_path, pad_size=128, args=args)
        totol_time += time.time() - start_time
        if true == "__label__0" and pred[0] == 0:
            accur_count += 1
        elif true == "__label__1" and pred[0] == 1:
            accur_count += 1
    logger.info(accur_count / total_count)
    logger.info(totol_time / total_count)
    model = BiLSTM(args)
    model.load_state_dict(torch.load(args.save_path))
    vocab, train_data, test_data = build_dataset(args, use_word=False)
    train_iter = build_iterator(train_data, args)
    test_iter = build_iterator(test_data, args)
    acc, _ = evaluate(args, model, test_iter)
    logger.info(acc)
    acc, _ = evaluate(args, model, train_iter)
    logger.info(acc)