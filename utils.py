import pandas as pd
import random
import os
import torch.nn as nn
import torch
import numpy as np
import pickle as pkl
import time
from datetime import timedelta
from tqdm import tqdm
from logger import logger

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip().strip('\n')
            if not line:continue
            content = line[11:]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dataset(args, use_word=False):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(args.vocab_path):
        vocab = pkl.load(open(args.vocab_path, 'rb'))
    else:
        vocab = build_vocab(args.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(args.vocab_path, 'wb'))
    logger.info(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip().strip('\n')
                if not lin:
                    continue
                label, content = lin[:10], lin[11:]
                if label == "__label__0": label = 0
                else: label = 1
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(args.train_path, args.max_seq_length)
    test = load_dataset(args.test_path, args.max_seq_length)
    return vocab, train, test

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def build_iterator(dataset, args):
    iter = DatasetIterater(dataset, args.batch_size, args.device)
    return iter

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def convert_test_csv_to_fasttext_format(origin_path, target_path):
    test_dict = {}
    df = pd.read_csv(origin_path)
    file = open(target_path, 'w', encoding='utf-8')
    for idx, value in df.iterrows():
        sentence = value["sentence"]
        label = value["label"]
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        test_dict[r"{}".format(sentence)] = 1
        if label == 0.0:
            file.write("__label__0 " + r"{}".format(sentence))
        else:
            file.write("__label__1 " + r"{}".format(sentence))
        file.write("\n")
    return test_dict

def convert_train_csv_to_fasttext_format(origin_path, target_path, test_dict):
    df = pd.read_csv(origin_path)
    file = open(target_path, 'w', encoding='utf-8')
    for idx, value in df.iterrows():
        sentence = value["MESSAGE"][1:-1]
        if not sentence or len(sentence) == 0: continue
        label = value["labels"]
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        if sentence in test_dict: continue
        if label == 0.0:
            file.write("__label__0 " + r"{}".format(sentence))
        else:
            file.write("__label__1 " + r"{}".format(sentence))
        file.write("\n")

def random_select(origin_path, target_path):
    target_file = open(target_path, 'w', encoding='utf-8')
    origin_file = open(origin_path, 'r', encoding='utf-8')
    lines = origin_file.readlines()
    pos_count = 0
    neg_count = 1
    for idx, line in enumerate(lines):
        label = line[:10]
        if label == "__label__0" and 0.0 <= random.random() <= 0.20:
            target_file.write(line)
            neg_count += 1
        elif label == "__label__1":
            target_file.write(line)
            pos_count += 1
        else:
            continue


if __name__ == "__main__":
    # test_dict = convert_test_csv_to_fasttext_format(origin_path="./data/all_data.csv", target_path="./data/train.txt")
    # convert_train_csv_to_fasttext_format(origin_path="./data/clean_data_with_label.csv", target_path="./data/test.txt", test_dict=test_dict)
    # random_select("./data/test.txt", "./data/balanced_test.txt")
    # train_dir = './data/train.txt'
    # config = Config()
    # build_dataset(config, ues_word=False)
    file = open("./data/temp.txt", encoding='utf-8')
    lines = file.readlines()
    neg_count, pos_count = 0, 0
    for line in lines:
        line = line.strip(' ').strip('\n')
        logger.info(line)
        logger.info(len(line))
        label = line[:10]
        sent = line[11:]
        logger.info(label + sent)

