import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(
            args.n_vocab,
            args.n_embed,
            padding_idx=args.n_vocab - 1)
        self.lstm = nn.LSTM(
            args.n_embed,
            args.hidden_size,
            args.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout)
        self.fc = nn.Linear(args.hidden_size * 2, 2)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out
