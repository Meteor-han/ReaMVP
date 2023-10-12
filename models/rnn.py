from abc import ABC
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from scipy.stats import truncnorm


# with dropout and bn
class MLP(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1, bn=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bn = bn
        self.fc = torch.nn.ModuleList()

        if self.num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif self.num_layers == 1:
            self.fc.append(nn.Linear(input_dim, output_dim))
        else:
            self.fc.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 1):
                self.fc.append(nn.Dropout(p=dropout))
                self.fc.append(nn.ReLU())
                if self.bn:
                    self.fc.append(nn.BatchNorm1d(self.hidden_dim))
                if layer < num_layers - 2:
                    self.fc.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)


# SMILES (index) input, embedding layer
def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return torch.from_numpy(values).float()


class SMILESRNN(nn.Module, ABC):
    def __init__(self, vocab_size, embed_size, hidden_size, use_bidirectional,
                 num_layers, dropout, use_lstm, batch_first=True, **kwargs):
        super(SMILESRNN, self).__init__()
        # 0.02, the same as X-mol, 0.05, the same as Zhao etc.
        # self.word_emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embed_size)
        weight_word = truncated_normal((vocab_size, embed_size), threshold=0.05)
        self.word_emb = nn.Embedding.from_pretrained(weight_word, freeze=False)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_bidirectional = use_bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_lstm = use_lstm
        self.batch_first = batch_first
        self.rnn = None

    def forward(self, left, left_len):
        left = self.word_emb(left)
        left_pack = \
            rnn_utils.pack_padded_sequence(left, left_len.cpu(), batch_first=True, enforce_sorted=False)
        if self.use_lstm == 1:
            left_out, (left_hidden, left_cell) = self.rnn(left_pack)
        else:
            left_out, left_hidden = self.rnn(left_pack)
        # take the output of the last time step
        if self.use_bidirectional:
            out = torch.cat([left_hidden[-1, :, :], left_hidden[-2, :, :]], dim=1)
        else:
            out = left_hidden[-1, :, :]
        return out


class EmbedLSTMSMILES(SMILESRNN, ABC):
    def __init__(self, *args):
        super(EmbedLSTMSMILES, self).__init__(*args)
        self.rnn = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            bidirectional=self.use_bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=self.batch_first
        )


class EmbedGRUSMILES(SMILESRNN, ABC):
    def __init__(self, *args):
        super(EmbedGRUSMILES, self).__init__(*args)
        self.rnn = nn.GRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            bidirectional=self.use_bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=self.batch_first
        )


class GeoRNN(nn.Module, ABC):
    def __init__(self, input_size, hidden_size, use_bidirectional,
                 num_layers, dropout, use_lstm, batch_first=True):
        super(GeoRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bidirectional = use_bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_lstm = use_lstm
        self.batch_first = batch_first
        self.rnn = None

    # stupid, name it input and len?????
    def forward(self, input_, len_):
        len_ = len_.tolist()
        input_pack = rnn_utils.pad_sequence(torch.split(input_, len_, dim=0), batch_first=True, padding_value=0)
        if self.use_lstm:
            out, (hidden, cell) = self.rnn(input_pack)
        else:
            out, hidden = self.rnn(input_pack)
        # take the output of the last time step
        if self.use_bidirectional:
            out = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=1)
        else:
            out = hidden[-1, :, :]
        return out


class GeoLSTM(GeoRNN, ABC):
    def __init__(self, *args):
        super(GeoLSTM, self).__init__(*args)
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=self.use_bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=self.batch_first
        )


class GeoGRU(GeoRNN, ABC):
    def __init__(self, *args):
        super(GeoGRU, self).__init__(*args)
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=self.use_bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=self.batch_first
        )


if __name__ == '__main__':
    print()
