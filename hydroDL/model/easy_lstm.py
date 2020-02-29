"""@author: keitakurita -- https://github.com/keitakurita """
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *
import torch.nn.functional as F


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class PytorchLstm(nn.Module):
    def __init__(self, *, nx, ny, hidden_size, dropouti=0, dropouto=0.5):
        super(PytorchLstm, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dropout_rate = dropouto
        self.lstm = nn.LSTM(input_size=self.nx, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=True,
                            dropout=dropouto)
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x):
        out_lstm, (h_n, c_n) = self.lstm(x)
        out = self.linearOut(out_lstm)
        return out


class EasyLstm(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, dropouti=0, dropouto=0.5):
        super(EasyLstm, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.lstm = LSTM(nx, hidden_size, dropouti=dropouti, dropouto=dropouto)
        self.linearOut = torch.nn.Linear(hidden_size, ny)

    def forward(self, x):
        out_lstm, state = self.lstm(x)
        out = self.linearOut(out_lstm)
        return out


class StackedEasyLstm(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, dropouti=0.2, dropouto=0.5):
        super(StackedEasyLstm, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.lstm1 = LSTM(nx, hidden_size, dropouti=dropouti, dropouto=dropouto)
        self.lstm2 = LSTM(hidden_size, ny, dropouti=dropouti, dropouto=dropouto)

    def forward(self, x):
        out_lstm1, state1 = self.lstm1(x)
        out_lstm2, state2 = self.lstm2(out_lstm1)
        return out_lstm2


class LinearEasyLstm(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, dropouti=0.5, dropouto=0.5):
        super(LinearEasyLstm, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        self.lstm = LSTM(hidden_size, hidden_size, dropouti=dropouti, dropouto=dropouto)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        self.gpu = 1

    def forward(self, x):
        # self.lstm.flatten_parameters()
        x0 = F.relu(self.linearIn(x))
        out_lstm, state = self.lstm(x0)
        out = self.linearOut(out_lstm)
        return out
