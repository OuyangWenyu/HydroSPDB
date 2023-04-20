from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from hydrospdb.models.head import get_head


class OfficialLstm(nn.Module):
    """LSTM model from Pytorch library"""

    def __init__(self, nx, ny, hidden_size, num_layers=2, dr=0.5):
        """
        A simple multi-layer LSTM model

        Parameters
        ----------
        nx
            number of input neurons
        ny
            number of output neurons
        hidden_size
            a list/tuple which contains number of neurons in each hidden layer;
            if int, only one hidden layer except for hidden_size=0
        num_layers
            how many layers in LSTM
        dr
            dropout rate
        """
        super(OfficialLstm, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hidden_size
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        # batch_first is False, so here we use seq_first mode
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dr,
        )
        self.linearOut = torch.nn.Linear(hidden_size, ny)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        out = self.linearOut(out_lstm)
        return out


class CudaLSTM(nn.Module):
    """
    From https://github.com/neuralhydrology/neuralhydrology/blob/a49f2e43cb2b25800adde2601ebf365db79dd745/neuralhydrology/modelzoo/cudalstm.py#L12
    LSTM model class, which relies on PyTorch's CUDA LSTM class.
    This class implements the standard LSTM combined with a model head, as specified in the params.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaLSTM` class only supports single-timescale predictions.
    """

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ["input_layer", "lstm", "head"]

    def __init__(
        self,
        input_size: int,
        output_size=1,
        hidden_size: int = 20,
        dr=0.5,
        mode="Nto1",
        initial_forget_bias=None,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )
        self.dropout = nn.Dropout(p=dr)
        self.head = get_head(n_in=hidden_size, n_out=output_size)
        self.mode = mode
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[
                self.hidden_size : 2 * self.hidden_size
            ] = self.initial_forget_bias

    def forward(self, data: [torch.Tensor]) -> [torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        Returns
        -------
        [torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.input_layer(data)
        # although LSTM has been set to batch_first, h_n and c_n are seq_first, but lstm_output is batch_first
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)
        if self.mode == "Nto1":
            pred = self.head(self.dropout(h_n[-1, :, :]))["y_hat"]
        elif self.mode == "NtoN":
            pred = self.head(self.dropout(lstm_output))["y_hat"]
        else:
            raise NotImplementedError("No such mode!!!")
        return pred
