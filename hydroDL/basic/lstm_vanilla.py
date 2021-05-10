from typing import Dict

import torch
from torch import nn

from hydroDL.basic.head import get_head


class CudaLSTM(nn.Module):
    """From https://github.com/neuralhydrology/neuralhydrology/blob/a49f2e43cb2b25800adde2601ebf365db79dd745/neuralhydrology/modelzoo/cudalstm.py#L12
    LSTM model class, which relies on PyTorch's CUDA LSTM class.
    This class implements the standard LSTM combined with a model head, as specified in the params.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaLSTM` class only supports single-timescale predictions.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['input_layer', 'lstm', 'head']

    def __init__(self, n_time_series: int,
                 output_seq_len=1,
                 hidden_size: int = 20,
                 dr=0.5,
                 mode="Nto1",
                 initial_forget_bias=None):
        super().__init__()
        self.input_layer = nn.Linear(n_time_series, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dr)
        self.head = get_head(n_in=hidden_size, n_out=output_seq_len)
        self.mode = mode
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = self.initial_forget_bias

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


class LSTMForecast(nn.Module):
    """
    A very simple baseline LSTM model that returns
    an output sequence given a multi-dimensional input seq. Inspired by the StackOverflow link below.
    https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
    """

    def __init__(
            self,
            seq_length: int,
            n_time_series: int,
            output_seq_len=1,
            hidden_states: int = 20,
            num_layers=2,
            bias=True,
            batch_size=100,
            probabilistic=False,
            mode="Nto1"):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.hidden_dim = hidden_states
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(n_time_series, hidden_states, num_layers, bias, batch_first=True)
        self.probabilistic = probabilistic
        if self.probabilistic:
            output_seq_len = 2
        if mode == "Nto1":
            # TODO: Nto1 mode not finished
            self.final_layer = torch.nn.Linear(seq_length * hidden_states, output_seq_len)
        elif mode == "NtoN":
            self.final_layer = torch.nn.Linear(hidden_states, output_seq_len)
        else:
            raise NotImplementedError("No such mode yet!")
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_hidden(batch_size)

    def init_hidden(self, batch_size: int) -> None:
        """[summary]

        :param batch_size: [description]
        :type batch_size: int
        """
        # This is what we'll initialise our hidden state
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        out_x, self.hidden = self.lstm(x, self.hidden)
        if self.mode == "Nto1":
            output = self.final_layer(out_x.contiguous().view(batch_size, -1))
        elif self.mode == "NtoN":
            output = self.final_layer(out_x)
        else:
            raise NotImplementedError("No such mode yet!")

        if self.probabilistic:
            # TODO: check it
            mean = output[..., 0][..., None]
            std = torch.clamp(output[..., 1][..., None], min=0.01)
            output = torch.distributions.Normal(mean, std)
        return output
