"""
Author: MHPI group, Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:53:31
LastEditors: Wenyu Ouyang
Description: LSTM with dropout implemented by Kuai Fang and more LSTMs using it
FilePath: /HydroSPDB/hydrospdb/models/cudnnlstm.py
Copyright (c) 2021-2022 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import math

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from hydrospdb.models.ann import SimpleAnn
from hydrospdb.models.dropout import DropMask, create_mask


class LstmCellTied(torch.nn.Module):
    """
    LSTM with dropout implemented by Kuai Fang: https://github.com/mhpi/hydroDL/blob/release/hydroDL/model/rnn.py

    the name of "Tied" comes from this paper:
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    which means the weights of all gates will be tied together to be used (eq. 6 in this paper).
    this code is mainly used as a CPU version of CudnnLstm
    """

    def __init__(
        self,
        *,
        input_size,
        hidden_size,
        mode="train",
        dr=0.5,
        dr_method="drX+drW+drC",
        gpu=1
    ):
        super(LstmCellTied, self).__init__()

        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))

        self.drMethod = dr_method.split("+")
        self.gpu = gpu
        self.mode = mode
        if mode == "train":
            self.train(mode=True)
        elif mode == "test":
            self.train(mode=False)
        elif mode == "drMC":
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.mask_x = create_mask(x, self.dr)
        self.mask_h = create_mask(h, self.dr)
        self.mask_c = create_mask(c, self.dr)
        self.mask_w_ih = create_mask(self.w_ih, self.dr)
        self.mask_w_hh = create_mask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, do_reset_mask=True, do_drop_mc=False):
        if self.dr > 0 and (do_drop_mc is True or self.training is True):
            do_drop = True
        else:
            do_drop = False

        batch_size = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batch_size, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batch_size, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and do_reset_mask is True:
            self.reset_mask(x, h0, c0)

        if do_drop is True and "drH" in self.drMethod:
            h0 = DropMask.apply(h0, self.mask_h, True)

        if do_drop is True and "drX" in self.drMethod:
            x = DropMask.apply(x, self.mask_x, True)

        if do_drop is True and "drW" in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.mask_w_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.mask_w_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and "drC" in self.drMethod:
            gate_c = gate_c.mul(self.mask_c)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CpuLstmModel(torch.nn.Module):
    """
    Cpu version of CudnnLstmModel , ,
    """

    def __init__(self, *, n_input_features, n_output_features, n_hidden_states, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hiddenSize = n_hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(n_input_features, n_hidden_states)
        self.lstm = LstmCellTied(
            input_size=n_hidden_states,
            hidden_size=n_hidden_states,
            dr=dr,
            dr_method="drW",
            gpu=-1,
        )
        self.linearOut = torch.nn.Linear(n_hidden_states, n_output_features)
        self.gpu = -1

    def forward(self, x, do_drop_mc=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        reset_mask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), do_reset_mask=reset_mask)
            yt = self.linearOut(ht)
            reset_mask = False
            out[t, :, :] = yt
        return out


class CudnnLstm(torch.nn.Module):
    """
    LSTM with dropout implemented by Kuai Fang: https://github.com/mhpi/hydroDL/blob/release/hydroDL/model/rnn.py

    Only run in GPU; the CPU version is LstmCellTied in this file
    """

    def __init__(self, *, input_size, hidden_size, dr=0.5):
        """

        Parameters
        ----------
        input_size
            number of neurons in input layer
        hidden_size
            number of neurons in hidden layer
        dr
            dropout rate
        """
        super(CudnnLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        # self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):  # this func will be called when loading the model
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.mask_w_ih = create_mask(self.w_ih, self.dr)
        self.mask_w_hh = create_mask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, do_drop_mc=False, dropout_false=False):
        # dropout_false: it will ensure do_drop is false, unless do_drop_mc is true
        if dropout_false and (not do_drop_mc):
            do_drop = False
        elif self.dr > 0 and (do_drop_mc is True or self.training is True):
            do_drop = True
        else:
            do_drop = False

        batch_size = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)

        # cuDNN backend - disabled flat weight
        freeze_mask = False
        # handle = torch.backends.cudnn.get_handle()
        if do_drop is True:
            if not freeze_mask:
                self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, n_input_features, n_output_features, n_hidden_states, dr=0.5):
        """
        An LSTM model writen by Kuai Fang from this paper: https://doi.org/10.1002/2017GL075619

        only gpu version

        Parameters
        ----------
        n_input_features
            the number of input features
        n_output_features
            the number of output features
        n_hidden_states
            the number of hidden features
        dr
            dropout rate and its default is 0.5
        """
        super(CudnnLstmModel, self).__init__()
        self.nx = n_input_features
        self.ny = n_output_features
        self.hidden_size = n_hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = CudnnLstm(
            input_size=self.hidden_size, hidden_size=self.hidden_size, dr=dr
        )
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x, do_drop_mc=False, dropout_false=False, return_h_c=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(
            x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false
        )
        out = self.linearOut(out_lstm)
        return (out, (hn, cn)) if return_h_c else out


class KuaiLstm(torch.nn.Module):
    """
    An LSTM model writen by Kuai Fang from this paper: https://doi.org/10.1002/2017GL075619
    """

    def __init__(
        self, n_input_features, n_output_features, n_hidden_states, dr=0.5, gpu=0
    ):
        """
        CPU or GPU version of Kuai's LSTM

        Parameters
        ----------
        n_input_features
            the number of input features
        n_output_features
            the number of output features
        n_hidden_states
            the number of hidden features
        dr
            dropout rate and its default is 0.5
        gpu
            if gpu<0 or torch.cuda is not available, we will use CPU version
        """
        super(KuaiLstm, self).__init__()
        if gpu < 0 or (not torch.cuda.is_available()):
            self.kuai_lstm = CpuLstmModel(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                n_hidden_states=n_hidden_states,
                dr=dr,
            )
        else:
            self.kuai_lstm = CudnnLstmModel(
                n_input_features=n_input_features,
                n_output_features=n_output_features,
                n_hidden_states=n_hidden_states,
                dr=dr,
            )

    def forward(self, x):
        return self.kuai_lstm(x)


class LinearCudnnLstmModel(torch.nn.Module):
    """
    This model is nonlinear layer + CudnnLSTM/CudnnLstm-MultiOutput-Model. It is used for transfer learning.
    """

    def __init__(self, linear_size, model_name="kai_tl", tl_tag=True, **kwargs):
        """

        Parameters
        ----------
        linear_size
            the number of input features for the first input linear layer
        model_name
            we provide two types of transfer learning model：
            1. kai_tl: model from this paper by Ma et al. -- https://doi.org/10.1029/2020WR028600
            2. multi_output_tl: transfer learning model for multi-output-cudnnlstm
        tl_tag
            when the source model's layers are totally same as target model's, it is set True;
            or it will be False, by default True
        output_size
            the number of output features
        hidden_size
            the number of hidden features
        dr
            dropout rate and its default is 0.5
        """
        super(LinearCudnnLstmModel, self).__init__()
        self.former_linear = torch.nn.Linear(linear_size, kwargs["n_input_features"])
        # the name must be "tl_part"
        if model_name == "kai_tl":
            self.tl_part = CudnnLstmModel(**kwargs)
        elif model_name == "multi_output_tl":
            self.tl_part = CudnnLstmModelMultiOutput(**kwargs)
        else:
            raise NotImplementedError(
                "We don't have such a transfer learning model; please choose one from 'kai_tl' or 'multi_output_tl'"
            )
        # tl_tag means this is a transfer learning (tl) model
        # generally this is a tl model, but sometimes we directly load weight of a tl model to it,
        # and now it is just a normal model rather than a tl model
        self.tl_tag = tl_tag

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.former_linear(x))
        return self.tl_part(x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
