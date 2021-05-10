import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from .cnn_vanilla import CNN1dKernel, cal_conv_size, cal_pool_size
from .dropout import DropMask, create_mask


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, input_size, hidden_size, dr=0.5, dr_method='drW'):
        super(CudnnLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        # self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):  # this func will be called when loading the model
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

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
                DropMask.apply(self.w_hh, self.mask_w_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hidden_size, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hidden_size, 0, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, seq_length, n_time_series, output_seq_len, hidden_states, num_layers, bias, batch_size,
                 probabilistic, dr=0.5):
        # TODO: the parameters setting should be improved
        super(CudnnLstmModel, self).__init__()
        self.nx = n_time_series
        self.ny = output_seq_len
        self.hidden_size = hidden_states
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = CudnnLstm(input_size=self.hidden_size, hidden_size=self.hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
        out = self.linearOut(out_lstm)
        return out


class LinearCudnnLstmModel(torch.nn.Module):
    def __init__(self, seq_length, n_time_series, input_seq_len, output_seq_len, hidden_states, num_layers, bias,
                 batch_size, probabilistic, dr=0.5):
        super(LinearCudnnLstmModel, self).__init__()
        self.n_linear = n_time_series
        self.nx = input_seq_len
        self.ny = output_seq_len
        self.hidden_size = hidden_states
        self.ct = 0
        self.nLayer = 1
        self.former_linear = torch.nn.Linear(self.n_linear, self.nx)
        self.linearIn = torch.nn.Linear(self.nx, self.hidden_size)
        self.lstm = CudnnLstm(input_size=self.hidden_size, hidden_size=self.hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(self.hidden_size, self.ny)

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x = F.relu(self.former_linear(x))
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
        out = self.linearOut(out_lstm)
        return out


class CNN1dLCmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, nx, ny, nobs, hidden_size,
                 n_kernel: Union[list, tuple] = (10, 5), kernel_size: Union[list, tuple] = (3, 3),
                 stride: Union[list, tuple] = (2, 1), dr=0.5, pool_opt=None, cnn_dr=0.5, cat_first=True):
        """cat_first means: we will concatenate the CNN output with the x, then input them to the CudnnLstm model;
        if not cat_first, it is relu_first, meaning we will relu the CNN output firstly, then concatenate it with x"""
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hidden_size
        n_layer = len(n_kernel)
        self.features = nn.Sequential()
        n_in_chan = 1
        lout = nobs
        for ii in range(n_layer):
            conv_layer = CNN1dKernel(
                n_in_channel=n_in_chan, n_kernel=n_kernel[ii], kernel_size=kernel_size[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), conv_layer)
            if cnn_dr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnn_dr))
            n_in_chan = n_kernel[ii]
            lout = cal_conv_size(lin=lout, kernel=kernel_size[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if pool_opt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(pool_opt[ii]))
                lout = cal_pool_size(lin=lout, kernel=pool_opt[ii])
        self.N_cnn_out = int(lout * n_kernel[-1])  # total CNN feature number after convolution
        self.cat_first = cat_first
        if cat_first:
            nf = self.N_cnn_out + nx
            self.linearIn = torch.nn.Linear(nf, hidden_size)
            self.lstm = CudnnLstm(input_size=hidden_size, hidden_size=hidden_size, dr=dr)
        else:
            nf = self.N_cnn_out + hidden_size
            self.linearIn = torch.nn.Linear(nx, hidden_size)
            self.lstm = CudnnLstm(input_size=nf, hidden_size=hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        self.gpu = 1

    def forward(self, x, z, do_drop_mc=False):
        # z = n_grid*nVar add a channel dimension
        z = z.t()
        n_grid, nobs = z.shape
        rho, bs, n_var = x.shape
        # add a channel dimension
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (n_grid) * n_kernel * sizeafterconv
        z0 = z0.view(n_grid, self.N_cnn_out).repeat(rho, 1, 1)
        if self.cat_first:
            x = torch.cat((x, z0), dim=2)
            x0 = F.relu(self.linearIn(x))
            out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc)
        else:
            x = F.relu(self.linearIn(x))
            x0 = torch.cat((x, z0), dim=2)
            out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc)
        out = self.linearOut(out_lstm)
        # out = rho/time * batchsize * Ntargetvar
        return out


class CudnnLstmModelLstmKernel(torch.nn.Module):
    """use a trained/un-trained CudnnLstm as a kernel generator before another CudnnLstm."""

    def __init__(self, nx, ny, hidden_size, nk=None, hidden_size_later=None, cut=False, dr=0.5, delta_s=False):
        """delta_s means we will use the difference of the first lstm's output and the second's as the final output"""
        super(CudnnLstmModelLstmKernel, self).__init__()
        # These three layers are same with CudnnLstmModel to be used for transfer learning or just vanilla-use
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(input_size=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        # if cut is True, we will only select the final index in nk, and repeat it, then concatenate with x
        self.cut = cut
        # the second lstm has more input than the previous
        if nk is None:
            nk = ny
        if hidden_size_later is None:
            hidden_size_later = hidden_size
        self.linear_in_later = torch.nn.Linear(nx + nk, hidden_size_later)
        self.lstm_later = CudnnLstm(input_size=hidden_size_later, hidden_size=hidden_size_later, dr=dr)
        self.linear_out_later = torch.nn.Linear(hidden_size_later, ny)

        self.delta_s = delta_s
        # when delta_s is true, cut cannot be true, because they have to have same number params
        assert not (cut and delta_s)

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm1, (hn1, cn1) = self.lstm(x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
        gen = self.linearOut(out_lstm1)
        if self.cut:
            gen = gen[-1, :, :].repeat(x.shape[0], 1, 1)
        x1 = torch.cat((x, gen), dim=len(gen.shape) - 1)
        x2 = F.relu(self.linear_in_later(x1))
        out_lstm2, (hn2, cn2) = self.lstm_later(x2, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
        out = self.linear_out_later(out_lstm2)
        if self.delta_s:
            return gen - out
        # The first output must be the prediction
        return out, gen
