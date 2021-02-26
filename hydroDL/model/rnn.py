import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, create_mask
from . import cnn


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = create_mask(x, self.dr)
        self.maskX_f = create_mask(x, self.dr)
        self.maskX_c = create_mask(x, self.dr)
        self.maskX_o = create_mask(x, self.dr)

        self.maskH_i = create_mask(h, self.dr)
        self.maskH_f = create_mask(h, self.dr)
        self.maskH_c = create_mask(h, self.dr)
        self.maskH_o = create_mask(h, self.dr)

        self.maskC = create_mask(c, self.dr)

        self.maskW_xi = create_mask(self.w_xi, self.dr)
        self.maskW_xf = create_mask(self.w_xf, self.dr)
        self.maskW_xc = create_mask(self.w_xc, self.dr)
        self.maskW_xo = create_mask(self.w_xo, self.dr)
        self.maskW_hi = create_mask(self.w_hi, self.dr)
        self.maskW_hf = create_mask(self.w_hf, self.dr)
        self.maskW_hc = create_mask(self.w_hc, self.dr)
        self.maskW_ho = create_mask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
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
        self.maskX = create_mask(x, self.dr)
        self.maskH = create_mask(h, self.dr)
        self.maskC = create_mask(c, self.dr)
        self.maskW_ih = create_mask(self.w_ih, self.dr)
        self.maskW_hh = create_mask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
                F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, input_size, hidden_size, dr=0.5, dr_method='drW', gpu=2):
        super(CudnnLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

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
        handle = torch.backends.cudnn.get_handle()
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

        output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
            input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
            self.hidden_size, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, dr=0.5, gpu=1):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(input_size=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linearOut = torch.nn.Linear(hidden_size, ny)
        self.gpu = gpu

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0, do_drop_mc=do_drop_mc, dropout_false=dropout_false)
        out = self.linearOut(out_lstm)
        return out


class CudnnLstmModelPretrain(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, pretrian_model_file):
        super(CudnnLstmModelPretrain, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.lstm = torch.load(pretrian_model_file)

    def forward(self, x, do_drop_mc=False, dropout_false=False):
        out = self.lstm(x)
        return out


class CudnnLstmModel_R2P(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, filename):
        super(CudnnLstmModel_R2P, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        # self.linearR2P = torch.nn.Linear(nx[1], nx[2])
        self.linearR2Pa = torch.nn.Linear(nx[1], hiddenSize)
        self.linearR2Pb = torch.nn.Linear(hiddenSize, nx[2])
        self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize)

        # self.lstm = CudnnLstmModel(
        #    nx=nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.lstm = torch.load(filename)

        # self.lstm.eval()

        for param in self.lstm.parameters():
            param.requires_grad = False

        # self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        if type(x) is tuple or type(x) is list:
            Forcing, Raw = x

        # Param = F.relu(self.linearR2P(Raw))
        # Param = torch.atan(self.linearR2P(Raw))
        Param_a = torch.relu(self.linearR2Pa(Raw))
        # Param_a.permute(0,2,1)
        # Param_bn = self.bn1(Param_a)
        # Param_bn.permute(0,2,1)
        Param = torch.tanh(self.linearR2Pb(Param_a))
        x0 = torch.cat((Forcing, Param), dim=len(Param.shape) - 1)  # by default cat along dim=0
        # self.lstm.eval()
        outLSTM = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=True)
        # self.lstm.train()
        # out = self.linearOut(outLSTM)
        return outLSTM


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny
        self.gpu = 1

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            input_size=hiddenSize, hidden_size=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            input_size=hiddenSize, hidden_size=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2


class MLPModel(nn.Module):
    def __init__(self, num_features, *, dropout=0.25, n_hid=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, 1),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)


class CudnnLstmModelInv(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size, dr=0.5):
        super(CudnnLstmModelInv, self).__init__()
        self.nx = nx  # (xqch, xct, theta)
        self.ny = ny
        self.hiddenSize = hidden_size
        self.lstm_inv = CudnnLstmModel(nx=nx[0], ny=nx[2], hidden_size=int(hidden_size / 4),
                                       dr=dr)  # Input is forcing + Q + attr,
        self.lstm = CudnnLstmModel(nx=nx[1] + nx[2], ny=ny, hidden_size=hidden_size, dr=dr)
        self.gpu = 2

    def forward(self, xh, xt):
        param = self.lstm_inv(xh)
        x1 = torch.cat((xt, param), dim=len(param.shape) - 1)  # by default cat along dim=0
        out_lstm = self.lstm(x1)
        return out_lstm, param


class CudnnLstmModelInvKernel(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size, dr=0.5):
        super(CudnnLstmModelInvKernel, self).__init__()
        self.nx = nx  # (xqch, xct, theta)
        self.ny = ny
        self.hiddenSize = hidden_size
        self.lstm_inv = CudnnLstmModel(nx=nx[0], ny=nx[2], hidden_size=int(hidden_size / 4),
                                       dr=dr)  # Input is forcing + Q + attr,
        self.lstm = CudnnLstmModel(nx=nx[1] + nx[2], ny=ny, hidden_size=hidden_size, dr=dr)
        self.gpu = 2

    def forward(self, xh, xt):
        gen = self.lstm_inv(xh)
        # seq to one
        param = gen[-1, :, :].repeat(xt.shape[0], 1, 1)
        x1 = torch.cat((xt, param), dim=len(param.shape) - 1)  # by default cat along dim=0
        out_lstm = self.lstm(x1)
        return out_lstm, param


class CudnnLstmModelInvKernelPretrain(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size, pretrian_model_file):
        super(CudnnLstmModelInvKernelPretrain, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.lstm_inv = torch.load(pretrian_model_file)

    def forward(self, xh, xt, do_drop_mc=False, dropout_false=False):
        out_lstm, param = self.lstm_inv(xh, xt)
        return out_lstm, param


class CudnnLstmModelStorage(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size_stroage, hidden_size, dr=0.5):
        super(CudnnLstmModelStorage, self).__init__()
        self.nx = nx  # (qx+c, natflow(t-T:t)+c, theta=1)
        self.ny = ny
        self.hidden_size_stroage = hidden_size_stroage
        self.hiddenSize = hidden_size
        self.lstm_storage = CudnnLstmModel(nx=nx[1], ny=nx[2], hidden_size=hidden_size_stroage,
                                           dr=dr)
        self.lstm = CudnnLstmModel(nx=nx[0] + nx[2], ny=ny, hidden_size=hidden_size, dr=dr)

    def forward(self, qnc, qxc):
        gen = self.lstm_storage(qnc)
        # storage of different time and different sites should be different, so just concatenate gen with qxc
        x1 = torch.cat((qxc, gen), dim=len(gen.shape) - 1)  # by default cat along dim=0
        out_lstm = self.lstm(x1)
        return out_lstm, gen


class CudnnLstmModelStoragePretrain(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size_stroage, hidden_size, pretrian_model_file):
        super(CudnnLstmModelStoragePretrain, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size_stroage = hidden_size_stroage
        self.hidden_size = hidden_size
        self.lstm_storage = torch.load(pretrian_model_file)

    def forward(self, qnc, qxc, do_drop_mc=False, dropout_false=False):
        out_lstm, param = self.lstm_storage(qnc, qxc)
        return out_lstm, param


class CudnnLstmModelStorageSeq2One(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size_stroage, hidden_size, dr=0.5):
        super(CudnnLstmModelStorageSeq2One, self).__init__()
        self.nx = nx  # (qx+c, natflow(t-T:t)+c, theta=1)
        self.ny = ny
        self.hidden_size_stroage = hidden_size_stroage
        self.hiddenSize = hidden_size
        self.lstm_storage = CudnnLstmModel(nx=nx[1], ny=nx[2], hidden_size=hidden_size_stroage,
                                           dr=dr)
        self.lstm = CudnnLstmModel(nx=nx[0] + nx[2], ny=ny, hidden_size=hidden_size, dr=dr)

    def forward(self, qnc, qxc):
        gen = self.lstm_storage(qnc)
        # seq to one
        param = gen[-1, :, :].repeat(qxc.shape[0], 1, 1)
        x1 = torch.cat((qxc, param), dim=len(param.shape) - 1)  # by default cat along dim=0
        out_lstm = self.lstm(x1)
        return out_lstm, param


class CudnnLstmModelStorageSeq2OnePretrain(torch.nn.Module):
    def __init__(self, *, nx, ny, hidden_size_stroage, hidden_size, pretrian_model_file):
        super(CudnnLstmModelStorageSeq2OnePretrain, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size_stroage = hidden_size_stroage
        self.hidden_size = hidden_size
        self.lstm_storage = torch.load(pretrian_model_file)

    def forward(self, qnc, qxc, do_drop_mc=False, dropout_false=False):
        out_lstm, param = self.lstm_storage(qnc, qxc)
        return out_lstm, param
