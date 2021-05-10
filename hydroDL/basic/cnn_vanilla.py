import torch.nn as nn
import torch.nn.functional as F


# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class CNN1dKernel(nn.Module):
    def __init__(self,
                 *,
                 n_in_channel=1,
                 n_kernel=3,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(CNN1dKernel, self).__init__()
        self.cnn1d = nn.Conv1d(
            in_channels=n_in_channel,
            out_channels=n_kernel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output


class Cnn1d(nn.Module):
    def __init__(self, *, nx, nt, cnn_size=32, cp1=(64, 3, 2), cp2=(128, 5, 2)):
        super(Cnn1d, self).__init__()
        self.nx = nx
        self.nt = nt
        c_out, f, p = cp1
        self.conv1 = nn.Conv1d(nx, c_out, f)
        self.pool1 = nn.MaxPool1d(p)
        l_tmp = int(cal_conv_size(nt, f, 0, 1, 1) / p)

        c_in = c_out
        c_out, f, p = cp2
        self.conv2 = nn.Conv1d(c_in, c_out, f)
        self.pool2 = nn.MaxPool1d(p)
        l_tmp = int(cal_conv_size(l_tmp, f, 0, 1, 1) / p)

        self.flatLength = int(c_out * l_tmp)
        self.fc1 = nn.Linear(self.flatLength, cnn_size)
        self.fc2 = nn.Linear(cnn_size, cnn_size)

    def forward(self, x):
        # x- [nt,ngrid,nx]
        x1 = x
        x1 = x1.permute(1, 2, 0)
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.flatLength)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        return x1


def cal_conv_size(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def cal_pool_size(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def cal_final_size1d(nobs, n_out_k, k_size, stride, pool):
    n_layer = len(k_size)
    lout = nobs
    for ii in range(n_layer):
        lout = cal_conv_size(lin=lout, kernel=k_size[ii], stride=stride[ii])
        if pool is not None:
            lout = cal_pool_size(lin=lout, kernel=pool[ii])
    n_cnn_out = int(lout * n_out_k)  # total CNN feature number after convolution
    return n_cnn_out
