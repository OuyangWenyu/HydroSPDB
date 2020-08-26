import torch

from hydroDL.model.ann import AnnModel


class CudnnWaterBalanceNN(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size, iter_num, delta_t):
        super(CudnnWaterBalanceNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hidden_size
        self.iter_num = iter_num
        self.delta_t = delta_t
        self.nn_outflow = AnnModel(nx=nx, ny=ny, hidden_size=hidden_size)

    def forward(self, inflows, storage0):
        iter_num = self.iter_num
        delta_t = torch.from_numpy(self.delta_t).cuda()
        storages = torch.Tensor(iter_num + 1, list(storage0.shape)[0], 1).cuda()
        outflows = torch.Tensor(iter_num, list(storage0.shape)[0], 1).cuda()
        for i in range(iter_num):
            if i == 0:
                storages[i, :, 0] = storage0
            storflowi = torch.cat((inflows[i, :, :], storages[i, :, :]), 1)
            outflows[i, :, :] = self.nn_outflow(storflowi)
            storages[i + 1, :, :] = delta_t[0] * (inflows[i, :, :] - outflows[i, :, :]) + storages[i, :, :]
        return storages, outflows
