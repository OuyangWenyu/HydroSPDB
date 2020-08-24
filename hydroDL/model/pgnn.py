import torch

from hydroDL.model.ann import AnnModel


class CudnnWaterBalanceNN(torch.nn.Module):
    def __init__(self, nx, ny, hidden_size_stroage, hidden_size, iter_num):
        super(CudnnWaterBalanceNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hidden_size
        self.iter_num = iter_num
        self.nn_outflow = AnnModel(nx=nx[1], ny=nx[2], hidden_size=hidden_size_stroage)
        self.linear_water_balance = torch.nn.Linear(nx[0] + nx[2], ny)

    def forward(self, inflows, storage0):
        iter_num = self.iter_num
        storages = torch.empty(iter_num + 1)
        outflows = torch.empty(iter_num)
        for i in iter_num:
            if i == 0:
                storages[i] = storage0
            storflowi = torch.cat((inflows[i], storages[i]), 0)
            outflows[i] = self.nn_outflow(storflowi)
            outstorflowi = torch.cat((outflows[i], storflowi), 0)
            storages[i + 1] = self.linear_water_balance(outstorflowi)
        return storages
