import torch

from hydroDL.model.ann import AnnModel


class CudnnWaterBalanceNN(torch.nn.Module):
    # TODO: not done
    def __init__(self, nx_o, ny_o, hs_o, nx_s, ny_s, hs_s, delta_t):
        # nx_i: x,c,t ; nx_s: x,c,t,s0,d (set d as ReLU(cropet-precp))
        super(CudnnWaterBalanceNN, self).__init__()
        self.delta_t = delta_t
        self.nn_outflow = AnnModel(nx=nx_o, ny=ny_o, hidden_size=hs_o)
        self.nn_storage = AnnModel(nx=nx_s, ny=ny_s, hidden_size=hs_s)

    def forward(self, x, t, d):
        # x: forcing and attr; t: time index, the start date is 0; d: water demand
        eps = torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps))
        x_o = torch.cat((x, torch.full(x.shape[0], t)), dim=1)
        x_s1 = torch.cat((x, torch.full(x.shape[0], t), d), dim=1)
        # TODO: how to set s1 and s2?
        x_s2 = torch.cat((x, torch.full(x.shape[0], t + eps), d), dim=1)
        x_s1 = self.nn_storage(x_s1)
        x_s2 = self.nn_storage(x_s2)
        x_q = self.nn_outflow(x_o)
        out = torch.cat((x_q, (x_s2 - x_s1) / eps + x_q), dim=1)
        return out
