from typing import List

import torch

from hydroDL.model_dict_function import pytorch_criterion_dict


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t) ** 2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t) ** 2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST) + 0.1) ** 2)
                losssum = losssum + temp
                nsample = nsample + 1
        loss = losssum / nsample
        return loss


class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        seq_length = target.shape[0]
        Ngage = target.shape[1]
        p = output[:, :, 0]
        t = target[:, :, 0]
        tmean = torch.mean(t, dim=0)
        tmeans = tmean.repeat(seq_length, 1)
        SST = torch.sum((t - tmeans) ** 2, dim=0)
        SSRes = torch.sum((t - p) ** 2, dim=0)
        # Same as Fredrick 2019
        temp = SSRes / ((torch.sqrt(SST) + 0.1) ** 2)
        # original NSE
        # temp = SSRes / SST
        loss = torch.sum(temp) / Ngage
        return loss


class WarmupRmseLoss(torch.nn.Module):
    def __init__(self, warmup_len):
        super(WarmupRmseLoss, self).__init__()
        self.warmup_len = warmup_len

    def forward(self, output, target):
        warmup_len = self.warmup_len
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[warmup_len:, :, 0]
            t0 = target[warmup_len:, :, 0]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t) ** 2).mean())
            loss = loss + temp
        return loss


def make_criterion_functions(crit_list) -> List:
    """crit_list should be either dict or list"""
    # TODO: not used now
    final_list = []
    if type(crit_list) == list:
        for crit in crit_list:
            final_list.append(pytorch_criterion_dict[crit]())
    else:
        for k, v in crit_list.items():
            final_list.append(pytorch_criterion_dict[k](**v))
    return final_list