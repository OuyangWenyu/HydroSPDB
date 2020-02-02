import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data import GagesSource
from explore import trans_norm


class SimInputDataset(Dataset):
    """simulated streamflow input"""

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.data_flow = data_source.prepare_flow_data()
        self.data_target = data_source.read_outflow()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data_flow[index]
        y = self.data_target[index]
        return x, y

    def __len__(self):
        return len(self.data_flow)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape
            - trg_seq: torch tensor of shape
    Returns:
        src_seqs: torch tensor of shape (time_seq_length ,batch_size, one_unit_length).
        trg_seqs: torch tensor of shape (time_seq_length ,batch_size, one_unit_length).
    """

    def merge(sequences):
        padded_seqs = torch.zeros(sequences[0].shape[0], len(sequences), sequences[0].shape[1])
        for i, seq in enumerate(sequences):
            padded_seqs[:, i, :] = seq
        return padded_seqs

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # transform sequences (from 2D tensor to 3D tensor)
    src_seqs = merge(src_seqs)
    trg_seqs = merge(trg_seqs)
    return src_seqs, trg_seqs


def get_loader(dataset, batch_size=100, shuffle=False, num_workers=0):
    """Returns data loader for custom dataset.
    Args:
        dataset: dataset
        batch_size: mini-batch size.
        shuffle: is shuffle?
        num_workers: num of cpu core
    Returns:
        data_loader: data loader for custom dataset.
    """
    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    if num_workers < 1:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                 num_workers=num_workers)
    return data_loader


class SimNatureFlowInput(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.data_input = data_source.read_natural_inflow()

    def get_data_inflow(self, rm_nan=True):
        """径流数据读取及归一化处理，会处理成三维，最后一维长度为1，表示径流变量"""
        data = self.data_input
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        # transform x to 3d, the final dim's length is the seq_length
        seq_length = self.data_source.model_data.data_source.data_config.model_dict["model"]["seqLength"]
        data_inflow = np.zeros([data.shape[0], data.shape[1] - seq_length + 1, seq_length])
        for i in range(data_inflow.shape[1]):
            data_inflow[:, i, :] = data[:, i:i + seq_length]
        return data_inflow

    def load_data(self, model_dict):
        """transform x to 3d, the final dim's length is the seq_length, add forcing with natural flow"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length):
            """cut to size same as inflow's"""
            temp = temp_x[:, temp_seq_length - 1:, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        q = self.get_data_inflow(rm_nan=rm_nan_x)
        x, y, c = self.data_source.model_data.load_data(model_dict)
        seq_length = model_dict["model"]["seqLength"]

        if seq_length > 1:
            x = cut_data(x, rm_nan_x, seq_length)
            y = cut_data(y, rm_nan_y, seq_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c
