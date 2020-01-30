"""for stacked lstm"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data import GagesSource, DataModel
from explore import trans_norm


class GagesSourceDataset(GagesSource):
    """extend GagesSource to utilize its function"""

    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)
        self.gages_data = DataModel(self)

    def read_attr_forcing(self):
        """generate flow from model, reshape to a 3d array, and transform to tensor:
        1d: nx * ny_per_nx
        2d: miniBatch[1]
        3d: length of time sequence, now also miniBatch[1]
        """
        # read data for model of allref
        sim_model_data = self.gages_data
        sim_config_data = sim_model_data.data_source.data_config
        batch_size = sim_config_data.model_dict["train"]["miniBatch"][0]
        x, y, c = sim_model_data.load_data(sim_config_data.model_dict)
        # concatenate x with c
        input_data = np.concatenate(x, c)
        return input_data

    def read_outflow(self):
        """read streamflow data as observation data, transform array to tensor"""
        gages_model_data = self.gages_data
        data_flow = gages_model_data.data_flow
        data = np.expand_dims(data_flow, axis=2)
        stat_dict = gages_model_data.stat_dict
        data = trans_norm(data, 'usgsFlow', stat_dict, to_norm=True)
        # cut the first rho data to match generated flow time series
        rho = self.data_config.model_dict["train"]["miniBatch"][1]
        data_chosen = data[:, rho - 1:, :]
        nx = data_chosen.shape[0]
        ny_per_nx = data_chosen.shape[1] - rho + 1
        x_tensor = torch.zeros([nx * ny_per_nx, rho, 1], requires_grad=False)
        for i in range(nx):
            per_x_np = np.zeros([ny_per_nx, rho, 1])
            for j in range(ny_per_nx):
                per_x_np[j, :, :] = data_chosen[i, j:j + rho, :]
            x_tensor[(i * ny_per_nx):((i + 1) * ny_per_nx), :, :] = torch.from_numpy(per_x_np)
        print("outflow ready!")
        return x_tensor


class GagesInputDataset(Dataset):
    """simulated streamflow input"""

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.data_input = data_source.read_attr_forcing()
        self.data_target = data_source.read_outflow()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data_input[index]
        y = self.data_target[index]
        return x, y

    def __len__(self):
        return len(self.data_input)


class GagesInvDataset(Dataset):
    """Dataset for inv model"""

    def __init__(self, data_model1, data_model2):
        self.data_input = self.prepare_input_inv(data_model1, data_model2)
        self.data_target = self.prepare_output(data_model2)

    def __getitem__(self, index):
        x = self.data_input[index]
        y = self.data_target[index]
        return x, y

    def __len__(self):
        return len(self.data_input)

    def prepare_input_inv(self, data_model1, data_model2):
        """prepare input for lstm-inv"""
        print("prepare input")
        return []

    def prepare_output(self, data_model):
        print("prepare target")
        return []


def collate_fn_inv(data):
    """
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


def get_loader_inv(dataset, batch_size=100, shuffle=False, num_workers=0):
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
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_inv)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_inv,
                                 num_workers=num_workers)
    return data_loader
