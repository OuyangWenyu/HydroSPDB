import torch
from torch.utils.data import Dataset, DataLoader


class SimInputDataset(Dataset):
    """simulated streamflow input"""

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.data_flow = data_source.prepare_flow_data()
        self.transform = transform

    def __getitem__(self, index):
        return self.data_flow[index]

    def __len__(self):
        return len(self.data_flow)
