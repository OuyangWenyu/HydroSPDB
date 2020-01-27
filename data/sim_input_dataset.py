import torch
from torch.utils.data import Dataset, DataLoader


class SimInputDataset(Dataset):
    """simulated streamflow input"""

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.data_flow = data_source.prepare_flow_data()
        self.data_target = data_source.read_outflow()
        self.transform = transform

    def __getitem__(self, index):
        # TODO: data is 3d, however we need get the final dim for getting input data
        # miniBatch[1] is length of x
        x = self.data_flow[index]
        y = self.data_target[index]
        return x, y

    def __len__(self):
        return len(self.data_flow)
