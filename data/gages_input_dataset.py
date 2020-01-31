"""for stacked lstm"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data import GagesSource, DataModel
from explore import trans_norm
from utils.dataset_format import subset_of_dict


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
        return data


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


class GagesInvDataModel(object):
    """DataModel for inv model"""

    def __init__(self, data_model1, data_model2):
        self.model_dict = data_model2.data_source.data_config.model_dict
        all_data = self.prepare_input(data_model1, data_model2)
        input_keys = ['xh', 'ch', 'qh', 'xt', 'ct']
        output_keys = ['qt']
        self.data_input = subset_of_dict(all_data, input_keys)
        self.data_target = subset_of_dict(all_data, output_keys)

    def prepare_input(self, data_model1, data_model2):
        """prepare input for lstm-inv"""
        print("prepare input")
        model_dict1 = data_model1.data_source.data_config.model_dict
        xh, qh, ch = data_model1.load_data(model_dict1)
        model_dict2 = data_model2.data_source.data_config.model_dict
        xt, qt, ct = data_model2.load_data(model_dict2)
        return {'xh': xh, 'ch': ch, 'qh': qh, 'xt': xt, 'ct': ct, 'qt': qt}

    def load_data(self):
        return self.data_input, self.data_target
