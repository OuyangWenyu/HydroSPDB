from abc import ABC
from typing import Union

from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from data.data_base import DatasetBase
from data.data_scalers import ScalerHub, wrap_t_s_dict
from data.cache.cache_base import DatasetCache
from utils.hydro_utils import select_subset, select_subset_batch_first


class HydroDlTsDataModel(ABC):
    """time series data model in hydrodl"""

    def __init__(self, data_source: DatasetBase):
        self.data_source = data_source

    def get_item(self, i_grid, i_t, rho, batch_first=True):
        """id and time_id are given; rho is the time_length of a time-sequence; choose the correspond data"""
        raise NotImplementedError


class XczDataModel(HydroDlTsDataModel):
    """x,c,z are all inputs, where z are some special inputs, such as FDC"""

    def __init__(self, data_source: DatasetBase, dataset_params: dict, loader_type: str):
        super().__init__(data_source)
        data_flow, data_forcing, data_attr, data_other = read_yxc(data_source, dataset_params, loader_type)
        # normalization
        scaler_hub = ScalerHub(np.expand_dims(data_flow, axis=2), data_forcing, data_attr, other_vars=data_other,
                               dataset_params=dataset_params, loader_type=loader_type, data_model=data_source)
        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        # different XczDataModel has different way to handle with z values, here only handle with one-key case
        one_value = list(scaler_hub.z.items())[0][1]
        if one_value.ndim == 3 and one_value.shape[-1] == 1:
            # if the 3rd dim is just 1, it must be expanded for normalization, and it will be used as kernel,
            # which we will use it as 2d var
            one_value = one_value.reshape(one_value.shape[0], one_value.shape[1])
        self.z = one_value
        self.target_scaler = scaler_hub.target_scaler

    def get_item(self, i_grid, i_t, rho, batch_first=True):
        if batch_first:
            x_train = select_subset_batch_first(self.x, i_grid, i_t, rho, c=self.c)
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
            z_train = select_subset_batch_first(self.z, i_grid, i_t=None, rho=None)
        else:
            x_train = select_subset(self.x, i_grid, i_t, rho, c=self.c)
            y_train = select_subset(self.y, i_grid, i_t, rho)
            z_train = select_subset(self.z, i_grid, i_t=None, rho=None)
        # y_train must be the final!!!!!!
        return x_train, z_train, y_train


class BasinFlowDataModel(HydroDlTsDataModel):
    def __init__(self, data_model: DatasetBase, dataset_params: dict, loader_type: str):
        super().__init__(data_model)
        data_flow, data_forcing, data_attr = read_yxc(data_model, dataset_params, loader_type)
        # normalization
        scaler_hub = ScalerHub(np.expand_dims(data_flow, axis=2), data_forcing, data_attr,
                               dataset_params=dataset_params, loader_type=loader_type, data_model=data_model)
        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        self.target_scaler = scaler_hub.target_scaler

    def get_item(self, i_grid, i_t, rho, batch_first=True):
        if batch_first:
            x_train = select_subset_batch_first(self.x, i_grid, i_t, rho, c=self.c)
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
        else:
            x_train = select_subset(self.x, i_grid, i_t, rho, c=self.c)
            y_train = select_subset(self.y, i_grid, i_t, rho)
        return x_train, y_train


class BasinFlowDataset(Dataset):
    """Dataset for input of LSTM"""

    def __init__(self, data_model: DatasetBase, dataset_params: dict, loader_type: str):
        if loader_type == "train":
            train_mode = True
        else:
            train_mode = False
        data_flow, data_forcing, data_attr = read_yxc(data_model, dataset_params, loader_type)

        # normalization
        scaler_hub = ScalerHub(np.expand_dims(data_flow, axis=2), data_forcing, data_attr,
                               dataset_params=dataset_params, loader_type=loader_type, data_model=data_model)

        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        self.train_mode = train_mode
        self.rho = dataset_params["forecast_history"]
        self.target_scaler = scaler_hub.target_scaler

    def __getitem__(self, index):
        ngrid, nt, nx = self.x.shape
        rho = self.rho
        if self.train_mode:
            i_grid = index // (nt - rho + 1)
            i_t = index % (nt - rho + 1)
            x = self.x[i_grid, i_t:i_t + rho, :]
            c = self.c[i_grid, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            y = self.y[i_grid, i_t:i_t + rho, :]
        else:
            x = self.x[index, :, :]
            c = self.c[index, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            y = self.y[index, :, :]
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def __len__(self):
        if self.train_mode:
            return self.x.shape[0] * (self.x.shape[1] - self.rho + 1)
        else:
            return self.x.shape[0]


class BasinSingleFlowDataset(Dataset):
    """one time length output for each grid in a batch"""

    def __init__(self, data_model: DatasetBase, dataset_params: dict, loader_type: str):
        data_flow, data_forcing, data_attr = read_yxc(data_model, dataset_params, loader_type)

        # normalization
        scaler_hub = ScalerHub(np.expand_dims(data_flow, axis=2), data_forcing, data_attr,
                               dataset_params=dataset_params, loader_type=loader_type, data_model=data_model)

        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        self.rho = dataset_params["forecast_history"]
        self.target_scaler = scaler_hub.target_scaler

    def __getitem__(self, index):
        ngrid, nt, nx = self.x.shape
        rho = self.rho
        i_grid = index // (nt - rho + 1)
        i_t = index % (nt - rho + 1)
        x = self.x[i_grid, i_t:i_t + rho, :]
        c = self.c[i_grid, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        y = self.y[i_grid, i_t + rho - 1, :]
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0] * (self.x.shape[1] - self.rho + 1)


class TestDataModel(object):
    """Data model for test (denormalization)"""

    def __init__(self, test_data: Union[BasinFlowDataset, BasinFlowDataModel, XczDataModel]):
        """test_data is the data loader when initializing TimeSeriesModel"""
        self.test_data = test_data
        self.target_scaler = test_data.target_scaler

    def inverse_scale(self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]) -> np.array:
        """denormalization of data
        :param result_data: The data you want to unscale can handle multiple data types.
        :type result_data: Union[torch.Tensor, pd.Series, np.ndarray]
        :return: Returns the unscaled data as np.array.
        :rtype: np.array
        """
        if isinstance(result_data, pd.Series) or isinstance(result_data, pd.DataFrame):
            result_data_np = result_data.values
        elif isinstance(result_data, torch.Tensor):
            if len(result_data.shape) > 2:
                result_data = result_data.permute(2, 0, 1).reshape(result_data.shape[2], -1)
                result_data = result_data.permute(1, 0)
            result_data_np = result_data.numpy()
        elif isinstance(result_data, np.ndarray):
            result_data_np = result_data
        else:
            raise TypeError("No such data type for denormalization!")
        np_target_denorm = self.target_scaler.inverse_transform(result_data_np)
        return np_target_denorm

    def load_test_data(self):
        x = self.test_data.x
        c = self.test_data.c
        y = self.test_data.y
        if hasattr(self.test_data, 'z'):
            z = self.test_data.z
            # y must be the final!!!
            return x, c, z, y
        return x, c, y


def read_yxc(data_model: DatasetBase, dataset_params: dict, loader_type: str):
    t_s_dict = wrap_t_s_dict(data_model, dataset_params, loader_type)
    basins_id = t_s_dict["sites_id"]
    t_range_list = t_s_dict["t_final_range"]
    target_cols = dataset_params["target_cols"]
    relevant_cols = dataset_params["relevant_cols"]
    constant_cols = dataset_params["constant_cols"]
    cache_read = dataset_params["cache_read"]
    if "other_cols" in dataset_params.keys():
        other_cols = dataset_params["other_cols"]
    else:
        other_cols = None
    if cache_read:
        # Don't wanna the cache impact the implemention of dataset_models' read_xxx functions
        # Hence, here we follow "Convention over configuration", and set the cache files' name in DatasetCache
        dataset_cache = DatasetCache(dataset_params["cache_path"], loader_type, data_model)
        caches = dataset_cache.load_dataset()
        data_dict = caches[3]
        # judge if the configs are correct
        assert basins_id == data_dict[dataset_cache.key_sites]
        assert t_range_list == data_dict[dataset_cache.key_t_range]
        assert target_cols == data_dict[dataset_cache.key_target_cols]
        assert relevant_cols == data_dict[dataset_cache.key_relevant_cols]
        assert constant_cols == data_dict[dataset_cache.key_constant_cols]
        if other_cols is not None:
            for key, value in other_cols.items():
                assert value == data_dict[dataset_cache.key_other_cols][key]
            return caches[0], caches[1], caches[2], caches[4]
        return caches[0], caches[1], caches[2]
    if "forcing_type" in dataset_params.keys():
        forcing_type = dataset_params["forcing_type"]
    else:
        forcing_type = None
    # read streamflow
    data_flow = data_model.read_target_cols(basins_id, t_range_list, target_cols)
    # read forcing
    data_forcing = data_model.read_relevant_cols(basins_id, t_range_list, relevant_cols, forcing_type=forcing_type)
    # read attributes
    data_attr = data_model.read_constant_cols(basins_id, constant_cols)
    if other_cols is not None:
        # read other data
        data_other = data_model.read_other_cols(basins_id, other_cols=other_cols)
        return data_flow, data_forcing, data_attr, data_other
    return data_flow, data_forcing, data_attr
