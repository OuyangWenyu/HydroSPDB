"""
Author: Wenyu Ouyang
Date: 2022-02-13 21:20:18
LastEditTime: 2023-04-06 17:12:08
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: /HydroMTL/hydromtl/data/loader/data_sets.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import logging
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from hydrospdb.data.source.data_base import DataSourceBase
from hydrospdb.data.loader.dataloader_utils import read_yxc
from hydrospdb.data.loader.data_scalers import ScalerHub, wrap_t_s_dict
from hydrospdb.utils import hydro_utils

LOGGER = logging.getLogger(__name__)


def interpolate_y(y_before, basins_id):
    """Use pandas dataframe to interpolate y_before and get interpolated y_after.

    Parameters
    ----------
    y_before : _type_
        _description_
    basins_id : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    y_df = pd.DataFrame(
        y_before.reshape(y_before.shape[0], y_before.shape[1]).T, columns=basins_id
    )
    y_df_intepolate = y_df.interpolate(
        method="linear", limit_direction="forward", axis=0
    )
    return y_df_intepolate.values.T.reshape(y_before.shape)


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTroch's Dataset"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        """
        Parameters
        ----------
        data_source
            object for reading source data
        data_params
            parameters for reading source data
        loader_type
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_source = data_source
        self.data_params = data_params
        if loader_type in {"train", "valid", "test"}:
            self.loader_type = loader_type
        else:
            raise ValueError("'loader_type' must be one of 'train', 'valid' or 'test' ")
        # load and preprocess data
        self._load_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho, :]
        y = self.y[basin, idx : idx + self.rho, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _load_data(self):
        train_mode = self.loader_type == "train"
        self.t_s_dict = wrap_t_s_dict(
            self.data_source, self.data_params, self.loader_type
        )
        data_flow, data_forcing, data_attr = read_yxc(
            self.data_source, self.data_params, self.loader_type
        )

        # we need unnormalized data for the physical model;
        # data_flow, data_forcing and data_attr will be referred, so they may be changed
        # the final output of physical model is unnormalized output;
        self.x_origin = np.copy(data_forcing)
        self.y_origin = np.copy(data_flow)
        self.c_origin = np.copy(data_attr)

        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            data_params=self.data_params,
            loader_type=self.loader_type,
            data_source=self.data_source,
        )

        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c

        x_rm_nan = self.data_params["relevant_rm_nan"]
        c_rm_nan = self.data_params["constant_rm_nan"]
        y_rm_nan = self.data_params["target_rm_nan"]
        if x_rm_nan is True:
            self.x[np.where(np.isnan(self.x))] = 0
            self.x_origin[np.where(np.isnan(self.x_origin))] = 0
        if c_rm_nan is True:
            self.c[np.where(np.isnan(self.c))] = 0
            self.c_origin[np.where(np.isnan(self.c_origin))] = 0
        if y_rm_nan and train_mode:
            # for streamflow, we now provide a interpolation way, but only for training
            basins_id = self.t_s_dict["sites_id"]
            self.y = interpolate_y(self.y, basins_id=basins_id)
            self.y_origin = interpolate_y(self.y_origin, basins_id=basins_id)

        self.train_mode = train_mode
        self.rho = self.data_params["forecast_history"]
        self.target_scaler = scaler_hub.target_scaler
        self.warmup_length = self.data_params["warmup_length"]
        self._create_lookup_table()

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        time_length = len(hydro_utils.t_range_days(self.t_s_dict["t_final_range"]))
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            # some dataloader load data with warmup period, so leave some periods for it
            # [warmup_len] -> time_start -> [rho]
            lookup.extend(
                (basin, f)
                for f in range(warmup_length, time_length)
                if f < time_length - rho + 1
            )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        super(BasinSingleFlowDataset, self).__init__(
            data_source, data_params, loader_type
        )

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y

    def __len__(self):
        return self.num_samples


class BasinFlowDataset(BaseDataset):
    """Dataset for input of LSTM"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        super(BasinFlowDataset, self).__init__(data_source, data_params, loader_type)

    def __getitem__(self, index):
        if self.train_mode:
            return super(BasinFlowDataset, self).__getitem__(index)
        # TODO: not CHECK warmup_length yet because we don't use warmup_length for pure DL models
        x = self.x[index, :, :]
        y = self.y[index, :, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[index, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def __len__(self):
        return self.num_samples if self.train_mode else len(self.t_s_dict["sites_id"])
