"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:51:20
LastEditors: Wenyu Ouyang
Description: Self-made Data sets and loaders for DL models; references to https://github.com/mhpi/hydroDL
FilePath: /HydroSPDB/hydrospdb/data/loader/data_loaders.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from abc import ABC
from functools import wraps
from typing import Union

import numpy as np
import torch
import pandas as pd

from hydrospdb.data.cache.cache_base import DataSourceCache
from hydrospdb.data.loader.dataloader_utils import read_yxc, check_data_loader
from hydrospdb.data.source.data_base import DataSourceBase
from hydrospdb.data.loader.data_scalers import ScalerHub, wrap_t_s_dict
from hydrospdb.utils.hydro_utils import (
    select_subset,
    select_subset_batch_first,
    check_np_array_nan,
)


class HydroDlTsDataModel(ABC):
    """time series data model in hydrodl of the MHPI group -- https://github.com/mhpi/hydroDL"""

    def __init__(self, data_source: DataSourceBase):
        """
        Parameters
        ----------
        data_source
            object for reading from data source
        """
        self.data_source = data_source

    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor]:
        """
        Read data from data source and compose a mini-batch

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]
        Returns
        -------
        tuple[torch.Tensor]
            a mini-batch tensor
        """
        raise NotImplementedError


class XczDataModel(HydroDlTsDataModel):
    """x,c,z are all inputs, where z are some special inputs, such as FDC"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        """
        Read source data (x,c,z are inputs; y is output) and normalize them

        Parameters
        ----------
        data_source
            object for reading source data
        data_params
            parameters for reading source data
        loader_type
            train, vaild or test
        """
        super().__init__(data_source)
        data_flow, data_forcing, data_attr, data_other = read_yxc(
            data_source, data_params, loader_type
        )
        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            other_vars=data_other,
            data_params=data_params,
            loader_type=loader_type,
            data_source=data_source,
        )
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

    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one mini-batch

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]

        Returns
        -------
        tuple
            a mini-batch data; x_train, z_train, y_train
        """
        if batch_first:
            x_train = select_subset_batch_first(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            # no need for y_train to set warmup period as loss is not calculated for warmup periods
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
            z_train = select_subset_batch_first(self.z, i_grid, i_t=None, rho=None)
        else:
            x_train = select_subset(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            y_train = select_subset(self.y, i_grid, i_t, rho)
            z_train = select_subset(self.z, i_grid, i_t=None, rho=None)
        # y_train must be the final!!!!!!
        return x_train, z_train, y_train


class BasinFlowDataModel(HydroDlTsDataModel):
    """Basic basin's rainfall-runoff mini-batch data model"""

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
        super().__init__(data_source)
        data_flow, data_forcing, data_attr = read_yxc(
            data_source, data_params, loader_type
        )
        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            data_params=data_params,
            loader_type=loader_type,
            data_source=data_source,
        )
        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        self.target_scaler = scaler_hub.target_scaler

    @check_data_loader
    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one mini-batch tensor from np.array data samples

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]

        Returns
        -------
        tuple
            a mini-batch data; x_train (x concat with c), y_train
        """
        if batch_first:
            x_train = select_subset_batch_first(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            # y_train don't need warmup period since loss is only calculated for formal periods
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
        else:
            x_train = select_subset(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            y_train = select_subset(self.y, i_grid, i_t, rho)
        return x_train, y_train
