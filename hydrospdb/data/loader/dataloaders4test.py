"""
Author: Wenyu Ouyang
Date: 2023-04-20 17:21:42
LastEditTime: 2023-04-20 17:53:10
LastEditors: Wenyu Ouyang
Description: 
FilePath: /HydroSPDB/hydrospdb/data/loader/dataloaders4test.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from typing import Union

import numpy as np
import pandas as pd
import torch

from hydrospdb.data.loader.data_loaders import (
    BasinFlowDataModel,
    XczDataModel,
)
from hydrospdb.data.loader.data_sets import BasinFlowDataset


class TestDataModel(object):
    """Data model for test or validation (denormalization)"""

    def __init__(
        self, test_data: Union[BasinFlowDataModel, XczDataModel, BasinFlowDataset]
    ):
        """test_data is the data loader when initializing TimeSeriesModel"""
        self.test_data = test_data
        self.target_scaler = test_data.target_scaler

    def inverse_scale(
        self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    ) -> np.array:
        """
        denormalization of data

        Parameters
        ----------
        result_data
            The data you want to unscale can handle multiple data types.

        Returns
        -------
        np.array
            Returns the unscaled data as np.array.
        """
        if isinstance(result_data, (pd.Series, pd.DataFrame)):
            result_data_np = result_data.values
        elif isinstance(result_data, torch.Tensor):
            # TODO: not tested, useful when validating
            if len(result_data.shape) > 2:
                result_data = result_data.permute(2, 0, 1).reshape(
                    result_data.shape[2], -1
                )
                result_data = result_data.permute(1, 0)
            result_data_np = result_data.numpy()
        elif isinstance(result_data, np.ndarray):
            result_data_np = result_data
        else:
            raise TypeError("No such data type for denormalization!")
        np_target_denorm = self.target_scaler.inverse_transform(result_data_np)
        return np_target_denorm

    def load_test_data(self):
        # don't test for warmup period yet as no pbm use it now
        x = self.test_data.x
        c = self.test_data.c
        y = self.test_data.y
        if hasattr(self.test_data, "z"):
            z = self.test_data.z
            # y must be the final!!!
            return x, c, z, y
        return x, c, y

