import os.path
from abc import ABC
from typing import Union

import numpy as np
import xarray as xr


class DataSourceBase(ABC):
    def __init__(self, data_path):
        self.data_source_dir = data_path
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

    def get_name(self):
        raise NotImplementedError

    def set_data_source_describe(self):
        raise NotImplementedError

    def download_data_source(self):
        raise NotImplementedError

    def read_object_ids(self, object_params=None) -> np.array:
        raise NotImplementedError

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        raise NotImplementedError

    def read_relevant_cols(
        self, object_ids=None, t_range_list: list = None, relevant_cols=None, **kwargs
    ) -> Union[np.array, xr.Dataset, list]:
        """3d data (site_num * time_length * var_num), time-series data"""
        raise NotImplementedError

    def read_constant_cols(
        self, object_ids=None, constant_cols=None, **kwargs
    ) -> np.array:
        """2d data (site_num * var_num), non-time-series data"""
        raise NotImplementedError

    def read_other_cols(
        self, object_ids=None, other_cols: dict = None, **kwargs
    ) -> dict:
        """some data which cannot be easily treated as constant vars or time-series with same length as relevant vars
        CONVENTION: other_cols is a dict, where each item is also a dict with all params in it"""
        raise NotImplementedError

    def get_constant_cols(self) -> np.array:
        """the constant cols in this data_source"""
        raise NotImplementedError

    def get_relevant_cols(self) -> np.array:
        """the relevant cols in this data_source"""
        raise NotImplementedError

    def get_target_cols(self) -> np.array:
        """the target cols in this data_source"""
        raise NotImplementedError

    def get_other_cols(self) -> dict:
        """the other cols in this data_source"""
        raise NotImplementedError
