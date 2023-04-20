"""
Author: Wenyu Ouyang
Date: 2021-12-17 18:02:27
LastEditTime: 2023-04-20 22:59:40
LastEditors: Wenyu Ouyang
Description: Class and functions for data cache
FilePath: /HydroSPDB/hydrospdb/data/cache/cache_base.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os

from hydrospdb.data.source.data_base import DataSourceBase
from hydrospdb.utils.hydro_utils import (
    unserialize_json,
    unserialize_numpy,
    serialize_numpy,
    serialize_json,
    serialize_pickle,
    unserialize_pickle,
)


class DataSourceCache(object):
    """The parent data-cache class: in charge of load data_source"""

    def __init__(self, save_path, mode, data_source: DataSourceBase):
        """

        Parameters
        ----------
        save_path
            where we save data cache
        mode
            train, valid or test
        data_source
            instance of data source class
        """
        self.save_path = save_path
        self.mode = mode
        self.flow_file_name = "flow.npy"
        self.forcing_file_name = "forcing.npy"
        self.attr_file_name = "attr.npy"
        self.data_dict_file_name = "data_dict.json"
        self.other_data_file_name = "other_data.txt"
        self.key_sites = "sites_id"
        self.key_t_range = "t_final_range"
        self.key_target_cols = "target_cols"
        self.key_relevant_cols = "relevant_cols"
        self.key_constant_cols = "constant_cols"
        self.key_other_cols = "other_cols"
        self.data_source = data_source
        self.all_attr_types = self.data_source.get_constant_cols()
        self.all_forcing_types = self.data_source.get_relevant_cols()
        self.all_flow_types = self.data_source.get_target_cols()
        self.all_other_types = self.data_source.get_other_cols()

    def load_data_source(self):
        save_dir = self.save_path
        mode = self.mode
        assert mode in ["train", "valid", "test"]
        gages_dict_file = os.path.join(save_dir, f"{mode}_{self.data_dict_file_name}")
        flow_npy_file = os.path.join(save_dir, f"{mode}_{self.flow_file_name}")
        forcing_npy_file = os.path.join(save_dir, f"{mode}_{self.forcing_file_name}")
        attr_npy_file = os.path.join(save_dir, f"{mode}_{self.attr_file_name}")
        other_data_file = os.path.join(save_dir, f"{mode}_{self.other_data_file_name}")
        data_dict = unserialize_json(gages_dict_file)
        data_flow = unserialize_numpy(flow_npy_file)
        data_forcing = unserialize_numpy(forcing_npy_file)
        data_attr = unserialize_numpy(attr_npy_file)
        if os.path.isfile(other_data_file):
            other_data = unserialize_pickle(other_data_file)
            return data_flow, data_forcing, data_attr, data_dict, other_data
        else:
            return data_flow, data_forcing, data_attr, data_dict

    def save_data_source(
        self, data_flow, data_forcing, data_attr, cols_dict: {}, other_data=None
    ):
        """mode in ["train","valid","test"]"""
        mode = self.mode
        save_dir = self.save_path
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        flow_file = os.path.join(save_dir, mode + "_" + self.flow_file_name)
        forcing_file = os.path.join(save_dir, mode + "_" + self.forcing_file_name)
        attr_file = os.path.join(save_dir, mode + "_" + self.attr_file_name)
        gages_dict_file = os.path.join(save_dir, mode + "_" + self.data_dict_file_name)
        other_data_file = os.path.join(save_dir, mode + "_" + self.other_data_file_name)
        serialize_numpy(data_flow, flow_file)
        serialize_numpy(data_forcing, forcing_file)
        serialize_numpy(data_attr, attr_file)
        serialize_json(cols_dict, gages_dict_file)
        if other_data is not None:
            serialize_pickle(other_data, other_data_file)

    def read_save_data_source(
        self,
        basins_id: list,
        t_range_list: list,
        target_cols: list,
        forcing_cols: list,
        attr_cols: list,
        forcing_type="daymet",
        other_cols: dict = None,
    ):
        """
        Read data from data source and save them

        Parameters
        ----------
        basins_id
            basins' ids
        t_range_list
            the time range, such as ["1990-01-01","2000-01-01"]
        target_cols
            target variables
        forcing_cols
            forcing variables
        attr_cols
            attribute variables
        forcing_type
            forcing type
        other_cols
            other variables

        Returns
        -------
        None
        """
        data_source = self.data_source
        if not set(target_cols).issubset(set(self.all_flow_types)):
            raise NotImplementedError(
                "We don't have such target variables, please check your setting!"
            )
        if not set(forcing_cols).issubset(set(self.all_forcing_types)):
            raise NotImplementedError(
                "We don't have such forcing variables, please check your setting!"
            )
        if not set(attr_cols).issubset(set(self.all_attr_types)):
            raise NotImplementedError(
                "We don't have such attribute variables, please check your setting!"
            )
        cols_dict = {
            self.key_sites: basins_id,
            self.key_t_range: t_range_list,
            self.key_target_cols: target_cols,
            self.key_relevant_cols: forcing_cols,
            self.key_constant_cols: attr_cols,
        }

        data_other = None
        if other_cols is not None:
            assert set(other_cols.keys()).issubset(set(self.all_other_types.keys()))
            save_other_cols = {self.key_other_cols: other_cols}
            cols_dict = {**cols_dict, **save_other_cols}
            data_other = data_source.read_other_cols(basins_id, other_cols)

        # read target data
        data_target = data_source.read_target_cols(basins_id, t_range_list, target_cols)
        # read forcing
        data_forcing = data_source.read_relevant_cols(
            basins_id, t_range_list, forcing_cols, forcing_type=forcing_type
        )
        # read attributes
        data_attr = data_source.read_constant_cols(basins_id, attr_cols)
        self.save_data_source(
            data_target, data_forcing, data_attr, cols_dict, other_data=data_other
        )
