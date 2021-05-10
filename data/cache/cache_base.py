import os
import numpy as np

from data.data_base import DatasetBase
from utils.hydro_utils import unserialize_json, unserialize_numpy, serialize_numpy, serialize_json, serialize_pickle, \
    unserialize_pickle


class DatasetCache(object):
    """The parent data-cache class: in charge of load dataset"""

    def __init__(self, save_path, mode, dataset: DatasetBase):
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
        self.dataset = dataset
        self.all_attr_types = self.dataset.get_constant_cols()
        self.all_forcing_types = self.dataset.get_relevant_cols()
        self.all_flow_types = self.dataset.get_target_cols()
        self.all_other_types = self.dataset.get_other_cols()

    def load_dataset(self):
        save_dir = self.save_path
        mode = self.mode
        assert mode in ["train", "valid", "test"]
        gages_dict_file = os.path.join(save_dir, mode + "_" + self.data_dict_file_name)
        flow_npy_file = os.path.join(save_dir, mode + "_" + self.flow_file_name)
        forcing_npy_file = os.path.join(save_dir, mode + "_" + self.forcing_file_name)
        attr_npy_file = os.path.join(save_dir, mode + "_" + self.attr_file_name)
        other_data_file = os.path.join(save_dir, mode + "_" + self.other_data_file_name)
        data_dict = unserialize_json(gages_dict_file)
        data_flow = unserialize_numpy(flow_npy_file)
        data_forcing = unserialize_numpy(forcing_npy_file)
        data_attr = unserialize_numpy(attr_npy_file)
        if os.path.isfile(other_data_file):
            other_data = unserialize_pickle(other_data_file)
            return data_flow, data_forcing, data_attr, data_dict, other_data
        else:
            return data_flow, data_forcing, data_attr, data_dict

    def save_dataset(self, data_flow, data_forcing, data_attr, cols_dict: {}, other_data=None):
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

    def read_save_dataset(self, basins_id: list, t_range_list: list, flow_cols: list, forcing_cols: list,
                          attr_cols: list, forcing_type="daymet", other_cols: dict = None):
        dataset = self.dataset
        assert set(flow_cols).issubset(set(self.all_flow_types))
        assert set(forcing_cols).issubset(set(self.all_forcing_types))
        assert set(attr_cols).issubset(set(self.all_attr_types))
        cols_dict = {self.key_sites: basins_id, self.key_t_range: t_range_list, self.key_target_cols: flow_cols,
                     self.key_relevant_cols: forcing_cols, self.key_constant_cols: attr_cols}

        data_other = None
        if other_cols is not None:
            assert set(other_cols.keys()).issubset(set(self.all_other_types.keys()))
            save_other_cols = {self.key_other_cols: other_cols}
            cols_dict = {**cols_dict, **save_other_cols}
            data_other = dataset.read_other_cols(basins_id, other_cols)

        # read streamflow
        data_flow = dataset.read_target_cols(basins_id, t_range_list, flow_cols)
        # read forcing
        data_forcing = dataset.read_relevant_cols(basins_id, t_range_list, forcing_cols, forcing_type=forcing_type)
        # read attributes
        data_attr = dataset.read_constant_cols(basins_id, attr_cols)
        self.save_dataset(data_flow, data_forcing, data_attr, cols_dict, other_data=data_other)
