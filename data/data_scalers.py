import json
import os
import pickle as pkl
from collections import OrderedDict

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

from data.data_base import DatasetBase
from explore.stat import cal_stat_basin_norm, cal_stat_gamma, cal_stat, trans_norm

scaler_dict = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler
}


class ScalerHub(object):
    def __init__(self, target_vars: np.array, relevant_vars: np.array, constant_vars: np.array = None,
                 other_vars: dict = None, dataset_params: dict = None, loader_type: str = None, **kwargs):
        assert target_vars.ndim == 3
        assert relevant_vars.ndim == 3
        assert constant_vars.ndim == 2
        assert dataset_params is not None
        assert loader_type is not None
        scaler_type = dataset_params["scaler"]
        if scaler_type == "DapengScaler":
            # TODO: need better framework, such as providing a algorithm in scaler and leaving the conditions outside
            assert "data_model" in list(kwargs.keys())
            if other_vars is not None:
                scaler = DapengScaler(target_vars, relevant_vars, constant_vars, dataset_params, loader_type,
                                      kwargs["data_model"], other_vars=other_vars)
                x, y, c, z = scaler.load_data()
            else:
                scaler = DapengScaler(target_vars, relevant_vars, constant_vars, dataset_params, loader_type,
                                      kwargs["data_model"])
                x, y, c = scaler.load_data()
            self.target_scaler = scaler
        else:
            if other_vars is not None:
                raise Exception("not Implemented yet!!")
            all_vars = [target_vars, relevant_vars, constant_vars]
            norm_keys = ["target_vars", "relevant_vars", "constant_vars"]
            norm_dict = {}
            for i in range(len(all_vars)):
                data_tmp = all_vars[i]
                scaler = scaler_dict[scaler_type]()
                if data_tmp.ndim == 3:
                    num_instances, num_time_steps, num_features = data_tmp.shape
                    data_tmp = data_tmp.reshape(-1, num_features)
                    save_file = os.path.join(dataset_params["test_path"], norm_keys[i] + "_scaler.pkl")
                    if loader_type == "train":
                        data_norm = scaler.fit_transform(data_tmp)
                        # Save scaler in test_path for valid/test
                        with open(save_file, "wb") as outfile:
                            pkl.dump(scaler, outfile)
                    else:
                        assert os.path.isfile(save_file)
                        with open(save_file, "rb") as infile:
                            scaler = pkl.load(infile)
                            data_norm = scaler.transform(data_tmp)
                    data_norm = data_norm.reshape(num_instances, num_time_steps, num_features)
                else:
                    # TODO: check this "else" block
                    data_norm = scaler.fit_transform(data_tmp)
                norm_dict[norm_keys[i]] = data_norm
                if i == 0:
                    self.target_scaler = scaler
            x = norm_dict["relevant_vars"]
            y = norm_dict["target_vars"]
            c = norm_dict["constant_vars"]

        print("Finish Normalization\n")
        self.x = x
        self.y = y
        self.c = c
        if other_vars is not None:
            self.z = z


class DapengScaler(object):
    """the normalization and denormalization methods from Dapeng's 1st WRR paper.
    Some use StandardScaler, and some use special norm methods"""

    def __init__(self, target_vars: np.array, relevant_vars: np.array, constant_vars: np.array, dataset_params: dict,
                 loader_type: str, data_model: DatasetBase, other_vars: dict = None):
        # TODO: now only support "CAMELS", "GAGES", and "GAGES_PRO" dataset
        assert data_model.get_name() in ["CAMELS", "GAGES", "GAGES_PRO"]
        self.data_flow = target_vars
        self.data_forcing = relevant_vars
        self.data_attr = constant_vars
        self.data_source = data_model
        self.dataset_params = dataset_params
        self.t_s_dict = wrap_t_s_dict(data_model, dataset_params, loader_type)
        self.data_other = other_vars
        # save stat_dict of training period in test_path for valid/test
        stat_file = os.path.join(dataset_params["test_path"], 'dapengscaler_stat.json')
        if loader_type == "train":
            self.stat_dict = self.cal_stat_all()
            with open(stat_file, 'w') as fp:
                json.dump(self.stat_dict, fp)
        else:
            assert os.path.isfile(stat_file)
            with open(stat_file, 'r') as fp:
                self.stat_dict = json.load(fp)

    def inverse_transform(self, target_values):
        """denormalization for target variables"""
        stat_dict = self.stat_dict
        target_cols = self.dataset_params["target_cols"]
        # TODO: only support one output now
        pred = _trans_norm(target_values, target_cols[0], stat_dict, to_norm=False)
        basin_area = self.data_source.read_basin_area(self.t_s_dict["sites_id"])
        mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
        pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
        return pred

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes."""
        # streamflow
        flow = self.data_flow.reshape(self.data_flow.shape[0], -1)
        assert flow.ndim == 2
        stat_dict = dict()
        basin_area = self.data_source.read_basin_area(self.t_s_dict["sites_id"])
        mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
        target_cols = self.dataset_params["target_cols"]
        # TODO: now only support one-type output
        stat_dict[target_cols[0]] = cal_stat_basin_norm(flow, basin_area, mean_prep)

        # forcing
        # TODO: now hard code for precipitation_str: prcp in daymet, pr in gridmet
        precipitation_str = ["prcp", "pr"]
        forcing_lst = self.dataset_params["relevant_cols"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var in precipitation_str:
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.dataset_params["constant_cols"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])

        # other vars
        if self.data_other is not None:
            other_data = self.data_other
            for key, value in other_data.items():
                stat_dict[key] = cal_stat(value)

        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        data = self.data_flow
        basin_area = self.data_source.read_basin_area(self.t_s_dict["sites_id"])
        mean_prep = self.data_source.read_mean_prep(self.t_s_dict["sites_id"])
        data = _basin_norm(data, basin_area, mean_prep, to_norm=True)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)
        target_cols = self.dataset_params["target_cols"]
        # TODO: only support one output now
        data = _trans_norm(data, target_cols[0], stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        var_lst = self.dataset_params["relevant_cols"]
        data = self.data_forcing
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, rm_nan=True, to_norm=True):
        """attr data and normalization"""
        stat_dict = self.stat_dict
        var_lst = self.dataset_params["constant_cols"]
        data = self.data_attr
        data = trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_other(self, rm_nan=True, to_norm=True) -> dict:
        stat_dict = self.stat_dict
        var_lst = list(self.dataset_params["other_cols"].keys())
        data = self.data_other
        trans_data = {}
        for var in var_lst:
            data_tmp = data[var]
            if data_tmp.ndim == 2:
                data_tmp = np.expand_dims(data_tmp, axis=2)
            data_tmp = trans_norm(data_tmp, var, stat_dict, to_norm=to_norm)
            if rm_nan is True:
                data_tmp[np.where(np.isnan(data_tmp))] = 0
            trans_data[var] = data_tmp
        return trans_data

    def load_data(self):
        """read data and make normalization as input for the model
        :parameter
            model_dict: model params
        :return  np.array
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        x = self.get_data_ts()
        y = self.get_data_obs()
        c = self.get_data_const()
        if self.data_other is not None:
            z = self.get_data_other()
            return x, y, c, z
        return x, y, c


def _trans_norm(x, var_lst, stat_dict, *, to_norm):
    """normalization; when to_norm=False, anti-normalization
    :parameter
        xï¼šad or 3d
            2d: 1st dim is gauge  2nd dim is var type
            3d: 1st dim is gauge 2nd dim is time 3rd dim is var type
    """
    # TODO: now hard code
    need_log_str = ["prcp", "pr", "usgsFlow"]
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var in need_log_str:
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in need_log_str:
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in need_log_str:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in need_log_str:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def _basin_norm(x, basin_area, mean_prep, to_norm):
    """for regional training, gageid should be numpyarray"""
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))
    tempprep = np.tile(mean_prep, (1, x.shape[1]))
    if to_norm is True:
        flow = (x * 0.0283168 * 3600 * 24) / (
                (temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # (m^3/day)/(m^3/day)
    else:
        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3))) / (0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def wrap_t_s_dict(data_model: DatasetBase, dataset_params: dict, loader_type: str):
    basins_id = dataset_params["object_ids"]
    if type(basins_id) is str and basins_id == "ALL":
        basins_id = data_model.read_object_ids(dataset_params["object_ids"]).tolist()
    assert (all(x < y for x, y in zip(basins_id, basins_id[1:])))
    if "t_range_" + loader_type in dataset_params:
        t_range_list = dataset_params["t_range_" + loader_type]
    else:
        raise Exception(
            "Error! The mode " + str(loader_type) + " was not found in the dataset params dict. Please add it.")
    t_s_dict = OrderedDict(sites_id=basins_id, t_final_range=t_range_list)
    return t_s_dict
