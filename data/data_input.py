"""一个处理数据的模板方法"""
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from data import GagesSource
from data.data_config import update_config_item
from explore import *
from utils import serialize_pickle, serialize_json, serialize_numpy, unserialize_pickle, unserialize_json, \
    unserialize_numpy, hydro_time
from utils.hydro_math import copy_attr_array_in2d, concat_two_3darray, random_index, select_subset, \
    select_subset_batch_first


def save_result(save_dir, epoch, pred, obs):
    """save the pred value of testing period and obs value"""
    flow_pred_file = os.path.join(save_dir, 'epoch' + str(epoch) + 'flow_pred')
    flow_obs_file = os.path.join(save_dir, 'epoch' + str(epoch) + 'flow_obs')
    serialize_numpy(pred, flow_pred_file)
    serialize_numpy(obs, flow_obs_file)


def load_result(save_dir, epoch):
    """load the pred value of testing period and obs value"""
    flow_pred_file = os.path.join(save_dir, 'epoch' + str(epoch) + 'flow_pred.npy')
    flow_obs_file = os.path.join(save_dir, 'epoch' + str(epoch) + 'flow_obs.npy')
    pred = unserialize_numpy(flow_pred_file)
    obs = unserialize_numpy(flow_obs_file)
    return pred, obs


def save_datamodel(data_model, num_str=None, **kwargs):
    if num_str:
        dir_temp = os.path.join(data_model.data_source.data_config.data_path["Temp"], num_str)
    else:
        dir_temp = data_model.data_source.data_config.data_path["Temp"]
    if not os.path.isdir(dir_temp):
        os.makedirs(dir_temp)
    data_source_file = os.path.join(dir_temp, kwargs['data_source_file_name'])
    stat_file = os.path.join(dir_temp, kwargs['stat_file_name'])
    flow_file = os.path.join(dir_temp, kwargs['flow_file_name'])
    forcing_file = os.path.join(dir_temp, kwargs['forcing_file_name'])
    attr_file = os.path.join(dir_temp, kwargs['attr_file_name'])
    f_dict_file = os.path.join(dir_temp, kwargs['f_dict_file_name'])
    var_dict_file = os.path.join(dir_temp, kwargs['var_dict_file_name'])
    t_s_dict_file = os.path.join(dir_temp, kwargs['t_s_dict_file_name'])
    serialize_pickle(data_model.data_source, data_source_file)
    serialize_json(data_model.stat_dict, stat_file)
    serialize_numpy(data_model.data_flow, flow_file)
    serialize_numpy(data_model.data_forcing, forcing_file)
    serialize_numpy(data_model.data_attr, attr_file)
    # dictFactorize.json is the explanation of value of categorical variables
    serialize_json(data_model.f_dict, f_dict_file)
    serialize_json(data_model.var_dict, var_dict_file)
    serialize_json(data_model.t_s_dict, t_s_dict_file)


class DataModel(object):
    """data formatter， utilizing function of DataSource object to read data and transform"""

    def __init__(self, data_source, *args):
        """:parameter data_source: DataSource object"""
        self.data_source = data_source
        # call "read_xxx" functions of DataSource to read forcing，flow，attributes data
        if len(args) == 0:
            # read flow
            data_flow = data_source.read_usgs()
            usgs_id = data_source.all_configs["flow_screen_gage_id"]
            data_flow, usgs_id, t_range_list = data_source.usgs_screen_streamflow(data_flow, usgs_ids=usgs_id)
            assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
            self.data_flow = data_flow
            # read forcing
            data_forcing = data_source.read_forcing(usgs_id, t_range_list)
            self.data_forcing = data_forcing
            # read attributes
            attr_lst = data_source.all_configs.get("attr_chosen")
            data_attr, var_dict, f_dict = data_source.read_attr(usgs_id, attr_lst)
            self.data_attr = data_attr
            self.var_dict = var_dict
            self.f_dict = f_dict
            # wrap gauges and time range to a dict.
            # To guarantee the time range is a left-closed and right-open interval, t_range_list[-1] + 1 day
            self.t_s_dict = OrderedDict(sites_id=usgs_id,
                                        t_final_range=[np.datetime_as_string(t_range_list[0], unit='D'),
                                                       np.datetime_as_string(
                                                           t_range_list[-1] + np.timedelta64(1, 'D'), unit='D')])
            # statistics
            stat_dict = self.cal_stat_all()
            self.stat_dict = stat_dict


        else:
            self.data_flow = args[0]
            self.data_forcing = args[1]
            self.data_attr = args[2]
            self.var_dict = args[3]
            self.f_dict = args[4]
            self.stat_dict = args[5]
            self.t_s_dict = args[6]

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes"""
        # streamflow
        flow = self.data_flow
        stat_dict = dict()
        stat_dict['usgsFlow'] = cal_stat(flow)

        # forcing
        forcing_lst = self.data_source.all_configs["forcing_chosen"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_source.all_configs["attr_chosen"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        """径流数据读取及归一化处理，会处理成三维，最后一维长度为1，表示径流变量"""
        stat_dict = self.stat_dict
        data = self.data_flow
        # 为了调用trans_norm函数进行归一化，这里先把径流变为三维数据
        data = np.expand_dims(data, axis=2)
        data = trans_norm(data, 'usgsFlow', stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True):
        """时间序列数据，主要是驱动数据读取 and choose data in the given time interval 及归一化处理"""
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, rm_nan=True, to_norm=True):
        """属性数据读取及归一化处理"""
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("attr_chosen")
        data = self.data_attr
        data = trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def load_data(self, model_dict):
        """读取数据为模型输入的形式，完成归一化运算
        :parameter
            model_dict: 载入数据需要模型相关参数
        :return  np.array
            x: 3-d  gages_num*time_num*var_num
            y: 3-d  gages_num*time_num*1
            c: 2-d  gages_num*var_num
        """
        # 如果读取到统计数据的json文件，则不需要再次计算。
        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        rm_nan_c = opt_data['rmNan'][0]
        to_norm_x = opt_data['doNorm'][0]
        to_norm_y = opt_data['doNorm'][1]
        to_norm_c = opt_data['doNorm'][0]
        x = self.get_data_ts(rm_nan=rm_nan_x, to_norm=to_norm_x)
        y = self.get_data_obs(rm_nan=rm_nan_y, to_norm=to_norm_y)
        c = self.get_data_const(rm_nan=rm_nan_c, to_norm=to_norm_c)
        if opt_data['daObs'] > 0:
            nday = opt_data['daObs']
            obs = self.get_data_obs(rm_nan=True)
            x = np.concatenate([x, obs], axis=2)
        return x, y, c

    @classmethod
    def data_models_of_train_test(cls, data_model, t_train, t_test):
        """split the data_model that will be used in LSTM according to train and test
        Notice: you can't know anything about test dataset before evaluating, so we should use the statistic value of
        training period for normalization in test period"""

        def select_by_time(data_flow_temp, data_forcing_temp, data_model_origin, t_temp, train_stat_dict=None):
            data_attr_temp = data_model_origin.data_attr[:, :]
            stat_dict_temp = {}
            t_s_dict_temp = {}
            source_data_temp = copy.deepcopy(data_model_origin.data_source)
            source_data_temp.t_range = t_temp
            f_dict_temp = data_model_origin.f_dict
            var_dict_temp = data_model_origin.var_dict
            data_model_temp = cls(source_data_temp, data_flow_temp, data_forcing_temp, data_attr_temp,
                                  var_dict_temp, f_dict_temp, stat_dict_temp, t_s_dict_temp)
            t_s_dict_temp['sites_id'] = data_model_origin.t_s_dict['sites_id']
            t_s_dict_temp['t_final_range'] = t_temp
            data_model_temp.t_s_dict = t_s_dict_temp
            if train_stat_dict is None:
                stat_dict_temp = data_model_temp.cal_stat_all()
            else:
                stat_dict_temp = train_stat_dict
            data_model_temp.stat_dict = stat_dict_temp
            return data_model_temp

        t_lst_train = hydro_time.t_range_days(t_train)
        t_train_final_index = t_lst_train.size
        data_flow_train = data_model.data_flow[:, :t_train_final_index]
        data_forcing_train = data_model.data_forcing[:, :t_train_final_index, :]
        data_model_train = select_by_time(data_flow_train, data_forcing_train, data_model, t_train)

        data_flow_test = data_model.data_flow[:, t_train_final_index:]
        data_forcing_test = data_model.data_forcing[:, t_train_final_index:, :]
        data_model_test = select_by_time(data_flow_test, data_forcing_test, data_model, t_test,
                                         data_model_train.stat_dict)
        return data_model_train, data_model_test

    @classmethod
    def load_datamodel(cls, dir_temp_orgin, num_str=None, **kwargs):
        if num_str:
            dir_temp = os.path.join(dir_temp_orgin, num_str)
        else:
            dir_temp = dir_temp_orgin
        data_source_file = os.path.join(dir_temp, kwargs['data_source_file_name'])
        stat_file = os.path.join(dir_temp, kwargs['stat_file_name'])
        flow_npy_file = os.path.join(dir_temp, kwargs['flow_file_name'])
        forcing_npy_file = os.path.join(dir_temp, kwargs['forcing_file_name'])
        attr_npy_file = os.path.join(dir_temp, kwargs['attr_file_name'])
        f_dict_file = os.path.join(dir_temp, kwargs['f_dict_file_name'])
        var_dict_file = os.path.join(dir_temp, kwargs['var_dict_file_name'])
        t_s_dict_file = os.path.join(dir_temp, kwargs['t_s_dict_file_name'])
        source_data = unserialize_pickle(data_source_file)
        # 存储data_model，因为data_model里的数据如果直接序列化会比较慢，所以各部分分别序列化，dict的直接序列化为json文件，数据的HDF5
        stat_dict = unserialize_json(stat_file)
        data_flow = unserialize_numpy(flow_npy_file)
        data_forcing = unserialize_numpy(forcing_npy_file)
        data_attr = unserialize_numpy(attr_npy_file)
        # dictFactorize.json is the explanation of value of categorical variables
        var_dict = unserialize_json(var_dict_file)
        f_dict = unserialize_json(f_dict_file)
        t_s_dict = unserialize_json(t_s_dict_file)
        data_model = cls(source_data, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                         t_s_dict)
        return data_model

    @classmethod
    def every_model(cls, model_data):
        data_models = []
        sites_id_all = model_data.t_s_dict["sites_id"]
        for i in range(len(sites_id_all)):
            data_model_temp = cls.which_data_model(model_data, i)
            data_models.append(data_model_temp)
        return data_models

    @classmethod
    def which_data_model(cls, model_data, i):
        sites_id_all = model_data.t_s_dict["sites_id"]
        data_flow = model_data.data_flow[i:i + 1, :]
        data_forcing = model_data.data_forcing[i:i + 1, :, :]
        data_attr = model_data.data_attr[i:i + 1, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(i))
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(i))
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        site_id_lst = [sites_id_all[i]]
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=site_id_lst)
        f_dict = model_data.f_dict
        var_dict = model_data.var_dict
        data_model_i = cls(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                           t_s_dict)
        t_s_dict['sites_id'] = site_id_lst
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        return data_model_i


class GagesModel(DataModel):
    def __init__(self, data_source, *args):
        super().__init__(data_source, *args)

    @classmethod
    def compact_data_model(cls, data_model_lst, data_source):
        print("compact models")
        data_model1 = data_model_lst[0]
        for i in range(1, len(data_model_lst)):
            data_model2 = data_model_lst[i]
            data_attr1 = data_model1.data_attr
            data_attr2 = data_model2.data_attr
            data_attr = np.vstack((data_attr1, data_attr2))
            data_flow1 = data_model1.data_flow
            data_flow2 = data_model2.data_flow
            # param of vstack is a tuple
            data_flow = np.vstack((data_flow1, data_flow2))
            data_forcing1 = data_model1.data_forcing
            data_forcing2 = data_model2.data_forcing
            data_forcing = np.vstack((data_forcing1, data_forcing2))
            f_dict = data_model1.f_dict
            var_dict = data_model1.var_dict
            stat_dict_temp = {}
            t_s_dict = {}
            s_dict = data_model1.t_s_dict['sites_id'] + data_model2.t_s_dict['sites_id']
            t_s_dict['sites_id'] = s_dict
            t_s_dict['t_final_range'] = data_model1.t_s_dict['t_final_range']
            data_model1 = cls(data_source, data_flow, data_forcing, data_attr,
                              var_dict, f_dict, stat_dict_temp, t_s_dict)
            data_model1.t_s_dict = t_s_dict

        stat_dict = data_model1.cal_stat_all()
        data_model1.stat_dict = stat_dict
        return data_model1

    @classmethod
    def update_data_model(cls, config_data, data_model_origin, t_range_update=None, data_attr_update=False,
                          train_stat_dict=None):
        t_s_dict = data_model_origin.t_s_dict
        data_flow = data_model_origin.data_flow
        data_forcing = data_model_origin.data_forcing
        data_attr = data_model_origin.data_attr
        var_dict = data_model_origin.var_dict
        f_dict = data_model_origin.f_dict
        stat_dict = data_model_origin.stat_dict
        t_range = t_s_dict["t_final_range"]
        new_source_data = GagesSource(config_data, t_range)
        if data_attr_update:
            attr_lst = new_source_data.all_configs.get("attr_chosen")
            data_attr, var_dict, f_dict = new_source_data.read_attr(t_s_dict["sites_id"], attr_lst)
        data_model = cls(new_source_data, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                         t_s_dict)
        if t_range_update is not None:
            t_range = t_range_update
            stat_dict_temp = {}
            t_s_dict_temp = {}
            start_index = int(
                (np.datetime64(t_range[0]) - np.datetime64(t_s_dict["t_final_range"][0])) / np.timedelta64(1, 'D'))
            assert start_index >= 0
            t_lst_temp = hydro_time.t_range_days(t_range)
            end_index = start_index + t_lst_temp.size
            data_flow = data_model.data_flow[:, start_index:end_index]
            data_forcing = data_model.data_forcing[:, start_index:end_index, :]

            data_model = cls(new_source_data, data_flow, data_forcing, data_attr,
                             var_dict, f_dict, stat_dict_temp, t_s_dict_temp)
            t_s_dict_temp['sites_id'] = data_model_origin.t_s_dict['sites_id']
            t_s_dict_temp['t_final_range'] = t_range
            data_model.t_s_dict = t_s_dict_temp
            data_model.data_source.t_range = t_range
        if not data_model.data_source.gage_dict["STAID"].tolist() == data_model.t_s_dict['sites_id']:
            gage_dict_new = dict()
            usgs_all_sites = data_model.data_source.gage_dict["STAID"]
            sites_chosen = np.zeros(usgs_all_sites.shape[0])
            usgs_ids = data_model.t_s_dict['sites_id']
            sites_index = np.where(np.in1d(usgs_all_sites, usgs_ids))[0]
            sites_chosen[sites_index] = 1
            for key, value in data_model.data_source.gage_dict.items():
                value_new = np.array([value[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0])
                gage_dict_new[key] = value_new
            data_model.data_source.gage_dict = gage_dict_new
            assert (np.array(usgs_ids) == gage_dict_new["STAID"]).all()
        if train_stat_dict is None:
            stat_dict_temp = data_model.cal_stat_all()
        else:
            stat_dict_temp = train_stat_dict
        data_model.stat_dict = stat_dict_temp

        return data_model

    def update_model_param(self, opt, **kwargs):
        for key in kwargs:
            if key in self.data_source.data_config.model_dict[opt].keys():
                self.data_source.data_config.model_dict[opt][key] = kwargs[key]
                print("update", opt, key, kwargs[key])

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes of Gages"""
        # streamflow
        flow = self.data_flow
        stat_dict = dict()
        basin_area = self.data_source.read_attr(self.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
        mean_prep = self.data_source.read_attr(self.t_s_dict["sites_id"], ['PPTAVG_BASIN'], is_return_dict=False)
        # annual value to daily value and cm to mm
        mean_prep = mean_prep / 365 * 10
        stat_dict['usgsFlow'] = cal_stat_basin_norm(flow, basin_area, mean_prep)

        # forcing
        forcing_lst = self.data_source.all_configs["forcing_chosen"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var == 'prcp':
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_source.all_configs["attr_chosen"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        data = self.data_flow
        basin_area = self.data_source.read_attr(self.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
        mean_prep = self.data_source.read_attr(self.t_s_dict["sites_id"], ['PPTAVG_BASIN'], is_return_dict=False)
        mean_prep = mean_prep / 365 * 10
        data = _basin_norm(data, basin_area, mean_prep, to_norm=True)
        data = np.expand_dims(data, axis=2)
        data = _trans_norm(data, 'usgsFlow', stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data


def _trans_norm(x, var_lst, stat_dict, *, to_norm):
    """normalization; when to_norm=False, anti-normalization
    :parameter
        xï¼šad or 3d
            2d: 1st dim is gauge  2nd dim is var type
            3d: 1st dim is gauge 2nd dim is time 3rd dim is var type
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
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


class CamelsModel(DataModel):
    def __init__(self, data_source, *args):
        super().__init__(data_source, *args)

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes. 计算统计值，便于后面归一化处理。"""
        # streamflow
        flow = self.data_flow
        stat_dict = dict()
        basin_area = self.data_source.read_attr(self.t_s_dict["sites_id"], ['area_gages2'], is_return_dict=False)
        mean_prep = self.data_source.read_attr(self.t_s_dict["sites_id"], ['p_mean'], is_return_dict=False)
        stat_dict['usgsFlow'] = cal_stat_basin_norm(flow, basin_area, mean_prep)

        # forcing
        forcing_lst = self.data_source.all_configs["forcing_chosen"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var == 'prcp':
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_source.all_configs["attr_chosen"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        data = self.data_flow
        basin_area = self.data_source.read_attr(self.t_s_dict["sites_id"], ['area_gages2'], is_return_dict=False)
        mean_prep = self.data_source.read_attr(self.t_s_dict["sites_id"], ['p_mean'], is_return_dict=False)
        data = _basin_norm(data, basin_area, mean_prep, to_norm=True)
        data = np.expand_dims(data, axis=2)
        data = _trans_norm(data, 'usgsFlow', stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data


class StreamflowInputDataset(Dataset):
    """Dataset for input of LSTM"""

    def __init__(self, data_model, train_mode=True, transform=None):
        self.data_model = data_model
        self.train_mode = train_mode
        model_dict = data_model.data_source.data_config.model_dict
        self.batch_size, self.rho = model_dict["train"]["miniBatch"]
        x, self.y, c = data_model.load_data(model_dict)
        c = copy_attr_array_in2d(c, x.shape[1])
        self.xc = concat_two_3darray(x, c)
        self.transform = transform

    def __getitem__(self, index):
        ngrid, nt, nx = self.xc.shape
        rho = self.rho
        if self.train_mode:
            i_grid = index // (nt - rho + 1)
            i_t = index % (nt - rho + 1)
            xc = self.xc[i_grid, i_t:i_t + rho, :]
            y = self.y[i_grid, i_t:i_t + rho, :]
        else:
            xc = self.xc[index, :, :]
            y = self.y[index, :, :]
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def __len__(self):
        if self.train_mode:
            return self.xc.shape[0] * (self.xc.shape[1] - self.rho + 1)
        else:
            return self.xc.shape[0]


class StreamflowDataset(Dataset):
    """Dataset for input of LSTM for only one gauge"""

    def __init__(self, x, y, rho=30, train_mode=True):
        self.x = x
        self.y = y
        self.rho = rho
        self.train_mode = train_mode

    def __getitem__(self, index):
        if self.train_mode:
            x = self.x[index:index + self.rho, :]
            y = self.y[index:index + self.rho, :]
        else:
            x = self.x
            y = self.y
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        if self.train_mode:
            return self.x.shape[0] - self.rho + 1
        else:
            return 1


def create_datasets(data_model, valid_size=0.2, train_mode=True):
    model_dict = data_model.data_source.data_config.model_dict
    batch_size, rho = model_dict["train"]["miniBatch"]
    x3d, y3d, c = data_model.load_data(model_dict)
    x = x3d[0]
    y = y3d[0]
    if train_mode:
        train_data = StreamflowDataset(x, y, rho=rho)
        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # load training data in batches
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler)

        # load validation data in batches
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   sampler=valid_sampler)
        return train_loader, valid_loader
    else:
        test_data = StreamflowDataset(x, y, rho=rho, train_mode=False)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size)

        return test_loader
