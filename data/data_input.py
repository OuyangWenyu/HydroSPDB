"""一个处理数据的模板方法"""
from collections import OrderedDict

import numpy as np

from explore import *


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


class GagesModel(DataModel):
    def __init__(self, data_source, *args):
        super().__init__(data_source, *args)

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes of Gages"""
        # streamflow
        flow = self.data_flow
        stat_dict = dict()
        basin_area = self.data_source.read_attr(self.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
        mean_prep = self.data_source.read_attr(self.t_s_dict["sites_id"], ['PPTAVG_BASIN'], is_return_dict=False)
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
