"""一个处理数据的模板方法"""
import json
import os

import numpy as np

from explore.stat import trans_norm, cal_stat
from utils import unserialize_json, serialize_json


class DataModel(object):
    """数据格式化类，通过 SourceData 类对象函数实现 数据读取以及归一化处理等"""

    def __init__(self, data_source):
        """:parameter data_source: SourceData 类对象"""
        self.data_source = data_source
        # 调用SourceData的read_xxx型函数读取forcing，flow，attributes等数据
        # read flow
        data_flow = data_source.read_usgs()
        # data_flow = np.expand_dims(data_flow, axis=2)
        # 根据径流数据过滤掉一些站点，目前给的是示例参数，后面需修改
        data_flow, usgs_id, t_range_list = data_source.usgs_screen_streamflow(data_flow,
                                                                              usgs_ids=['03144816', '03145000',
                                                                                        '03156000', '03157000',
                                                                                        '03157500', '03219500',
                                                                                        '03220000', '03221000',
                                                                                        '03223000', '03224500',
                                                                                        '03225500', '03226800'],
                                                                              time_range=['1995-01-01', '1997-01-01'])
        self.data_flow = data_flow
        self.gages_id = usgs_id
        self.t_range_list = t_range_list
        # read forcing
        data_forcing = data_source.read_forcing(usgs_id, t_range_list)
        self.data_forcing = data_forcing
        # read attributes
        attr_lst = data_source.all_configs.get("attr_chosen")
        data_attr = data_source.read_attr(usgs_id, attr_lst)
        self.data_attr = data_attr
        # 初步计算统计值
        stat_dict = self.cal_stat_all()
        self.stat_dict = stat_dict

    def cal_stat_all(self):
        """计算统计值，便于后面归一化处理。"""
        forcing_lst = self.data_source.all_configs["forcing_chosen"]
        flow = self.data_flow
        stat_dict = dict()
        # 计算统计值
        stat_dict['usgsFlow'] = cal_stat(flow)

        # forcing数据
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_source.all_configs["attr_chosen"]
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            print(var)
            stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_obs(self, rm_nan=True):
        """径流数据读取及归一化处理"""
        stat_dict = self.stat_dict
        data = self.data_flow
        data = trans_norm(data, 'flow', stat_dict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, rm_nan=True):
        """时间序列数据，主要是驱动数据读取 and choose data in the given time interval 及归一化处理"""
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = trans_norm(data, var_lst, stat_dict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, rm_nan=True):
        """属性数据读取及归一化处理"""
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("attr_chosen")
        data = self.data_attr
        data = trans_norm(data, var_lst, stat_dict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def load_data(self):
        """读取数据为模型输入的形式，完成归一化运算"""
        # 如果读取到统计数据的json文件，则不需要再次计算。
        opt_data = self.data_source.all_configs
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        rm_nan_c = opt_data['rmNan'][0]
        x = self.get_data_ts(rm_nan=rm_nan_x)
        y = self.get_data_obs(rm_nan=rm_nan_y)
        c = self.get_data_const(rm_nan=rm_nan_c)
        if opt_data['daObs'] > 0:
            nday = opt_data['daObs']
            obs = self.get_data_obs(rm_nan=True)
            x = np.concatenate([x, obs], axis=2)
        return x, y, c
