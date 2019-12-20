"""一个处理数据的模板方法"""
import json
import os

import numpy as np

from explore.stat import trans_norm, cal_stat


class DataModel(object):
    """数据格式化类，通过 SourceData 类对象函数实现 数据读取以及归一化处理等"""

    def __init__(self, data_source):
        """:parameter data_source: SourceData 类对象"""
        self.data_source = data_source
        # 调用SourceData的read_xxx型函数读取forcing，flow，attributes等数据
        # read flow
        data_flow = data_source.read_usgs(data_source.gage_dict, data_source.gage_fld_lst, data_source.t_range)
        data_flow = np.expand_dims(data_flow, axis=2)
        # 根据径流数据过滤掉一些站点，目前给的是示例参数，后面需修改
        data_flow, usgs_id = data_source.usgs_screen_streamflow(data_flow, ["02349000", "08168797"],
                                                                [19950101, 20150101],
                                                                {'missing_data_ratio': 0.1, 'zero_value_ratio': 0.005,
                                                                 'basin_area_ceil': 'HUC4'})
        self.data_flow = data_flow
        # read forcing
        var_lst = data_source.all_configs.get("forcing_chosen")
        ref_nonref_regions = data_source.all_configs.get("gage_region_dir")
        data_forcing = data_source.read_forcing(usgs_id, data_source.t_range, var_lst, ref_nonref_regions)
        self.data_forcing = data_forcing
        # read attributes
        attr_lst = data_source.all_configs.get("attr_chosen")
        data_attr = data_source.read_attr(usgs_id, attr_lst)
        self.data_attr = data_attr
        # 初步计算统计值
        stat_dict = self.basic_statistic()
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

        dir_db = self.data_source.all_configs["root_dir"]
        stat_file = os.path.join(dir_db, 'Statistics.json')
        with open(stat_file, 'w') as FP:
            json.dump(stat_dict, FP, indent=4)

    def basic_statistic(self):
        """根据读取的数据进行基本统计运算，以便于归一化等运算"""
        # 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
        # module variable
        source_data = self.data_source
        dir_db = source_data.all_configs.get("root_dir")
        stat_file = os.path.join(dir_db, 'Statistics.json')
        # 如果统计值已经计算过了，就没必要再重新计算了
        if not os.path.isfile(stat_file):
            self.cal_stat_all()
        # 计算过了，就从存储的json文件中读取出统计结果
        with open(stat_file, 'r') as fp:
            stat_dict = json.load(fp)
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
