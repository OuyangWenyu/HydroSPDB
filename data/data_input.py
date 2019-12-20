"""一个处理数据的模板方法"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from explore.stat import cal_stat_all, trans_norm
import utils


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

    def basic_statistic(self):
        """根据读取的数据进行基本统计运算，以便于归一化等运算"""
        # 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
        # module variable
        source_data = self.data_source
        dirDB = source_data.all_configs.get("root_dir")
        statFile = os.path.join(dirDB, 'Statistics.json')
        gageDictOrigin = read_gage_info(GAGE_FILE, region_shapefiles=REF_NONREF_REGIONS, screen_basin_area='HUC4')
        # screen some sites
        usgs = read_usgs(gageDictOrigin, tRange4DownloadData)
        usgsFlow, gagesChosen = usgsd_screen_streamflow(
            pd.DataFrame(usgs, index=gageDictOrigin[GAGE_FLD_LST[0]], columns=tLstAll),
            time_range=tRangeTrain, missing_data_ratio=0.1, zero_value_ratio=0.005)
        # after screening, update the gageDict and idLst
        gageDict = read_gage_info(GAGE_FILE, region_shapefiles=REF_NONREF_REGIONS, ids_specific=gagesChosen)
        # 如果统计值已经计算过了，就没必要再重新计算了
        if not os.path.isfile(statFile):
            cal_stat_all(gageDict, tRangeTrain, FORCING_LST, usgsFlow, REF_NONREF_REGIONS)
        # 计算过了，就从存储的json文件中读取出统计结果
        with open(statFile, 'r') as fp:
            statDict = json.load(fp)
        return

    def get_data_obs(self, stat_dict, rm_nan=True):
        """径流数据读取及归一化处理"""
        data = self.data_flow
        data = trans_norm(data, 'flow', stat_dict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, stat_dict, rm_nan=True):
        """时间序列数据，主要是驱动数据读取 and choose data in the given time interval 及归一化处理"""
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = trans_norm(data, var_lst, stat_dict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, stat_dict, rm_nan=True):
        """属性数据读取及归一化处理"""
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
