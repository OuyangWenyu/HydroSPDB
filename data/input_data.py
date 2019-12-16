"""刚开始都需要获取并处理数据格式到项目数据格式，这里类的作用即定义项目数据"""
import json
import os

import numpy as np

from explore.gages2 import dirDB, GAGE_FLD_LST, FORCING_LST, tRange4DownloadData, gageDict, tLstAll, \
    statDict, REF_NONREF_REGIONS, ATTR_STR_SEL
from data.source_data import read_usgs, read_attr_all, read_attr, read_forcing
from hydroDL import utils
from hydroDL import cal_stat, trans_norm


def cal_stat_all(gage_dict, t_range, forcing_lst, flow=None, regions=None):
    """计算统计值，便于后面归一化处理。"""
    stat_dict = dict()
    id_lst = gage_dict[GAGE_FLD_LST[0]]
    if flow.size < 1:
        flow = read_usgs(gage_dict, tRange4DownloadData)

    # 计算统计值
    stat_dict['usgsFlow'] = cal_stat(flow)

    # forcing数据
    x = read_forcing(id_lst, t_range, forcing_lst, regions=regions)
    for k in range(len(forcing_lst)):
        var = forcing_lst[k]
        stat_dict[var] = cal_stat(x[:, :, k])

    # const attribute
    attr_data, attr_lst = read_attr_all(id_lst)
    for k in range(len(attr_lst)):
        var = attr_lst[k]
        print(var)
        stat_dict[var] = cal_stat(attr_data[:, k])

    stat_file = os.path.join(dirDB, 'Statistics.json')
    with open(stat_file, 'w') as FP:
        json.dump(stat_dict, FP, indent=4)


class InputData(object):
    def __init__(self, *, subset='All', t_range):
        self.rootDB = dirDB
        self.subset = subset
        self.gageDict = gageDict
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict[GAGE_FLD_LST[0]]
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict[GAGE_FLD_LST[4]]
            crd[:, 1] = gageDict[GAGE_FLD_LST[5]]
            self.crd = crd
        self.tRange = t_range
        self.time = utils.time.tRange2Array(t_range)

    def __init__(self, attributes, forcing, streamflow):
        self.__attributes = attributes
        self.__forcing = forcing
        self.__streamflow = streamflow

        """
        :InputData: a container of data
        """

    def get_c(self):
        """get属性数据"""
        return self.__attributes

    def get_x(self):
        """get驱动等时间序列数据"""
        return self.__forcing

    def get_y(self):
        return self.__streamflow

    def get_data_train(self):
        return self.get_x(), self.get_y(), self.get_c()

    def get_data_obs(self, *, do_norm=True, rm_nan=True):
        """径流数据读取及归一化处理"""
        data = read_usgs(self.gageDict, tRange4DownloadData)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstAll, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, 'usgsFlow', statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, *, var_lst=FORCING_LST, do_norm=True, rm_nan=True):
        """时间序列数据，主要是驱动数据读取 and choose data in the given time interval 及归一化处理"""
        if type(var_lst) is str:
            var_lst = [var_lst]
        # read ts forcing
        data = read_forcing(self.usgsId, self.tRange, var_lst, regions=REF_NONREF_REGIONS)
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, *, var_lst=ATTR_STR_SEL, do_norm=True, rm_nan=True):
        """属性数据读取及归一化处理"""
        if type(var_lst) is str:
            var_lst = [var_lst]
        data = read_attr(self.usgsId, var_lst)
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data
