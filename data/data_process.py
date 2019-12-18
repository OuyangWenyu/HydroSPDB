"""一个处理数据的模板方法"""
import datetime as dt
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from explore.stat import cal_stat_all
import utils


class DataFrameGages(object):
    """gages数据格式化类"""

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


def wrap_master(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def read_master_file(out):
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'r') as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + m_file)
    return m_dict


def write_master_file(m_dict):
    out = m_dict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'w') as fp:
        json.dump(m_dict, fp, indent=4)
    print('write master file ' + m_file)
    return out


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    if not os.path.exists(out):
        os.makedirs(out)
    if not os.path.exists(os.path.join(out, 'master.json')):
        return ['None']
    mDict = read_master_file(out)
    target = mDict['data']['target']
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict['loss']['name']
    if epoch is None:
        epoch = mDict['train']['nEpoch']

    fileNameLst = list()
    for k in range(nt):
        testName = '_'.join(
            [subset, str(tRange[0]),
             str(tRange[1]), 'ep' + str(epoch)])
        fileName = '_'.join([target[k], testName])
        fileNameLst.append(fileName)
        if lossName == 'hydroDL.model.crit.SigmaLoss':
            fileName = '_'.join([target[k] + 'SigmaX', testName])
            fileNameLst.append(fileName)

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + '_' + suffix
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    return filePathLst


def load_data(opt_data):
    if eval(opt_data['name']) is app.streamflow.data.gages2.DataframeGages2:
        df = app.streamflow.data.gages2.DataframeGages2(
            subset=opt_data['subset'],
            t_range=opt_data['tRange'])
    elif eval(opt_data['name']) is app.streamflow.data.camels.DataframeCamels:
        df = app.streamflow.data.camels.DataframeCamels(
            subset=opt_data['subset'], t_range=opt_data['tRange'])
    else:
        raise Exception('unknown database')
    x = df.get_data_ts(
        var_lst=opt_data['varT'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    y = df.get_data_obs(
        do_norm=opt_data['doNorm'][1], rm_nan=opt_data['rmNan'][1])
    c = df.get_data_const(
        var_lst=opt_data['varC'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    if opt_data['daObs'] > 0:
        nday = opt_data['daObs']
        sd = utils.time.t2dt(
            opt_data['tRange'][0]) - dt.timedelta(days=nday)
        ed = utils.time.t2dt(
            opt_data['tRange'][1]) - dt.timedelta(days=nday)
        if eval(opt_data['name']) is app.streamflow.data.gages2.DataframeGages2:
            df = app.streamflow.data.gages2.DataframeGages2(subset=opt_data['subset'], tRange=[sd, ed])
        elif eval(opt_data['name']) is app.streamflow.data.camels.DataframeCamels:
            df = app.streamflow.data.camels.DataframeCamels(subset=opt_data['subset'], tRange=[sd, ed])
        obs = df.get_data_obs(do_norm=opt_data['doNorm'][1], rm_nan=True)
        x = np.concatenate([x, obs], axis=2)
    return df, x, y, c


def read_usge_gage(huc, usgs_id, t_range, read_qc=False):
    """读取各个径流站的径流数据"""
    print(usgs_id)
    # 首先找到要读取的那个txt
    usgs_file = os.path.join(DIR_GAGE_FLOW, str(huc), usgs_id + '.txt')
    # 下载的数据文件，注释结束的行不一样
    row_comment_end = 27  # 从0计数的行数
    with open(usgs_file, 'r') as f:
        ind_temp = 0
        for line in f:
            if line[0] is not '#':
                row_comment_end = ind_temp
                break
            ind_temp += 1

    # 下载的时候，没有指定统计类型，因此下下来的数据表有的还包括径流在一个时段内的最值，这里仅适用均值
    skip_rows_index = list(range(0, row_comment_end))
    skip_rows_index.append(row_comment_end + 1)
    df_flow = pd.read_csv(usgs_file, skiprows=skip_rows_index, sep='\t', dtype={'site_no': str})
    if usgs_id == '07311600':
        print(
            "just for test, it only contains max and min flow of a day, but dont have a mean, there will be some "
            "warning, but it's fine. no impact for results.")
    # 原数据的列名并不好用，这里修改
    columns_names = df_flow.columns.tolist()
    for column_name in columns_names:
        # 00060表示径流值，00003表示均值
        # 还有一种情况：#        126801       00060     00003     Discharge, cubic feet per second (Mean)和
        # 126805       00060     00003     Discharge, cubic feet per second (Mean), PUBLISHED 都是均值，但有两套数据，这里暂时取第一套
        if '_00060_00003' in column_name and '_00060_00003_cd' not in column_name:
            df_flow.rename(columns={column_name: 'flow'}, inplace=True)
            break
    for column_name in columns_names:
        if '_00060_00003_cd' in column_name:
            df_flow.rename(columns={column_name: 'mode'}, inplace=True)
            break

    columns = ['agency_cd', 'site_no', 'datetime', 'flow', 'mode']
    if df_flow.empty:
        df_flow = pd.DataFrame(columns=columns)

    data_temp = df_flow.loc[:, columns]

    # 处理下负值
    obs = data_temp['flow'].astype('float').values
    # 看看warning是哪个站点：01606500，时间索引为2828的站点为nan，不过不影响计算。
    if usgs_id == '01606500':
        print(obs)
        print(np.argwhere(np.isnan(obs)))
    obs[obs < 0] = np.nan
    if read_qc is True:
        qc_dict = {'A': 1, 'A:e': 2, 'M': 3}
        qc = np.array([qc_dict[x] for x in data_temp[4]])
    # 如果时间序列长度和径流数据长度不一致，说明有missing值，先补充nan值
    t_lst = utils.time.tRange2Array(t_range)
    nt = len(t_lst)
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        # df中的date是字符串，转换为datetime，方可与tLst求交集
        df_date = data_temp['datetime']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs
        if read_qc is True:
            out_qc = np.full([nt], np.nan)
            out_qc[ind2] = qc
    else:
        out = obs
        if read_qc is True:
            out_qc = qc

    if read_qc is True:
        return out, out_qc
    else:
        return out


def usgs_screen_streamflow(usgs, usgs_ids=None, time_range=None, **kwargs):
    """according to the criteria and its ancillary condition--thresh of streamflow data,
        choose appropriate ones from all usgs sites
        Parameters
        ----------
        usgs : pd.DataFrame -- all usgs sites' data, its index are 'sites', its columns are 'day',
                               if there is some missing value, usgs should already be filled by nan
        usgs_ids: list -- chosen sites' ids
        time_range: list -- chosen time range
        kwargs: all criteria

        Returns
        -------
        usgs_out : ndarray -- streamflow  1d-var is gage, 2d-var is day
        sites_chosen: [] -- ids of chosen gages

        Examples
        --------
        usgs_screen(usgs, ["02349000","08168797"], [19950101,20150101],
        {'missing_data_ratio':0.1,'zero_value_ratio':0.005,'basin_area_ceil':'HUC4'})
    """
    sites_chosen = np.zeros(usgs.shape[0])
    # choose the given sites
    usgs_all_sites = usgs.index.values
    if usgs_ids:
        sites_index = np.where(np.in1d(usgs_ids, usgs_all_sites))[0]
        sites_chosen[sites_index] = 1
    else:
        sites_index = np.arange(usgs.shape[0])
        sites_chosen = np.ones(usgs.shape[0])
    # choose data in given time range
    all_t_list = usgs.columns.values
    t_lst = all_t_list
    if time_range:
        # calculate the day length
        t_lst = utils.time.tRange2Array(time_range)
    ts, ind1, ind2 = np.intersect1d(all_t_list, t_lst, return_indices=True)
    usgs_values = usgs.iloc[sites_index, ind1]

    for site_index in sites_index:
        # loop for every site
        runoff = usgs_values.iloc[site_index, :]
        for criteria in kwargs:
            # if any criteria is not matched, we can filter this site
            if sites_chosen[site_index] == 0:
                break
            if criteria == 'missing_data_ratio':
                nan_length = len(runoff[np.isnan(runoff)])
                # then calculate the length of consecutive nan
                thresh = kwargs[criteria]
                if nan_length / runoff.size > thresh:
                    sites_chosen[site_index] = 0
                else:
                    sites_chosen[site_index] = 1

            elif criteria == 'zero_value_ratio':
                sites_chosen[site_index] = 1
            else:
                print("Oops!  That is not valid value.  Try again...")
    # get discharge data of chosen sites, and change to ndarray
    usgs_out = usgs_values.iloc[np.where(sites_chosen > 0)].values
    gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]

    return usgs_out, gages_chosen_id


def basic_statistic():
    # 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
    # module variable
    statFile = os.path.join(dirDB, 'Statistics.json')
    gageDictOrigin = read_gage_info(GAGE_FILE, region_shapefiles=REF_NONREF_REGIONS, screen_basin_area='HUC4')
    # screen some sites
    usgs = read_usgs(gageDictOrigin, tRange4DownloadData)
    usgsFlow, gagesChosen = usgs_screen_streamflow(
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