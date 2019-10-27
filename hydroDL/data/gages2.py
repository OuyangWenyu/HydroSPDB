"""read gages-ii data"""

# 读取GAGES-II数据，需要指定文件路径、时间范围、属性类型、需要计算配置的项是forcing data。
# module variable
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_numeric_dtype, is_string_dtype

from hydroDL import pathGages2
from hydroDL import utils
from hydroDL.data import Dataframe
from hydroDL.data.camels import read_forcing
from hydroDL.utils.statistics import cal_stat, trans_norm

dirDB = pathGages2['DB']
dirGageflow = os.path.join(dirDB, 'gages_streamflow')
dirGageAttr = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
streamflowUrl = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-01-01&end_date={}-12-31'
tRange = [19800101, 20150101]
tLst = utils.time.tRange2Array(tRange)
nt = len(tLst)
# 671个流域的forcing值需要重新计算，但是训练先用着671个流域，可以先用CAMELS的计算。
forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
# gages的attributes可以先按照CAMELS的这几项去找，因为使用了forcing数据，因此attributes里就没用气候的数据，因为要进行预测，所以也没用水文的
attrLstAll = os.listdir(dirGageAttr)
# 因为是对CONUS分析，所以只用conterm开头的
attrLst = []
for attrLstAllTemp in attrLstAll:
    if 'conterm' in attrLstAllTemp:
        attrLstTemp = attrLstAllTemp[8:].lower()
        attrLst.append(attrLstTemp)

# land cover部分：forest_frac对应FORESTNLCD06；lai没有，这里暂时用所有forest的属性；land_cover暂时用除人为种植之外的其他所有属性。
# soil：soil_depth相关的有：ROCKDEPAVE；soil_porosity类似的可能是：AWCAVE；soil_conductivity可能相关的：PERMAVE；max_water_content没有，暂时用RFACT
# geology在GAGES-II中一共两类，来自两个数据源，用第一种，
gageField = ['HUC02', 'STAID', 'STANAME', 'LAT_GAGE', 'LNG_GAGE', 'DRAIN_SQKM']
attrBasin = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM']
attrLandcover = ['FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06',
                 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
attrSoil = ['ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT', ]
attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE']
attrLstSel = attrBasin + attrLandcover + attrSoil + attrGeol


# 然后根据配置读取所需的gages-ii站点信息
def read_gage_info(dir_db, field_lst):
    """读取gages-ii站点及流域基本location等信息。
    从中选出field_lst中属性名称对应的值，存入dic中。
    """
    gage_file = os.path.join(dir_db, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format',
                             'conterm_basinid.txt')
    # 数据从第二行开始，因此跳过第一行。
    data = pd.read_csv(gage_file, sep=',', header=0)
    out = dict()
    for s in field_lst:
        if s is gageField[2]:
            out[s] = data[field_lst.index(s)].values.tolist()
        else:
            out[s] = data[field_lst.index(s)].values
    return out


# module variable
gageDict = read_gage_info(dirDB, gageField)

# 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
# module variable
statFile = os.path.join(dirDB, 'Statistics.json')


def read_usgs_gage(usgs_id, *, read_qc=False):
    """读取各个径流站的径流数据"""
    # 首先找到要读取的那个txt
    ind = np.argwhere(gageDict['id'] == usgs_id)[0][0]
    huc = gageDict[gageField[0]][ind]
    usgs_file = os.path.join(dirGageflow, str(huc).zfill(2), '%08d.txt' % (usgs_id))
    data_temp = pd.read_csv(usgs_file, skiprows=29, sep='\t',
                            names=['agency_cd', 'site_no', 'datetime', 'flow', 'mode'], index_col=0)
    # 处理下负值
    obs = data_temp[3].values
    obs[obs < 0] = np.nan
    if read_qc is True:
        qc_dict = {'A': 1, 'A:e': 2, 'M': 3}
        qc = np.array([qc_dict[x] for x in data_temp[4]])
    # 如果时间序列长度和径流数据长度不一致，说明有missing值，先补充nan值
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        df_date = data_temp[[1, 2, 3]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        [C, ind1, ind2] = np.intersect1d(date, tLst, return_indices=True)
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


def read_usgs(usgs_id_lst):
    """读取USGS的径流数据，首先判断哪些径流站点的数据已经读取并存入本地，如果没有，就从网上下载并读入txt文件。
    Parameter:
        usgs_id_lst：站点列表

    Return：
        各个站点的径流数据
    """
    if not os.path.isdir(dirGageflow):
        os.mkdir(dirGageflow)
    dir_list = os.listdir(dirGageflow)
    # 区域一共有18个，因此，为了便于后续处理，还是要把属于不同region的站点的文件放到不同的文件夹下面
    # 判断usgs_id_lst中没有对应径流文件的要从网上下载
    for usgs_id in usgs_id_lst:
        # 首先判断站点属于哪个region
        ind = np.argwhere(gageDict['id'] == usgs_id)[0][0]
        huc_02 = gageDict[gageField[0]][ind]
        dir_huc_02 = str(huc_02)
        if dir_huc_02 not in dir_list:
            os.mkdir(dir_huc_02)
        file_usgs_id = str(usgs_id) + ".txt"
        file_list = os.listdir(dir_huc_02)
        if file_usgs_id not in file_list:
            # 通过直接读取网页的方式获取数据，然后存入txt文件
            url = streamflowUrl.format(usgs_id, tRange[0], tRange[1])
            r = requests.get(url)
            # 存放的位置是对应HUC02区域的文件夹下
            temp_file = dirGageflow + dir_huc_02 + usgs_id + '.txt'
            with open(temp_file, 'w') as f:
                f.write(r.text)
            print("成功写入" + temp_file + "径流数据！")
    t0 = time.time()
    y = np.empty([len(usgs_id_lst), nt])
    for k in range(len(usgs_id_lst)):
        dataObs = read_usgs_gage(usgs_id_lst[k])
        y[k, :] = dataObs
    print("read usgs streamflow", time.time() - t0)
    return y


def read_attr_all(save_dict=False):
    """读取GAGES-II下的属性数据"""
    data_folder = dirGageAttr
    f_dict = dict()  # factorize dict
    var_dict = dict()
    var_lst = list()
    out_lst = list()
    # 属性暂时只用一部分，判断来自哪个文件
    key_lst = []
    var_des = pd.read_csv(os.path.join(dirGageAttr, 'variable_description.txt'), sep=',')
    var_des_map_keys = var_des['VARIABLE_NAME']
    var_des_map_values = var_des['VARIABLE_TYPE']
    var_des_map = {}
    for i in range(len(var_des)):
        var_des_map[var_des_map_keys[i]] = var_des_map_values[i]
    for attrLstSelTemp in attrLstSel:
        key_lst.append(var_des_map[attrLstSelTemp].lower())

    for key in key_lst:
        data_file = os.path.join(data_folder, 'conterm_' + key + '.txt')
        data_temp = pd.read_csv(data_file, sep=',')
        var_lst_temp = list(data_temp.columns[1:])
        var_dict[key] = var_lst_temp
        var_lst.extend(var_lst_temp)
        k = 0
        n_gage = len(gageDict['STAID'])
        out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)
        for field in var_lst_temp:
            if is_string_dtype(data_temp[field]):
                value, ref = pd.factorize(data_temp[field], sort=True)
                out_temp[:, k] = value
                f_dict[field] = ref.tolist()
            elif is_numeric_dtype(data_temp[field]):
                out_temp[:, k] = data_temp[field].values
            k = k + 1
        out_lst.append(out_temp)
    out = np.concatenate(out_lst, 1)
    if save_dict is True:
        file_name = os.path.join(data_folder, 'dictFactorize.json')
    with open(file_name, 'w') as fp:
        json.dump(f_dict, fp, indent=4)
    file_name = os.path.join(data_folder, 'dictAttribute.json')
    with open(file_name, 'w') as fp:
        json.dump(var_dict, fp, indent=4)
    return out, var_lst


def cal_stat_all():
    """计算统计值，便于后面归一化处理。
    目前驱动数据暂时使用camels的驱动数据，因此forcing的统计计算可以直接使用camels的；
    这里仅针对GAGES-II属性值进行统计计算；
    USGS径流数据也使用GAGES-II的数据：先确定要
    """
    stat_dict = dict()
    id_lst = gageDict['STAID']
    # usgs streamflow
    y = read_usgs(id_lst)
    # 计算统计值，可以和camels下的共用同一个函数
    stat_dict['usgsFlow'] = cal_stat(y)
    # forcing数据可以暂时使用camels下的，后续有了新数据，也可以组织成和camels一样的
    x = read_forcing(id_lst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        stat_dict[var] = cal_stat(x[:, :, k])
    # const attribute
    attr_data, attr_lst = read_attr_all()
    for k in range(len(attr_lst)):
        var = attr_lst[k]
        stat_dict[var] = cal_stat(attr_data[:, k])
    stat_file = os.path.join(dirDB, 'Statistics.json')
    with open(stat_file, 'w') as FP:
        json.dump(stat_dict, FP, indent=4)


# 如果统计值已经计算过了，就没必要再重新计算了
if not os.path.isfile(statFile):
    cal_stat_all()
# 计算过了，就从存储的json文件中读取出统计结果
with open(statFile, 'r') as fp:
    statDict = json.load(fp)


def read_attr(usgs_id_lst, var_lst):
    attr_all, var_lst_all = read_attr_all()
    ind_var = list()
    for var in var_lst:
        ind_var.append(var_lst_all.index(var))
    id_lst_all = gageDict['id']
    c, ind_grid, ind2 = np.intersect1d(id_lst_all, usgs_id_lst, return_indices=True)
    temp = attr_all[ind_grid, :]
    out = temp[:, ind_var]
    return out


class DataframeGages2(Dataframe):
    def __init__(self, *, subset='All', tRange):
        self.rootDB = dirDB
        self.subset = subset
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict['lat']
            crd[:, 1] = gageDict['lon']
            self.crd = crd
        self.time = utils.time.tRange2Array(tRange)

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def get_data_obs(self, *, do_norm=True, rm_nan=True):
        """径流数据读取及归一化处理"""
        data = read_usgs(self.usgsId)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, 'usgsFlow', toNorm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, *, var_lst=forcingLst, do_norm=True, rm_nan=True):
        """时间序列数据，主要是驱动数据读取及归一化处理"""
        if type(var_lst) is str:
            var_lst = [var_lst]
        # read ts forcing
        data = read_forcing(self.usgsId, var_lst)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, var_lst, toNorm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, *, var_lst=attrLstSel, do_norm=True, rm_nan=True):
        """属性数据读取及归一化处理"""
        if type(var_lst) is str:
            var_lst = [var_lst]
        data = read_attr(self.usgsId, var_lst)
        if do_norm is True:
            data = trans_norm(data, var_lst, toNorm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data
