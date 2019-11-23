"""read gages-ii data以计算统计值 and ready for model"""

# 读取GAGES-II数据需要指定文件路径、时间范围、属性类型、需要计算配置的项是forcing data。
# module variable
import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_numeric_dtype, is_string_dtype
import geopandas as gpd

from hydroDL import pathGages2
from hydroDL import utils
from hydroDL.data import Dataframe
from hydroDL.post.stat import cal_stat, trans_norm
from hydroDL.utils.time import t2dt


# -----------------------------------functions for configure of gages-------------------------------------------------

# 根据配置读取所需的gages-ii站点信息
def read_gage_info(dir_db, region_shapefiles=None, ids_specific=None):
    """读取gages-ii站点及流域基本location等信息。
    从中选出field_lst中属性名称对应的值，存入dic中。

    Parameter:
        dir_db: file of gages' information
        region_shapefile: choose some regions
        ids_specific： given sites' ids
    Return：
        各个站点的attibutes in basinid.txt and 径流数据
    """
    # 数据从第二行开始，因此跳过第一行。
    data = pd.read_csv(dir_db, sep=',', header=None, skiprows=1, dtype={0: str})
    out = dict()
    if len(region_shapefiles):
        # read sites from shapefile of region, get id from it.
        # Read file using gpd.read_file()  # TODO: read multi shapefiles
        shapefile = os.path.join(GAGE_SHAPE_DIR, region_shapefiles[0])
        shape_data = gpd.read_file(shapefile)
        print(shape_data.columns)
        gages_id = shape_data['GAGE_ID'].values
        df_id_region = data.iloc[:, 0].values
        c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
        data = data.iloc[ind1, :]
    if ids_specific:
        df_id_test = data.iloc[:, 0].values
        c, ind1, ind2 = np.intersect1d(df_id_test, ids_specific, return_indices=True)
        data = data.iloc[ind1, :]
    for s in GAGE_FLD_LST:
        if s is GAGE_FLD_LST[1]:
            out[s] = data[GAGE_FLD_LST.index(s)].values.tolist()
        else:
            out[s] = data[GAGE_FLD_LST.index(s)].values
    return out


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


def read_usgs(gage_dict, t_range):
    """读取USGS的daily average 径流数据 according to id and time,
        首先判断哪些径流站点的数据已经读取并存入本地，如果没有，就从网上下载并读入txt文件。
    Parameter:
        gage_dict：站点 information
        t_range: must be time range for downloaded data
    Return：
        y: ndarray--各个站点的径流数据, 1d-axis: gages, 2d-axis: day
    """
    if not os.path.isdir(DIR_GAGE_FLOW):
        os.mkdir(DIR_GAGE_FLOW)
    dir_list = os.listdir(DIR_GAGE_FLOW)
    # 区域一共有18个，因此，为了便于后续处理，还是要把属于不同region的站点的文件放到不同的文件夹下面
    # 判断usgs_id_lst中没有对应径流文件的要从网上下载
    usgs_id_lst = gage_dict[GAGE_FLD_LST[0]]
    for ind in range(len(usgs_id_lst)):
        # different hucs different directories
        huc_02 = gage_dict[GAGE_FLD_LST[3]][ind]
        dir_huc_02 = str(huc_02)
        if dir_huc_02 not in dir_list:
            dir_huc_02 = os.path.join(DIR_GAGE_FLOW, str(huc_02))
            os.mkdir(dir_huc_02)
            dir_list = os.listdir(DIR_GAGE_FLOW)
        dir_huc_02 = os.path.join(DIR_GAGE_FLOW, str(huc_02))
        file_list = os.listdir(dir_huc_02)
        file_usgs_id = str(usgs_id_lst[ind]) + ".txt"
        if file_usgs_id not in file_list:
            # 通过直接读取网页的方式获取数据，然后存入txt文件
            start_time_str = t2dt(t_range[0])
            end_time_str = t2dt(t_range[1]) - timedelta(days=1)
            url = STREAMFLOW_URL.format(usgs_id_lst[ind], start_time_str.year, start_time_str.month, start_time_str.day,
                                        end_time_str.year, end_time_str.month, end_time_str.day)
            r = requests.get(url)
            # 存放的位置是对应HUC02区域的文件夹下
            temp_file = os.path.join(dir_huc_02, str(usgs_id_lst[ind]) + '.txt')
            with open(temp_file, 'w') as f:
                f.write(r.text)
            print("成功写入 " + temp_file + " 径流数据！")
    t_lst = utils.time.tRange2Array(t_range)
    nt = len(t_lst)
    t0 = time.time()
    y = np.empty([len(usgs_id_lst), nt])
    for k in range(len(usgs_id_lst)):
        huc_02 = gage_dict[GAGE_FLD_LST[3]][k]
        data_obs = read_usge_gage(huc_02, usgs_id_lst[k], t_range)
        y[k, :] = data_obs
    print("time of reading usgs streamflow: ", time.time() - t0)
    return y


def usgs_screen(usgs, usgs_ids=None, time_range=None, **kwargs):
    """according to the criteria and its ancillary condition--thresh, choose appropriate ones from all usgs sites
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
            elif criteria == 'basin_area_ceil':
                sites_chosen[site_index] = 1
            else:
                print("Oops!  That is not valid value.  Try again...")
    # get discharge data of chosen sites, and change to ndarray
    usgs_out = usgs_values.iloc[np.where(sites_chosen > 0)].values
    gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]

    return usgs_out, gages_chosen_id


def read_attr_all(gages_ids):
    """读取GAGES-II下的属性数据，目前是将用到的几个属性所属的那个属性大类下的所有属性的统计值都计算一下"""
    data_folder = DIR_GAGE_ATTR
    f_dict = dict()  # factorize dict
    # 每个key-value对是一个文件（str）下的所有属性（list）
    var_dict = dict()
    # 所有属性放在一起
    var_lst = list()
    out_lst = list()
    # 读取所有属性，直接按类型判断要读取的文件名
    var_des = pd.read_csv(os.path.join(DIR_GAGE_ATTR, 'variable_descriptions.txt'), sep=',')
    var_des_map_values = var_des['VARIABLE_TYPE'].tolist()
    for i in range(len(var_des)):
        var_des_map_values[i] = var_des_map_values[i].lower()
    # 按照读取的时候的顺序对type排序
    key_lst = list(set(var_des_map_values))
    key_lst.sort(key=var_des_map_values.index)
    # x_region_names属性暂不需要读入
    key_lst.remove('x_region_names')

    for key in key_lst:
        # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
        if key == 'flow_record':
            key = 'flowrec'
        data_file = os.path.join(data_folder, 'conterm_' + key + '.txt')
        # 各属性值的“参考来源”是不需读入的
        if key == 'bas_classif':
            data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str}, usecols=range(0, 4))
        else:
            data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str})
        if key == 'flowrec':
            # 最后一列为空，舍弃
            data_temp = data_temp.iloc[:, range(0, data_temp.shape[1] - 1)]
        # 该文件下的所有属性
        var_lst_temp = list(data_temp.columns[1:])
        var_dict[key] = var_lst_temp
        var_lst.extend(var_lst_temp)
        k = 0
        n_gage = len(gages_ids)
        out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)  # 所有站点是一维，当前data_file下所有属性是第二维
        # 因为选择的站点可能是站点的一部分，所以需要求交集，ind2是所选站点在conterm_文件中所有站点里的index，把这些值放到out_temp中
        range1 = gages_ids
        range2 = data_temp.iloc[:, 0].astype(str).tolist()
        c, ind1, ind2 = np.intersect1d(range1, range2, return_indices=True)
        for field in var_lst_temp:
            if is_string_dtype(data_temp[field]):  # 字符串值就当做是类别变量，赋值给变量类型value，以及类型说明ref
                value, ref = pd.factorize(data_temp.loc[ind2, field], sort=True)
                out_temp[:, k] = value
                f_dict[field] = ref.tolist()
            elif is_numeric_dtype(data_temp[field]):
                out_temp[:, k] = data_temp.loc[ind2, field].values
            k = k + 1
        out_lst.append(out_temp)
    out = np.concatenate(out_lst, 1)
    # dictFactorize.json is the explanation of value of categorical variables
    file_name = os.path.join(dirDB, 'dictFactorize.json')
    with open(file_name, 'w') as fp:
        json.dump(f_dict, fp, indent=4)
    file_name = os.path.join(dirDB, 'dictAttribute.json')
    with open(file_name, 'w') as fp:
        json.dump(var_dict, fp, indent=4)
    return out, var_lst


def read_attr(usgs_id_lst, var_lst):
    attr_all, var_lst_all = read_attr_all(usgs_id_lst)
    ind_var = list()
    for var in var_lst:
        ind_var.append(var_lst_all.index(var))
    out = attr_all[:, ind_var]
    return out


def read_forcing(usgs_id_lst, t_range, var_lst, dataset='daymet'):
    """读取gagesII_forcing文件夹下的驱动数据(data processed from GEE)
    :return
    x: ndarray -- 1d-axis:gages, 2d-axis: day, 3d-axis: forcing vst
    """
    t0 = time.time()
    data_folder = os.path.join(dirDB, 'gagesII_forcing')
    if dataset is 'nldas':
        print("no data now!!!")
    # different files for different years
    t_start = str(t_range[0])[0:4]
    t_end = str(t_range[1])[0:4]
    t_lst_chosen = utils.time.tRange2Array(t_range)
    t_lst_years = np.arange(t_start, t_end, dtype='datetime64[Y]').astype(str)
    data_temps = pd.DataFrame()
    for year in t_lst_years:
        data_file = os.path.join(data_folder, dataset, 'daymet_MxWdShld_mean_' + year + '.csv')
        data_temp = pd.read_csv(data_file, sep=',', dtype={'gage_id': int})
        frames_temp = [data_temps, data_temp]
        data_temps = pd.concat(frames_temp)
    # choose data in given time and sites. if there is no value for site in usgs_id_lst, just error(because every
    # site should have forcing). using dataframe mostly will make data type easy to handle with
    sites_forcing = data_temps.iloc[:, 0].values
    sites_index = [i for i in range(sites_forcing.size) if sites_forcing[i] in usgs_id_lst.astype(int)]
    data_sites_chosen = data_temps.iloc[sites_index, :]
    t_range_forcing = np.array(data_sites_chosen.iloc[:, 1].values.astype(str), dtype='datetime64[D]')
    t_index = [j for j in range(t_range_forcing.size) if t_range_forcing[j] in t_lst_chosen]
    data_chosen = data_sites_chosen.iloc[t_index, :]
    # when year is a leap year, only 365d will be provided by gee datasets. better to fill it with nan
    # number of days are different in different years, so reshape can't be used
    x = np.empty([len(usgs_id_lst), t_lst_chosen.size, len(var_lst)])
    data_chosen_t_length = np.unique(data_chosen.iloc[:, 1].values).size
    for k in range(len(usgs_id_lst)):
        data_k = data_chosen.iloc[k * data_chosen_t_length:(k + 1) * data_chosen_t_length, :]
        out = np.full([t_lst_chosen.size, len(FORCING_LST)], np.nan)
        # df中的date是字符串，转换为datetime，方可与tLst求交集
        df_date = data_k.iloc[:, 1]
        date = df_date.values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(t_lst_chosen, date, return_indices=True)
        out[ind1, :] = data_k.iloc[ind2, 2:].values
        x[k, :, :] = out

    print("time of reading usgs forcing data", time.time() - t0)
    return x


def cal_stat_all(gage_dict, t_range, forcing_lst, flow=None):
    """计算统计值，便于后面归一化处理。"""
    stat_dict = dict()
    id_lst = gage_dict[GAGE_FLD_LST[0]]
    if flow.size < 1:
        flow = read_usgs(gage_dict, tRange4DownloadData)

    # 计算统计值
    stat_dict['usgsFlow'] = cal_stat(flow)

    # forcing数据
    x = read_forcing(id_lst, t_range, forcing_lst)
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


# ------------------- some scripts for config of gages-ii datasets  -----------------------

dirDB = pathGages2['DB']

# USGS所有站点 file
GAGE_FILE = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format',
                         'conterm_basinid.txt')
GAGE_SHAPE_DIR = os.path.join(dirDB, 'boundaries-shapefiles-by-aggeco')

GAGE_FLD_LST = ['STAID', 'STANAME', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE', 'STATE', 'BOUND_SOURCE', 'HCDN-2009',
                'HBN36', 'OLD_HCDN', 'NSIP_SENTINEL', 'FIPS_SITE', 'COUNTYNAME_SITE', 'NAWQA_SUID']
# gageFldLst = camels.gageFldLst
DIR_GAGE_FLOW = os.path.join(dirDB, 'gages_streamflow')
DIR_GAGE_ATTR = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
STREAMFLOW_URL = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={' \
                 '}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{} '

# 671个流域的forcing值需要重新计算，但是训练先用着671个流域，可以先用CAMELS的计算。
FORCING_LST = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

# all attributes:
# attrLstAll = os.listdir(DIR_GAGE_ATTR)
# # 因为是对CONUS分析，所以只用conterm开头的
# ATTR_LST = []
# for attrLstAllTemp in attrLstAll:
#     if 'conterm' in attrLstAllTemp:
#         attrLstTemp = attrLstAllTemp[8:].lower()
#         ATTR_LST.append(attrLstTemp)

# gages的attributes可以先按照CAMELS的这几项去找，因为使用了forcing数据，因此attributes里就没用气候的数据，因为要进行预测，所以也没用水文的
# land cover部分：forest_frac对应FORESTNLCD06；lai没有，这里暂时用所有forest的属性；land_cover暂时用除人为种植之外的其他所有属性。
# soil：soil_depth相关的有：ROCKDEPAVE；soil_porosity类似的可能是：AWCAVE；soil_conductivity可能相关的：PERMAVE；max_water_content没有，暂时用RFACT
# geology在GAGES-II中一共两类，来自两个数据源，用第一种，
attrBasin = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM']
attrLandcover = ['FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06',
                 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
attrSoil = ['ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT', ]
attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE']
# attributes include: Hydro, HydroMod_Dams, HydroMod_Other,Landscape_Pat,
# LC06_Basin,LC06_Mains100,LC06_Mains800,LC06_Rip100,LC_Crops,Pop_Infrastr,Prot_Areas
attrHydro = ['STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
             'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE', 'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT']
# 'RUNAVE7100', 'WB5100_JAN_MM', 'WB5100_FEB_MM', 'WB5100_MAR_MM', 'WB5100_APR_MM', 'WB5100_MAY_MM',
# 'WB5100_JUN_MM', 'WB5100_JUL_MM', 'WB5100_AUG_MM', 'WB5100_SEP_MM', 'WB5100_OCT_MM', 'WB5100_NOV_MM',
# 'WB5100_DEC_MM', 'WB5100_ANN_MM', 'PCT_1ST_ORDER', 'PCT_2ND_ORDER', 'PCT_3RD_ORDER', 'PCT_4TH_ORDER',
# 'PCT_5TH_ORDER', 'PCT_6TH_ORDER_OR_MORE', 'PCT_NO_ORDER'
# "pre19xx" attributes need be given to right period
attrHydroModDams = ['NDAMS_2009', 'DDENS_2009', 'STOR_NID_2009', 'STOR_NOR_2009', 'MAJ_NDAMS_2009', 'MAJ_DDENS_2009',
                    'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS', 'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS']
# 'pre1940_NDAMS', 'pre1950_NDAMS', 'pre1960_NDAMS', 'pre1970_NDAMS', 'pre1980_NDAMS',
#                     'pre1990_NDAMS', 'pre1940_DDENS', 'pre1950_DDENS', 'pre1960_DDENS', 'pre1970_DDENS',
#                     'pre1980_DDENS', 'pre1990_DDENS', 'pre1940_STOR', 'pre1950_STOR', 'pre1960_STOR', 'pre1970_STOR',
#                     'pre1980_STOR', 'pre1990_STOR',
attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'RAW_AVG_DIS_ALLCANALS', 'CANALS_MAINSTEM_PCT',
                     'NPDES_MAJ_DENS', 'RAW_DIS_NEAREST_MAJ_NPDES', 'RAW_AVG_DIS_ALL_MAJ_NPDES', 'FRESHW_WITHDRAWAL',
                     'MINING92_PCT', 'PCT_IRRIG_AG', 'POWER_NUM_PTS', 'POWER_SUM_MW']
attrLandscapePat = ['FRAGUN_BASIN', 'HIRES_LENTIC_NUM', 'HIRES_LENTIC_DENS', 'HIRES_LENTIC_MEANSIZ']
attrLC06Basin = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'DEVOPENNLCD06',
                 'DEVLOWNLCD06', 'DEVMEDNLCD06', 'DEVHINLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
                 'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'PASTURENLCD06', 'CROPSNLCD06', 'WOODYWETNLCD06',
                 'EMERGWETNLCD06']
attrLC06Mains100 = ['MAINS100_DEV', 'MAINS100_FOREST', 'MAINS100_PLANT', 'MAINS100_11', 'MAINS100_12', 'MAINS100_21',
                    'MAINS100_22', 'MAINS100_23', 'MAINS100_24', 'MAINS100_31', 'MAINS100_41', 'MAINS100_42',
                    'MAINS100_43', 'MAINS100_52', 'MAINS100_71', 'MAINS100_81', 'MAINS100_82', 'MAINS100_90',
                    'MAINS100_95', ]
attrLC06Mains800 = ['MAINS800_DEV', 'MAINS800_FOREST', 'MAINS800_PLANT', 'MAINS800_11', 'MAINS800_12', 'MAINS800_21',
                    'MAINS800_22', 'MAINS800_23', 'MAINS800_24', 'MAINS800_31', 'MAINS800_41', 'MAINS800_42',
                    'MAINS800_43', 'MAINS800_52', 'MAINS800_71', 'MAINS800_81', 'MAINS800_82', 'MAINS800_90',
                    'MAINS800_95']
attrLC06Rip100 = ['RIP100_DEV', 'RIP100_FOREST', 'RIP100_PLANT', 'RIP100_11', 'RIP100_12', 'RIP100_21', 'RIP100_22',
                  'RIP100_23', 'RIP100_24', 'RIP100_31', 'RIP100_41', 'RIP100_42', 'RIP100_43', 'RIP100_52',
                  'RIP100_71', 'RIP100_81', 'RIP100_82', 'RIP100_90', 'RIP100_95']
attrLCCrops = ['RIP800_DEV', 'RIP800_FOREST', 'RIP800_PLANT', 'RIP800_11', 'RIP800_12', 'RIP800_21', 'RIP800_22',
               'RIP800_23', 'RIP800_24', 'RIP800_31', 'RIP800_41', 'RIP800_42', 'RIP800_43', 'RIP800_52', 'RIP800_71',
               'RIP800_81', 'RIP800_82', 'RIP800_90', 'RIP800_95']
attrPopInfrastr = ['CDL_CORN', 'CDL_COTTON', 'CDL_RICE', 'CDL_SORGHUM', 'CDL_SOYBEANS', 'CDL_SUNFLOWERS', 'CDL_PEANUTS',
                   'CDL_BARLEY', 'CDL_DURUM_WHEAT', 'CDL_SPRING_WHEAT', 'CDL_WINTER_WHEAT', 'CDL_WWHT_SOY_DBL_CROP',
                   'CDL_OATS', 'CDL_ALFALFA', 'CDL_OTHER_HAYS', 'CDL_DRY_BEANS', 'CDL_POTATOES', 'CDL_FALLOW_IDLE',
                   'CDL_PASTURE_GRASS', 'CDL_ORANGES', 'CDL_OTHER_CROPS', 'CDL_ALL_OTHER_LAND']
attrProtAreas = ['PDEN_2000_BLOCK', 'PDEN_DAY_LANDSCAN_2007', 'PDEN_NIGHT_LANDSCAN_2007', 'ROADS_KM_SQ_KM',
                 'RD_STR_INTERS', 'IMPNLCD06', 'NLCD01_06_DEV']
ATTR_STR_SEL = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams + attrHydroModOther + \
               attrLandscapePat + attrLC06Basin + attrLC06Mains100 + attrLC06Mains800 + attrLC06Rip100 + attrLCCrops + \
               attrPopInfrastr + attrProtAreas

# GAGES-II的所有站点 and all time, for first using of this code to download streamflow datasets
tRange4DownloadData = [19800101, 20150101]  # 左闭右开
tLstAll = utils.time.tRange2Array(tRange4DownloadData)
# gageDict = read_gage_info(gageField)

# training time range
tRangeTrain = [19950101, 20000101]

# regions
REF_NONREF_REGIONS_SHP = ['bas_nonref_MxWdShld.shp']

# 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
# module variable
statFile = os.path.join(dirDB, 'Statistics.json')
gageDictOrigin = read_gage_info(GAGE_FILE, REF_NONREF_REGIONS_SHP,
                                ['07311600', '04074548', '04121000', '04121500', '05211000'])
# screen some sites
usgs = read_usgs(gageDictOrigin, tRange4DownloadData)
usgsFlow, gagesChosen = usgs_screen(pd.DataFrame(usgs, index=gageDictOrigin[GAGE_FLD_LST[0]], columns=tLstAll),
                                    time_range=tRangeTrain, missing_data_ratio=0.1, zero_value_ratio=0.005,
                                    basin_area_ceil='HUC4')
# after screening, update the gageDict and idLst
gageDict = read_gage_info(GAGE_FILE, region_shapefiles=REF_NONREF_REGIONS_SHP, ids_specific=gagesChosen)
# 如果统计值已经计算过了，就没必要再重新计算了
if not os.path.isfile(statFile):
    cal_stat_all(gageDict, tRangeTrain, FORCING_LST, usgsFlow)
# 计算过了，就从存储的json文件中读取出统计结果
with open(statFile, 'r') as fp:
    statDict = json.load(fp)


# ------------------- DataframeGages2  -----------------------

class DataframeGages2(Dataframe):
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

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

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
        data = read_forcing(self.usgsId, self.tRange, var_lst)
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
