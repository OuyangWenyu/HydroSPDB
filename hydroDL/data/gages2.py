"""read gages-ii data以计算统计值，为归一化备用"""

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

from hydroDL import pathGages2
from hydroDL import utils
from hydroDL.data import Dataframe
from hydroDL.post.stat import cal_stat, trans_norm
from hydroDL.utils.time import t2dt

dirDB = pathGages2['DB']
# USGS所有站点
gageFile = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format',
                        'conterm_basinid.txt')
gageChosen = []
gageFldLst = ['STAID', 'STANAME', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE', 'STATE', 'BOUND_SOURCE', 'HCDN-2009',
              'HBN36', 'OLD_HCDN', 'NSIP_SENTINEL', 'FIPS_SITE', 'COUNTYNAME_SITE', 'NAWQA_SUID']
# gageFldLst = camels.gageFldLst
dirGageflow = os.path.join(dirDB, 'gages_streamflow')
dirGageAttr = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
streamflowUrl = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}'
# 左闭右开
tRange = [19800101, 20150101]
tLst = utils.time.tRange2Array(tRange)
nt = len(tLst)
tRangeChosen = [19950101, 19950103]
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
attrBasin = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM']
attrLandcover = ['FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06',
                 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
attrSoil = ['ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT', ]
attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE']
# attributes include: Hydro, HydroMod_Dams, HydroMod_Other,Landscape_Pat,
# LC06_Basin,LC06_Mains100,LC06_Mains800,LC06_Rip100,LC_Crops,Pop_Infrastr,Prot_Areas
attrHydro = ['STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
             'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE', 'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT',
             'RUNAVE7100', 'WB5100_JAN_MM', 'WB5100_FEB_MM', 'WB5100_MAR_MM', 'WB5100_APR_MM', 'WB5100_MAY_MM',
             'WB5100_JUN_MM', 'WB5100_JUL_MM', 'WB5100_AUG_MM', 'WB5100_SEP_MM', 'WB5100_OCT_MM', 'WB5100_NOV_MM',
             'WB5100_DEC_MM', 'WB5100_ANN_MM', 'PCT_1ST_ORDER', 'PCT_2ND_ORDER', 'PCT_3RD_ORDER', 'PCT_4TH_ORDER',
             'PCT_5TH_ORDER', 'PCT_6TH_ORDER_OR_MORE', 'PCT_NO_ORDER']
attrHydroModDams = ['NDAMS_2009', 'DDENS_2009', 'STOR_NID_2009', 'STOR_NOR_2009', 'MAJ_NDAMS_2009', 'MAJ_DDENS_2009',
                    'pre1940_NDAMS', 'pre1950_NDAMS', 'pre1960_NDAMS', 'pre1970_NDAMS', 'pre1980_NDAMS',
                    'pre1990_NDAMS', 'pre1940_DDENS', 'pre1950_DDENS', 'pre1960_DDENS', 'pre1970_DDENS',
                    'pre1980_DDENS', 'pre1990_DDENS', 'pre1940_STOR', 'pre1950_STOR', 'pre1960_STOR', 'pre1970_STOR',
                    'pre1980_STOR', 'pre1990_STOR', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
                    'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS']
attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'RAW_AVG_DIS_ALLCANALS', 'CANALS_MAINSTEM_PCT',
                     'NPDES_MAJ_DENS', 'RAW_DIS_NEAREST_MAJ_NPDES', 'RAW_AVG_DIS_ALL_MAJ_NPDES', 'FRESHW_WITHDRAWAL',
                     'MINING92_PCT', 'PCT_IRRIG_AG', 'POWER_NUM_PTS', 'POWER_SUM_MW']
attrLandscapePat = ['FRAGUN_BASIN', 'HIRES_LENTIC_NUM', 'HIRES_LENTIC_DENS', 'HIRES_LENTIC_MEANSIZ']
attrLC06Basin = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'DEVOPENNLCD06',
                 'DEVLOWNLCD06', 'DEVMEDNLCD06', 'DEVHINLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
                 'MIXEDFORNLCD06', 'DWARFNLCD', 'SHRUBNLCD06', 'GRASSNLCD06', 'SEDGENLCD', 'MOSSNLCD', 'PASTURENLCD06',
                 'CROPSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
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
attrLstSel = attrBasin + attrLandcover + attrSoil + attrGeol


# 然后根据配置读取所需的gages-ii站点信息
def read_gage_info(dir_db, id4test=None):
    """读取gages-ii站点及流域基本location等信息。
    从中选出field_lst中属性名称对应的值，存入dic中。
    it will take so much time to read all sites' data which make test inconvenient, so add 'id4test' param.
    """
    # 数据从第二行开始，因此跳过第一行。
    data = pd.read_csv(dir_db, sep=',', header=None, skiprows=1, dtype={0: str})
    out = dict()
    if id4test:
        df_id = data.iloc[:, 0].values
        c, ind1, ind2 = np.intersect1d(df_id, id4test, return_indices=True)
        data = data.iloc[ind1, :]
    for s in gageFldLst:
        if s is gageFldLst[1]:
            out[s] = data[gageFldLst.index(s)].values.tolist()
        else:
            out[s] = data[gageFldLst.index(s)].values
    return out


# module variable
# GAGES-II的所有站点
# gageDict = read_gage_info(gageField)
# id_lst = gageDict['STAID']
# gageDict = camels.read_gage_info(pathCamels['DB'])
gageDict = read_gage_info(gageFile, ['07311600', '04074548', '04121000', '04121500', '05211000'])
idLst = gageDict[gageFldLst[0]]

# 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
# module variable
statFile = os.path.join(dirDB, 'Statistics.json')


def read_usgs_gage(usgs_id, *, read_qc=False):
    """读取各个径流站的径流数据"""
    print(usgs_id)
    # 首先找到要读取的那个txt
    ind = np.argwhere(gageDict[gageFldLst[0]] == usgs_id)[0][0]
    huc = gageDict[gageFldLst[3]][ind]
    usgs_file = os.path.join(dirGageflow, str(huc), usgs_id + '.txt')
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
            "just for test, it only contains max and min flow of a day, but dont have a mean, there will be some warning, but it's fine. no impact for results.")
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
    if len(obs) != nt:
        out = np.full([nt], np.nan)
        # df中的date是字符串，转换为datetime，方可与tLst求交集
        df_date = data_temp['datetime']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(date, tLst, return_indices=True)
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
        ind = np.argwhere(idLst == usgs_id)[0][0]
        # different hucs different directories
        huc_02 = gageDict[gageFldLst[3]][ind]
        dir_huc_02 = str(huc_02)
        if dir_huc_02 not in dir_list:
            dir_huc_02 = os.path.join(dirGageflow, str(huc_02))
            os.mkdir(dir_huc_02)
            dir_list = os.listdir(dirGageflow)
        dir_huc_02 = os.path.join(dirGageflow, str(huc_02))
        file_list = os.listdir(dir_huc_02)
        file_usgs_id = str(usgs_id) + ".txt"
        if file_usgs_id not in file_list:
            # 通过直接读取网页的方式获取数据，然后存入txt文件
            start_time_str = t2dt(tRange[0])
            #
            end_time_str = t2dt(tRange[1]) - timedelta(days=1)
            url = streamflowUrl.format(usgs_id, start_time_str.year, start_time_str.month, start_time_str.day,
                                       end_time_str.year, end_time_str.month, end_time_str.day)
            r = requests.get(url)
            # 存放的位置是对应HUC02区域的文件夹下
            temp_file = os.path.join(dir_huc_02, str(usgs_id) + '.txt')
            with open(temp_file, 'w') as f:
                f.write(r.text)
            print("成功写入 " + temp_file + " 径流数据！")
    t0 = time.time()
    y = np.empty([len(usgs_id_lst), nt])
    for k in range(len(usgs_id_lst)):
        dataObs = read_usgs_gage(usgs_id_lst[k])
        y[k, :] = dataObs
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
        usgs_out : ndarray -- the ids of usgs sites which match the criteria
        sites_chosen: []
        time_range: []
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
    # calculate the day length
    t_lst = utils.time.tRange2Array(time_range)
    # choose usgs data in these period
    all_t_list = usgs.columns.values
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

    return usgs_out, sites_chosen


def read_attr_all(gages_ids):
    """读取GAGES-II下的属性数据，目前是将用到的几个属性所属的那个属性大类下的所有属性的统计值都计算一下"""
    data_folder = dirGageAttr
    f_dict = dict()  # factorize dict
    # 每个key-value对是一个文件（str）下的所有属性（list）
    var_dict = dict()
    # 所有属性放在一起
    var_lst = list()
    out_lst = list()
    # 读取所有属性，直接按类型判断要读取的文件名
    var_des = pd.read_csv(os.path.join(dirGageAttr, 'variable_descriptions.txt'), sep=',')
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


def read_forcing(usgs_id_lst, t_range, var_lst, dataset='daymet'):
    """读取gagesII_forcing文件夹下的驱动数据(data processed from GEE)"""
    t0 = time.time()
    t_lst_chosen = utils.time.tRange2Array(t_range)
    data_folder = os.path.join(dirDB, 'gagesII_forcing')
    if dataset is 'nldas':
        print("no data now!!!")
    data_file = os.path.join(data_folder, dataset, 'daymet_MxWdShld_mean.csv')
    data_temp = pd.read_csv(data_file, sep=',', dtype={'gage_id': int})
    # choose data in given time and sites. if there is no value for site in usgs_id_lst, just error(because every
    # site should have forcing). using dataframe mostly will make data type easy to handle with
    sites_forcing = data_temp.iloc[:, 0].values
    sites_index = [i for i in range(sites_forcing.size) if sites_forcing[i] in usgs_id_lst.astype(int)]
    data_sites_chosen = data_temp.iloc[sites_index, :]
    t_range_forcing = np.array(data_sites_chosen.iloc[:, 1].values.astype(str), dtype='datetime64[D]')
    t_index = [j for j in range(t_range_forcing.size) if t_range_forcing[j] in t_lst_chosen]
    data_chosen = data_sites_chosen.iloc[t_index, :]
    data_2dim = data_chosen.iloc[:, 2:].values
    data_3dim = data_2dim.reshape(len(usgs_id_lst), len(t_lst_chosen), len(var_lst))

    print("time of reading usgs forcing data", time.time() - t0)
    return data_3dim


def cal_stat_all(id_lst):
    """计算统计值，便于后面归一化处理。
    目前驱动数据暂时使用camels的驱动数据，因此forcing的统计计算可以直接使用camels的；
    这里仅针对GAGES-II属性值进行统计计算；
    USGS径流数据也使用GAGES-II的数据：先确定要
    """
    stat_dict = dict()

    # usgs streamflow and choose sites which match conditions
    usgs = read_usgs(id_lst)
    y, sites_chosen = usgs_screen(pd.DataFrame(usgs, index=id_lst, columns=tLst), time_range=tRangeChosen,
                                  missing_data_ratio=0.1, zero_value_ratio=0.005, basin_area_ceil='HUC4')
    # 计算统计值，可以和camels下的共用同一个函数
    stat_dict['usgsFlow'] = cal_stat(y)
    gage_chosen_ids = id_lst[np.where(sites_chosen == 1)]

    # forcing数据
    x = read_forcing(gage_chosen_ids, tRangeChosen, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        stat_dict[var] = cal_stat(x[:, :, k])

    # const attribute
    attr_data, attr_lst = read_attr_all(gage_chosen_ids)
    for k in range(len(attr_lst)):
        var = attr_lst[k]
        print(var)
        stat_dict[var] = cal_stat(attr_data[:, k])

    stat_file = os.path.join(dirDB, 'Statistics.json')
    with open(stat_file, 'w') as FP:
        json.dump(stat_dict, FP, indent=4)


# 如果统计值已经计算过了，就没必要再重新计算了
if not os.path.isfile(statFile):
    cal_stat_all(idLst)
# 计算过了，就从存储的json文件中读取出统计结果
with open(statFile, 'r') as fp:
    statDict = json.load(fp)


def read_attr(usgs_id_lst, var_lst):
    attr_all, var_lst_all = read_attr_all(gageChosen)
    ind_var = list()
    for var in var_lst:
        ind_var.append(var_lst_all.index(var))
    id_lst_all = idLst
    c, ind_grid, ind2 = np.intersect1d(id_lst_all, usgs_id_lst, return_indices=True)
    temp = attr_all[ind_grid, :]
    out = temp[:, ind_var]
    return out


class DataframeGages2(Dataframe):
    def __init__(self, *, subset='All', t_range):
        self.rootDB = dirDB
        self.subset = subset
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict['lat']
            crd[:, 1] = gageDict['lon']
            self.crd = crd
        self.time = utils.time.tRange2Array(t_range)

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
            data = trans_norm(data, 'usgsFlow', statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_ts(self, *, var_lst=forcingLst, do_norm=True, rm_nan=True):
        """时间序列数据，主要是驱动数据读取 and choose data in the given time interval 及归一化处理"""
        if type(var_lst) is str:
            var_lst = [var_lst]
        # read ts forcing
        data = read_forcing(self.usgsId, var_lst)
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def get_data_const(self, *, var_lst=attrLstSel, do_norm=True, rm_nan=True):
        """属性数据读取及归一化处理, some attributes have time, which need to repeat in different periods in this function of
        'gages2.py' """
        if type(var_lst) is str:
            var_lst = [var_lst]
        data = read_attr(self.usgsId, var_lst)
        if do_norm is True:
            data = trans_norm(data, var_lst, statDict, to_norm=True)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data
