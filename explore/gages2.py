"""read gages-ii data以计算统计值 and ready for model 的脚本代码，
some scripts for config of gages-ii datasets"""

# 读取GAGES-II数据需要指定文件路径、时间范围、属性类型、需要计算配置的项是forcing data。
# module variable
import json
import os

import pandas as pd

from data.input_data import cal_stat_all
from data.source_data import usgs_screen_streamflow, read_gage_info, read_usgs


def read_gages_config():
    """读取gages数据项的配置"""
    dirDB = pathGages2['DB']
    # USGS所有站点 file
    GAGE_FILE = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format',
                             'conterm_basinid.txt')
    GAGE_SHAPE_DIR = os.path.join(dirDB, 'boundaries-shapefiles-by-aggeco')
    # 读取id文件，得到属性值
    GAGE_FLD_LST = ['STAID', 'STANAME', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE', 'STATE', 'BOUND_SOURCE',
                    'HCDN-2009',
                    'HBN36', 'OLD_HCDN', 'NSIP_SENTINEL', 'FIPS_SITE', 'COUNTYNAME_SITE', 'NAWQA_SUID']
    # gageFldLst = camels.gageFldLst
    DIR_GAGE_FLOW = os.path.join(dirDB, 'gages_streamflow')
    DIR_GAGE_ATTR = os.path.join(dirDB, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
    # all attributes:
    # attrLstAll = os.listdir(DIR_GAGE_ATTR)
    # # 因为是对CONUS分析，所以只用conterm开头的
    # ATTR_LST = []
    # for attrLstAllTemp in attrLstAll:
    #     if 'conterm' in attrLstAllTemp:
    #         attrLstTemp = attrLstAllTemp[8:].lower()
    #         ATTR_LST.append(attrLstTemp)
    ATTR_STR_SEL = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams + attrHydroModOther
    # + attrLandscapePat + attrLC06Basin + attrLC06Mains100 + attrLC06Mains800 + attrLC06Rip100 + attrLCCrops + \
    # attrPopInfrastr + attrProtAreas

    # GAGES-II的所有站点 and all time, for first using of this code to download streamflow datasets
    tRange4DownloadData = [19800101, 20150101]  # 左闭右开
    tLstAll = utils.time.tRange2Array(tRange4DownloadData)
    # gageDict = read_gage_info(gageField)

    # training time range
    tRangeTrain = [19950101, 20000101]

    # regions
    # TODO: now just for one region
    REF_NONREF_REGIONS = ['bas_nonref_CntlPlains']
    REF_NONREF_REGIONS_SHPFILES_DIR = "gagesII_basin_shapefile_wgs84"
    GAGESII_POINTS_DIR = "gagesII_9322_point_shapefile"
    GAGESII_POINTS_FILE = "gagesII_9322_sept30_2011.shp"
    HUC4_SHP_DIR = "huc4"
    HUC4_SHP_FILE = "HUC4.shp"
    return


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
