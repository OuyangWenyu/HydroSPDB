import argparse
import json
import os
from easydict import EasyDict as edict
import pandas as pd
import definitions
from data.download_data import download_google_drive, download_one_zip, download_small_file, download_excel
from datetime import datetime, timedelta

from utils import unzip_nested_zip

__C = edict()
cfg = __C  # type: edict()

# ----------------------- First part: mainly for data source  ------------------------
# This part should NOT be modified, unless you are clear about what you are doing
__C.DATASET = "gages"
# Project directory
__C.ROOT_DIR = os.path.join(definitions.ROOT_DIR, "example")

__C.DATA_PATH = os.path.join(__C.ROOT_DIR, 'data', __C.DATASET)
if not os.path.exists(__C.DATA_PATH):
    os.makedirs(__C.DATA_PATH)

# data config
__C.GAGES = edict()
# __C.GAGES.DOWNLOAD = True
__C.GAGES.DOWNLOAD = False
__C.GAGES.DOWNLOAD_FROM_OWEN = True
__C.GAGES.ARE_YOU_OWEN = True
__C.GAGES.DOWNLOAD_FROM_WEB = False

__C.GAGES.tRangeAll = ['1980-01-01', '2020-01-01']

__C.GAGES.forcingDir = os.path.join(__C.DATA_PATH, "basin_mean_forcing", "basin_mean_forcing")
__C.GAGES.forcingType = "daymet"
__C.GAGES.forcingDir = os.path.join(__C.GAGES.forcingDir, __C.GAGES.forcingType)

__C.GAGES.streamflowDir = os.path.join(__C.DATA_PATH, "gages_streamflow", "gages_streamflow")
__C.GAGES.streamflowUrl = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}"

__C.GAGES.attrDir = os.path.join(__C.DATA_PATH, "basinchar_and_report_sept_2011")
__C.GAGES.attrUrl = ["https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
                     "https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
                     "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
                     "https://www.sciencebase.gov/catalog/file/get/59692a64e4b0d1f9f05fbd39"]

# region shapefiles
__C.GAGES.gage_region_dir = os.path.join(__C.DATA_PATH, 'boundaries_shapefiles_by_aggeco',
                                         'boundaries-shapefiles-by-aggeco')
# point shapefile
__C.GAGES.gagesii_points_dir = os.path.join(__C.DATA_PATH, "gagesII_9322_point_shapefile")
__C.GAGES.gagesii_points_file = os.path.join(__C.GAGES.gagesii_points_dir, "gagesII_9322_sept30_2011.shp")
# HUC shapefile
__C.GAGES.huc4_shp_dir = os.path.join(__C.DATA_PATH, "wbdhu4-a-us-september2019-shpfile")
__C.GAGES.huc4_shp_file = os.path.join(__C.GAGES.huc4_shp_dir, "HUC4.shp")

# all USGS sites
__C.GAGES.gage_files_dir = os.path.join(__C.GAGES.attrDir, 'spreadsheets-in-csv-format')
__C.GAGES.gage_id_file = os.path.join(__C.GAGES.gage_files_dir, 'conterm_basinid.txt')

# GAGES-II time series dataset dir
__C.GAGES.gagests_dir = os.path.join(__C.DATA_PATH, "59692a64e4b0d1f9f05f")
__C.GAGES.population_file = os.path.join(__C.GAGES.gagests_dir, "Dataset8_Population-Housing",
                                         "Dataset8_Population-Housing", "PopulationHousing.txt")
__C.GAGES.wateruse_file = os.path.join(__C.GAGES.gagests_dir, "Dataset10_WaterUse", "Dataset10_WaterUse",
                                       "WaterUse_1985-2010.txt")

# NID database
__C.NID = edict()
__C.NID.NID_DIR = os.path.join(__C.DATA_PATH, "nid")
__C.NID.NID_FILE = os.path.join(__C.NID.NID_DIR, "NID2018_U.xlsx")
__C.NID.NID_URL = 'https://nid.sec.usace.army.mil/ords/NID_R.DOWNLOADFILE?InFileName={nidFile}'.format(
    nidFile="NID2018_U.xlsx")
# EPSG:4269 --  https://epsg.io/4269
__C.NID.NID_EPSG = 4269

# Search for the initial data
find_gages_data_path = True
if __C.GAGES.DOWNLOAD:
    find_gages_data_path = False
    if __C.GAGES.DOWNLOAD_FROM_OWEN and __C.GAGES.DOWNLOAD_FROM_WEB:
        raise RuntimeError("Don't download data by two ways at the same time!")

if not find_gages_data_path:
    if __C.GAGES.DOWNLOAD_FROM_OWEN:
        if __C.GAGES.ARE_YOU_OWEN:
            print("Downloading dataset from google drive ... "
                  "if there is any interruption when downloading, just rerun this script code again, it will continue."
                  "All files in google drive are zip files")
            # Firstly, move the creds file to the following directory manually
            client_secrets_file = os.path.join(definitions.ROOT_DIR, "data", "mycreds.txt")
            if not os.path.isfile(client_secrets_file):
                raise RuntimeError("Please put the credential file to the root directory. "
                                   "To generate it, please see: https://github.com/OuyangWenyu/aqualord/blob/master/CloudStor/googledrive.ipynb")
            google_drive_dir_name = "hydro-dl-reservoir-data"
            # google_drive_dir_name = "test"
            download_dir_name = __C.DATA_PATH
            download_google_drive(client_secrets_file, google_drive_dir_name, download_dir_name)
            # after downloading from google drive, unzip all files:
            print("unzip all files")
            entries = os.listdir(__C.DATA_PATH)
            for entry in entries:
                if os.path.isdir(entry):
                    continue
                zipfile_path = os.path.join(download_dir_name, entry)
                filename = entry[0:-4]
                unzip_dir = os.path.join(download_dir_name, filename)
                unzip_nested_zip(zipfile_path, unzip_dir)
        else:
            raise RuntimeError("If you have the data, please put them in the correct directories."
                               "Or else, Please connect with hust2014owen@gmail.com."
                               "Then set  __C.GAGES.DOWNLOAD = False")
    elif __C.GAGES.DOWNLOAD_FROM_WEB:
        print("Not all dataset could be downloaded from website directly, so I didn't test this part completely."
              "Hence, please be careful!")
        # download zip files
        [download_one_zip(attr_url, __C.DATA_PATH) for attr_url in __C.GAGES.attrUrl]
        # download NID file
        download_excel(__C.NID.NID_URL, __C.NID.NID_FILE)
        # download streamflow data from USGS website
        dir_gage_flow = __C.GAGES.streamflowDir
        streamflow_url = __C.GAGES.streamflowUrl
        t_download_range = __C.GAGES.tRangeAll
        if not os.path.isdir(dir_gage_flow):
            os.makedirs(dir_gage_flow)
        dir_list = os.listdir(dir_gage_flow)
        # 区域一共有18个，为了便于后续处理，把属于不同region的站点的文件放到不同的文件夹下面
        # 判断usgs_id_lst中没有对应径流文件的要从网上下载
        data_all = pd.read_csv(__C.GAGES.gage_id_file, sep=',', dtype={0: str})
        usgs_id_lst = data_all.iloc[:, 0].values.tolist()
        gage_fld_lst = data_all.columns.values
        for ind in range(len(usgs_id_lst)):
            # different hucs different directories
            huc_02 = data_all[gage_fld_lst[3]][ind]
            dir_huc_02 = str(huc_02)
            if dir_huc_02 not in dir_list:
                dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
                os.mkdir(dir_huc_02)
                dir_list = os.listdir(dir_gage_flow)
            dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
            file_list = os.listdir(dir_huc_02)
            file_usgs_id = str(usgs_id_lst[ind]) + ".txt"
            if file_usgs_id not in file_list:
                # 通过直接读取网页的方式获取数据，然后存入txt文件
                start_time_str = datetime.strptime(t_download_range[0], '%Y-%m-%d')
                end_time_str = datetime.strptime(t_download_range[1], '%Y-%m-%d') - timedelta(days=1)
                url = streamflow_url.format(usgs_id_lst[ind], start_time_str.year, start_time_str.month,
                                            start_time_str.day, end_time_str.year, end_time_str.month, end_time_str.day)

                # 存放的位置是对应HUC02区域的文件夹下
                temp_file = os.path.join(dir_huc_02, str(usgs_id_lst[ind]) + '.txt')
                download_small_file(url, temp_file)
                print("成功写入 " + temp_file + " 径流数据！")
    else:
        raise RuntimeError("Initial database is not found! Please download the data")

# ----------------------- Second part: some changeable configs------------------------
# data config
__C.GAGES.regions = ['bas_ref_all', 'bas_nonref_CntlPlains', 'bas_nonref_EastHghlnds', 'bas_nonref_MxWdShld',
                     'bas_nonref_NorthEast', 'bas_nonref_SECstPlain', 'bas_nonref_SEPlains', 'bas_nonref_WestMnts',
                     'bas_nonref_WestPlains', 'bas_nonref_WestXeric']

__C.GAGES.gageIdScreen = None
__C.GAGES.streamflowScreenParams = {'missing_data_ratio': 0, 'zero_value_ratio': 1}
__C.GAGES.attrScreenParams = None  # {'DOR': -0.02, 'dam_num': 0}

attrBasin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
attrLandcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
attrSoil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
attrHydro = ['STREAMS_KM_SQ_KM']
attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
attrPopInfrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
__C.GAGES.varC = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams + attrHydroModOther + attrPopInfrastr

__C.GAGES.varT = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

# sub experiment
__C.SUBSET = "basic"
__C.SUB_EXP = "exp11"
__C.TEMP_PATH = os.path.join(__C.ROOT_DIR, 'temp', __C.DATASET, __C.SUBSET, __C.SUB_EXP)
if not os.path.exists(__C.TEMP_PATH):
    os.makedirs(__C.TEMP_PATH)
__C.OUT_PATH = os.path.join(__C.ROOT_DIR, 'output', __C.DATASET, __C.SUBSET, __C.SUB_EXP)
if not os.path.exists(__C.OUT_PATH):
    os.makedirs(__C.OUT_PATH)

# computer config
__C.RANDOM_SEED = 1234
__C.CTX = 0
__C.TEST_EPOCH = 300
__C.TRAIN_MODE = True
__C.PUB_PLAN = None
__C.PLUS = None
__C.SPLIT_NUM = None
__C.DAM_PLAN = None

# model config
__C.MODEL = edict()
__C.MODEL.tRangeTrain = ['1990-01-01', '2000-01-01']
__C.MODEL.tRangeTest = ['2000-01-01', '2010-01-01']
__C.MODEL.doNorm = [True, True]
__C.MODEL.rmNan = [True, False]
__C.MODEL.daObs = 0
__C.MODEL.miniBatch = [100, 365]
__C.MODEL.nEpoch = 340
__C.MODEL.saveEpoch = 20
__C.MODEL.name = "CudnnLstmModel"
__C.MODEL.hiddenSize = 256
__C.MODEL.doReLU = True
__C.MODEL.loss = "RmseLoss"
__C.MODEL.prior = "gauss"

# cache data
__C.CACHE = edict()
# generate quick data?
__C.CACHE.GEN_QUICK_DATA = False
# QUICKDATA means already cache a binary version for the data source
__C.CACHE.QUICK_DATA = False
__C.CACHE.QUICK_DATA_DIR = os.path.join(__C.DATA_PATH, "quickdata")
__C.CACHE.DATA_DIR = os.path.join(__C.CACHE.QUICK_DATA_DIR, "conus-all_90-10_nan-0.0_00-1.0")
# 1 means, the data model of sub exp will be cached
__C.CACHE.STATE = False
__C.CACHE.HAS = False


def cmd():
    """input args from cmd"""
    parser = argparse.ArgumentParser(description='Train the CONUS model')
    parser.add_argument('--sub', dest='sub', help='subset and sub experiment', default=None, type=str)
    parser.add_argument('--ctx', dest='ctx',
                        help='Running Context -- gpu num. E.g `--ctx 0` means run code in the context of gpu 0',
                        type=int, default=None)
    parser.add_argument('--rs', dest='rs', help='random seed', default=None, type=int)
    parser.add_argument('--te', dest='te', help='test epoch', default=None, type=int)
    # There is something wrong with "bool", so I used 1 as True, 0 as False
    parser.add_argument('--train_mode', dest='train_mode', help='train or test', default=None, type=int)
    parser.add_argument('--train_epoch', dest='train_epoch', help='epoches of training period', default=None, type=int)
    parser.add_argument('--save_epoch', dest='save_epoch', help='save for every save_epoch epoches', default=None,
                        type=int)
    parser.add_argument('--regions', dest='regions',
                        help='There are 10 regions in GAGES-II. One is reference region, others are non-ref regions',
                        default=None, nargs='+')
    parser.add_argument('--gage_id', dest='gage_id', help='just select some sites',
                        default=None, nargs='+')
    parser.add_argument('--flow_screen', dest='flow_screen',
                        help='screen some sites according to their streamflow record',
                        default=None, type=json.loads)
    parser.add_argument('--attr_screen', dest='attr_screen',
                        help='screen some sites according to their attributes',
                        default=None, type=json.loads)
    parser.add_argument('--var_c', dest='var_c', help='types of attributes', default=None, nargs='+')
    parser.add_argument('--var_t', dest='var_t', help='types of forcing', default=None, nargs='+')
    parser.add_argument('--gen_quick_data', dest='gen_quick_data', help='do I generate quick data?', default=0,
                        type=int)
    parser.add_argument('--quick_data', dest='quick_data', help='Has quick data existed?', default=1, type=int)
    parser.add_argument('--cache_state', dest='cache_state', help='Does save the data model for the sub experiment?',
                        default=0, type=int)

    parser.add_argument('--pub_plan', dest='pub_plan',
                        help='4 plans:0-camels->non-camels 1-no dam->small dor;2:no dam->large dor;3:small_dor->large_dor',
                        default=None, type=int)
    parser.add_argument('--plus', dest='plus', help='Do training dataset contain data from both A and B?',
                        default=None, type=int)
    parser.add_argument('--split_num', dest='split_num', help='the split number when doing PUB test',
                        default=None, type=int)

    parser.add_argument('--dam_plan', dest='dam_plan',
                        help='combination of dam cases: 1--no dam+small dam;2--no dam+large dam;3--small dam+large dam',
                        default=None, type=int)
    args = parser.parse_args()
    return args


def update_cfg_item(cfg_file, new_args):
    print("update an item of config file")
    if new_args.sub is not None:
        subset, subexp = new_args.sub.split("/")
        cfg_file.SUBSET = subset
        cfg_file.SUB_EXP = subexp
        cfg_file.TEMP_PATH = os.path.join(cfg_file.ROOT_DIR, 'temp', cfg_file.DATASET, cfg_file.SUBSET,
                                          cfg_file.SUB_EXP)
        if not os.path.exists(cfg_file.TEMP_PATH):
            os.makedirs(cfg_file.TEMP_PATH)
        cfg_file.OUT_PATH = os.path.join(cfg_file.ROOT_DIR, 'output', cfg_file.DATASET, cfg_file.SUBSET,
                                         cfg_file.SUB_EXP)
        if not os.path.exists(cfg_file.OUT_PATH):
            os.makedirs(cfg_file.OUT_PATH)
    else:
        print("no update")


def update_cfg(cfg_file, new_args):
    print("update config file")
    if new_args.sub is not None:
        update_cfg_item(cfg_file, new_args)
    if new_args.ctx is not None:
        cfg_file.CTX = new_args.ctx
    if new_args.rs is not None:
        cfg_file.RANDOM_SEED = new_args.rs
    if new_args.te is not None:
        cfg_file.TEST_EPOCH = new_args.te
    if new_args.train_mode is not None:
        if new_args.train_mode > 0:
            cfg_file.TRAIN_MODE = True
        else:
            cfg_file.TRAIN_MODE = False
    if new_args.regions is not None:
        cfg_file.GAGES.regions = new_args.regions
    if new_args.gage_id is not None:
        cfg_file.GAGES.gageIdScreen = new_args.gage_id
    if new_args.flow_screen is not None:
        cfg_file.GAGES.streamflowScreenParams = new_args.flow_screen
    if new_args.attr_screen is not None:
        cfg_file.GAGES.attrScreenParams = new_args.attr_screen
    if new_args.var_c is not None:
        cfg_file.GAGES.varC = new_args.var_c
    if new_args.var_t is not None:
        cfg_file.GAGES.varT = new_args.var_t
    if new_args.train_epoch is not None:
        cfg_file.MODEL.nEpoch = new_args.train_epoch
    if new_args.save_epoch is not None:
        cfg_file.MODEL.saveEpoch = new_args.save_epoch
    if new_args.gen_quick_data is not None:
        if new_args.gen_quick_data > 0:
            cfg_file.CACHE.GEN_QUICK_DATA = True
        else:
            cfg_file.CACHE.GEN_QUICK_DATA = False
    if new_args.quick_data is not None:
        if new_args.quick_data > 0:
            cfg_file.CACHE.QUICK_DATA = True
        else:
            cfg_file.CACHE.QUICK_DATA = False
    if new_args.cache_state is not None:
        if new_args.cache_state > 0:
            cfg_file.CACHE.STATE = True
        else:
            cfg_file.CACHE.STATE = False
    if new_args.pub_plan is not None:
        cfg_file.PUB_PLAN = new_args.pub_plan
    if new_args.plus is not None:
        cfg_file.PLUS = new_args.plus
    if new_args.split_num is not None:
        cfg_file.SPLIT_NUM = new_args.split_num
    if new_args.dam_plan is not None:
        cfg_file.DAM_PLAN = new_args.dam_plan
