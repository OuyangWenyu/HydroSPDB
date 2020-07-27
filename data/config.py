import os

from easydict import EasyDict as edict

import definitions

__C = edict()
cfg = __C  # type: edict()

# Random seed
__C.SEED = 1234

# Dataset name
# Used by symbols factories who need to adjust for different
# inputs based on dataset used. Should be set by the script.
__C.DATASET = "gages"

# Project directory
__C.ROOT_DIR = os.path.join(definitions.ROOT_DIR, "example")

__C.DATA_PATH = os.path.join(__C.ROOT_DIR, 'data', __C.DATASET)
if not os.path.exists(__C.DATA_PATH):
    os.makedirs(__C.DATA_PATH)
__C.TEMP_PATH = os.path.join(__C.ROOT_DIR, 'temp', __C.DATASET)
if not os.path.exists(__C.TEMP_PATH):
    os.makedirs(__C.TEMP_PATH)
__C.OUT_PATH = os.path.join(__C.ROOT_DIR, 'output', __C.DATASET)
if not os.path.exists(__C.OUT_PATH):
    os.makedirs(__C.OUT_PATH)

# data config
__C.GAGES.tRangeAll = ['1980-01-01', '2020-01-01']
__C.GAGES.regions = ['bas_nonref_CntlPlains', 'bas_nonref_EastHghlnds']

__C.GAGES.forcingDir = os.path.join(__C.DATA_PATH, "basin_mean_forcing")
__C.GAGES.forcingType = "daymet"
__C.GAGES.forcingDir = os.path.join(__C.GAGES.forcingDir, __C.GAGES.forcingType)
if not os.path.isdir(__C.GAGES.forcingDir):
    os.mkdir(__C.GAGES.forcingDir)

__C.GAGES.forcingUrl = None
__C.GAGES.varT = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

__C.GAGES.streamflowDir = os.path.join(__C.DATA_PATH, "gages_streamflow")
__C.GAGES.streamflowUrl = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}"
__C.GAGES.gageIdScreen = ['03144816', '03145000', '03156000', '03157000', '03157500', '03219500', '03220000',
                          '03221000', '03223000', '03224500', '03225500', '03226800', '02383000', '02383500',
                          '02384500', '02385170', '02385500', '02385800', '02387000', '02387500', '02387600',
                          '02388300', '02388320', '02388350', '02388500']
__C.GAGES.streamflowScreenParams = {'missing_data_ratio': 0, 'zero_value_ratio': 1}

__C.GAGES.attrDir = os.path.join(__C.DATA_PATH, "basinchar_and_report_sept_2011")
__C.GAGES.attrUrl = ["https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
                     "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
                     "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
                     "https://water.usgs.gov/GIS/dsdl/mainstem_line_covers.zip"]
attrBasin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
attrLandcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
attrSoil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
attrHydro = ['STREAMS_KM_SQ_KM']
attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
attrLandscapePat = ['FRAGUN_BASIN']
attrLC06Basin = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06']
attrPopInfrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
attrProtAreas = ['PADCAT1_PCT_BASIN', 'PADCAT2_PCT_BASIN']
__C.GAGES.varC = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams

# gages data path
gages_data_paths = [os.path.join(__C.DATA_PATH, )]
# region文件夹
gage_region_dir = os.path.join(__C.DATA_PATH, 'boundaries_shapefiles_by_aggeco', 'boundaries-shapefiles-by-aggeco')
# 站点的point文件文件夹
gagesii_points_file = os.path.join(__C.DATA_PATH, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp")
# 调用download_kaggle_file从kaggle上下载,
huc4_shp_dir = os.path.join(__C.DATA_PATH, "huc4")
huc4_shp_file = os.path.join(huc4_shp_dir, "HUC4.shp")
kaggle_src = definitions.KAGGLE_FILE
name_of_dataset = "owenyy/wbdhu4-a-us-september2019-shpfile"
# download_kaggle_file(kaggle_src, name_of_dataset, huc4_shp_dir, huc4_shp_file)

# USGS所有站点的文件，gages文件夹下载下来之后文件夹都是固定的
gage_files_dir = os.path.join(__C.GAGES.attrDir, 'spreadsheets-in-csv-format')
gage_id_file = os.path.join(gage_files_dir, 'conterm_basinid.txt')

# GAGES-II time series dataset dir
gagests_dir = os.path.join(__C.DATA_PATH, "59692a64e4b0d1f9f05f")
population_file = os.path.join(gagests_dir, "Dataset8_Population-Housing", "Dataset8_Population-Housing",
                               "PopulationHousing.txt")
wateruse_file = os.path.join(gagests_dir, "Dataset10_WaterUse", "Dataset10_WaterUse", "WaterUse_1985-2010.txt")

# Search for the initial data
find_gages_data_path = False
for ele in gages_data_paths:
    if os.path.exists(ele):
        find_gages_data_path = True
        __C.HKO_PNG_PATH = ele
        break
if not find_gages_data_path:
    raise RuntimeError("Initial database is not found! You can download the data using `bash download_gages.bash`")

__C.MODEL.tRangeTrain = ['1995-01-01', '1997-01-01']
__C.MODEL.tRangeTest = ['1997-01-01', '1999-01-01']
__C.MODEL.doNorm = [True, True]
__C.MODEL.rmNan = [True, False]
__C.MODEL.daObs = 0
__C.MODEL.miniBatch = [100, 30]
__C.MODEL.nEpoch = 100
__C.MODEL.saveEpoch = 10
__C.MODEL.name = "CudnnLstmModel"
__C.MODEL.hiddenSize = 256
__C.MODEL.doReLU = True
__C.MODEL.loss = "RmseLoss"
__C.MODEL.prior = "gauss"

# cache data
__C.HKO_VALID_DATETIME_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'valid_datetime.pkl')
__C.HKO_SORTED_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'sorted_day.pkl')
__C.HKO_RAINY_TRAIN_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_train_days.txt')
__C.HKO_RAINY_VALID_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_valid_days.txt')
__C.HKO_RAINY_TEST_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_test_days.txt')
