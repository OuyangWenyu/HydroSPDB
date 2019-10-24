"""read gages-ii data"""

# 读取GAGES-II数据，需要指定文件路径、时间范围、属性类型、需要计算配置的项是forcing data。
# module variable
import os

import pandas as pd

from hydroDL import pathGages2
from hydroDL import utils

dirDB = pathGages2['DB']
tRange = [19800101, 20150101]
tLst = utils.time.tRange2Array(tRange)
nt = len(tLst)
# 671个流域的forcing值需要重新计算，但是训练先用着671个流域，可以先用CAMELS的计算。
forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
# gages的attributes可以先按照CAMELS的这几项去找，因为使用了forcing数据，因此attributes里就没用气候的数据，因为要进行预测，所以也没用水文的。
# land cover部分：forest_frac对应FORESTNLCD06；lai没有，这里暂时用所有forest的属性；land_cover暂时用除人为种植之外的其他所有属性。
# soil：soil_depth相关的有：ROCKDEPAVE；soil_porosity类似的可能是：AWCAVE；soil_conductivity可能相关的：PERMAVE；max_water_content没有，暂时用RFACT
# geology在GAGES-II中一共两类，来自两个数据源，用第一种，
attrLstSel = [
    'ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM',
    'FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06',
    'WOODYWETNLCD06', 'EMERGWETNLCD06',
    'ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT',
    'GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE'
]


# 然后根据配置读取所需的gages-ii站点信息
def read_gage_info(dir_db):
    """读取gages-ii站点及流域基本location等信息"""
    gage_file = os.path.join(dir_db, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format',
                             'conterm_basinid.txt')
    # 数据从第二行开始，因此跳过第一行。
    data = pd.read_csv(gage_file, sep=',', header=0)
    # header gives some troubles. Skip and hardcode
    field_lst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
    out = dict()
    for s in field_lst:
        if s is 'name':
            out[s] = data[field_lst.index(s)].values.tolist()
        else:
            out[s] = data[field_lst.index(s)].values
    return out


# module variable
gageDict = read_gage_info(dirDB)

# 为了便于后续的归一化计算，这里需要计算流域attributes、forcings和streamflows统计值。
# module variable
statFile = os.path.join(dirDB, 'Statistics.json')
if not os.path.isfile(statFile):
    calStatAll()
with open(statFile, 'r') as fp:
    statDict = json.load(fp)
