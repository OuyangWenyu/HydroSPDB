"""获取源数据，源数据不考虑格式，只是最原始所需下载的数据，先以gages数据集测试编写，后面其他数据集采用继承方式修改"""

# 数据类型包括：径流数据（从usgs下载），forcing数据（从daymet或者nldas下载），属性数据（从usgs属性表读取）
# 定义选择哪些源数据
import collections
import fnmatch
import json
import time
import os
from datetime import datetime, timedelta
import kaggle
import requests
from six.moves import urllib
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
import shutil
from app.common.default import init_path, init_data_param
from data.data_process import read_usge_gage
from urllib import parse
from hydroDL import utils


def download_small_zip(data_url, data_dir):
    """下载文件较小的zip文件并解压"""
    data_url_str = data_url.split('/')
    filename = parse.unquote(data_url_str[-1])
    filepath = os.path.join(data_dir, filename)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    filepath, _ = urllib.request.urlretrieve(data_url, filepath)

    with zipfile.ZipFile(filepath, 'r') as _zip:
        _zip.extractall(data_dir)


def download_small_file(data_url, temp_file):
    """根据url下载数据到temp_file中"""
    r = requests.get(data_url)
    with open(temp_file, 'w') as f:
        f.write(r.text)


def download_kaggle_file(name_of_dataset, path_download):
    """下载kaggle上的数据"""
    home_dir = os.environ['HOME']
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    print(home_dir)
    print(kaggle_dir)
    print(os.path.isdir(kaggle_dir))
    if not os.path.isdir(kaggle_dir):
        os.mkdir(os.path.join(home_dir, '.kaggle'))
    print(os.path.isdir(kaggle_dir))

    src = './kaggle.json'
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    dst = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.isfile(dst):
        print("copying file...")
        shutil.copy(src, dst)

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(name_of_dataset, path=path_download)


class SourceData(object):
    def __init__(self, url_attr, url_forcing, url_flow, dir_attr, dir_forcing, dir_flow, t_range, site_ids):
        self.t_range = t_range
        self.site_ids = site_ids
        self.url_attr = url_attr
        self.url_forcing = url_forcing
        self.url_flow = url_flow
        self.dir_attr = dir_attr
        self.dir_forcing = dir_forcing
        self.dir_flow = dir_flow

    def prepare_attr_data(self):
        """根据时间读取数据，没有的数据下载"""
        attr_dir = self.dir_attr
        if not os.path.isdir(attr_dir):
            os.mkdir(attr_dir)
        attr_url = self.url_attr
        download_small_zip(attr_url, attr_dir)

    def prepare_forcing_data(self):
        """如果没有给url或者查到没有数据，就只能报错了，提示需要手动下载"""
        if self.url_forcing is None:
            print("please read downloaded dataset directly")

    def prepare_flow_data(self, gage_fld_lst, dir_gage_flow, t_range):
        """检查数据是否齐全，不够的话进行下载，下载数据的时间范围要设置的大一些，这里暂时例子都是以1980-01-01到2015-12-31"""
        streamflow_dir = self.dir_flow
        streamflow_url = self.url_flow
        if not os.path.isdir(streamflow_dir):
            os.mkdir(streamflow_dir)
        dir_list = os.listdir(streamflow_dir)
        # 区域一共有18个，为了便于后续处理，把属于不同region的站点的文件放到不同的文件夹下面
        # 判断usgs_id_lst中没有对应径流文件的要从网上下载
        usgs_id_lst = gage_dict[gage_fld_lst[0]]
        for ind in range(len(usgs_id_lst)):
            # different hucs different directories
            huc_02 = gage_dict[gage_fld_lst[3]][ind]
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
                start_time_str = datetime.strptime(t_range[0], '%Y-%m-%d')
                end_time_str = datetime.strptime(t_range[1]) - timedelta(days=1)
                url = streamflow_url.format(usgs_id_lst[ind], start_time_str.year, start_time_str.month,
                                            start_time_str.day, end_time_str.year, end_time_str.month, end_time_str.day)

                # 存放的位置是对应HUC02区域的文件夹下
                temp_file = os.path.join(dir_huc_02, str(usgs_id_lst[ind]) + '.txt')
                download_small_file(url, temp_file)
                print("成功写入 " + temp_file + " 径流数据！")

    def read_gages_config(self, config_file):
        """读取gages数据项的配置，即各项所需数据的文件夹，返回到一个dict中"""
        dir_db = init_path(config_file)
        # USGS所有站点 file
        DIR_GAGE_ATTR = os.path.join(dir_db, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
        gage_id_file = os.path.join(DIR_GAGE_ATTR, 'conterm_basinid.txt')
        GAGE_SHAPE_DIR = os.path.join(dir_db, 'boundaries-shapefiles-by-aggeco')
        # 读取id文件，得到属性值
        gage_ids = pd.read_csv(gage_id_file)
        GAGE_FLD_LST = gage_ids.columns.tolist()
        DIR_GAGE_FLOW = self.dir_flow

        ATTR_LST = []
        for f_name in os.listdir(DIR_GAGE_ATTR):
            if f_name.startswith('conterm'):
                ATTR_LST.append(f_name)

        data_params = init_data_param(config_file)
        ATTR_chosen = data_params.get("varC")
        # gageDict = read_gage_info(gageField)

        # training time range
        tRangeTrain = data_params.get("tRange")

        # regions TODO: now just for one region
        REF_NONREF_REGIONS = ['bas_nonref_CntlPlains']
        REF_NONREF_REGIONS_SHPFILES_DIR = "gagesII_basin_shapefile_wgs84"
        GAGESII_POINTS_DIR = "gagesII_9322_point_shapefile"
        GAGESII_POINTS_FILE = "gagesII_9322_sept30_2011.shp"
        HUC4_SHP_DIR = "huc4"  # 后面判断该文件夹下是否有数据，没有的话调用download_kaggle_file从kaggle上下载
        HUC4_SHP_FILE = "HUC4.shp"
        return collections.OrderedDict()

    def read_gage_info(self, dir_db, region_shapefiles=None, ids_specific=None, screen_basin_area=None):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area

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
            # Read file using gpd.read_file()
            shapefile = os.path.join(GAGE_SHAPE_DIR, region_shapefiles[0] + '.shp')
            shape_data = gpd.read_file(shapefile)
            print(shape_data.columns)
            gages_id = shape_data['GAGE_ID'].values
            if screen_basin_area == 'HUC4':
                # using shapefile of all basins to check if their basin area satisfy the criteria
                # remove stations with catchment areas greater than the HUC4 basins in which they are located
                # firstly, get the HUC4 basin's area of the site
                print("screen big area basins")
                points_file = os.path.join(dirDB, GAGESII_POINTS_DIR, GAGESII_POINTS_FILE)
                polys_file = os.path.join(dirDB, HUC4_SHP_DIR, HUC4_SHP_FILE)
                join_points = spatial_join(points_file, polys_file)
                # get "AREASQKM" attribute data to filter
                join_points = join_points[join_points["DRAIN_SQKM"] < join_points["AREASQKM"]]
                gages_huc4_id = join_points['STAID'].values
                gages_id, ind1, ind2 = np.intersect1d(gages_id, gages_huc4_id, return_indices=True)
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

    def read_usgs(huc_02s, usgs_id_lst, t_range):
        """读取USGS的daily average 径流数据 according to id and time,
            首先判断哪些径流站点的数据已经读取并存入本地，如果没有，就从网上下载并读入txt文件。
        Parameter:
            gage_dict：站点 information
            t_range: must be time range for downloaded data
        Return：
            y: ndarray--各个站点的径流数据, 1d-axis: gages, 2d-axis: day
        """
        t_lst = utils.time.tRange2Array(t_range)
        nt = len(t_lst)
        t0 = time.time()
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            huc_02 = huc_02s[k]
            data_obs = read_usge_gage(huc_02, usgs_id_lst[k], t_range)
            y[k, :] = data_obs
        print("time of reading usgs streamflow: ", time.time() - t0)
        return y

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

    def read_forcing(usgs_id_lst, t_range, var_lst, dataset='daymet', regions=None):
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
            # to match the file of the given year
            data_dir = os.path.join(data_folder, dataset, regions[0])
            data_file = ''
            for f_name in os.listdir(data_dir):
                if fnmatch.fnmatch(f_name, dataset + '_*_mean_' + year + '.csv'):
                    print(f_name)
                    data_file = os.path.join(data_dir, f_name)
                    break
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
