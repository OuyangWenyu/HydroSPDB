"""获取源数据，源数据不考虑格式，只是最原始所需下载的数据，先以gages数据集测试编写，后面其他数据集采用继承方式修改"""

# 数据类型包括：径流数据（从usgs下载），forcing数据（从daymet或者nldas下载），属性数据（从usgs属性表读取）
# 定义选择哪些源数据
import fnmatch
import json
import time
import os
from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from utils import *
from data.download_data import download_small_zip, download_small_file, download_google_drive
from data.read_config import read_gages_config


class SourceData(object):
    """获取源数据的思路是：
    首先准备好属性文件，主要是从网上下载获取；
    然后读取配置文件及相关属性文件了解到模型计算的对象；
    接下来获取forcing数据和streamflow数据
    """

    def __init__(self, config_file, t_range):
        """读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        gages_config = read_gages_config(config_file)
        self.all_configs = gages_config
        # t_range: 训练数据还是测试数据，需要外部指定
        self.t_range = t_range
        self.prepare_attr_data()
        gage_dict, gage_fld_lst = self.read_gage_info()
        self.prepare_forcing_data()
        self.prepare_flow_data(gage_dict, gage_fld_lst)
        # 一些后面常用的变量也在这里赋值到SourceData对象中
        self.gage_dict = gage_dict
        self.gage_fld_lst = gage_fld_lst

    def prepare_attr_data(self):
        """根据时间读取数据，没有的数据下载"""
        configs = self.all_configs
        attr_dir = configs.get('attr_dir')
        if not os.path.isdir(attr_dir):
            os.mkdir(attr_dir)
        attr_url = configs.get('attr_url')
        download_small_zip(attr_url, attr_dir)

    def read_gage_info(self, ids_specific=None, screen_basin_area_huc4=False):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area

        Parameter:
            ids_specific： given sites' ids
            screen_basin_area_huc4: 是否取出流域面积大于等于所处HUC流域的面积的流域
        Return：
            各个站点的attibutes in basinid.txt
        """
        # 数据从第二行开始，因此跳过第一行。
        gage_id_file = self.all_configs.get("gage_id_file")
        points_file = self.all_configs.get("gage_point_file")
        huc4_shp_file = self.all_configs.get("huc4_shp_file")
        gage_region_dir = self.all_configs.get("gage_region_dir")
        region_shapefiles = self.all_configs.get("regions")
        data = pd.read_csv(gage_id_file, sep=',', header=None, skiprows=1, dtype={0: str})
        gage_fld_lst = data.columns.values
        out = dict()
        if len(region_shapefiles):
            # read sites from shapefile of region, get id from it.
            # Read file using gpd.read_file() TODO:多个regins情况还未完成
            shapefile = os.path.join(gage_region_dir, region_shapefiles[0] + '.shp')
            shape_data = gpd.read_file(shapefile)
            print(shape_data.columns)
            gages_id = shape_data['GAGE_ID'].values
            if screen_basin_area_huc4:
                # using shapefile of all basins to check if their basin area satisfy the criteria
                # remove stations with catchment areas greater than the HUC4 basins in which they are located
                # firstly, get the HUC4 basin's area of the site
                print("screen big area basins")
                join_points = spatial_join(points_file, huc4_shp_file)
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
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data[gage_fld_lst.index(s)].values.tolist()
            else:
                out[s] = data[gage_fld_lst.index(s)].values
        return out, gage_fld_lst

    def prepare_forcing_data(self):
        """如果没有给url或者查到没有数据，就只能报错了，提示需要手动下载. 可以使用google drive下载数据"""
        url_forcing = self.all_configs.get("forcing_url")
        if url_forcing is None:
            print("Downloading dataset from google drive directly...")
        # 个人定义的：google drive上的forcing数据文件夹名和forcing类型一样的
        dir_name = self.all_configs.get("forcing_type")
        download_dir_name = os.path.join(self.all_configs.get("forcing_dir"), dir_name)
        if not os.path.isdir(download_dir_name):
            os.mkdir(download_dir_name)
        # 然后下载数据到这个文件夹下，这里从google drive下载数据
        download_google_drive(dir_name, download_dir_name)

    def prepare_flow_data(self, gage_dict, gage_fld_lst):
        """检查数据是否齐全，不够的话进行下载，下载数据的时间范围要设置的大一些，这里暂时例子都是以1980-01-01到2015-12-31
        parameters:
            gage_dict: read_gage_info返回值--out
            gage_dict: read_gage_info返回值--gage_fld_lst
        """
        dir_gage_flow = self.all_configs.get("flow_dir")
        streamflow_url = self.all_configs.get("flow_url")
        t_range = self.t_range
        if not os.path.isdir(dir_gage_flow):
            os.mkdir(dir_gage_flow)
        dir_list = os.listdir(dir_gage_flow)
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

    def read_usge_gage(self, huc, usgs_id, t_range, read_qc=False):
        """读取各个径流站的径流数据"""
        print(usgs_id)
        dir_gage_flow = self.all_configs.get("flow_dir")
        # 首先找到要读取的那个txt
        usgs_file = os.path.join(dir_gage_flow, str(huc), usgs_id + '.txt')
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
        t_lst = hydro_time.t_range_days(t_range)
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

    def read_usgs(self, gage_dict, gage_fld_lst, t_range):
        """读取USGS的daily average 径流数据 according to id and time,
            首先判断哪些径流站点的数据已经读取并存入本地，如果没有，就从网上下载并读入txt文件。
        Parameter:
            gage_dict：站点 information
            t_range: must be time range for downloaded data
        Return：
            y: ndarray--各个站点的径流数据, 1d-axis: gages, 2d-axis: day
        """
        t_lst = hydro_time.t_range_days(t_range)
        nt = len(t_lst)
        t0 = time.time()
        usgs_id_lst = gage_dict[gage_fld_lst[0]]
        huc_02s = gage_dict[gage_fld_lst[3]]
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            huc_02 = huc_02s[k]
            data_obs = self.read_usge_gage(huc_02, usgs_id_lst[k], t_range)
            y[k, :] = data_obs
        print("time of reading usgs streamflow: ", time.time() - t0)
        return y

    def usgs_screen_streamflow(self, usgs, usgs_ids=None, time_range=None, **kwargs):
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
            t_lst = hydro_time.t_range_days(time_range)
        else:
            time_range = self.t_range
            t_lst = hydro_time.t_range_days(time_range)
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

    def read_attr_all(self, gages_ids):
        """读取GAGES-II下的属性数据，目前是将用到的几个属性所属的那个属性大类下的所有属性的统计值都计算一下
        parameters:
            gages_ids:可以指定几个gages站点
        """
        dir_gage_attr = self.all_configs.get("attr_dir")
        dir_db = self.all_configs.get("root_dir")
        f_dict = dict()  # factorize dict
        # 每个key-value对是一个文件（str）下的所有属性（list）
        var_dict = dict()
        # 所有属性放在一起
        var_lst = list()
        out_lst = list()
        # 读取所有属性，直接按类型判断要读取的文件名
        var_des = pd.read_csv(os.path.join(dir_gage_attr, 'variable_descriptions.txt'), sep=',')
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
            data_file = os.path.join(dir_gage_attr, 'conterm_' + key + '.txt')
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
        file_name = os.path.join(dir_db, 'dictFactorize.json')
        with open(file_name, 'w') as fp:
            json.dump(f_dict, fp, indent=4)
        file_name = os.path.join(dir_db, 'dictAttribute.json')
        with open(file_name, 'w') as fp:
            json.dump(var_dict, fp, indent=4)
        return out, var_lst

    def read_attr(self, usgs_id_lst, var_lst):
        """指定读取某些站点的某些属性"""
        attr_all, var_lst_all = self.read_attr_all(usgs_id_lst)
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        out = attr_all[:, ind_var]
        return out

    def read_forcing(self, usgs_id_lst, t_range, var_lst, regions):
        """读取gagesII_forcing文件夹下的驱动数据(data processed from GEE)
        :return
        x: ndarray -- 1d-axis:gages, 2d-axis: day, 3d-axis: forcing vst
        """
        t0 = time.time()
        data_folder = os.path.join(self.all_configs.get("root_dir"), self.all_configs.get("forcing_dir"))
        dataset = self.all_configs.get("forcing_type")
        forcing_lst = self.all_configs.get("varT")
        # different files for different years
        t_start = str(t_range[0])[0:4]
        t_end = str(t_range[1])[0:4]
        t_lst_chosen = hydro_time.t_range_days(t_range)
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
            out = np.full([t_lst_chosen.size, len(forcing_lst)], np.nan)
            # df中的date是字符串，转换为datetime，方可与tLst求交集
            df_date = data_k.iloc[:, 1]
            date = df_date.values.astype('datetime64[D]')
            c, ind1, ind2 = np.intersect1d(t_lst_chosen, date, return_indices=True)
            out[ind1, :] = data_k.iloc[ind2, 2:].values
            x[k, :, :] = out

        print("time of reading usgs forcing data", time.time() - t0)
        return x
