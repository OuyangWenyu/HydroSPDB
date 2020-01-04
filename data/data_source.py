"""获取源数据，源数据不考虑格式，只是最原始所需下载的数据，先以gages数据集测试编写，后面其他数据集采用继承方式修改"""

# 数据类型包括：径流数据（从usgs下载），forcing数据（从daymet或者nldas下载），属性数据（从usgs属性表读取）
# 定义选择哪些源数据
import fnmatch
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from data.download_data import download_one_zip
from utils import *


class DataSource(object):
    """获取源数据的思路是：
    首先准备好属性文件，主要是从网上下载获取；
    然后读取配置文件及相关属性文件了解到模型计算的对象；
    接下来获取forcing数据和streamflow数据
    """

    def __init__(self, config_data, t_range):
        """read configuration of data source. 读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        self.data_config = config_data
        self.all_configs = config_data.read_data_config()
        # t_range: 训练数据还是测试数据，需要外部指定
        self.t_range = t_range
        self.prepare_attr_data()
        gage_dict, gage_fld_lst = self.read_site_info()
        self.prepare_forcing_data()
        self.prepare_flow_data(gage_dict, gage_fld_lst)
        # 一些后面常用的变量也在这里赋值到SourceData对象中
        self.gage_dict = gage_dict
        self.gage_fld_lst = gage_fld_lst

    def prepare_attr_data(self):
        """根据时间读取数据，没有的数据下载"""
        configs = self.all_configs
        data_dir = configs.get('root_dir')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        attr_url = configs.get('attr_url')
        download_one_zip(attr_url, data_dir)
        print("属性数据准备好了...")

    def read_site_info(self, ids_specific=None, screen_basin_area_huc4=True):
        """read basic information of sites"""
        pass

    def prepare_forcing_data(self):
        """DOWNLOAD forcing data from website"""
        pass

    def prepare_flow_data(self, gage_dict, gage_fld_lst):
        """download streamflow data"""
        pass

    def read_usge_gage(self, huc, usgs_id, t_lst, read_qc=False):
        """读取各个径流站的径流数据"""
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
            # TODO：我暂时把非平均值的径流数据的读取取出了，所以read_qc暂时没用
            qc_dict = {'A': 1, 'A:e': 2, 'M': 3}
            qc = np.array([qc_dict[x] for x in data_temp[4]])
        # 首先判断时间范围是否一致，读取的径流数据的时间序列和要读取的径流数据时间序列交集，获取径流数据，剩余没有数据的要读取的径流数据时间序列是nan
        nt = len(t_lst)
        out = np.full([nt], np.nan)
        # df中的date是字符串，转换为datetime，方可与tLst求交集
        df_date = data_temp['datetime']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
        if read_qc is True:
            out_qc = np.full([nt], np.nan)
            out_qc[ind2] = qc

        if read_qc is True:
            return out, out_qc
        else:
            return out

    @my_logger
    @my_timer
    def read_usgs(self):
        """读取USGS的daily average 径流数据 according to id and time,
            首先判断哪些径流站点的数据已经读取并存入本地，如果没有，就从网上下载并读入txt文件。
        Parameter:
            gage_dict：站点 information
            t_range: must be time range for downloaded data
        Return：
            y: ndarray--各个站点的径流数据, 1d-axis: gages, 2d-axis: day
        """
        gage_dict = self.gage_dict
        gage_fld_lst = self.gage_fld_lst
        t_range = self.t_range
        t_lst = hydro_time.t_range_days(t_range)
        nt = len(t_lst)
        usgs_id_lst = gage_dict[gage_fld_lst[0]]
        huc_02s = gage_dict[gage_fld_lst[3]]
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            huc_02 = huc_02s[k]
            data_obs = self.read_usge_gage(huc_02, usgs_id_lst[k], t_lst)
            y[k, :] = data_obs
        return y

    def usgs_screen_streamflow(self, streamflow, usgs_ids=None, time_range=None):
        """according to the criteria and its ancillary condition--thresh of streamflow data,
            choose appropriate ones from all usgs sites
            Parameters
            ----------
            streamflow : numpy ndarray -- all usgs sites(config)' data, its index are 'sites', its columns are 'day',
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
            usgs_screen(usgs, ["02349000","08168797"], [‘1995-01-01’,‘2015-01-01’])
        """
        kwargs = self.all_configs["flow_screen_param"]
        sites_chosen = np.zeros(streamflow.shape[0])
        # choose the given sites
        usgs_all_sites = self.gage_dict[self.gage_fld_lst[0]]
        if usgs_ids:
            sites_index = np.where(np.in1d(usgs_ids, usgs_all_sites))[0]
            sites_chosen[sites_index] = 1
        else:
            sites_index = np.arange(streamflow.shape[0])
            sites_chosen = np.ones(streamflow.shape[0])
        # choose data in given time range
        all_t_list = hydro_time.t_range_days(self.t_range)
        t_lst = all_t_list
        if time_range:
            # calculate the day length
            t_lst = hydro_time.t_range_days(time_range)
        ts, ind1, ind2 = np.intersect1d(all_t_list, t_lst, return_indices=True)
        # 取某几行的某几列数据稍微麻烦一点点
        streamflow_temp = streamflow[sites_index]  # 先取出想要的行数据
        usgs_values = streamflow_temp[:, ind1]  # 再取出要求的列数据

        for site_index in sites_index:
            # loop for every site
            runoff = usgs_values[site_index, :]
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
        usgs_out = usgs_values[np.where(sites_chosen > 0)]
        gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]

        return usgs_out, gages_chosen_id, ts

    def read_attr_all(self, gages_ids):
        """读取GAGES-II下的属性数据，目前是将用到的几个属性所属的那个属性大类下的所有属性的统计值都计算一下
        parameters:
            gages_ids:可以指定几个gages站点
        :return
            out：ndarray
        """
        dir_gage_attr = self.all_configs.get("gage_files_dir")
        dir_out = self.all_configs.get("out_dir")
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
        return out, var_lst, var_dict, f_dict

    def read_attr(self, usgs_id_lst, var_lst):
        """指定读取某些站点的某些属性"""
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all(usgs_id_lst)
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        out = attr_all[:, ind_var]
        return out, var_dict, f_dict

    @my_timer
    def read_forcing(self, usgs_id_lst, t_range_lst):
        """读取gagesII_forcing文件夹下的驱动数据(data processed from GEE)
        :return
        x: ndarray -- 1d-axis:gages, 2d-axis: day, 3d-axis: forcing vst
        """
        data_folder = os.path.join(self.all_configs.get("forcing_dir"))
        dataset = self.all_configs.get("forcing_type")
        var_lst = self.all_configs.get("forcing_chosen")
        regions = self.all_configs.get("regions")
        # different files for different years
        t_start_year = hydro_time.get_year(t_range_lst[0])
        t_end_year = hydro_time.get_year(t_range_lst[-1])
        # arange是左闭右开的，所以+1
        t_lst_years = np.arange(t_start_year, t_end_year + 1).astype(str)
        data_temps = pd.DataFrame()
        for year in t_lst_years:
            # to match the file of the given year
            data_file = ''
            for f_name in os.listdir(data_folder):
                # 首先判断是不是在给定的region内的，TODO 这里先用一个region
                region = regions[0].split("_")[-1]
                if fnmatch.fnmatch(f_name, dataset + '_' + region + '_mean_' + year + '.csv'):
                    print(f_name)
                    data_file = os.path.join(data_folder, f_name)
                    break
            data_temp = pd.read_csv(data_file, sep=',', dtype={'gage_id': str})
            frames_temp = [data_temps, data_temp]
            data_temps = pd.concat(frames_temp)
        # choose data in given time and sites. if there is no value for site in usgs_id_lst, just error(because every
        # site should have forcing). using dataframe mostly will make data type easy to handle with
        sites_forcing = data_temps.iloc[:, 0].values
        sites_index = [i for i in range(sites_forcing.size) if sites_forcing[i] in usgs_id_lst]
        data_sites_chosen = data_temps.iloc[sites_index, :]
        t_range_forcing = np.array(data_sites_chosen.iloc[:, 1].values.astype(str), dtype='datetime64[D]')
        t_index = [j for j in range(t_range_forcing.size) if t_range_forcing[j] in t_range_lst]
        data_chosen = data_sites_chosen.iloc[t_index, :]
        # when year is a leap year, only 365d will be provided by gee datasets. better to fill it with nan
        # number of days are different in different years, so reshape can't be used
        x = np.empty([len(usgs_id_lst), t_range_lst.size, len(var_lst)])
        for k in range(len(usgs_id_lst)):
            data_k = data_chosen[data_chosen['gage_id'] == usgs_id_lst[k]]
            out = np.full([t_range_lst.size, len(var_lst)], np.nan)
            # df中的date是字符串，转换为datetime，方可与tLst求交集
            df_date = data_k.iloc[:, 1]
            date = df_date.values.astype('datetime64[D]')
            c, ind1, ind2 = np.intersect1d(t_range_lst, date, return_indices=True)
            out[ind1, :] = data_k.iloc[ind2, 2:].values
            x[k, :, :] = out

        return x
