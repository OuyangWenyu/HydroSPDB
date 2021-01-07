import os
from datetime import datetime, timedelta
import geopandas as gpd
from data import *
from data.download_data import download_one_zip, download_google_drive, download_small_file
from utils import *
import fnmatch
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from utils.hydro_math import is_any_elem_in_a_lst
from utils.hydro_util import hydro_logger


class GagesSource(DataSource):
    def __init__(self, config_data, t_range, screen_basin_area_huc4=True):
        super().__init__(config_data, t_range, screen_basin_area_huc4)

    @classmethod
    def choose_some_basins_multi_crit(cls, config_data, t_range, screen_basin_area_huc4=False, **kwargs):
        """choose some basins according to given conditions"""
        data_source = cls(config_data, t_range, screen_basin_area_huc4=screen_basin_area_huc4)
        sites = data_source.gage_dict["STAID"]
        for key, value in kwargs.items():
            d = {"screen_basin_area_huc4": screen_basin_area_huc4, key: value}
            data_source_temp = cls.choose_some_basins(config_data, t_range, **d)
            sites_id_temp = data_source_temp.all_configs["flow_screen_gage_id"]
            sites = np.intersect1d(sites, sites_id_temp)
        data_source.all_configs["flow_screen_gage_id"] = sites
        return data_source

    @classmethod
    def choose_some_basins(cls, config_data, t_range, **kwargs):
        """choose some basins according to given condition, different conditions but only one for once"""
        screen_basin_area_huc4 = True
        for screen_basin_area_huc4_key in kwargs:
            if screen_basin_area_huc4_key == "screen_basin_area_huc4":
                screen_basin_area_huc4 = kwargs[screen_basin_area_huc4_key]
                break
        new_data_source = cls(config_data, t_range, screen_basin_area_huc4=screen_basin_area_huc4)
        for criteria in kwargs:
            if criteria == "basin_area":
                new_data_source.small_basins_chosen(kwargs[criteria])
            elif criteria == "sites_id":
                if not (all(x < y for x, y in zip(kwargs[criteria], kwargs[criteria][1:]))):
                    kwargs[criteria].sort()
                assert type(kwargs[criteria]) == list
                if new_data_source.all_configs["flow_screen_gage_id"] is not None:
                    chosen_id = (np.intersect1d(np.array(kwargs[criteria]),
                                                new_data_source.all_configs["flow_screen_gage_id"])).tolist()
                    assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
                new_data_source.all_configs["flow_screen_gage_id"] = kwargs[criteria]
            elif criteria == "DOR":
                new_data_source.dor_reservoirs_chosen(kwargs[criteria])
            elif criteria == 'dam_num':
                new_data_source.dam_num_chosen(kwargs[criteria])
            elif criteria == 'major_dam_num':
                new_data_source.major_dams_chosen(kwargs[criteria])
            elif criteria == 'ref':
                new_data_source.ref_or_nonref_chosen(kwargs[criteria])
            elif criteria == 'STORAGE':
                new_data_source.storage_reservors_chosen(kwargs[criteria])
            elif criteria == 'ecoregion':
                new_data_source.ecoregion_chosen(kwargs[criteria])
            elif criteria == 'diversion':
                new_data_source.diversion_chosen(kwargs[criteria])
        return new_data_source

    def diversion_chosen(self, is_divert=True):
        diversion_strs = ["diversion", "divert"]
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
        data_attr = self.read_attr_origin(usgs_id, attr_lst)
        diversion_strs_lower = [elem.lower() for elem in diversion_strs]
        data_attr0_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[0]])
        data_attr1_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[1]])
        data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
        if is_divert:
            chosen_id = [usgs_id[i] for i in range(len(usgs_id)) if
                         is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True)]
        else:
            chosen_id = [usgs_id[i] for i in range(len(usgs_id)) if
                         not is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True)]
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def ecoregion_chosen(self, ecoregion):
        assert type(ecoregion) == tuple
        assert ecoregion[0] in ["ECO2_CODE", "ECO3_CODE"]
        if ecoregion[0] == "ECO2_CODE":
            assert ecoregion[1] in [5.2, 5.3, 6.2, 7.1, 8.1, 8.2, 8.3, 8.4, 8.5, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2,
                                    10.4, 11.1, 12.1, 13.1]
            attr_name = "ECO2_BAS_DOM"
        else:
            assert ecoregion[1] in np.arange(1, 85)
            attr_name = "ECO3_BAS_DOM"
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        attr_lst = [attr_name]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        eco_names = data_attr[:, 0]
        chosen_id = [usgs_id[i] for i in range(eco_names.size) if eco_names[i] == ecoregion[1]]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def storage_reservors_chosen(self, storage=None):
        """choose basins of specified normal storage range"""
        if storage is None:
            storage = [0, 50]
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        # megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
        attr_lst = ["STOR_NOR_2009"]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        nor_storage = data_attr[:, 0]
        storage_lower = storage[0]
        storage_upper = storage[1]
        chosen_id = [usgs_id[i] for i in range(nor_storage.size) if storage_lower <= nor_storage[i] < storage_upper]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def dam_num_chosen(self, dam_num=0):
        """choose basins of dams"""
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        attr_lst = ["NDAMS_2009"]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        if type(dam_num) == list:
            chosen_id = [usgs_id[i] for i in range(data_attr.size) if dam_num[0] <= data_attr[:, 0][i] < dam_num[1]]
        else:
            chosen_id = [usgs_id[i] for i in range(data_attr.size) if data_attr[:, 0][i] == dam_num]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def ref_or_nonref_chosen(self, ref="Ref"):
        assert ref in ["Ref", "Non-ref"]
        if ref == "Ref":
            ref_num = 1
        else:
            ref_num = 0
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        attr_lst = ["CLASS"]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        chosen_id = [usgs_id[i] for i in range(data_attr.size) if data_attr[:, 0][i] == ref_num]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def major_dams_chosen(self, major_dam_num=0):
        """choose basins of major dams"""
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        attr_lst = ["MAJ_NDAMS_2009"]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        if type(major_dam_num) == list:
            chosen_id = [usgs_id[i] for i in range(data_attr.size) if
                         major_dam_num[0] <= data_attr[:, 0][i] < major_dam_num[1]]
        else:
            chosen_id = [usgs_id[i] for i in range(data_attr.size) if data_attr[:, 0][i] == major_dam_num]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def dor_reservoirs_chosen(self, dor_chosen):
        """choose basins of small DOR(calculated by NOR_STORAGE/RUNAVE7100)"""
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        usgs_id = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        # mm/year 1-km grid,  megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
        # attr_lst = ["RUNAVE7100", "STOR_NID_2009"]
        attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
        data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        run_avg = data_attr[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
        nor_storage = data_attr[:, 1] * 1000  # m^3
        dors = nor_storage / run_avg
        if dor_chosen < 0:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] < -dor_chosen]
        else:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] >= dor_chosen]
        if self.all_configs["flow_screen_gage_id"] is not None:
            chosen_id = (np.intersect1d(np.array(chosen_id), self.all_configs["flow_screen_gage_id"])).tolist()
            assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
        self.all_configs["flow_screen_gage_id"] = chosen_id

    def small_basins_chosen(self, basin_area):
        """choose small basins"""
        # set self.all_configs["flow_screen_gage_id"]
        all_points_file = self.all_configs.get("gage_point_file")
        all_points = gpd.read_file(all_points_file)
        all_points_chosen = all_points[all_points["DRAIN_SQKM"] < basin_area]
        small_gages_chosen_id = all_points_chosen['STAID'].values
        # if arrays are not ascending order, np.intersect1d can't be used, because it will resort the arrays
        assert (all(x < y for x, y in zip(small_gages_chosen_id, small_gages_chosen_id[1:])))
        assert (all(x < y for x, y in
                    zip(self.all_configs["flow_screen_gage_id"], self.all_configs["flow_screen_gage_id"][1:])))
        if self.all_configs["flow_screen_gage_id"]:  # TODO: check
            small_gages_chosen_id, ind1, ind2 = np.intersect1d(self.all_configs["flow_screen_gage_id"],
                                                               small_gages_chosen_id, return_indices=True)
        self.all_configs["flow_screen_gage_id"] = small_gages_chosen_id.tolist()

    def read_site_info(self, screen_basin_area_huc4=True):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area

        Parameter:
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
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        gage_fld_lst = data_all.columns.values
        out = dict()
        if screen_basin_area_huc4:
            # using shapefile of all basins to check if their basin area satisfy the criteria
            # remove stations with catchment areas greater than the HUC4 basins in which they are located
            # firstly, get the HUC4 basin's area of the site
            join_points_all = spatial_join(points_file, huc4_shp_file)
            # get "AREASQKM" attribute data to filter
            join_points = join_points_all[join_points_all["DRAIN_SQKM"] < join_points_all["AREASQKM"]]
            join_points_sorted = join_points.sort_values(by="STAID")
            gages_huc4_id = join_points_sorted['STAID'].values
        # read sites from shapefile of region, get id from it.
        shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                      region_shapefiles]
        data = pd.DataFrame()
        df_id_region = data_all.iloc[:, 0].values
        assert (all(x < y for x, y in zip(df_id_region, df_id_region[1:])))
        if len(shapefiles) == 10:  # there are 10 regions in GAGES-II dataset in all
            hydro_logger.debug("all regions included, CONUS\n")
            if screen_basin_area_huc4:
                assert (all(x < y for x, y in zip(gages_huc4_id, gages_huc4_id[1:])))
                c, ind1, ind2 = np.intersect1d(df_id_region, gages_huc4_id, return_indices=True)
                data = data_all.iloc[ind1, :]
                data_all = data.sort_values(by="STAID")
        else:
            for shapefile in shapefiles:
                shape_data = gpd.read_file(shapefile)
                shape_data_sort = shape_data.sort_values(by="GAGE_ID")
                gages_id = shape_data_sort['GAGE_ID'].values
                assert (all(x < y for x, y in zip(gages_id, gages_id[1:])))
                if screen_basin_area_huc4:
                    gages_id = np.intersect1d(gages_id, gages_huc4_id)
                c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
                assert (all(x < y for x, y in zip(ind1, ind1[1:])))
                data = pd.concat([data, data_all.iloc[ind1, :]])
            # after screen for every regions, resort the dataframe by sites_id
            data_all = data.sort_values(by="STAID")

        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data_all[s].values.tolist()
            else:
                out[s] = data_all[s].values
        return out, gage_fld_lst

    def read_usge_gage(self, huc, usgs_id, t_lst):
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
        # 原数据的列名并不好用，这里修改
        columns_names = df_flow.columns.tolist()
        columns_flow = []
        columns_flow_cd = []
        for column_name in columns_names:
            # 00060 means "discharge"，00003 represents "mean value"
            # one special case： 126801       00060     00003     Discharge, cubic feet per second (Mean) and
            # 126805       00060     00003     Discharge, cubic feet per second (Mean), PUBLISHED
            # Both are mean values, here I will choose the column with more records
            if '_00060_00003' in column_name and '_00060_00003_cd' not in column_name:
                columns_flow.append(column_name)
        for column_name in columns_names:
            if '_00060_00003_cd' in column_name:
                columns_flow_cd.append(column_name)

        if len(columns_flow) > 1:
            print("there are some columns for flow, choose one\n")
            df_date_temp = df_flow['datetime']
            date_temp = pd.to_datetime(df_date_temp).values.astype('datetime64[D]')
            c_temp, ind1_temp, ind2_temp = np.intersect1d(date_temp, t_lst, return_indices=True)
            num_nan_lst = []
            for i in range(len(columns_flow)):
                out_temp = np.full([len(t_lst)], np.nan)
                df_flow_temp = df_flow[columns_flow[i]].copy()
                out_temp[ind2_temp] = df_flow_temp[ind1_temp]
                num_nan = np.isnan(out_temp).sum()
                num_nan_lst.append(num_nan)
            num_nan_np = np.array(num_nan_lst)
            index_flow_num = np.argmin(num_nan_np)
            df_flow.rename(columns={columns_flow[index_flow_num]: 'flow'}, inplace=True)
            df_flow.rename(columns={columns_flow_cd[index_flow_num]: 'mode'}, inplace=True)
        else:
            for column_name in columns_names:
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
        if not ('flow' in df_flow.columns.intersection(columns)):
            data_temp = df_flow.loc[:, df_flow.columns.intersection(columns)]
            # add nan column to data_temp
            data_temp = pd.concat([data_temp, pd.DataFrame(columns=['flow', 'mode'])])
        else:
            data_temp = df_flow.loc[:, columns]
        # fix flow which is not numeric data
        data_temp.loc[data_temp['flow'] == "Ice", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Ssn", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Tst", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Eqp", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Rat", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Dis", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Bkw", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "***", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "Mnt", 'flow'] = np.nan
        data_temp.loc[data_temp['flow'] == "ZFL", 'flow'] = np.nan
        print(usgs_id)
        # set negative value -- nan
        obs = data_temp['flow'].astype('float').values
        # 看看warning是哪个站点：01606500 and other 3-5 ones. For 01606500, 时间索引为2828的站点为nan，不过不影响计算。
        if usgs_id == '01606500':
            print(obs)
            print(np.argwhere(np.isnan(obs)))
        obs[obs < 0] = np.nan
        # 首先判断时间范围是否一致，读取的径流数据的时间序列和要读取的径流数据时间序列交集，获取径流数据，剩余没有数据的要读取的径流数据时间序列是nan
        nt = len(t_lst)
        out = np.full([nt], np.nan)
        # df中的date是字符串，转换为datetime，方可与tLst求交集
        df_date = data_temp['datetime']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
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
        assert len(usgs_all_sites) == streamflow.shape[0]
        if usgs_ids is not None:
            sites_index = np.where(np.in1d(usgs_all_sites, usgs_ids))[0]
            sites_chosen[sites_index] = 1
        else:
            sites_index = np.arange(streamflow.shape[0])
            sites_chosen = np.ones(streamflow.shape[0])
        # choose data in given time range
        all_t_list = hydro_time.t_range_days(self.t_range)
        assert all_t_list.size == streamflow.shape[1]
        t_lst = all_t_list
        if time_range:
            # calculate the day length
            t_lst = hydro_time.t_range_days(time_range)
        ts, ind1, ind2 = np.intersect1d(all_t_list, t_lst, return_indices=True)
        # 取某几行的某几列数据稍微麻烦一点点
        streamflow_temp = streamflow[sites_index]  # 先取出想要的行数据
        usgs_values = streamflow_temp[:, ind1]  # 再取出要求的列数据

        for i in range(sites_index.size):
            # loop for every site
            runoff = usgs_values[i, :]
            for criteria in kwargs:
                # if any criteria is not matched, we can filter this site
                if sites_chosen[sites_index[i]] == 0:
                    break
                if criteria == 'missing_data_ratio':
                    nan_length = runoff[np.isnan(runoff)].size
                    # then calculate the length of consecutive nan
                    thresh = kwargs[criteria]
                    if nan_length / runoff.size > thresh:
                        sites_chosen[sites_index[i]] = 0
                    else:
                        sites_chosen[sites_index[i]] = 1

                elif criteria == 'zero_value_ratio':
                    zero_length = runoff.size - np.count_nonzero(runoff)
                    thresh = kwargs[criteria]
                    if zero_length / runoff.size > thresh:
                        sites_chosen[sites_index[i]] = 0
                    else:
                        sites_chosen[sites_index[i]] = 1
                else:
                    print("Oops!  That is not valid value.  Try again...")
        # get discharge data of chosen sites, and change to ndarray
        usgs_out = np.array([usgs_values[i] for i in range(sites_index.size) if sites_chosen[sites_index[i]] > 0])
        gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]
        gage_dict_new = dict()
        for key, value in self.gage_dict.items():
            value_new = np.array([value[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0])
            gage_dict_new[key] = value_new
        self.gage_dict = gage_dict_new
        assert (gages_chosen_id == gage_dict_new["STAID"]).all()
        assert (all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:])))
        return usgs_out, gages_chosen_id, ts

    @my_timer
    def read_forcing(self, usgs_id_lst, t_range_lst):
        """读取gagesII_forcing文件夹下的驱动数据(data processed from GEE)
        :return
        x: ndarray -- 1d-axis:gages, 2d-axis: day, 3d-axis: forcing vst
        """
        data_folder = os.path.join(self.all_configs.get("forcing_dir"))
        download_dir_name = os.path.join(self.all_configs.get("root_dir"), "gagesII_forcing",
                                         self.all_configs.get("forcing_type"))
        assert (all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:])))
        assert (all(x < y for x, y in zip(t_range_lst, t_range_lst[1:])))
        if data_folder == download_dir_name:
            dataset = self.all_configs.get("forcing_type")
            var_lst = self.all_configs.get("forcing_chosen")
            regions = self.all_configs.get("regions")
            # different files for different years
            t_start_year = hydro_time.get_year(t_range_lst[0])
            t_end_year = hydro_time.get_year(t_range_lst[-1])
            # arange是左闭右开的，所以+1
            t_lst_years = np.arange(t_start_year, t_end_year + 1).astype(str)
            data_temps = pd.DataFrame()
            region_names = [region_temp.split("_")[-1] for region_temp in regions]
            # forcing data file generated is named as "allref", so rename the "all"
            region_names = ["allref" if r == "all" else r for r in region_names]
            for year in t_lst_years:
                # to match the file of the given year
                for f_name in os.listdir(data_folder):
                    # 首先判断是不是在给定的region内的
                    for region_name in region_names:
                        if fnmatch.fnmatch(f_name, dataset + '_' + region_name + '_mean_' + year + '.csv'):
                            data_file = os.path.join(data_folder, f_name)
                            data_temp = pd.read_csv(data_file, sep=',', dtype={'gage_id': str})
                            frames_temp = [data_temps, data_temp]
                            data_temps = pd.concat(frames_temp)
            # choose data in given time and sites. if there is no value for site in usgs_id_lst, just error(because
            # every site should have forcing). using dataframe mostly will make data type easy to handle with
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
                data_chosen_var = data_k[var_lst]
                out[ind1, :] = data_chosen_var.iloc[ind2, :].values
                x[k, :, :] = out

            return x
        else:
            print("reading formatted data:")
            var_lst = self.all_configs["forcing_chosen"]
            nt = t_range_lst.shape[0]
            x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
            for k in range(len(usgs_id_lst)):
                data = self.read_forcing_gage(usgs_id_lst[k], var_lst, t_range_lst,
                                              dataset=self.all_configs["forcing_type"])
                x[k, :, :] = data
            return x

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='daymet'):
        gage_dict = self.gage_dict
        ind = np.argwhere(gage_dict['STAID'] == usgs_id)[0][0]
        huc = gage_dict['HUC02'][ind]

        data_folder = self.all_configs["forcing_dir"]
        # original daymet file not for leap year, there is no data in 12.31 in leap year, so files which have been interpolated for nan value have name "_leap"
        data_file = os.path.join(data_folder, huc, '%s_lump_%s_forcing_leap.txt' % (usgs_id, dataset))
        print("reading", usgs_id, "forcing data")
        data_temp = pd.read_csv(data_file, sep=r'\s+', header=None, skiprows=1)

        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        nf = len(var_lst)
        assert (all(x < y for x, y in zip(date, date[1:])))
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
        nt = t_range_list.size
        out = np.empty([nt, nf])
        var_lst_in_file = ["dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = [i for i in range(len(var_lst_in_file)) if var_lst[k] in var_lst_in_file[i]][0]
            out[ind2, k] = data_temp[ind + 4].values[ind1]
        return out

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
            assert (all(x < y for x, y in zip(range2, range2[1:])))
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

    def read_attr(self, usgs_id_lst, var_lst, is_return_dict=True):
        """指定读取某些站点的某些属性"""
        # assert type(usgs_id_lst) == list
        assert (all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:])))
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all(usgs_id_lst)
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        out = attr_all[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out

    def read_attr_origin(self, gages_ids, attr_lst):
        """:return np.array -- the first dim is types of attrs, and the second one is sites"""
        dir_gage_attr = self.all_configs.get("gage_files_dir")
        data_temp_chosen_lst = list()
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
        var_lst = list()
        out_lst = []
        for i in range(len(attr_lst)):
            out_lst.append([])
        # 因为选择的站点可能是站点的一部分，所以需要求交集，ind2是所选站点在conterm_文件中所有站点里的index，把这些值放到out_temp中
        range1 = gages_ids
        gage_id_file = self.all_configs.get("gage_id_file")
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        range2 = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(range2, range2[1:])))
        c, ind1, ind2 = np.intersect1d(range1, range2, return_indices=True)

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == 'flow_record':
                key = 'flowrec'
            data_file = os.path.join(dir_gage_attr, 'conterm_' + key + '.txt')
            # 各属性值的“参考来源”是不需读入的
            if key == 'bas_classif':
                data_temp = pd.read_csv(data_file, sep=',',
                                        dtype={'STAID': str, "WR_REPORT_REMARKS": str, "ADR_CITATION": str,
                                               "SCREENING_COMMENTS": str}, engine='python')
            elif key == 'bound_qa':
                # "DRAIN_SQKM" already exists
                data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str},
                                        usecols=["STAID", "BASIN_BOUNDARY_CONFIDENCE", "NWIS_DRAIN_SQKM",
                                                 "PCT_DIFF_NWIS", "HUC10_CHECK"])
            else:
                data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str})
            if key == 'flowrec':
                # 最后一列为空，舍弃
                data_temp = data_temp.iloc[:, range(0, data_temp.shape[1] - 1)]
            # 该文件下的所有属性
            var_lst_temp = list(data_temp.columns[1:])
            do_exist, idx_lst = is_any_elem_in_a_lst(attr_lst, var_lst_temp, return_index=True)
            if do_exist:
                for idx in idx_lst:
                    idx_in_var = var_lst_temp.index(attr_lst[idx]) + 1  # +1 because the first col of data_temp is ID
                    out_lst[idx] = data_temp.iloc[ind2, idx_in_var].values
            else:
                continue
        out = np.array(out_lst)
        return out
