import collections
import os

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict
import pytz
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from data.data_base import DatasetBase
from explore.stat import cal_fdc
from utils import hydro_utils
from utils.hydro_utils import hydro_logger, download_one_zip, download_small_file, unzip_nested_zip, \
    is_any_elem_in_a_lst


class Gages(DatasetBase):
    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.dataset_description = self.set_dataset_describe()
        if download:
            self.download_dataset()
        self.gages_sites = self.read_site_info()

    def get_name(self):
        return "GAGES"

    def get_constant_cols(self) -> np.array:
        """all readable attrs in GAGES-II"""
        dir_gage_attr = self.dataset_description["GAGES_ATTR_DIR"]
        var_desc_file = os.path.join(dir_gage_attr, "variable_descriptions.txt")
        var_desc = pd.read_csv(var_desc_file)
        return var_desc["VARIABLE_NAME"].values

    def get_relevant_cols(self):
        # TODO: now only these 7 forcings
        return np.array(['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'])

    def get_target_cols(self):
        # TODO: now only usgsFlow
        return np.array(["usgsFlow"])

    def get_other_cols(self) -> dict:
        return {"FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}}

    def set_dataset_describe(self):
        gages_db = self.dataset_dir
        # region shapefiles
        gage_region_dir = os.path.join(gages_db, 'boundaries_shapefiles_by_aggeco', 'boundaries-shapefiles-by-aggeco')
        gages_regions = ['bas_ref_all', 'bas_nonref_CntlPlains', 'bas_nonref_EastHghlnds', 'bas_nonref_MxWdShld',
                         'bas_nonref_NorthEast', 'bas_nonref_SECstPlain', 'bas_nonref_SEPlains', 'bas_nonref_WestMnts',
                         'bas_nonref_WestPlains', 'bas_nonref_WestXeric']
        # point shapefile
        gagesii_points_file = os.path.join(gages_db, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp")

        # config of flow data
        flow_dir = os.path.join(gages_db, "gages_streamflow", "gages_streamflow")
        # forcing
        forcing_dir = os.path.join(gages_db, "basin_mean_forcing", "basin_mean_forcing")
        forcing_types = ["daymet"]
        # attr
        attr_dir = os.path.join(gages_db, "basinchar_and_report_sept_2011", 'spreadsheets-in-csv-format')
        gauge_id_file = os.path.join(attr_dir, 'conterm_basinid.txt')

        download_url_lst = [
            "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
            "https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
            "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
            "https://www.sciencebase.gov/catalog/file/get/59692a64e4b0d1f9f05fbd39"]
        usgs_streamflow_url = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}"
        # GAGES-II time series dataset dir
        gagests_dir = os.path.join(gages_db, "59692a64e4b0d1f9f05f")
        population_file = os.path.join(gagests_dir, "Dataset8_Population-Housing",
                                       "Dataset8_Population-Housing", "PopulationHousing.txt")
        wateruse_file = os.path.join(gagests_dir, "Dataset10_WaterUse", "Dataset10_WaterUse",
                                     "WaterUse_1985-2010.txt")
        return collections.OrderedDict(GAGES_DIR=gages_db, GAGES_FLOW_DIR=flow_dir, GAGES_FORCING_DIR=forcing_dir,
                                       GAGES_FORCING_TYPE=forcing_types, GAGES_ATTR_DIR=attr_dir,
                                       GAGES_GAUGE_FILE=gauge_id_file, GAGES_DOWNLOAD_URL_LST=download_url_lst,
                                       GAGES_REGIONS_SHP_DIR=gage_region_dir, GAGES_REGION_LIST=gages_regions,
                                       GAGES_POINT_SHP_FILE=gagesii_points_file, GAGES_POPULATION_FILE=population_file,
                                       GAGES_WATERUSE_FILE=wateruse_file, USGS_FLOW_URL=usgs_streamflow_url)

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs) -> dict:
        # TODO: not finish
        out_dict = {}
        for key, value in other_cols.items():
            if key == "FDC":
                assert "time_range" in value.keys()
                if "quantile_num" in value.keys():
                    quantile_num = value["quantile_num"]
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"),
                                  quantile_num=quantile_num)
                else:
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"))
            else:
                raise NotImplementedError("No this item yet!!")
            out_dict[key] = out
        return out_dict

    def read_attr_all(self, gages_ids):
        """read data from GAGES-II
        parameters:
            gages_ids:gages sites' ids
        :return
            out：ndarray
        """
        dir_gage_attr = self.dataset_description["GAGES_ATTR_DIR"]
        f_dict = dict()  # factorize dict
        # each key-value pair for atts in a file (list）
        var_dict = dict()
        # all attrs
        var_lst = list()
        out_lst = list()
        # read all attrs
        var_des = pd.read_csv(os.path.join(dir_gage_attr, 'variable_descriptions.txt'), sep=',')
        var_des_map_values = var_des['VARIABLE_TYPE'].tolist()
        for i in range(len(var_des)):
            var_des_map_values[i] = var_des_map_values[i].lower()
        # sort by type
        key_lst = list(set(var_des_map_values))
        key_lst.sort(key=var_des_map_values.index)
        # remove x_region_names
        key_lst.remove('x_region_names')

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == 'flow_record':
                key = 'flowrec'
            data_file = os.path.join(dir_gage_attr, 'conterm_' + key + '.txt')
            # remove some unused atttrs in bas_classif
            if key == 'bas_classif':
                data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str}, usecols=range(0, 4))
            else:
                data_temp = pd.read_csv(data_file, sep=',', dtype={'STAID': str})
            if key == 'flowrec':
                # remove final column which is nan
                data_temp = data_temp.iloc[:, range(0, data_temp.shape[1] - 1)]
            # all attrs in files
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gages_ids)
            out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)  # 1d:sites，2d: attrs in current data_file
            # sites intersection，ind2 is the index of sites in conterm_ files，set them in out_temp
            range1 = gages_ids
            range2 = data_temp.iloc[:, 0].astype(str).tolist()
            assert (all(x < y for x, y in zip(range2, range2[1:])))
            c, ind1, ind2 = np.intersect1d(range1, range2, return_indices=True)
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):  # str vars -> categorical vars
                    value, ref = pd.factorize(data_temp.loc[ind2, field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp.loc[ind2, field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_lst, var_dict, f_dict

    def read_constant_cols(self, object_ids=None, constant_cols: list = None, **kwargs) -> np.array:
        """read some attrs of some sites"""
        assert (all(x < y for x, y in zip(object_ids, object_ids[1:])))
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all(object_ids)
        ind_var = list()
        for var in constant_cols:
            ind_var.append(var_lst_all.index(var))
        out = attr_all[:, ind_var]
        return out

    def read_attr_origin(self, gages_ids, attr_lst):
        """this function read the attrs data in GAGES-II but not transform them to int when they are str
        :return np.array -- the first dim is types of attrs, and the second one is sites"""
        dir_gage_attr = self.dataset_description["GAGES_ATTR_DIR"]
        var_des = pd.read_csv(os.path.join(dir_gage_attr, 'variable_descriptions.txt'), sep=',')
        var_des_map_values = var_des['VARIABLE_TYPE'].tolist()
        for i in range(len(var_des)):
            var_des_map_values[i] = var_des_map_values[i].lower()
        key_lst = list(set(var_des_map_values))
        key_lst.sort(key=var_des_map_values.index)
        key_lst.remove('x_region_names')
        out_lst = []
        for i in range(len(attr_lst)):
            out_lst.append([])
        range1 = gages_ids
        gage_id_file = self.dataset_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        range2 = data_all["STAID"].values.tolist()
        assert (all(x < y for x, y in zip(range2, range2[1:])))
        c, ind1, ind2 = np.intersect1d(range1, range2, return_indices=True)

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == 'flow_record':
                key = 'flowrec'
            data_file = os.path.join(dir_gage_attr, 'conterm_' + key + '.txt')
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
                data_temp = data_temp.iloc[:, range(0, data_temp.shape[1] - 1)]
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

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='daymet'):
        gage_dict = self.gages_sites
        ind = np.argwhere(gage_dict['STAID'] == usgs_id)[0][0]
        huc = gage_dict['HUC02'][ind]

        data_folder = os.path.join(self.dataset_description["GAGES_FORCING_DIR"], dataset)
        # original daymet file not for leap year, there is no data in 12.31 in leap year,
        # so files which have been interpolated for nan value have name "_leap"
        data_file = os.path.join(data_folder, huc, '%s_lump_%s_forcing_leap.txt' % (usgs_id, dataset))
        print("reading", dataset, "forcing data ", usgs_id)
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

    def read_relevant_cols(self, object_ids=None, t_range_list=None, var_lst=None, **kwargs) -> np.array:
        # TODO: now only support daymet
        assert (all(x < y for x, y in zip(object_ids, object_ids[1:])))
        assert (all(x < y for x, y in zip(t_range_list, t_range_list[1:])))
        print("reading formatted data:")
        t_lst = hydro_utils.t_range_days(t_range_list)
        nt = t_lst.shape[0]
        x = np.empty([len(object_ids), nt, len(var_lst)])
        for k in range(len(object_ids)):
            data = self.read_forcing_gage(object_ids[k], var_lst, t_lst,
                                          dataset=self.dataset_description["GAGES_FORCING_TYPE"][0])
            x[k, :, :] = data
        return x

    def read_target_cols(self, usgs_id_lst=None, t_range_list=None, target_cols=None, **kwargs) -> np.array:
        """read USGS daily average streamflow data according to id and time
                Parameter:
                    gage_dict：site information
                    t_range: must be time range for downloaded data
                Return：
                    y: ndarray--streamflow data, 1d-axis: gages, 2d-axis: day
                """
        t_lst = hydro_utils.t_range_days(t_range_list)
        nt = t_lst.shape[0]
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            data_obs = self.read_usgs_gage(usgs_id_lst[k], t_lst)
            y[k, :] = data_obs
        return y

    def read_usgs_gage(self, usgs_id, t_lst):
        """read data for one gage"""
        print(usgs_id)
        dir_gage_flow = self.dataset_description["GAGES_FLOW_DIR"]
        gage_id_df = pd.DataFrame(self.gages_sites)
        huc = gage_id_df[gage_id_df["STAID"] == usgs_id]["HUC02"].values[0]
        usgs_file = os.path.join(dir_gage_flow, str(huc), usgs_id + '.txt')
        # ignore the comment lines and the first non-value row
        df_flow = pd.read_csv(usgs_file, comment='#', sep='\t', dtype={'site_no': str}).iloc[1:, :]
        # change the original column names
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

                df_flow.loc[df_flow[columns_flow[i]] == "Ice", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Ssn", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Tst", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Eqp", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Rat", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Dis", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Bkw", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "***", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "Mnt", columns_flow[i]] = np.nan
                df_flow.loc[df_flow[columns_flow[i]] == "ZFL", columns_flow[i]] = np.nan

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
        # set negative value -- nan
        obs = data_temp['flow'].astype('float').values
        obs[obs < 0] = np.nan
        # time range intersection. set points without data nan values
        nt = len(t_lst)
        out = np.full([nt], np.nan)
        # date in df is str，so transform them to datetime
        df_date = data_temp['datetime']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
        return out

    def read_object_ids(self, object_params=None) -> np.array:
        return self.gages_sites['STAID']

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ['DRAIN_SQKM'], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        mean_prep = self.read_constant_cols(object_ids, ['PPTAVG_BASIN'], is_return_dict=False)
        mean_prep = mean_prep / 365 * 10
        return mean_prep

    def download_dataset(self):
        print("Please download data manually!")
        if not os.path.isdir(self.dataset_description["GAGES_DIR"]):
            os.makedirs(self.dataset_description["GAGES_DIR"])
        zip_files = ["59692a64e4b0d1f9f05fbd39", "basin_mean_forcing.zip", "basinchar_and_report_sept_2011.zip",
                     "boundaries_shapefiles_by_aggeco.zip", "gages_streamflow.zip", "gagesII_9322_point_shapefile.zip"]
        download_zip_files = [os.path.join(self.dataset_description["GAGES_DIR"], zip_file) for zip_file in zip_files]
        for download_zip_file in download_zip_files:
            if not os.path.isfile(download_zip_file):
                raise RuntimeError(download_zip_file + " not found! Please download the data")
        unzip_dirs = [os.path.join(self.dataset_description["GAGES_DIR"], zip_file[:-4]) for zip_file in zip_files]
        for i in range(len(unzip_dirs)):
            if not os.path.isdir(unzip_dirs[i]):
                print("unzip directory:" + unzip_dirs[i])
                unzip_nested_zip(download_zip_files[i], unzip_dirs[i])
            else:
                print("unzip directory -- " + unzip_dirs[i] + " has existed")

    def read_site_info(self):
        gage_id_file = self.dataset_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        gage_fld_lst = data_all.columns.values
        out = dict()
        df_id_region = data_all.iloc[:, 0].values
        assert (all(x < y for x, y in zip(df_id_region, df_id_region[1:])))
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data_all[s].values.tolist()
            else:
                out[s] = data_all[s].values
        return out


def prepare_usgs_data(dataset_description: Dict):
    hydro_logger.info("NOT all dataset could be downloaded from website directly!")
    # download zip files
    [download_one_zip(attr_url, dataset_description["GAGES_DIR"]) for attr_url in
     dataset_description["GAGES_DOWNLOAD_URL_LST"]]
    # download streamflow data from USGS website
    dir_gage_flow = dataset_description["GAGES_FLOW_DIR"]
    streamflow_url = dataset_description["USGS_FLOW_URL"]
    # TODO: now a hard code for t_download_range
    t_download_range = ["1980-01-01", "2021-01-01"]
    if not os.path.isdir(dir_gage_flow):
        os.makedirs(dir_gage_flow)
    dir_list = os.listdir(dir_gage_flow)
    # if no streamflow data for the usgs_id_lst, then download them from the USGS website
    data_all = pd.read_csv(dataset_description["GAGES_GAUGE_FILE"], sep=',', dtype={0: str})
    usgs_id_lst = data_all.iloc[:, 0].values.tolist()
    gage_fld_lst = data_all.columns.values
    for ind in range(len(usgs_id_lst)):  # different hucs different directories
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
            # download data and save as txt file
            start_time_str = datetime.strptime(t_download_range[0], '%Y-%m-%d')
            end_time_str = datetime.strptime(t_download_range[1], '%Y-%m-%d') - timedelta(days=1)
            url = streamflow_url.format(usgs_id_lst[ind], start_time_str.year, start_time_str.month,
                                        start_time_str.day, end_time_str.year, end_time_str.month,
                                        end_time_str.day)

            # save in its HUC02 dir
            temp_file = os.path.join(dir_huc_02, str(usgs_id_lst[ind]) + '.txt')
            download_small_file(url, temp_file)
            print("successfully download " + temp_file + " streamflow data！")


def make_usgs_data(start_date: datetime, end_date: datetime, site_number: str) -> pd.DataFrame:
    """This method could also be used to download usgs streamflow data"""
    base_url = "https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065&format=rdb&"
    full_url = base_url + "site_no=" + site_number + "&period=&begin_date=" + \
               start_date.strftime("%Y-%m-%d") + "&end_date=" + end_date.strftime("%Y-%m-%d")
    print("Getting request from USGS")
    print(full_url)
    r = requests.get(full_url)
    with open(site_number + ".txt", "w") as f:
        f.write(r.text)
    print("Request finished")
    response_data = process_response_text(site_number + ".txt")
    create_csv(response_data[0], response_data[1], site_number)
    return pd.read_csv(site_number + "_flow_data.csv")


def process_response_text(file_name: str) -> Tuple[str, Dict]:
    extractive_params = {}
    with open(file_name, "r") as f:
        lines = f.readlines()
        i = 0
        params = False
        while "#" in lines[i]:
            # TODO figure out getting height and discharge code efficently
            the_split_line = lines[i].split()[1:]
            if params:
                print(the_split_line)
                if len(the_split_line) < 2:
                    params = False
                else:
                    extractive_params[the_split_line[0] + "_" + the_split_line[1]] = df_label(the_split_line[2])
            if len(the_split_line) > 2:
                if the_split_line[0] == "TS":
                    params = True
            i += 1
        with open(file_name.split(".")[0] + "data.tsv", "w") as t:
            t.write("".join(lines[i:]))
        return file_name.split(".")[0] + "data.tsv", extractive_params


def df_label(usgs_text: str) -> str:
    usgs_text = usgs_text.replace(",", "")
    if usgs_text == "Discharge":
        return "cfs"
    elif usgs_text == "Gage":
        return "height"
    else:
        return usgs_text


def create_csv(file_path: str, params_names: dict, site_number: str):
    """
    Function that creates the final version of the CSV files
    """
    df = pd.read_csv(file_path, sep="\t")
    for key, value in params_names.items():
        df[value] = df[key]
    df.to_csv(site_number + "_flow_data.csv")


def get_timezone_map():
    timezone_map = {
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "MDT": "America/Denver",
        "MST": "America/Denver",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles"}
    return timezone_map


def process_intermediate_csv(df: pd.DataFrame) -> (pd.DataFrame, int, int, int):
    # Remove garbage first row
    # TODO check if more rows are garbage
    df = df.iloc[1:]
    time_zone = df["tz_cd"].iloc[0]
    time_zone = get_timezone_map()[time_zone]
    old_timezone = pytz.timezone(time_zone)
    new_timezone = pytz.timezone("UTC")
    # This assumes timezones are consistent throughout the USGS stream (this should be true)
    df["datetime"] = df["datetime"].map(lambda x: old_timezone.localize(
        datetime.strptime(x, "%Y-%m-%d %H:%M")).astimezone(new_timezone))
    df["cfs"] = pd.to_numeric(df['cfs'], errors='coerce')
    max_flow = df["cfs"].max()
    min_flow = df["cfs"].min()
    count_nan = len(df["cfs"]) - df["cfs"].count()
    print(f"there are {count_nan} nan values")
    return df[df.datetime.dt.minute == 0], max_flow, min_flow
