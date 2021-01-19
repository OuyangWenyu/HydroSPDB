import numpy as np
import pandas as pd
import os
from data.config import cfg
from explore import cal_stat_gamma, cal_stat
from explore.stat import trans_norm4gridmet
from utils import hydro_time, unzip_nested_zip, serialize_pickle, serialize_json, serialize_numpy, unserialize_pickle, \
    unserialize_json, unserialize_numpy
import geopandas as gpd
import fnmatch


def unzip_gridmet_zip(zip_file):
    unzip_dir = zip_file[:-4]
    if not os.path.isdir(unzip_dir):
        print("unzip directory:" + unzip_dir)
        unzip_nested_zip(zip_file, unzip_dir)
    else:
        print("unzip directory -- " + unzip_dir + " has existed")


class GridmetConfig(object):

    def __init__(self, gridmet_dir, et_dir_name, et_shp_file_name,
                 gridmet_forcing_var_lst=["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "etr"]):
        self.gridmet_dir = gridmet_dir
        self.gridmet_forcing_dir = os.path.join(self.gridmet_dir, "gridmet")
        self.crop_et_dir = os.path.join(self.gridmet_dir, et_dir_name, "daily_stats", "daily_stats")
        self.crop_area_shp = os.path.join(self.gridmet_dir, et_dir_name, et_shp_file_name + ".shp")
        self.gridmet_forcing_var_lst = gridmet_forcing_var_lst

        if not os.path.isdir(gridmet_dir):
            os.makedirs(gridmet_dir)
        cropet_zip_file = os.path.join(gridmet_dir, et_dir_name, "daily_stats.zip")
        gridmet_zip_file = os.path.join(gridmet_dir, "gridmet.zip")

        unzip_gridmet_zip(gridmet_zip_file)
        unzip_gridmet_zip(cropet_zip_file)


class GridmetSource(object):

    def __init__(self, config_data, gages_id):
        """read configuration of data source. 读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        self.data_config = config_data
        gage_id_file = cfg.GAGES.gage_id_file
        data_all = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        gage_fld_lst = data_all.columns.values
        df_id_region = data_all.iloc[:, 0].values
        c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
        data = data_all.iloc[ind1, :]
        gage_dict = dict()
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                gage_dict[s] = data[s].values.tolist()
            else:
                gage_dict[s] = data[s].values
        self.gage_dict = gage_dict
        self.basin_id_lst = c.tolist()

    def read_forcing(self, t_range_lst):
        basin_id_lst = self.basin_id_lst
        assert (all(x < y for x, y in zip(basin_id_lst, basin_id_lst[1:])))
        assert (all(x < y for x, y in zip(t_range_lst, t_range_lst[1:])))

        print("reading formatted data:")
        var_lst = self.data_config.gridmet_forcing_var_lst
        nt = t_range_lst.shape[0]
        x = np.empty([len(basin_id_lst), nt, len(var_lst)])
        for k in range(len(basin_id_lst)):
            data = self.read_forcing_gage(basin_id_lst[k], var_lst, t_range_lst)
            x[k, :, :] = data
        return x

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list):
        gage_dict = self.gage_dict
        ind = np.argwhere(gage_dict['STAID'] == usgs_id)[0][0]
        huc = gage_dict['HUC02'][ind]

        data_folder = self.data_config.gridmet_forcing_dir
        data_file = os.path.join(data_folder, huc, '%s_lump_gridmet_forcing.txt' % usgs_id)
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
        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = [i for i in range(len(var_lst)) if var_lst[k] in var_lst[i]][0]
            out[ind2, k] = data_temp[ind + 4].values[ind1]
        return out

    def read_cet(self, t_range_lst):
        crop_area_shp = self.data_config.crop_area_shp

        gdf_origin = gpd.read_file(crop_area_shp)
        gdf = gdf_origin.sort_values(by='GAGE_ID')
        all_columns = gdf.columns.tolist()
        crop_field_list = [a_col for a_col in all_columns if "CROP" in a_col]

        crops_et_vals = []
        for irri_basin in self.basin_id_lst:
            print("read crop et of " + str(irri_basin))
            # if irri_basin != "01434025":
            #     continue
            basin_crop_areas = gdf.loc[gdf['GAGE_ID'] == irri_basin, crop_field_list].values.flatten()
            # crop with too small area or not crop type has area but no crop et data, so exclude them
            crop_et_lst = []
            crop_et_names = []
            crop_et_file_num = 0

            # get the date list
            file4date = os.listdir(self.data_config.crop_et_dir)[0]
            tmp4date = pd.read_csv(os.path.join(self.data_config.crop_et_dir, file4date), comment='#')
            date = pd.to_datetime(tmp4date["Date"]).values.astype('datetime64[D]')
            [c, ind3, ind4] = np.intersect1d(date, t_range_lst, return_indices=True)
            assert date[0] <= t_range_lst[0] and date[-1] >= t_range_lst[-1]
            nt = t_range_lst.size

            for f_name in os.listdir(self.data_config.crop_et_dir):
                if fnmatch.fnmatch(f_name, irri_basin + '_*'):
                    crop_et_names.append("CROP_" + f_name[-7:-4])
                    data_tmp = pd.read_csv(os.path.join(self.data_config.crop_et_dir, f_name), comment='#')
                    crop_et_tmp = data_tmp["ETact"].values
                    crop_et_lst.append(crop_et_tmp)
                    crop_et_file_num += 1
            if crop_et_file_num < 1:
                print("no real crop")
                chose_time = np.full(nt, 0)
                crops_et_vals.append(chose_time)
                continue
            # Find the indices
            inter_crop, ind1, ind2 = np.intersect1d(crop_field_list, crop_et_names, return_indices=True)
            basin_crop_areas_not0 = basin_crop_areas[ind1]
            crop_et_arr_tmp = np.array(crop_et_lst)
            crop_et_arr = np.zeros(crop_et_arr_tmp.shape)
            # the sequence of f_name is random, so rearrange them as the correct sequence
            crop_et_arr[:] = crop_et_arr_tmp[ind2]
            weighted_crop_et_vals = np.sum(crop_et_arr.T * basin_crop_areas_not0, axis=1) / np.sum(
                basin_crop_areas_not0)
            # get the data within the time range
            chose_time = np.empty(nt)
            chose_time[ind4] = weighted_crop_et_vals[ind3]
            crops_et_vals.append(chose_time)
        crops_et_arr = np.array(crops_et_vals)
        return crops_et_arr


class GridmetModel(object):
    def __init__(self, gridmet_source, t_range, is_test=False, stat_train=None, stat_cet_train=None, cet_cache=None,
                 data_forcing_cache=None, stat_forcing_cache=None, stat_cet_cache=None):
        self.gridmet_source = gridmet_source
        self.t_range = t_range
        t_range_lst = hydro_time.t_range_days(self.t_range)
        if data_forcing_cache is None:
            self.data_forcing = gridmet_source.read_forcing(t_range_lst)
        else:
            self.data_forcing = data_forcing_cache
        if cet_cache is None:
            cet = gridmet_source.read_cet(t_range_lst)
            self.cet = cet.reshape(cet.shape[0], cet.shape[1], 1)
        else:
            self.cet = cet_cache
        if (stat_forcing_cache is None) and (stat_cet_cache is None):
            if is_test:
                assert stat_train is not None
                assert stat_cet_train is not None
                self.stat_forcing_dict = stat_train
                self.stat_cet_dict = stat_cet_train
            else:
                self.stat_forcing_dict = self.cal_stat()
                var_lst = ['ETc']
                gages_et_stat_dict = {var_lst[0]: cal_stat(self.cet)}
                self.stat_cet_dict = gages_et_stat_dict
        else:
            assert stat_forcing_cache is not None
            assert stat_cet_cache is not None
            self.stat_forcing_dict = stat_forcing_cache
            self.stat_cet_dict = stat_cet_cache

    def cal_stat(self):
        # cal the statistical values
        stat_dict = dict()
        forcing_lst = self.gridmet_source.data_config.gridmet_forcing_var_lst
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var == 'pr':
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])
        return stat_dict

    def load_data(self, rm_nan=True):
        # normalization
        stat_dict = self.stat_forcing_dict
        var_lst = self.gridmet_source.data_config.gridmet_forcing_var_lst
        data = self.data_forcing
        x = trans_norm4gridmet(data, var_lst, stat_dict, to_norm=True)
        cet_data = self.cet
        stat_cet_dict = self.stat_cet_dict
        cet_norm = trans_norm4gridmet(cet_data, ['ETc'], stat_cet_dict, to_norm=True)
        if rm_nan:
            x[np.where(np.isnan(data))] = 0
            cet_norm[np.where(np.isnan(cet_data))] = 0
        return x, cet_norm

    @classmethod
    def load_gridmet_datamodel(cls, dir_temp, **kwargs):
        gridmet_source_file = os.path.join(dir_temp, kwargs['gridmet_source_file_name'])
        stat_cet_file = os.path.join(dir_temp, kwargs['gridmet_stat_cet_file_name'])
        stat_forcing_file = os.path.join(dir_temp, kwargs['gridmet_stat_forcing_file_name'])
        forcing_file = os.path.join(dir_temp, kwargs['gridmet_forcing_file_name'])
        cet_file = os.path.join(dir_temp, kwargs['gridmet_cet_file_name'])
        time_range_file = os.path.join(dir_temp, kwargs['gridmet_time_range_file_name'])
        gridmet_source = unserialize_pickle(gridmet_source_file)
        stat_cet_dict = unserialize_json(stat_cet_file)
        stat_forcing_dict = unserialize_json(stat_forcing_file)
        data_forcing = unserialize_numpy(forcing_file)
        data_cet = unserialize_numpy(cet_file)
        t_range = unserialize_pickle(time_range_file)
        gridmetdata_model = cls(gridmet_source, t_range, cet_cache=data_cet, data_forcing_cache=data_forcing,
                                stat_forcing_cache=stat_forcing_dict, stat_cet_cache=stat_cet_dict)
        return gridmetdata_model


def save_gridmet_datamodel(dir_temp, grdimet_datamodel, **kwargs):
    if not os.path.isdir(dir_temp):
        os.makedirs(dir_temp)
    gridmet_source_file = os.path.join(dir_temp, kwargs['gridmet_source_file_name'])
    stat_cet_file = os.path.join(dir_temp, kwargs['gridmet_stat_cet_file_name'])
    stat_forcing_file = os.path.join(dir_temp, kwargs['gridmet_stat_forcing_file_name'])
    forcing_file = os.path.join(dir_temp, kwargs['gridmet_forcing_file_name'])
    cet_file = os.path.join(dir_temp, kwargs['gridmet_cet_file_name'])
    time_range_file = os.path.join(dir_temp, kwargs['gridmet_time_range_file_name'])

    serialize_pickle(grdimet_datamodel.gridmet_source, gridmet_source_file)
    serialize_json(grdimet_datamodel.stat_cet_dict, stat_cet_file)
    serialize_json(grdimet_datamodel.stat_forcing_dict, stat_forcing_file)
    serialize_numpy(grdimet_datamodel.data_forcing, forcing_file)
    serialize_numpy(grdimet_datamodel.cet, cet_file)
    serialize_pickle(grdimet_datamodel.t_range, time_range_file)
