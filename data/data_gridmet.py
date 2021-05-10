import collections
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import fnmatch
from data.data_base import DatasetBase
from utils import hydro_utils


class Gridmet(DatasetBase):
    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.dataset_description = self.set_dataset_describe()
        if download:
            self.download_dataset()
        self.all_sites = self.read_site_info()

    def get_name(self):
        return "GRIDMET"

    def set_dataset_describe(self):
        gridmet_db = self.dataset_dir
        gridmet_crop_et_dir = os.path.join(gridmet_db, "conus3557", "daily_stats", "daily_stats")
        gridmet_crop_et_shpfile = os.path.join(gridmet_db, "conus3557", "some_from_3557.shp")
        # forcing
        forcing_dir = os.path.join(gridmet_db, "gridmet")

        return collections.OrderedDict(GRIDMET_DIR=gridmet_db, GRIDMET_CROPET_DIR=gridmet_crop_et_dir,
                                       GRIDMET_FORCING_DIR=forcing_dir, GRIDMET_CROPET_SHPFILE=gridmet_crop_et_shpfile)

    def read_object_ids(self, object_params=None) -> np.array:
        basepath = self.dataset_description["GRIDMET_FORCING_DIR"]
        subdirs = []
        for entry in os.listdir(basepath):
            if os.path.isdir(os.path.join(basepath, entry)):
                subdirs.append(entry)
        usgs_ids = []
        for dirpath, dirname, files in os.walk(self.dataset_description["GRIDMET_FORCING_DIR"]):
            sub_dir_name = dirpath.split("/")[-1]
            if sub_dir_name in subdirs:
                for file_name in files:
                    usgs_ids.append(file_name.split("_")[0])
        sort_usgs_ids = np.sort(usgs_ids)
        return sort_usgs_ids

    def read_site_info(self) -> pd.DataFrame:
        basepath = self.dataset_description["GRIDMET_FORCING_DIR"]
        subdirs = []
        for entry in os.listdir(basepath):
            if os.path.isdir(os.path.join(basepath, entry)):
                subdirs.append(entry)
        dict_id_huc = {}
        for dirpath, dirname, files in os.walk(self.dataset_description["GRIDMET_FORCING_DIR"]):
            huc_num = dirpath.split("/")[-1]
            if huc_num in subdirs:
                for file_name in files:
                    dict_id_huc[file_name.split("_")[0]] = huc_num
        usgs_ids = self.read_object_ids()
        sort_usgs_ids = np.sort(usgs_ids)
        id_hucs = [dict_id_huc[id_tmp] for id_tmp in sort_usgs_ids]
        data = pd.DataFrame({"gauge_id": sort_usgs_ids, "huc_02": id_hucs})
        return data

    def download_dataset(self):
        print("The GRIDMET data are self-made. Please set it manually!")

    def get_constant_cols(self) -> np.array:
        pass

    def get_relevant_cols(self):
        # TODO: now only these 9 forcings
        return np.array(["pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "etr", "cet"])

    def get_target_cols(self):
        pass

    def get_other_cols(self):
        pass

    def read_target_cols(self, usgs_id_lst=None, t_range=None, target_cols=None, **kwargs):
        pass

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list):
        print("reading gridmet forcing data", usgs_id)
        forcing_lst = ["Year", "Mnth", "Day", "Hr", "pr", "rmin", "srad", "tmmn", "tmmx", "vs", "eto", "etr"]
        assert np.array(np.intersect1d(var_lst, forcing_lst)).size == np.array(var_lst).size
        gage_id_df = self.all_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
        data_folder = self.dataset_description["GRIDMET_FORCING_DIR"]
        data_file = os.path.join(data_folder, huc, '%s_lump_gridmet_forcing.txt' % usgs_id)
        data_temp = pd.read_csv(data_file, sep=r'\s+')
        df_date = data_temp[["Year", "Mnth", "Day"]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            out[:, k] = data_temp[var_lst[k]].values[ind1]
        return out

    def read_cropet(self, usgs_id, t_range_lst):
        crop_area_shp = self.dataset_description["GRIDMET_CROPET_SHPFILE"]
        gdf_origin = gpd.read_file(crop_area_shp)
        gdf = gdf_origin.sort_values(by='GAGE_ID')
        all_columns = gdf.columns.tolist()
        crop_field_list = [a_col for a_col in all_columns if "CROP" in a_col]

        crops_et_vals = []
        for irri_basin in usgs_id:
            print("read crop et of " + str(irri_basin))
            # if irri_basin != "01434025":
            #     continue
            basin_crop_areas = gdf.loc[gdf['GAGE_ID'] == irri_basin, crop_field_list].values.flatten()
            # crop with too small area or not crop type has area but no crop et data, so exclude them
            crop_et_lst = []
            crop_et_names = []
            crop_et_file_num = 0

            # get the date list
            file4date = os.listdir(self.dataset_description["GRIDMET_CROPET_DIR"])[0]
            tmp4date = pd.read_csv(os.path.join(self.dataset_description["GRIDMET_CROPET_DIR"], file4date), comment='#')
            date = pd.to_datetime(tmp4date["Date"]).values.astype('datetime64[D]')
            [c, ind3, ind4] = np.intersect1d(date, t_range_lst, return_indices=True)
            assert date[0] <= t_range_lst[0] and date[-1] >= t_range_lst[-1]
            nt = t_range_lst.size

            for f_name in os.listdir(self.dataset_description["GRIDMET_CROPET_DIR"]):
                if fnmatch.fnmatch(f_name, irri_basin + '_*'):
                    crop_et_names.append("CROP_" + f_name[-7:-4])
                    data_tmp = pd.read_csv(os.path.join(self.dataset_description["GRIDMET_CROPET_DIR"], f_name),
                                           comment='#')
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

    def read_relevant_cols(self, usgs_id_lst=None, t_range=None, var_lst=None, **kwargs):
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
        if "cet" in var_lst:
            forcing_lst = var_lst[:]
            forcing_lst.remove("cet")
            cet_idx = var_lst.index("cet")
            other_idx = np.delete(np.arange(len(var_lst)), cet_idx)
            cropet = self.read_cropet(usgs_id_lst, t_range_list)
            for k in range(len(usgs_id_lst)):
                data = self.read_forcing_gage(usgs_id_lst[k], forcing_lst, t_range_list)
                x[k, :, other_idx] = data.T
                x[k, :, cet_idx] = cropet[k]
        else:
            for k in range(len(usgs_id_lst)):
                data = self.read_forcing_gage(usgs_id_lst[k], var_lst, t_range_list)
                x[k, :, :] = data
        return x

    def read_constant_cols(self, usgs_id_lst=None, var_lst=None, is_return_dict=False):
        pass

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        pass
