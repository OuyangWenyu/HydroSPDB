import collections
import itertools
import os
from scipy import stats
import geopandas as gpd
import numpy as np
import pandas as pd

from data.data_base import DatasetBase
from data.data_gages import Gages
from data.data_gridmet import Gridmet
from data.data_nid import Nid
from utils.hydro_geo import coefficient_of_variation
from utils.hydro_utils import is_any_elem_in_a_lst, hydro_logger, serialize_json, unserialize_geopandas, \
    unserialize_json_ordered, nanlog


class GagesPro(DatasetBase):
    def __init__(self, data_path: list, download=False):
        # data_path is a list: now the Convention -- 0->gages_pro 1->gages, 2->nid, 3->gridmet
        # gages_pro is a virtual dataset, not from any real datasets, just for this class
        super().__init__(data_path[0])
        gages = Gages(data_path[1], download=download)
        self.gages = gages
        self.nid = Nid(data_path[2], download=download)
        self.gridmet = Gridmet(data_path[3], download=download)
        self.dataset_description = self.set_dataset_describe()
        self.init_cache()

    def init_cache(self):
        # check if the cache exists
        if not os.path.isfile(self.dataset_description["DAM_MAIN_PURPOSE_FILE"]) or not os.path.isfile(
                self.dataset_description["DAM_COORDINATION_FILE"]) or not os.path.isfile(
            self.dataset_description["DAM_STORAGE_FILE"]):
            self.spatial_join_dam()
            print("Generate gage_dam_coordinates file, gage_dam_coordinates file and gage_dam_storages file")
        else:
            print("The  gage_dam_coordinates file, gage_dam_points file and gage_dam_storages file have existed")

    def spatial_join_dam(self):
        # ALL_PURPOSES = ['C', 'D', 'F', 'G', 'H', 'I', 'N', 'O', 'P', 'R', 'S', 'T', 'X']
        gage_region_dir = self.dataset_description["GAGES_REGIONS_SHP_DIR"]
        region_shapefiles = self.dataset_description["GAGES_REGION_LIST"]
        # read sites from shapefile of region, get id from it.
        shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                      region_shapefiles]
        dam_dict = {}
        dam_points_dict = {}
        dam_storages_dict = {}
        points = unserialize_geopandas(self.dataset_description["NID_GEO_FILE"])
        for shapefile in shapefiles:
            polys = gpd.read_file(shapefile)
            print(points.crs)
            print(polys.crs)
            if not (points.crs == polys.crs):
                points = points.to_crs(polys.crs)
            print(points.head())
            print(polys.head())
            # Make a spatial join
            spatial_dam = gpd.sjoin(points, polys, how="inner", op="within")
            gages_id_dam = spatial_dam['GAGE_ID'].values
            u1 = np.unique(gages_id_dam)
            u1 = self.no_clear_diff_between_nid_gages(u1)
            main_purposes = []
            dam_points = []
            storages = []
            for u1_i in u1:
                purposes = []
                storages4purpose = []
                dam_point = []
                storage = []
                for index_i in range(gages_id_dam.shape[0]):
                    if gages_id_dam[index_i] == u1_i:
                        now_purpose = spatial_dam["PURPOSES"].iloc[index_i]
                        # if purpose is nan, then set it to X
                        NoneType = type(None)
                        if type(now_purpose) == float or type(now_purpose) == NoneType:
                            now_purpose = 'X'
                        purposes.append(now_purpose)
                        # storages4purpose.append(spatial_dam["NID_STORAGE"].iloc[index_i])
                        # NOR STORAGE
                        storages4purpose.append(spatial_dam["NORMAL_STORAGE"].iloc[index_i])
                        dam_point_lat = spatial_dam["LATITUDE"].iloc[index_i]
                        dam_point_lon = spatial_dam["LONGITUDE"].iloc[index_i]
                        dam_point.append((dam_point_lat, dam_point_lon))
                        storage.append(spatial_dam["NORMAL_STORAGE"].iloc[index_i])
                # main_purpose = which_is_main_purpose(storages4purpose, storages, care_1purpose=care_1purpose)
                main_purpose = only_one_main_purpose(purposes, storages4purpose)
                main_purposes.append(main_purpose)
                dam_points.append(dam_point)
                storages.append(storage)
            d = dict(zip(u1.tolist(), main_purposes))
            dam_dict = {**dam_dict, **d}
            d1 = dict(zip(u1.tolist(), dam_points))
            dam_points_dict = {**dam_points_dict, **d1}
            d2 = dict(zip(u1.tolist(), storages))
            dam_storages_dict = {**dam_storages_dict, **d2}
        # sorted by keys(gages_id)
        dam_dict_sorted = {}
        for key in sorted(dam_dict.keys()):
            dam_dict_sorted[key] = dam_dict[key]
        serialize_json(dam_dict_sorted, self.dataset_description["DAM_MAIN_PURPOSE_FILE"])
        dam_points_dict_sorted = {}
        for key in sorted(dam_points_dict.keys()):
            dam_points_dict_sorted[key] = dam_points_dict[key]
        dam_storages_dict_sorted = {}
        for key in sorted(dam_storages_dict.keys()):
            dam_storages_dict_sorted[key] = dam_storages_dict[key]
        serialize_json(dam_points_dict_sorted, self.dataset_description["DAM_COORDINATION_FILE"])
        serialize_json(dam_storages_dict_sorted, self.dataset_description["DAM_STORAGE_FILE"])

    def no_clear_diff_between_nid_gages(self, u1):
        """if there is clear diff for some basins in dam number between NID dataset and GAGES-II dataset,
        these basins will be excluded in the analysis"""
        print("excluede some basins")
        gages = self.gages
        # there are some basins from shapefile which are not in CONUS, that will cause some bug, so do an intersection before search the attibutes. Also decrease the number needed to be calculated
        usgs_id = np.intersect1d(u1, gages.gages_sites['STAID'])
        attr_lst_dam_num = ["NDAMS_2009"]
        data_gages_dam_num = gages.read_constant_cols(usgs_id, attr_lst_dam_num)
        u2 = [usgs_id[i] for i in range(len(usgs_id)) if data_gages_dam_num[i][0] > 0]
        return np.array(u2)

    def get_name(self):
        return "GAGES_PRO"

    def set_dataset_describe(self):
        gages_dataset_description = self.gages.dataset_description
        nid_dataset_description = self.nid.dataset_description
        gages_dam_description = collections.OrderedDict(
            DAM_MAIN_PURPOSE_FILE=os.path.join(self.dataset_dir, "dam_main_purpose_dict.json"),
            DAM_COORDINATION_FILE=os.path.join(self.dataset_dir, "dam_coordination_dict.json"),
            DAM_STORAGE_FILE=os.path.join(self.dataset_dir, "dam_storage_dict.json"))
        gridmet_dataset_description = self.gridmet.dataset_description
        return collections.OrderedDict(**gages_dataset_description, **nid_dataset_description, **gages_dam_description,
                                       **gridmet_dataset_description)

    def download_dataset(self):
        self.gages.download_dataset()
        self.nid.download_dataset()
        self.gridmet.download_dataset()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.gages.read_object_ids()

    def read_target_cols(self, object_ids=None, t_range_list=None, target_cols=None, **kwargs) -> np.array:
        return self.gages.read_target_cols(object_ids, t_range_list, target_cols)

    def read_relevant_cols(self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs) -> np.array:
        if "forcing_type" in kwargs.keys():
            # TODO: now only support one forcing type (daymet or gridmet) each time
            if kwargs["forcing_type"] == "gridmet":
                return self.gridmet.read_relevant_cols(object_ids, t_range_list, relevant_cols)
            elif kwargs["forcing_type"] == "daymet":
                return self.gages.read_relevant_cols(object_ids, t_range_list, relevant_cols)
            else:
                raise NotImplementedError("No more forcing data now!")
        else:
            return self.gages.read_relevant_cols(object_ids, t_range_list, relevant_cols)

    def read_constant_cols(self, object_ids=None, constant_cols: list = None, **kwargs) -> np.array:
        c = np.empty([len(object_ids), len(constant_cols)])
        all_attrs = self.get_constant_cols()
        gages_direct_attrs = self.gages.get_constant_cols()
        assert np.all(np.isin(np.array(constant_cols), all_attrs))
        # keep the sequence of attrs, so don't use the set-operation functions in numpy, which will sort the arrays
        gages_direct_index = [i for i in range(len(constant_cols)) if constant_cols[i] in gages_direct_attrs]
        constant_cols_in_gages_direct = [tmp for tmp in constant_cols if tmp in gages_direct_attrs]
        gages_indirect_index = [j for j in range(len(constant_cols)) if
                                (constant_cols[j] in all_attrs) and (constant_cols[j] not in gages_direct_attrs)]
        not_direct_attrs = [temp for temp in constant_cols if (temp in all_attrs) and (temp not in gages_direct_attrs)]
        gages_direct_data = self.gages.read_constant_cols(object_ids, constant_cols_in_gages_direct)
        indirect_data = self.read_new_attr(object_ids, not_direct_attrs)
        c[:, gages_direct_index] = gages_direct_data
        # TODO: Now no preprocess for NaN values
        c[:, gages_indirect_index] = indirect_data
        return c

    def read_new_attr(self, object_ids, not_direct_attrs) -> np.array:
        if len(not_direct_attrs) == 0:
            return np.array([])
        count = 0
        for attr in not_direct_attrs:
            if attr == "DAM_GAGE_DIS_VAR":
                data_temp = get_dam_dis_var(self, object_ids)
            elif attr == "DAM_MAIN_PURPOSE":
                gage_main_dam_purpose = get_dam_main_purpose(self, object_ids)
                data_temp = pd.Series(gage_main_dam_purpose).factorize(sort=True)[0]
            elif attr == "DAM_STORAGE_STD":
                data_temp = get_dam_storage_std(self, object_ids)
            elif attr == "DIVERSION":
                diversions = get_diversion(self.gages, object_ids)
                data_temp = pd.Series(diversions).factorize(sort=True)[0]
            elif attr == "DOR":
                data_temp = get_dor_values(self.gages, object_ids)
            else:
                raise NotImplementedError("No such attribute now!")
            if count == 0:
                new_attr_arr = data_temp
            else:
                new_attr_arr = np.vstack((new_attr_arr, data_temp))
            count = +1
        if count == 1:
            new_attr_arr = new_attr_arr.reshape(1, new_attr_arr.size)
        return new_attr_arr.T

    def get_constant_cols(self) -> np.array:
        # TODO: now just 5 more attrs:["DOR", "DAM_MAIN_PURPOSE", "DIVERSION", "DAM_GAGE_DIS_VAR", "DAM_STORAGE_STD"]
        """all readable attrs in GAGES-II, including more attributes not directly acquired from GAGES-II"""
        dir_gage_attr = self.dataset_description["GAGES_ATTR_DIR"]
        var_desc_file = os.path.join(dir_gage_attr, "variable_descriptions.txt")
        var_desc = pd.read_csv(var_desc_file)
        origin_attr = var_desc["VARIABLE_NAME"].values
        added_attr = ["DOR", "DAM_MAIN_PURPOSE", "DIVERSION", "DAM_GAGE_DIS_VAR", "DAM_STORAGE_STD"]
        attrs = np.append(origin_attr, added_attr)
        return attrs

    def get_relevant_cols(self):
        # TODO: now only support daymet and gridmet
        daymet_forcing_cols = self.gages.get_relevant_cols()
        gridmet_forcing_cols = self.gridmet.get_relevant_cols()
        forcing_cols = np.append(daymet_forcing_cols, gridmet_forcing_cols)
        return forcing_cols

    def get_target_cols(self):
        return self.gages.get_target_cols()

    def get_other_cols(self) -> dict:
        return {"RES_STOR_HIST": {"bins": 50, "quantile_limits": [0, 1]},
                "RES_DOR_HIST": {"bins": 50, "quantile_limits": [0, 1]},
                "FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}}

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        out_dict = {}
        for key, value in other_cols.items():
            if key == "FDC":
                fdc_dict = self.gages.read_other_cols(object_ids, {key: other_cols[key]})
                out_dict = {**out_dict, **fdc_dict}
                continue
            elif key == "RES_STOR_HIST":
                # cal the res_stor_hist for each basin
                assert "bins" in value.keys()
                num_bins = value["bins"]
                if "quantile_limits" in value.keys():
                    out = get_reservoir_storage_hist(self, object_ids, num_bins=num_bins,
                                                     quantile_limits=value["quantile_limits"])
                else:
                    out = get_reservoir_storage_hist(self, object_ids, num_bins=num_bins)
            elif key == "RES_DOR_HIST":
                # cal the res_stor_hist for each basin
                assert "bins" in value.keys()
                num_bins = value["bins"]
                if "quantile_limits" in value.keys():
                    out = get_reservoir_dor_hist(self, object_ids, num_bins=num_bins,
                                                 quantile_limits=value["quantile_limits"])
                else:
                    out = get_reservoir_dor_hist(self, object_ids, num_bins=num_bins)
            else:
                raise NotImplementedError("No this item yet!!")
            out_dict[key] = out
        return out_dict

    def read_basin_area(self, object_ids) -> np.array:
        return self.gages.read_basin_area(object_ids)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.gages.read_mean_prep(object_ids)


def get_dor_values(gages: Gages, usgs_id) -> np.array:
    """get dor values from gages for the usgs_id-sites"""
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    # mm/year 1-km grid,  megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
    # attr_lst = ["RUNAVE7100", "STOR_NID_2009"]
    attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
    data_attr = gages.read_constant_cols(usgs_id, attr_lst)
    run_avg = data_attr[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
    nor_storage = data_attr[:, 1] * 1000  # m^3
    dors = nor_storage / run_avg
    return dors


def get_diversion(gages: Gages, usgs_id) -> np.array:
    diversion_strs = ["diversion", "divert"]
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
    data_attr = gages.read_attr_origin(usgs_id, attr_lst)
    diversion_strs_lower = [elem.lower() for elem in diversion_strs]
    data_attr0_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[0]])
    data_attr1_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[1]])
    data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
    diversions = [is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True) for i in
                  range(len(usgs_id))]
    return np.array(diversions)


def get_dam_main_purpose(gages_pro: GagesPro, usgs_id: list) -> np.array:
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    dam_main_purposes = unserialize_json_ordered(gages_pro.dataset_description["DAM_MAIN_PURPOSE_FILE"])
    all_dam_gages_id = list(dam_main_purposes.keys())
    inter_id, all_dam_gages_id_ind, usgs_id_dam_ind = np.intersect1d(all_dam_gages_id, usgs_id, return_indices=True)
    if len(inter_id) != len(usgs_id):
        main_purposes = []
        print("some undammed basins are included, set their dam purpose to None")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_id:
                main_purposes.append(None)
            else:
                main_purposes.append(dam_main_purposes[usgs_id[i]])
    else:
        main_purposes = [dam_main_purposes[i] for i in usgs_id]
    return np.array(main_purposes)


def get_dam_dis_var(gages_pro: GagesPro, usgs_id: list) -> np.array:
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    gages_ids = gages_pro.gages.gages_sites["STAID"]
    inter_usgs_id, gages_ids_ind, usgs_id_ind = np.intersect1d(gages_ids, usgs_id, return_indices=True)
    assert len(inter_usgs_id) == len(usgs_id)
    gages_loc_lat = gages_pro.gages.gages_sites["LAT_GAGE"]
    gages_loc_lon = gages_pro.gages.gages_sites["LNG_GAGE"]
    gages_loc = [[gages_loc_lat[i], gages_loc_lon[i]] for i in gages_ids_ind]
    # calculate index of dispersion, then plot the NSE-dispersion scatterplot
    # Geo coord system of gages_loc and dam_coords are both NAD83
    dam_coords = unserialize_json_ordered(gages_pro.dataset_description["DAM_COORDINATION_FILE"])
    all_coord_gages_id = list(dam_coords.keys())
    inter_coord_id, all_coord_gages_id_ind, usgs_id_coord_ind = np.intersect1d(all_coord_gages_id, usgs_id,
                                                                               return_indices=True)
    if len(inter_coord_id) != len(usgs_id):
        coords = []
        print("some undammed basins are included, set their dis_var to NaN")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_coord_id:
                coords.append([])
            else:
                coords.append(dam_coords[usgs_id[i]])
    else:
        coords = [dam_coords[inter_id] for inter_id in inter_usgs_id]

    coefficient_of_var = list(map(coefficient_of_variation, gages_loc, coords))
    return np.array(coefficient_of_var)


def get_max_dam_norm_stor(gages_pro: GagesPro, usgs_id: list) -> np.array:
    """the max dam normal storage in a basin (unit: Acre-Feet)"""
    dam_storages = unserialize_json_ordered(gages_pro.dataset_description["DAM_STORAGE_FILE"])
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    all_dam_gages_id = list(dam_storages.keys())
    inter_id, all_dam_gages_id_ind, usgs_id_dam_ind = np.intersect1d(all_dam_gages_id, usgs_id, return_indices=True)
    if len(inter_id) != len(usgs_id):
        storage = []
        print("some undammed basins are included, set their storage to NaN")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_id:
                storage.append(np.nan)
            else:
                storage.append(dam_storages[usgs_id[i]])
    else:
        storage = [dam_storages[i] for i in usgs_id]
    max_in_a_basin = list(map(np.max, storage))
    return max_in_a_basin


def get_dam_storage_std(gages_pro: GagesPro, usgs_id: list) -> np.array:
    dam_storages = unserialize_json_ordered(gages_pro.dataset_description["DAM_STORAGE_FILE"])
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    all_dam_gages_id = list(dam_storages.keys())
    inter_id, all_dam_gages_id_ind, usgs_id_dam_ind = np.intersect1d(all_dam_gages_id, usgs_id, return_indices=True)
    if len(inter_id) != len(usgs_id):
        storage = []
        print("some undammed basins are included, set their storage to NaN")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_id:
                storage.append(np.nan)
            else:
                storage.append(dam_storages[usgs_id[i]])
    else:
        storage = [dam_storages[i] for i in usgs_id]
    std_storage_in_a_basin = list(map(np.nanstd, storage))
    log_std_storage_in_a_basin = list(map(nanlog, np.array(std_storage_in_a_basin) + 1))
    return log_std_storage_in_a_basin


def get_reservoir_storage_hist(gages_pro: GagesPro, usgs_id: list, num_bins=50, quantile_limits=[0, 1]) -> np.array:
    """cal the hist of reservoirs' storage in a basin"""
    dam_storages = unserialize_json_ordered(gages_pro.dataset_description["DAM_STORAGE_FILE"])
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    all_dam_gages_id = list(dam_storages.keys())
    inter_id, all_dam_gages_id_ind, usgs_id_dam_ind = np.intersect1d(all_dam_gages_id, usgs_id, return_indices=True)
    if len(inter_id) != len(usgs_id):
        storage = []
        print("some undammed basins are included, set their storage to 0")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_id:
                storage.append([0])
            else:
                storage.append(dam_storages[usgs_id[i]])
    else:
        storage = [dam_storages[i] for i in usgs_id]
    all_storage = np.array(list(itertools.chain(*storage)))
    lower_limit = np.quantile(all_storage[~np.isnan(all_storage)], quantile_limits[0])
    upper_limit = np.quantile(all_storage[~np.isnan(all_storage)], quantile_limits[1])
    res = [stats.relfreq(stor, numbins=num_bins, defaultreallimits=(lower_limit, upper_limit)).frequency for stor in
           storage]
    return np.array(res)


def get_reservoir_dor_hist(gages_pro: GagesPro, usgs_id: list, num_bins=50, quantile_limits=[0, 1]) -> np.array:
    """cal the hist of reservoirs' dor in a basin"""
    dam_storages = unserialize_json_ordered(gages_pro.dataset_description["DAM_STORAGE_FILE"])

    attr_lst = ["RUNAVE7100", "DRAIN_SQKM"]
    data_attr = gages_pro.read_constant_cols(usgs_id, attr_lst)
    run_avg = data_attr[:, 0] * (10 ** (-3)) * (10 ** 6)  # mm/year->m^3/(year * km^2)
    basin_area = data_attr[:, 1]  # square km
    run_stor = run_avg * basin_area  # m^3/year

    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    all_dam_gages_id = list(dam_storages.keys())
    inter_id, all_dam_gages_id_ind, usgs_id_dam_ind = np.intersect1d(all_dam_gages_id, usgs_id, return_indices=True)

    if len(inter_id) != len(usgs_id):
        dor = []
        print("some undammed basins are included, set their storage to 0")
        for i in range(len(usgs_id)):
            if usgs_id[i] not in inter_id:
                dor.append([0])
            else:
                # unit of stor in NID dataset is Acre-Feet (from:
                # https://nid.sec.usace.army.mil/ords/f?p=105:10:16514683433056::NO::P10_COLUMN_NAME:NORMAL_STORAGE)
                # 1 Acre-Feet = 1233.48183754752 cubic-meter (from: https://en.wikipedia.org/wiki/Acre-foot)
                dor.append(np.array(dam_storages[usgs_id[i]]) * 1233.48183754752 / run_stor[i])  # m^3/m^3 per year
    else:
        dor = [np.array(dam_storages[i]) * 1233.48183754752 / run_stor[i] for i in usgs_id]
    all_dor = np.array(list(itertools.chain(*dor)))
    lower_limit = np.quantile(all_dor[~np.isnan(all_dor)], quantile_limits[0])
    upper_limit = np.quantile(all_dor[~np.isnan(all_dor)], quantile_limits[1])
    res = [stats.relfreq(a_dor, numbins=num_bins, defaultreallimits=(lower_limit, upper_limit)).frequency for a_dor in
           dor]
    return np.array(res)


def multi_max_indices(nums):
    """nums could be a 2d array, where length of every 1d array is same"""
    max_of_nums = max(nums)
    tup = [(i, nums[i]) for i in range(len(nums))]
    indices = [i for i, n in tup if n == max_of_nums]
    return indices


def only_one_main_purpose(dams_purposes_of_a_basin, storages_of_a_basin):
    assert type(dams_purposes_of_a_basin) == list
    assert type(storages_of_a_basin) == list
    assert len(dams_purposes_of_a_basin) == len(storages_of_a_basin)

    all_purposes = []
    for j in range(len(dams_purposes_of_a_basin)):
        purposes_str_i = [dams_purposes_of_a_basin[j][i:i + 1] for i in range(0, len(dams_purposes_of_a_basin[j]), 1)]
        all_purposes = all_purposes + purposes_str_i
    all_purposes_unique = np.unique(all_purposes)
    purpose_storages = []
    for purpose in all_purposes_unique:
        purpose_storage = 0
        for i in range(len(dams_purposes_of_a_basin)):
            if purpose in dams_purposes_of_a_basin[i]:
                purpose_storage = purpose_storage + storages_of_a_basin[i]
        purpose_storages.append(purpose_storage)
    # define a new max function, which return multiple indices when some values are same
    max_indices = multi_max_indices(purpose_storages)
    if len(max_indices) > 1:
        print("choose only one")
        every_level_purposes = []
        max_multi_purpose_types_num = max([len(purpose_temp) for purpose_temp in dams_purposes_of_a_basin])
        for k in range(len(max_indices)):
            key_temp = all_purposes_unique[max_indices[k]]
            # calculate storage for every purpose with different importance
            key_temp_array = np.full(max_multi_purpose_types_num, -1e-6).tolist()
            for j in range(len(dams_purposes_of_a_basin)):
                if key_temp not in dams_purposes_of_a_basin[j]:
                    continue
                index_temp = \
                    [i for i in range(len(dams_purposes_of_a_basin[j])) if dams_purposes_of_a_basin[j][i] == key_temp][
                        0]
                if key_temp_array[index_temp] < 0:
                    # here we use this 'if' to diff 0 and nothing (we use -1e-6 to represent nothing as initial value)
                    key_temp_array[index_temp] = 0
                key_temp_array[index_temp] = key_temp_array[index_temp] + storages_of_a_basin[j]
            every_level_purposes.append(key_temp_array)
        main_purpose_lst = multi_max_indices(every_level_purposes)
        if len(main_purpose_lst) > 1:
            print("multiple main purposes")
            main_purposes_temp = [all_purposes_unique[max_indices[i]] for i in main_purpose_lst]
            main_purpose = ''.join(main_purposes_temp)
        else:
            main_purpose = all_purposes_unique[max_indices[main_purpose_lst[0]]]
    else:
        main_purpose = all_purposes_unique[purpose_storages.index(max(purpose_storages))]
    return main_purpose
