import collections
import os
from configparser import ConfigParser
import geopandas as gpd
import numpy as np
import pandas as pd

from data import wrap_master, DataConfig, DataSource, DataModel
from data.data_input import _trans_norm
from explore import cal_stat_gamma, cal_stat
from utils import hydro_time


class SusquehannaSource(DataSource):
    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)

    def prepare_attr_data(self):
        print("attribute data Ready! ...")

    def read_site_info(self, **kwargs):
        if "HUC10" == self.all_configs["huc"]:
            gage_file = self.all_configs["huc10_shpfile"]
        else:
            gage_file = self.all_configs["huc8_shpfile"]

        data_read = gpd.read_file(gage_file)
        if "HUC10" == self.all_configs["huc"]:
            data = data_read.sort_values(by="HUC10")
            field_lst = ['HUC10', 'AreaSqKm']
        else:
            data = data_read.sort_values(by="HUC8")
            field_lst = ['HUC8', 'AreaSqKm']
        out = dict()
        for s in field_lst:
            out[s] = data[s].values
        return out, field_lst

    def usgs_screen_streamflow(self, streamflow, usgs_ids=None, time_range=None):
        usgs_out = None
        gages_chosen_id = self.gage_dict["HUC10"]
        ts = hydro_time.t_range_days(self.t_range)
        return usgs_out, gages_chosen_id, ts

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='daymet'):
        forcing_lst = self.all_configs["forcing_chosen"]
        data_folder = self.all_configs["forcing_dir"]
        data_file = os.path.join(data_folder, '%s_lump_%s_forcing_leap.txt' % (usgs_id, dataset))
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
            ind = [i for i in range(len(var_lst_in_file)) if var_lst[k] in var_lst_in_file[i]][0]
            out[ind2, k] = data_temp[ind + 4].values[ind1]
        return out

    def read_forcing(self, usgs_id_lst, t_range_list):
        var_lst = self.all_configs["forcing_chosen"]
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
        for k in range(len(usgs_id_lst)):
            data = self.read_forcing_gage(usgs_id_lst[k], var_lst, t_range_list,
                                          dataset=self.all_configs["forcing_type"])
            x[k, :, :] = data
        return x

    def read_attr(self, usgs_id_lst, var_lst):
        f_dict = dict()  # factorize dict
        var_dict = self.all_configs["attr_chosen"]
        var_lst = self.all_configs["attr_chosen"].copy()
        if "PPTAVG_BASIN" in var_lst:
            var_lst.remove("PPTAVG_BASIN")
        out_lst = list()
        gage_dict = self.gage_dict
        if "HUC10" == self.all_configs["huc"]:
            data_file = self.all_configs["huc10_shpfile"]
        else:
            data_file = self.all_configs["huc8_shpfile"]
        data_read = gpd.read_file(data_file)
        if "HUC10" == self.all_configs["huc"]:
            data_temp = data_read.sort_values(by="HUC10")
            n_gage = len(gage_dict['HUC10'])
        else:
            data_temp = data_read.sort_values(by="HUC8")
            n_gage = len(gage_dict['HUC8'])
        k = 0
        out_temp = np.full([n_gage, len(var_lst)], np.nan)
        for field in var_lst:
            out_temp[:, k] = data_temp[field].values
            k = k + 1
        out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_dict, f_dict


class SusquehannaModel(DataModel):
    def __init__(self, data_source, *args):
        super().__init__(data_source, *args)
        print("no related attr data, now read from forcing data")
        forcing_data = self.data_forcing.copy()
        forcing_day_mean = np.mean(forcing_data, axis=1)
        index_ppt = [i for i in range(len(data_source.all_configs["forcing_chosen"])) if
                     data_source.all_configs["forcing_chosen"][i] == "prcp"]
        ppt_vg_basin = forcing_day_mean[:, index_ppt[0]:index_ppt[0] + 1]
        self.ppt_avg_basin = ppt_vg_basin
        if "PPTAVG_BASIN" in data_source.all_configs["attr_chosen"]:
            data_attr_origin = self.data_attr
            data_attr = np.concatenate((data_attr_origin, ppt_vg_basin), axis=1)
            self.data_attr = data_attr
            stat_dict = self.cal_stat_all()
            self.stat_dict = stat_dict

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes."""
        stat_dict = dict()

        # forcing
        forcing_lst = self.data_source.all_configs["forcing_chosen"]
        x = self.data_forcing
        for k in range(len(forcing_lst)):
            var = forcing_lst[k]
            if var == 'prcp':
                stat_dict[var] = cal_stat_gamma(x[:, :, k])
            else:
                stat_dict[var] = cal_stat(x[:, :, k])

        # const attribute
        attr_data = self.data_attr
        attr_lst = self.data_source.all_configs["attr_chosen"]
        if attr_data.shape[1] != len(attr_lst):
            attr_lst_chosen = attr_lst.copy()
            attr_lst_chosen.remove("PPTAVG_BASIN")
            for k in range(len(attr_lst_chosen)):
                var = attr_lst[k]
                stat_dict[var] = cal_stat(attr_data[:, k])
        else:
            for k in range(len(attr_lst)):
                var = attr_lst[k]
                stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_obs(self, rm_nan=True, to_norm=True):
        data = self.data_flow
        return data

    def get_data_ts(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data


class SusquehannaConfig(DataConfig):
    def __init__(self, config_file):
        super().__init__(config_file)
        opt_data, opt_train, opt_model, opt_loss = self.init_model_param()
        self.model_dict = wrap_master(self.data_path, opt_data, opt_model, opt_loss, opt_train)

    @classmethod
    def set_subdir(cls, config_file, subdir):
        """ set_subdir for "temp" and "output" """
        new_data_config = cls(config_file)
        new_data_config.data_path["Out"] = os.path.join(new_data_config.data_path["Out"], subdir)
        new_data_config.data_path["Temp"] = os.path.join(new_data_config.data_path["Temp"], subdir)
        if not os.path.isdir(new_data_config.data_path["Out"]):
            os.makedirs(new_data_config.data_path["Out"])
        if not os.path.isdir(new_data_config.data_path["Temp"]):
            os.makedirs(new_data_config.data_path["Temp"])
        return new_data_config

    def init_data_param(self):
        """read camels or gages dataset configuration"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)
        # huc10 or huc8
        huc = cfg.get(section, options[0])
        # forcing
        forcing_dir = cfg.get(section, options[1])
        forcing_type = cfg.get(section, options[2])
        forcing_url = cfg.get(section, options[3])
        forcing_lst = eval(cfg.get(section, options[4]))

        # attribute
        attr_dir = cfg.get(section, options[5])
        attr_url = eval(cfg.get(section, options[6]))
        attr_str_sel = eval(cfg.get(section, options[7]))

        opt_data = collections.OrderedDict(huc=huc, varT=forcing_lst, forcingDir=forcing_dir, forcingType=forcing_type,
                                           forcingUrl=forcing_url, varC=attr_str_sel, attrDir=attr_dir,
                                           attrUrl=attr_url)

        return opt_data

    def read_data_config(self):
        dir_db = self.data_path.get("DB")
        dir_out = self.data_path.get("Out")
        dir_temp = self.data_path.get("Temp")

        data_params = self.init_data_param()
        susquehanna_huc10_shp_file = os.path.join(dir_db, "shpfile", "HUC10_Susquehanna.shp")
        susquehanna_huc8_shp_file = os.path.join(dir_db, "shpfile", "HUC8_Susquehanna.shp")
        huc = data_params["huc"]
        forcing_chosen = data_params.get("varT")
        forcing_dir = os.path.join(dir_db, data_params.get("forcingDir"))
        forcing_type = data_params.get("forcingType")
        forcing_dir = os.path.join(forcing_dir, forcing_type)
        forcing_url = data_params.get("forcingUrl")
        attr_url = data_params.get("attrUrl")
        attr_chosen = data_params.get("varC")
        attr_dir = os.path.join(dir_db, data_params.get("attrDir"))

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp, huc=huc,
                                       forcing_chosen=forcing_chosen, forcing_dir=forcing_dir,
                                       forcing_type=forcing_type, forcing_url=forcing_url,
                                       attr_url=attr_url, attr_chosen=attr_chosen, attr_dir=attr_dir,
                                       huc10_shpfile=susquehanna_huc10_shp_file, huc8_shpfile=susquehanna_huc8_shp_file,
                                       flow_screen_gage_id=None)
