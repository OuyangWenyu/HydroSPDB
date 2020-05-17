import collections
import json
import os
from configparser import ConfigParser
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from data import wrap_master, DataConfig, DataSource, DataModel
from data.data_input import _trans_norm
from explore import cal_stat_gamma, cal_stat


class SusquehannaSource(DataSource):
    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)

    def prepare_attr_data(self):
        print("attribute data Ready! ...")

    def read_site_info(self, **kwargs):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area
        param **kwargs: none
        Return：
            各个站点的attibutes in basinid.txt

        """
        if "HUC10" in self.all_configs["attr_chosen"]:
            gage_file = self.all_configs["huc10_shpfile"]
        else:
            gage_file = self.all_configs["huc8_shpfile"]

        data_read = gpd.read_file(gage_file)
        data = data_read.sort_values(by="HUC10")
        # header gives some troubles. Skip and hardcode
        field_lst = ['HUC10', 'AreaSqKm']
        out = dict()
        for s in field_lst:
            out[s] = data[s].values
        return out, field_lst

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='daymet'):
        forcing_lst = self.all_configs["forcing_chosen"]
        gage_dict = self.gage_dict
        ind = np.argwhere(gage_dict['id'] == usgs_id)[0][0]
        huc = gage_dict['huc'][ind]

        data_folder = self.all_configs["forcing_dir"]
        data_file = os.path.join(data_folder, str(huc).zfill(2), '%s_lump_%s_forcing_leap.txt' % (usgs_id, dataset))
        data_temp = pd.read_csv(data_file, sep=r'\s+', header=None, skiprows=4)

        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = forcing_lst.index(var_lst[k])
            out[:, k] = data_temp[ind + 4].values[ind1]
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

    def read_attr_all(self, *, save_dict=False):
        data_folder = self.all_configs["attr_dir"]
        f_dict = dict()  # factorize dict
        var_dict = dict()
        var_lst = list()
        out_lst = list()
        key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
        gage_dict = self.gage_dict
        for key in key_lst:
            data_file = os.path.join(data_folder, 'camels_' + key + '.txt')
            data_temp = pd.read_csv(data_file, sep=';')
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gage_dict['id'])
            out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):
                    value, ref = pd.factorize(data_temp[field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp[field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        if save_dict is True:
            file_name = os.path.join(data_folder, 'dictFactorize.json')
            with open(file_name, 'w') as fp:
                json.dump(f_dict, fp, indent=4)
            file_name = os.path.join(data_folder, 'dictAttribute.json')
            with open(file_name, 'w') as fp:
                json.dump(var_dict, fp, indent=4)
        return out, var_lst, var_dict, f_dict

    def read_attr(self, usgs_id_lst, var_lst, is_return_dict=True):
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        gage_dict = self.gage_dict
        id_lst_all = gage_dict['id']
        c, ind_grid, ind2 = np.intersect1d(id_lst_all, usgs_id_lst, return_indices=True)
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out


class SusquehannaModel(DataModel):
    def __init__(self, data_source, *args):
        super().__init__(data_source, *args)

    def cal_stat_all(self):
        """calculate statistics of streamflow, forcing and attributes. 计算统计值，便于后面归一化处理。"""
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
        for k in range(len(attr_lst)):
            var = attr_lst[k]
            stat_dict[var] = cal_stat(attr_data[:, k])
        return stat_dict

    def get_data_ts(self, rm_nan=True, to_norm=True):
        stat_dict = self.stat_dict
        var_lst = self.data_source.all_configs.get("forcing_chosen")
        data = self.data_forcing
        data = _trans_norm(data, var_lst, stat_dict, to_norm=to_norm)
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        return data


class SusquehannaModels(object):
    def __init__(self, config_data):
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = SusquehannaSource(config_data, t_train_test)
        # 构建输入数据类对象
        data_model = SusquehannaModel(source_data)
        self.data_model_train, self.data_model_test = SusquehannaModel.data_models_of_train_test(data_model, t_train,
                                                                                                 t_test)


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
        """read camels or gages dataset configuration
        根据配置文件读取有关输入数据的各项参数"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)

        # forcing
        forcing_dir = cfg.get(section, options[0])
        forcing_type = cfg.get(section, options[1])
        forcing_url = cfg.get(section, options[2])
        forcing_lst = eval(cfg.get(section, options[3]))

        # attribute
        attr_dir = cfg.get(section, options[4])
        attr_url = eval(cfg.get(section, options[5]))
        attr_str_sel = eval(cfg.get(section, options[6]))

        opt_data = collections.OrderedDict(varT=forcing_lst, forcingDir=forcing_dir, forcingType=forcing_type,
                                           forcingUrl=forcing_url,
                                           varC=attr_str_sel, attrDir=attr_dir, attrUrl=attr_url)

        return opt_data

    def read_data_config(self):
        dir_db = self.data_path.get("DB")
        dir_out = self.data_path.get("Out")
        dir_temp = self.data_path.get("Temp")

        data_params = self.init_data_param()
        # 站点的shp file
        susquehanna_huc10_shp_file = os.path.join(dir_db, "shpfile", "HUC10_Susquehanna.shp")
        susquehanna_huc8_shp_file = os.path.join(dir_db, "shpfile", "HUC8_Susquehanna.shp")
        # 所选forcing
        forcing_chosen = data_params.get("varT")
        forcing_dir = os.path.join(dir_db, data_params.get("forcingDir"))
        forcing_type = data_params.get("forcingType")
        # 有了forcing type之后，确定到真正的forcing数据文件夹
        forcing_dir = os.path.join(forcing_dir, forcing_type)
        forcing_url = data_params.get("forcingUrl")
        # 所选属性
        attr_url = data_params.get("attrUrl")
        attr_chosen = data_params.get("varC")
        attr_dir = os.path.join(dir_db, data_params.get("attrDir"))

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       forcing_chosen=forcing_chosen, forcing_dir=forcing_dir,
                                       forcing_type=forcing_type, forcing_url=forcing_url,
                                       attr_url=attr_url, attr_chosen=attr_chosen, attr_dir=attr_dir,
                                       huc10_shpfile=susquehanna_huc10_shp_file, huc8_shpfile=susquehanna_huc8_shp_file)
