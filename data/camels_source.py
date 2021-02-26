import json
import os
import numpy as np
import pandas as pd
from data import DataSource
from data.download_data import download_one_zip
from utils import *
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype


class CamelsSource(DataSource):
    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)

    def read_site_info(self, **kwargs):
        """basic info of gages-ii basins
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area
        param **kwargs: none
        Return：
            attibutes in basinid.txt

        """
        gage_file = self.all_configs["gauge_id_file"]

        data = pd.read_csv(gage_file, sep='\t', header=None, skiprows=1, dtype={1: str})
        # header gives some troubles. Skip and hardcode
        field_lst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
        out = dict()
        for s in field_lst:
            if s is 'name':
                out[s] = data[field_lst.index(s)].values.tolist()
            else:
                out[s] = data[field_lst.index(s)].values
        return out, field_lst

    def prepare_forcing_data(self):
        configs = self.all_configs
        data_dir = configs.get('root_dir')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        forcing_url = configs.get('forcing_url')
        download_one_zip(forcing_url, data_dir)
        print("forcing data Ready! ...")

    def read_usgs_gage(self, usgs_id, *, read_qc=False):
        gage_dict = self.gage_dict
        ind = np.argwhere(gage_dict['id'] == usgs_id)[0][0]
        huc = gage_dict['huc'][ind]
        usgs_file = os.path.join(self.all_configs["flow_dir"], str(huc).zfill(2), usgs_id + '_streamflow_qc.txt')
        data_temp = pd.read_csv(usgs_file, sep=r'\s+', header=None)
        obs = data_temp[4].values
        obs[obs < 0] = np.nan
        if read_qc is True:
            qc_dict = {'A': 1, 'A:e': 2, 'M': 3}
            qc = np.array([qc_dict[x] for x in data_temp[5]])
        t_lst = hydro_time.t_range_days(self.t_range)
        nt = t_lst.shape[0]
        if len(obs) != nt:
            out = np.full([nt], np.nan)
            df_date = data_temp[[1, 2, 3]]
            df_date.columns = ['year', 'month', 'day']
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            [C, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
            out[ind2] = obs[ind1]
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

    def read_usgs(self):
        usgs_id_lst = self.gage_dict["id"]
        nt = hydro_time.t_range_days(self.t_range).shape[0]
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            data_obs = self.read_usgs_gage(usgs_id_lst[k])
            y[k, :] = data_obs
        return y

    def usgs_screen_streamflow(self, streamflow, usgs_ids=None, time_range=None):
        """ choose appropriate ones from all usgs sites
            Parameters
            ----------
            streamflow : numpy ndarray -- all usgs sites(config)' data, its index are 'sites', its columns are 'day',
                                   if there is some missing value, usgs should already be filled by nan
            usgs_ids: list -- chosen sites' ids
            time_range: list -- chosen time range

            Returns
            -------
            usgs_out : ndarray -- streamflow  1d-var is gage, 2d-var is day
            sites_chosen: [] -- ids of chosen gages

            Examples
            --------
            usgs_screen(usgs, ["02349000","08168797"], [‘1995-01-01’,‘2015-01-01’])
        """
        sites_chosen = np.zeros(streamflow.shape[0])
        # choose the given sites
        usgs_all_sites = self.gage_dict[self.gage_fld_lst[1]]
        if usgs_ids:
            sites_index = np.where(np.in1d(usgs_all_sites, usgs_ids))[0]
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

        streamflow_temp = streamflow[sites_index]  # raw data
        usgs_values = streamflow_temp[:, ind1]  # column data

        # get discharge data of chosen sites, and change to ndarray
        usgs_out = usgs_values[np.where(sites_chosen > 0)]
        gages_chosen_id = [usgs_all_sites[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]

        return usgs_out, gages_chosen_id, ts

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='nldas'):
        # dataset = daymet or maurer or nldas
        forcing_lst = self.all_configs["forcing_chosen"]
        gage_dict = self.gage_dict
        ind = np.argwhere(gage_dict['id'] == usgs_id)[0][0]
        huc = gage_dict['huc'][ind]

        data_folder = self.all_configs["forcing_dir"]
        if dataset is 'daymet':
            temp_s = 'cida'
        else:
            temp_s = dataset
        data_file = os.path.join(data_folder, str(huc).zfill(2), '%s_lump_%s_forcing_leap.txt' % (usgs_id, temp_s))
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
