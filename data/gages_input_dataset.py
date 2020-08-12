"""for stacked lstm"""
import copy
import operator
import os
from calendar import isleap
from functools import reduce

import pandas as pd
import torch
import numpy as np
import geopandas as gpd
from scipy import constants, interpolate
import definitions
from data import DataModel, GagesSource, GagesConfig
from data.data_config import update_config_item
from data.data_input import GagesModel, _trans_norm, GagesModelWoBasinNorm, load_result
from explore import trans_norm, cal_stat
from explore.hydro_cluster import cluster_attr_train
from explore.stat import statError
from hydroDL import master_train
from hydroDL.model import model_run
from utils import hydro_time
from utils.dataset_format import subset_of_dict
from utils.hydro_math import concat_two_3darray, copy_attr_array_in2d


def load_pub_ensemble_result(pub_exp, trained_exp_lst, test_epoch, split_num=None, return_value=False):
    preds = []
    obss = []
    dor_config_data = load_dataconfig_case_exp(pub_exp)
    if split_num is None:
        for i in range(len(trained_exp_lst)):
            pretrained_model_name = trained_exp_lst[i] + "_pretrained_model"
            save_dir_i = os.path.join(dor_config_data.data_path['Out'], pretrained_model_name)
            pred_i, obs_i = load_result(save_dir_i, test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            print(obs_i)
            preds.append(pred_i)
            obss.append(obs_i)

        preds_np = np.array(preds)
        obss_np = np.array(obss)
        pred_mean = np.mean(preds_np, axis=0)
        obs_mean = np.mean(obss_np, axis=0)
        inds = statError(obs_mean, pred_mean)
        inds_df = pd.DataFrame(inds)
        if return_value:
            return inds_df, pred_mean, obs_mean
        return inds_df
    else:
        for i in range(split_num):
            pretrained_model_name = trained_exp_lst[0] + "_pretrained_model" + str(i)
            save_dir_i = os.path.join(dor_config_data.data_path['Out'], pretrained_model_name)
            pred_i, obs_i = load_result(save_dir_i, test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            print(obs_i)
            preds.append(pred_i)
            obss.append(obs_i)
        predsbase_np = reduce(lambda a, b: np.vstack((a, b)), preds)
        obssbase_np = reduce(lambda a, b: np.vstack((a, b)), obss)
        indsbase = statError(obssbase_np, predsbase_np)
        inds_df_abase = pd.DataFrame(indsbase)
        if return_value:
            return inds_df_abase, predsbase_np, obssbase_np
        return inds_df_abase


def load_ensemble_result(cases_exps, test_epoch, return_value=False):
    preds = []
    obss = []
    for case_exp in cases_exps:
        config_data_i = load_dataconfig_case_exp(case_exp)
        pred_i, obs_i = load_result(config_data_i.data_path['Temp'], test_epoch)
        pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
        obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
        print(obs_i)
        preds.append(pred_i)
        obss.append(obs_i)
    preds_np = np.array(preds)
    obss_np = np.array(obss)
    pred_mean = np.mean(preds_np, axis=0)
    obs_mean = np.mean(obss_np, axis=0)
    inds = statError(obs_mean, pred_mean)
    inds_df = pd.DataFrame(inds)
    if return_value:
        return inds_df, pred_mean, obs_mean
    return inds_df


def load_pub_test_result(config_data, i, test_epoch):
    """i means the ith experiment of k-fold"""
    data_model = GagesModel.load_datamodel(config_data.data_path["Temp"], str(i),
                                           data_source_file_name='test_data_source.txt',
                                           stat_file_name='test_Statistics.json',
                                           flow_file_name='test_flow.npy',
                                           forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                           f_dict_file_name='test_dictFactorize.json',
                                           var_dict_file_name='test_dictAttribute.json',
                                           t_s_dict_file_name='test_dictTimeSpace.json')
    data_model_majordam = GagesModel.load_datamodel(config_data.data_path["Temp"], str(i),
                                                    data_source_file_name='test_data_source_largedor.txt',
                                                    stat_file_name='test_Statistics_largedor.json',
                                                    flow_file_name='test_flow_largedor.npy',
                                                    forcing_file_name='test_forcing_largedor.npy',
                                                    attr_file_name='test_attr_largedor.npy',
                                                    f_dict_file_name='test_dictFactorize_largedor.json',
                                                    var_dict_file_name='test_dictAttribute_largedor.json',
                                                    t_s_dict_file_name='test_dictTimeSpace_largedor.json')
    pred, obs = load_result(data_model.data_source.data_config.data_path['Temp'], test_epoch)
    pred = pred.reshape(pred.shape[0], pred.shape[1])
    obs = obs.reshape(obs.shape[0], obs.shape[1])
    inds = statError(obs, pred)
    inds['STAID'] = data_model.t_s_dict["sites_id"]
    inds_df = pd.DataFrame(inds)

    pred_majordam, obs_majordam = load_result(data_model_majordam.data_source.data_config.data_path['Temp'],
                                              test_epoch, pred_name='flow_pred_largedor',
                                              obs_name='flow_obs_largedor')
    pred_majordam = pred_majordam.reshape(pred_majordam.shape[0], pred_majordam.shape[1])
    obs_majordam = obs_majordam.reshape(obs_majordam.shape[0], obs_majordam.shape[1])
    inds_majordam = statError(obs_majordam, pred_majordam)
    inds_majordam['STAID'] = data_model_majordam.t_s_dict["sites_id"]
    inds_majordam_df = pd.DataFrame(inds_majordam)
    return inds_df, inds_majordam_df


def load_dataconfig_case_exp(case_exp):
    config_dir = definitions.CONFIG_DIR
    (case, exp) = case_exp.split("_")
    if case == "inv" or case == "simulate" or case == "storage":
        config_file_i = os.path.join(config_dir, case + "/config2_" + exp + ".ini")
        subdir = case + "/" + exp
    elif case == "siminv":
        config_file_i = os.path.join(config_dir, case + "/config3_" + exp + ".ini")
        subdir = case + "/" + exp
    elif case == 'dam':
        config_file_i = os.path.join(config_dir, case + "/config_" + exp + ".ini")
        subdir = case + "/" + exp
        if not os.path.isfile(config_file_i):
            config_file_i = os.path.join(config_dir, case + "/config2_" + exp + ".ini")
            if not os.path.isfile(config_file_i):
                config_file_i = os.path.join(config_dir, case + "/config3_" + exp + ".ini")
    else:
        config_file_i = os.path.join(config_dir, case + "/config_" + exp + ".ini")
        subdir = case + "/" + exp
    config_data_i = GagesConfig.set_subdir(config_file_i, subdir)
    return config_data_i


def load_datamodel_case_exp(case_exp):
    (case, exp) = case_exp.split("_")
    config_data_i = load_dataconfig_case_exp(case_exp)
    if case == "inv" or case == "simulate":
        data_model_i = GagesModel.load_datamodel(config_data_i.data_path["Temp"], "2",
                                                 data_source_file_name='test_data_source.txt',
                                                 stat_file_name='test_Statistics.json',
                                                 flow_file_name='test_flow.npy',
                                                 forcing_file_name='test_forcing.npy',
                                                 attr_file_name='test_attr.npy',
                                                 f_dict_file_name='test_dictFactorize.json',
                                                 var_dict_file_name='test_dictAttribute.json',
                                                 t_s_dict_file_name='test_dictTimeSpace.json')
    elif case == "siminv":
        data_model_i = GagesModel.load_datamodel(config_data_i.data_path["Temp"], "3",
                                                 data_source_file_name='test_data_source.txt',
                                                 stat_file_name='test_Statistics.json',
                                                 flow_file_name='test_flow.npy',
                                                 forcing_file_name='test_forcing.npy',
                                                 attr_file_name='test_attr.npy',
                                                 f_dict_file_name='test_dictFactorize.json',
                                                 var_dict_file_name='test_dictAttribute.json',
                                                 t_s_dict_file_name='test_dictTimeSpace.json')
    else:
        data_model_i = GagesModel.load_datamodel(config_data_i.data_path["Temp"],
                                                 data_source_file_name='test_data_source.txt',
                                                 stat_file_name='test_Statistics.json',
                                                 flow_file_name='test_flow.npy',
                                                 forcing_file_name='test_forcing.npy',
                                                 attr_file_name='test_attr.npy',
                                                 f_dict_file_name='test_dictFactorize.json',
                                                 var_dict_file_name='test_dictAttribute.json',
                                                 t_s_dict_file_name='test_dictTimeSpace.json')
    return data_model_i


class GagesModels(object):
    """the data model for GAGES-II dataset"""

    def __init__(self, config_data, screen_basin_area_huc4=True, **kwargs):
        # 准备训练数据
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = GagesSource.choose_some_basins(config_data, t_train_test,
                                                     screen_basin_area_huc4=screen_basin_area_huc4, **kwargs)
        # 构建输入数据类对象
        data_model = GagesModel(source_data)
        self.data_model_train, self.data_model_test = GagesModel.data_models_of_train_test(data_model, t_train, t_test)


class GagesModelsWoBasinNorm(object):
    def __init__(self, config_data, screen_basin_area_huc4=True, **kwargs):
        # 准备训练数据
        t_train = config_data.model_dict["data"]["tRangeTrain"]
        t_test = config_data.model_dict["data"]["tRangeTest"]
        t_train_test = [t_train[0], t_test[1]]
        source_data = GagesSource.choose_some_basins(config_data, t_train_test,
                                                     screen_basin_area_huc4=screen_basin_area_huc4, **kwargs)
        # 构建输入数据类对象
        data_model = GagesModelWoBasinNorm(source_data)
        self.data_model_train, self.data_model_test = GagesModelWoBasinNorm.data_models_of_train_test(data_model,
                                                                                                      t_train, t_test)


class GagesTsDataModel(object):
    """the data model for GAGES-II dataset with GAGES-II time series data"""

    def __init__(self, data_model):
        self.data_model = data_model
        self.data_source = data_model.data_source
        self.stat_dict = data_model.stat_dict
        self.t_s_dict = data_model.t_s_dict
        self.water_use_years, self.pop_years = self.read_gagesii_tsdata()

    def read_gagesii_tsdata(self):
        # deal with water use and population data, interpolate the data for every year of 1980-2015
        t_range_all = self.data_source.all_configs["t_range_all"]
        all_start_year = int(t_range_all[0].split("-")[0])
        # left closed right open interval
        all_end_year = int(t_range_all[1].split("-")[0])
        num_of_years = int(
            (np.datetime64(str(all_end_year)) - np.datetime64(str(all_start_year))) / np.timedelta64(1, 'Y'))
        # closed interval
        water_use_start_year = 1985
        water_use_end_year = 2010
        assert all_start_year < water_use_start_year
        assert water_use_end_year < all_end_year
        water_use_start_year_idx = water_use_start_year - all_start_year
        water_use_end_year_idx = water_use_end_year - all_start_year
        # closed interval
        pop_start_year = 1990
        pop_end_year = 2010
        assert all_start_year < pop_start_year
        assert pop_end_year < all_end_year
        pop_start_year_idx = pop_start_year - all_start_year
        pop_end_year_idx = pop_end_year - all_start_year

        ids_now = self.data_model.t_s_dict["sites_id"]
        assert (all(x < y for x, y in zip(ids_now, ids_now[1:])))
        attr_lst = ["DRAIN_SQKM"]
        basins_area, var_dict, f_dict = self.data_source.read_attr(ids_now, attr_lst)

        # mean freshwater withdrawals in units of millions of gallons per day per square kilometer for five-year periods from 1985 to 2010
        water_use_df = pd.read_csv(self.data_source.all_configs["wateruse_file"], sep=',', dtype={0: str})
        water_use_df.sort_values(by=['STAID'])
        water_use_np = water_use_df.iloc[:, 1:].values
        vectorized_unit_trans = np.vectorize(lambda gal: gal * (10 ** 6) * constants.gallon)
        water_use_np = vectorized_unit_trans(water_use_np)

        def interpolate_wateruse(water_use_y):
            # 1985-1990-1995-2000-2005-2010
            water_use_x = np.linspace(water_use_start_year_idx, water_use_end_year_idx, num=6)
            water_use_xs = np.linspace(0, num_of_years - 1, num=num_of_years)
            # interpolate by (x,y)，apply to xs
            water_use_ys = interpolate.UnivariateSpline(water_use_x, water_use_y, s=0)(water_use_xs)
            return water_use_ys

        water_use_interpolates = np.apply_along_axis(interpolate_wateruse, 1, water_use_np)
        water_use_interpolates[np.where(water_use_interpolates < 0)] = 0
        water_use_newdf = pd.DataFrame(water_use_interpolates, index=water_use_df["STAID"])
        water_use_chosen_df = water_use_newdf.loc[ids_now]
        # avg value for per sqkm -> avg value for a basin
        water_use_basin = water_use_chosen_df.values * basins_area
        time_range = np.arange(all_start_year, all_end_year)
        water_use_chosen = pd.DataFrame(water_use_basin, index=water_use_chosen_df.index, columns=time_range)

        # one series for population density, in units of persons per square kilometer (sq km) and one for housing unit density, in units of housing units per sq km
        population_and_house_df = pd.read_csv(self.data_source.all_configs["population_file"], sep=',', dtype={0: str})
        population_and_house_df.sort_values(by=['STAID'])
        population_np = population_and_house_df.iloc[:, 1:4].values

        def interpolate_pop(pop_y):
            """only 3 years data, UnivariateSpline can't work, so just repeat it:
            80-95 use 90 data; 95-05 use 00 data; 05-15 use 10 data"""
            pop_ys = np.array(
                pop_y[0].repeat(15).tolist() + pop_y[1].repeat(10).tolist() + pop_y[2].repeat(10).tolist())
            return pop_ys

        population_interpolates = np.apply_along_axis(interpolate_pop, 1, population_np)
        population_interpolates[np.where(population_interpolates < 0)] = 0
        population_newdf = pd.DataFrame(population_interpolates, index=population_and_house_df["STAID"])
        pop_chosen_df = population_newdf.loc[ids_now]
        # avg value for per sqkm -> avg value for a basin
        population_basin = pop_chosen_df.values * basins_area
        pop_chosen = pd.DataFrame(population_basin, index=pop_chosen_df.index, columns=time_range)

        return water_use_chosen, pop_chosen

    def load_data(self, model_dict):
        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        x, y, c = self.data_model.load_data(model_dict)
        # concatenate water use and pop data with attr data
        t_range_all = self.data_source.all_configs["t_range_all"]
        all_start_year = int(t_range_all[0].split("-")[0])
        water_use_df = self.water_use_years
        start_date_str = self.data_model.t_s_dict["t_final_range"][0]
        end_date_str = self.data_model.t_s_dict["t_final_range"][1]

        def copy_every_year(start_date_str_tmp, end_date_str_tmp, all_start_year_tmp, data_df_tmp, rm_nan_tmp):
            data_lst = []
            start_year = int(start_date_str_tmp.split("-")[0])
            end_year = int(end_date_str_tmp.split("-")[0])
            start_date = np.datetime64(start_date_str_tmp)
            start_year_final_day = np.datetime64(str(start_year) + '-12-31')
            # start_date should be contained
            first_year_days_num = int((start_year_final_day - start_date) / np.timedelta64(1, 'D')) + 1
            first_year_data_idx = start_year - all_start_year_tmp
            first_year_np = data_df_tmp.iloc[:, first_year_data_idx].values
            first_year_data = np.tile(first_year_np.reshape(first_year_np.size, 1), (1, first_year_days_num))
            data_lst.append(first_year_data)
            end_date = np.datetime64(end_date_str_tmp)
            end_year_first_day = np.datetime64(str(end_year) + '-01-01')
            # rignt open interval, so end_date should NOT be contained
            end_year_days_num = int((end_date - end_year_first_day) / np.timedelta64(1, 'D'))
            end_year_data_idx = end_year - all_start_year_tmp
            end_year_np = data_df_tmp.iloc[:, end_year_data_idx].values
            end_year_data = np.tile(end_year_np.reshape(end_year_np.size, 1), (1, end_year_days_num))
            for idx in range(first_year_data_idx + 1, end_year_data_idx):
                data_year_np = data_df_tmp.iloc[:, idx].values
                if isleap(all_start_year_tmp + idx):
                    year_days_num = 366
                else:
                    year_days_num = 365
                year_water_use = np.tile(data_year_np.reshape(data_year_np.size, 1), (1, year_days_num))
                data_lst.append(year_water_use)
            data_lst.append(end_year_data)
            data_x = reduce(lambda x_tmp, y_tmp: np.hstack((x_tmp, y_tmp)), data_lst)
            if rm_nan_tmp:
                data_x[np.where(np.isnan(data_x))] = 0
            return data_x

        water_use_x = copy_every_year(start_date_str, end_date_str, all_start_year, water_use_df, rm_nan_x)
        # firstly, water_use_c should be normalized, then concatenate with x
        water_use_x = water_use_x.reshape(water_use_x.shape[0], water_use_x.shape[1], 1)
        var_lst = ['wu', 'pop']
        gagests_stat_dict = {}
        gagests_stat_dict[var_lst[0]] = cal_stat(water_use_x)
        pop_df = self.pop_years
        pop_x = copy_every_year(start_date_str, end_date_str, all_start_year, pop_df, rm_nan_x)
        pop_x = pop_x.reshape(pop_x.shape[0], pop_x.shape[1], 1)
        gagests_stat_dict[var_lst[1]] = cal_stat(pop_x)
        wu_pop_x = concat_two_3darray(water_use_x, pop_x)
        wu_pop_x = _trans_norm(wu_pop_x, var_lst, gagests_stat_dict, to_norm=True)
        new_x = concat_two_3darray(x, wu_pop_x)
        return new_x, y, c


class GagesJulianDataModel(object):
    def __init__(self, data_model_natflow, data_model_lstm):
        self.sim_data_model = GagesSimDataModel(data_model_natflow, data_model_lstm)
        self.data_model2 = self.sim_data_model.data_model2

    def load_data(self, model_dict):
        qx, y, c = self.sim_data_model.load_data(model_dict)
        julian_date = hydro_time.t_range_to_julian(self.sim_data_model.t_s_dict["t_final_range"])
        sites_num = c.shape[0]
        sites_julian = np.tile(julian_date, (sites_num, 1))
        sites_julian_concat = sites_julian.reshape(sites_julian.shape[0], sites_julian.shape[1], 1)
        var_lst = ['julian']
        julian_stat_dict = {}
        julian_stat_dict[var_lst[0]] = cal_stat(sites_julian_concat)
        julian_x = _trans_norm(sites_julian_concat, var_lst, julian_stat_dict, to_norm=True)
        x = concat_two_3darray(qx, julian_x)
        return x, y, c


class GagesStorageDataModel(object):
    def __init__(self, data_model_natflow, data_model_storage):
        self.sim_data_model = GagesSimDataModel(data_model_natflow, data_model_storage)
        self.data_model_natflow = data_model_natflow
        self.data_model_storage = data_model_storage

    def load_data(self):
        model_dict = self.data_model_storage.data_source.data_config.model_dict
        qx, y, c = self.sim_data_model.load_data(model_dict)
        # cut data: because the input of LSTM_storage need previous days' data
        storage_seq_length = model_dict['model']["storageLength"]
        qx_cut = qx[:, storage_seq_length - 1:, :]
        y_cut = y[:, storage_seq_length - 1:, :]
        natflow = self.sim_data_model.natural_flow
        return qx_cut, c, natflow, y_cut


class GagesSimDataModel(object):
    """DataModel for sim model"""

    def __init__(self, data_model1, data_model2):
        self.data_model1 = data_model1
        self.data_model2 = data_model2
        self.t_s_dict = data_model2.t_s_dict
        self.natural_flow = self.read_natural_inflow()

    def read_natural_inflow(self):
        sim_model_data = self.data_model1
        sim_config_data = sim_model_data.data_source.data_config
        # read model
        # firstly, check if the model used to generate natural flow has existed
        out_folder = sim_config_data.data_path["Out"]
        epoch = sim_config_data.model_dict["train"]["nEpoch"]
        model_file = os.path.join(out_folder, 'model_Ep' + str(epoch) + '.pt')
        if not os.path.isfile(model_file):
            master_train(sim_model_data)
        model = torch.load(model_file)
        # run the model
        model_data = self.data_model2
        config_data = model_data.data_source.data_config
        model_dict = config_data.model_dict
        batch_size = model_dict["train"]["miniBatch"][0]
        x, y, c = model_data.load_data(model_dict)
        t_range = self.t_s_dict["t_final_range"]
        natural_epoch = model_dict["train"]["nEpoch"]
        file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(natural_epoch)])
        file_path = os.path.join(out_folder, file_name) + '.csv'
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        # read natural_flow from file
        np_natural_flow = pd.read_csv(file_path, dtype=np.float, header=None).values
        return np_natural_flow

    def get_data_inflow(self, rm_nan=True):
        """径流数据读取及归一化处理，会处理成三维，最后一维长度为1，表示径流变量"""
        data = self.natural_flow
        if rm_nan is True:
            data[np.where(np.isnan(data))] = 0
        # transform x to 3d, the final dim's length is the seq_length
        seq_length = self.data_model2.data_source.data_config.model_dict["model"]["seqLength"]
        data_inflow = np.zeros([data.shape[0], data.shape[1] - seq_length + 1, seq_length])
        for i in range(data_inflow.shape[1]):
            data_inflow[:, i, :] = data[:, i:i + seq_length]
        return data_inflow

    def load_data(self, model_dict):
        """transform x to 3d, the final dim's length is the seq_length, add forcing with natural flow"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length):
            """cut to size same as inflow's"""
            temp = temp_x[:, temp_seq_length - 1:, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        q = self.get_data_inflow(rm_nan=rm_nan_x)
        x, y, c = self.data_model2.load_data(model_dict)
        seq_length = model_dict["model"]["seqLength"]

        if seq_length > 1:
            x = cut_data(x, rm_nan_x, seq_length)
            y = cut_data(y, rm_nan_y, seq_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


class GagesInvDataModel(object):
    """DataModel for inv model"""

    def __init__(self, data_model1, data_model2):
        self.model_dict1 = data_model1.data_source.data_config.model_dict
        self.model_dict2 = data_model2.data_source.data_config.model_dict
        self.stat_dict = data_model2.stat_dict
        self.t_s_dict = data_model2.t_s_dict
        all_data = self.prepare_input(data_model1, data_model2)
        input_keys = ['xh', 'ch', 'qh', 'xt', 'ct']
        output_keys = ['qt']
        self.data_input = subset_of_dict(all_data, input_keys)
        self.data_target = subset_of_dict(all_data, output_keys)

    def prepare_input(self, data_model1, data_model2):
        """prepare input for lstm-inv, gages_id may be different, fix it here"""
        print("prepare input")
        sites_id1 = data_model1.t_s_dict['sites_id']
        sites_id2 = data_model2.t_s_dict['sites_id']
        assert sites_id1 == sites_id2
        # # if sites_id1 != sites_id2, need to be handled, but notice that it's very easy to misuse "intersect1d"
        # sites_id, ind1, ind2 = np.intersect1d(sites_id1, sites_id2, return_indices=True)
        # data_model1.data_attr = data_model1.data_attr[ind1, :]
        # data_model1.data_flow = data_model1.data_flow[ind1, :]
        # data_model1.data_forcing = data_model1.data_forcing[ind1, :]
        # data_model2.data_attr = data_model2.data_attr[ind2, :]
        # data_model2.data_flow = data_model2.data_flow[ind2, :]
        # data_model2.data_forcing = data_model2.data_forcing[ind2, :]
        # data_model1.t_s_dict['sites_id'] = sites_id
        # data_model2.t_s_dict['sites_id'] = sites_id
        model_dict1 = data_model1.data_source.data_config.model_dict
        xh, qh, ch = data_model1.load_data(model_dict1)
        model_dict2 = data_model2.data_source.data_config.model_dict
        xt, qt, ct = data_model2.load_data(model_dict2)
        return {'xh': xh, 'ch': ch, 'qh': qh, 'xt': xt, 'ct': ct, 'qt': qt}

    def load_data(self):
        data_input = self.data_input
        data_inflow_h = data_input['qh']
        data_inflow_h = data_inflow_h.reshape(data_inflow_h.shape[0], data_inflow_h.shape[1])
        # transform x to 3d, the final dim's length is the seq_length
        seq_length = self.model_dict1["model"]["seqLength"]
        data_inflow_h_new = np.zeros([data_inflow_h.shape[0], data_inflow_h.shape[1] - seq_length + 1, seq_length])
        for i in range(data_inflow_h_new.shape[1]):
            data_inflow_h_new[:, i, :] = data_inflow_h[:, i:i + seq_length]

        # because data_inflow_h_new is assimilated, time sequence length has changed
        data_forcing_h = data_input['xh'][:, seq_length - 1:, :]
        xqh = concat_two_3darray(data_inflow_h_new, data_forcing_h)

        attr_h = data_input['ch']
        attr_h_new = copy_attr_array_in2d(attr_h, xqh.shape[1])

        # concatenate xqh with ch
        xqch = concat_two_3darray(xqh, attr_h_new)

        # concatenate xt with ct
        data_forcing_t = data_input['xt']
        attr_t = data_input['ct']
        attr_t_new = copy_attr_array_in2d(attr_t, data_forcing_t.shape[1])
        xct = concat_two_3darray(data_forcing_t, attr_t_new)

        qt = self.data_target["qt"]
        return xqch, xct, qt


class GagesSimInvDataModel(object):
    """DataModel for siminv model"""

    def __init__(self, data_model1, data_model2, data_model3):
        self.sim_model = data_model1
        self.inv_model = data_model2
        self.lstm_model = data_model3
        all_data = self.prepare_input(data_model1, data_model2, data_model3)
        input_keys = ['xh', 'ch', 'qh', 'qnh', 'xt', 'ct', 'qnt']
        output_keys = ['qt']
        self.data_input = subset_of_dict(all_data, input_keys)
        self.data_target = subset_of_dict(all_data, output_keys)

    def read_natural_inflow(self, sim_model_data, model_data):
        sim_config_data = sim_model_data.data_source.data_config
        # read model
        # firstly, check if the model used to generate natural flow has existed
        out_folder = sim_config_data.data_path["Out"]
        epoch = sim_config_data.model_dict["train"]["nEpoch"]
        model_file = os.path.join(out_folder, 'model_Ep' + str(epoch) + '.pt')
        if not os.path.isfile(model_file):
            master_train(sim_model_data)
        model = torch.load(model_file)
        # run the model
        config_data = model_data.data_source.data_config
        model_dict = config_data.model_dict
        batch_size = model_dict["train"]["miniBatch"][0]
        x, y, c = model_data.load_data(model_dict)
        t_range = model_data.t_s_dict["t_final_range"]
        natural_epoch = model_dict["train"]["nEpoch"]
        file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(natural_epoch)])
        file_path = os.path.join(out_folder, file_name) + '.csv'
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        # read natural_flow from file
        np_natural_flow = pd.read_csv(file_path, dtype=np.float, header=None).values
        return np_natural_flow

    def prepare_input(self, data_model1, data_model2, data_model3):
        """prepare input for lstm-inv, gages_id of data_model2 and data_model3 must be same"""
        print("prepare input")
        sites_id2 = data_model2.t_s_dict['sites_id']
        sites_id3 = data_model3.t_s_dict['sites_id']
        assert sites_id2 == sites_id3
        sim_flow = self.read_natural_inflow(data_model1, data_model2)
        qnh = np.expand_dims(sim_flow, axis=2)
        model_dict2 = data_model2.data_source.data_config.model_dict
        xh, qh, ch = data_model2.load_data(model_dict2)
        sim_flow43rd_model = self.read_natural_inflow(data_model1, data_model3)
        qnt = np.expand_dims(sim_flow43rd_model, axis=2)
        model_dict3 = data_model3.data_source.data_config.model_dict
        xt, qt, ct = data_model3.load_data(model_dict3)
        return {'xh': xh, 'ch': ch, 'qh': qh, 'qnh': qnh, 'xt': xt, 'ct': ct, 'qt': qt, 'qnt': qnt}

    def load_data(self):
        data_input = self.data_input
        data_inflow_h = data_input['qh']
        data_nat_inflow_h = data_input['qnh']
        seq_length = self.inv_model.data_source.data_config.model_dict["model"]["seqLength"]

        def trans_to_tim_seq(data_now, seq_length_now):
            data_now = data_now.reshape(data_now.shape[0], data_now.shape[1])
            # the final dim's length is the seq_length
            data_now_new = np.zeros([data_now.shape[0], data_now.shape[1] - seq_length_now + 1, seq_length_now])
            for i in range(data_now_new.shape[1]):
                data_now_new[:, i, :] = data_now[:, i:i + seq_length_now]
            return data_now_new

        data_inflow_h_new = trans_to_tim_seq(data_inflow_h, seq_length)
        data_nat_inflow_h_new = trans_to_tim_seq(data_nat_inflow_h, seq_length)
        qqnh = concat_two_3darray(data_inflow_h_new, data_nat_inflow_h_new)
        # because data_inflow_h_new is assimilated, time sequence length has changed
        data_forcing_h = data_input['xh'][:, seq_length - 1:, :]
        xqqnh = concat_two_3darray(qqnh, data_forcing_h)

        def copy_attr_array_in2d(arr1, len_of_2d):
            arr2 = np.zeros([arr1.shape[0], len_of_2d, arr1.shape[1]])
            for k in range(arr1.shape[0]):
                arr2[k] = np.tile(arr1[k], arr2.shape[1]).reshape(arr2.shape[1], arr1.shape[1])
            return arr2

        attr_h = data_input['ch']
        attr_h_new = copy_attr_array_in2d(attr_h, xqqnh.shape[1])

        # concatenate xqh with ch
        xqqnch = concat_two_3darray(xqqnh, attr_h_new)

        # concatenate xt with ct
        # data_forcing_t = data_input['xt']
        # attr_t = data_input['ct']
        # attr_t_new = copy_attr_array_in2d(attr_t, data_forcing_t.shape[1])
        # xct = concat_two_3darray(data_forcing_t, attr_t_new)
        # return xqqnch, xct, qt

        # use natural flow in 3rd lstm
        data_nat_inflow_t = data_input['qnt']
        data_nat_inflow_t_new = trans_to_tim_seq(data_nat_inflow_t, seq_length)
        data_forcing_t = data_input['xt'][:, seq_length - 1:, :]
        xqnt = concat_two_3darray(data_nat_inflow_t_new, data_forcing_t)
        attr_t = data_input['ct']
        attr_t_new = copy_attr_array_in2d(attr_t, data_forcing_t.shape[1])
        xqnct = concat_two_3darray(xqnt, attr_t_new)
        qt = self.data_target["qt"]
        return xqqnch, xqnct, qt


class GagesDaDataModel(object):
    """DataModel for da model"""

    def __init__(self, data_model):
        self.data_model = data_model

    def load_data(self, model_dict):
        """Notice that don't cover the data of today when loading history data"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length):
            """cut to size same as inflow's. Don't cover the data of today when loading history data"""
            temp = temp_x[:, temp_seq_length:, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        x, y, c = self.data_model.load_data(model_dict)

        seq_length = model_dict["model"]["seqLength"]
        # don't cover the data of today when loading history data, so the length of 2nd dim is 'y.shape[1] - seq_length'
        flow = y.reshape(y.shape[0], y.shape[1])
        q = np.zeros([flow.shape[0], flow.shape[1] - seq_length, seq_length])
        for i in range(q.shape[1]):
            q[:, i, :] = flow[:, i:i + seq_length]

        if rm_nan_x is True:
            q[np.where(np.isnan(q))] = 0

        if seq_length > 1:
            x = cut_data(x, rm_nan_x, seq_length)
            y = cut_data(y, rm_nan_y, seq_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


class GagesForecastDataModel(object):
    """DataModel for assimilation of forecast data"""

    def __init__(self, sim_data_model, data_model):
        self.sim_data_model = sim_data_model
        self.model_data = data_model
        self.natural_flow = self.read_natural_inflow_and_forecast()

    def read_natural_inflow_and_forecast(self):
        sim_model_data = self.sim_data_model
        sim_config_data = sim_model_data.data_source.data_config
        # read model
        # firstly, check if the model used to generate natural flow has existed
        out_folder = sim_config_data.data_path["Out"]
        epoch = sim_config_data.model_dict["train"]["nEpoch"]
        model_file = os.path.join(out_folder, 'model_Ep' + str(epoch) + '.pt')
        if not os.path.isfile(model_file):
            master_train(sim_model_data)
        model = torch.load(model_file)
        # run the model
        model_data = self.model_data
        config_data = model_data.data_source.data_config
        model_dict = config_data.model_dict
        batch_size = model_dict["train"]["miniBatch"][0]
        x, y, c = model_data.load_data(model_dict)
        t_range = self.model_data.t_s_dict["t_final_range"]
        natural_epoch = model_dict["train"]["nEpoch"]
        file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(natural_epoch)])
        file_path = os.path.join(out_folder, file_name) + '.csv'
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
        # read natural_flow from file
        np_natural_flow = pd.read_csv(file_path, dtype=np.float, header=None).values
        return np_natural_flow

    def load_data(self, model_dict):
        """Notice that don't cover the data of today when loading history data"""

        def cut_data(temp_x, temp_rm_nan, temp_seq_length, temp_fcst_length):
            """cut to size same as inflow's. Cover future natural flow"""
            temp = temp_x[:, temp_seq_length - 1: -temp_fcst_length, :]
            if temp_rm_nan:
                temp[np.where(np.isnan(temp))] = 0
            return temp

        opt_data = model_dict["data"]
        rm_nan_x = opt_data['rmNan'][0]
        rm_nan_y = opt_data['rmNan'][1]
        x, y, c = self.model_data.load_data(model_dict)

        seq_length = model_dict["model"]["seqLength"]
        fcst_length = model_dict["model"]["fcstLength"]
        # don't cover the data of today when loading history data, so the length of 2nd dim is 'y.shape[1] - seq_length'
        flow = self.natural_flow
        q = np.zeros([flow.shape[0], flow.shape[1] - seq_length - fcst_length + 1, seq_length + fcst_length])
        for i in range(q.shape[1]):
            q[:, i, :] = flow[:, i:i + seq_length + fcst_length]

        if rm_nan_x is True:
            q[np.where(np.isnan(q))] = 0

        if seq_length >= 1 or fcst_length >= 1:
            x = cut_data(x, rm_nan_x, seq_length, fcst_length)
            y = cut_data(y, rm_nan_y, seq_length, fcst_length)
        qx = np.array([np.concatenate((q[j], x[j]), axis=1) for j in range(q.shape[0])])
        return qx, y, c


def divide_to_classes(label_dict, model_data, num_cluster, sites_id_all, with_dam_purpose=False):
    data_models = []
    var_dict = model_data.var_dict
    f_dict = model_data.f_dict
    for i in range(num_cluster):
        sites_label_i = [key for key, value in label_dict.items() if value == i]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_label_i]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(i))
        if not os.path.isdir(out_dir_new):
            os.makedirs(out_dir_new)
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(i))
        if not os.path.isdir(temp_dir_new):
            os.makedirs(temp_dir_new)
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_label_i)
        f_dict_new = copy.deepcopy(f_dict)
        if with_dam_purpose:
            if num_cluster > len(f_dict['GAGE_MAIN_DAM_PURPOSE']):
                # there is a "None" type
                if i == 0:
                    f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = None
                else:
                    f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = [f_dict['GAGE_MAIN_DAM_PURPOSE'][i - 1]]
            else:
                f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = [f_dict['GAGE_MAIN_DAM_PURPOSE'][i]]
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict_new, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_label_i
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        data_models.append(data_model_i)
    return data_models


class GagesExploreDataModel(object):
    def __init__(self, data_model):
        self.data_model = data_model

    def cluster_datamodel(self, num_cluster, start_dam_var='NDAMS_2009', with_dam_purpose=False,
                          sites_ids_list=None):
        """according to attr, cluster dataset"""
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        label_dict = {}
        if sites_ids_list:
            for k in range(len(sites_ids_list)):
                for site_id_temp in sites_ids_list[k]:
                    label_dict[site_id_temp] = k
        else:
            stat_dict = model_data.stat_dict
            var_lst = model_data.data_source.all_configs.get("attr_chosen")
            data = trans_norm(model_data.data_attr, var_lst, stat_dict, to_norm=True)
            index_start_anthro = 0
            for i in range(len(var_lst)):
                if var_lst[i] == start_dam_var:
                    index_start_anthro = i
                    break
            norm_data = data[:, index_start_anthro:]
            kmeans, labels = cluster_attr_train(norm_data, num_cluster)
            label_dict = dict(zip(sites_id_all, labels))
        if with_dam_purpose:
            data_models = divide_to_classes(label_dict, model_data, num_cluster, sites_id_all,
                                            with_dam_purpose=True)
        else:
            data_models = divide_to_classes(label_dict, model_data, num_cluster, sites_id_all)
        return data_models

    def classify_datamodel_by_dam_purpose(self, sites_ids_list=None):
        """classify data into classes one of which include all gage with same main dam purpose"""
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        data_attrs = model_data.data_attr[:, -1]
        data_attrs_unique, indices = np.unique(data_attrs, return_index=True)
        label_dict = {}
        if sites_ids_list:
            for k in range(len(sites_ids_list)):
                for site_id_temp in sites_ids_list[k]:
                    label_dict[site_id_temp] = k
        else:
            for i in range(data_attrs_unique.size):
                for j in range(len(sites_id_all)):
                    if data_attrs[j] == data_attrs_unique[i]:
                        label_dict[sites_id_all[j]] = i
        data_models = divide_to_classes(label_dict, model_data, data_attrs_unique.size, sites_id_all,
                                        with_dam_purpose=True)
        return data_models

    def choose_datamodel(self, sites_ids, f_dict_dam_purpose, sub_dir_num):
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_ids]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(sub_dir_num))
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(sub_dir_num))
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        sites_id_all_np = np.array(sites_id_all)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_id_all_np[sites_label_i_index].tolist())
        f_dict = model_data.f_dict
        f_dict_new = copy.deepcopy(f_dict)
        if f_dict_dam_purpose is None:
            f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = None
        else:
            f_dict_new['GAGE_MAIN_DAM_PURPOSE'] = f_dict_dam_purpose
        var_dict = model_data.var_dict
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict_new, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_id_all_np[sites_label_i_index].tolist()
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        return data_model_i

    def choose_datamodel_nodam(self, sites_ids, sub_dir_num):
        model_data = self.data_model
        sites_id_all = model_data.t_s_dict["sites_id"]
        sites_label_i_index = [j for j in range(len(sites_id_all)) if sites_id_all[j] in sites_ids]
        data_flow = model_data.data_flow[sites_label_i_index, :]
        data_forcing = model_data.data_forcing[sites_label_i_index, :, :]
        data_attr = model_data.data_attr[sites_label_i_index, :]
        stat_dict = {}
        t_s_dict = {}
        source_data_i = copy.deepcopy(model_data.data_source)
        out_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Out"], str(sub_dir_num))
        temp_dir_new = os.path.join(source_data_i.data_config.model_dict['dir']["Temp"], str(sub_dir_num))
        update_config_item(source_data_i.data_config.data_path, Out=out_dir_new, Temp=temp_dir_new)
        sites_id_all_np = np.array(sites_id_all)
        update_config_item(source_data_i.all_configs, out_dir=out_dir_new, temp_dir=temp_dir_new,
                           flow_screen_gage_id=sites_id_all_np[sites_label_i_index].tolist())
        f_dict = model_data.f_dict
        var_dict = model_data.var_dict
        data_model_i = DataModel(source_data_i, data_flow, data_forcing, data_attr, var_dict, f_dict, stat_dict,
                                 t_s_dict)
        t_s_dict['sites_id'] = sites_id_all_np[sites_label_i_index].tolist()
        t_s_dict['t_final_range'] = source_data_i.t_range
        data_model_i.t_s_dict = t_s_dict
        stat_dict_i = data_model_i.cal_stat_all()
        data_model_i.stat_dict = stat_dict_i
        return data_model_i


def which_is_main_purpose(dams_purposes_of_a_basin, storages_of_a_basin, care_1purpose=False):
    """if care_1purpose=True, consider every purpose seperately in multi-target dam"""
    assert type(dams_purposes_of_a_basin) == list
    assert type(storages_of_a_basin) == list
    assert len(dams_purposes_of_a_basin) == len(storages_of_a_basin)
    if care_1purpose:
        all_purposes = []
        for j in range(len(dams_purposes_of_a_basin)):
            if type(dams_purposes_of_a_basin[j]) == float:
                print("this purpose is unknown, set it to X")
                dams_purposes_of_a_basin[j] = 'X'
            purposes_str_i = [dams_purposes_of_a_basin[j][i:i + 1] for i in
                              range(0, len(dams_purposes_of_a_basin[j]), 1)]
            all_purposes = all_purposes + purposes_str_i
        all_purposes_unique = np.unique(all_purposes)
        purpose_storages = []
        for purpose in all_purposes_unique:
            purpose_storage = 0
            for i in range(len(dams_purposes_of_a_basin)):
                if purpose in dams_purposes_of_a_basin[i]:
                    purpose_storage = purpose_storage + storages_of_a_basin[i]
            purpose_storages.append(purpose_storage)
        main_purpose = all_purposes_unique[purpose_storages.index(max(purpose_storages))]
    else:
        purposes = np.array(dams_purposes_of_a_basin)
        storages = np.array(storages_of_a_basin)
        u, indices = np.unique(purposes, return_inverse=True)
        max_index = np.amax(indices)
        dict_i = {}
        for i in range(max_index + 1):
            dict_i[u[i]] = np.sum(storages[np.where(indices == i)])
        main_purpose = max(dict_i.items(), key=operator.itemgetter(1))[0]
    return main_purpose


def only_one_main_purpose(dams_purposes_of_a_basin, storages_of_a_basin):
    assert type(dams_purposes_of_a_basin) == list
    assert type(storages_of_a_basin) == list
    assert len(dams_purposes_of_a_basin) == len(storages_of_a_basin)

    all_purposes = []
    for j in range(len(dams_purposes_of_a_basin)):
        purposes_str_i = [dams_purposes_of_a_basin[j][i:i + 1] for i in
                          range(0, len(dams_purposes_of_a_basin[j]), 1)]
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


def multi_max_indices(nums):
    """nums could be a 2d array, where length of every 1d array is same"""
    max_of_nums = max(nums)
    tup = [(i, nums[i]) for i in range(len(nums))]
    indices = [i for i, n in tup if n == max_of_nums]
    return indices


def choose_which_purpose(gages_dam_datamodel, purpose=None):
    assert type(gages_dam_datamodel) == GagesDamDataModel
    if purpose is None:
        # choose all purpose
        sites_id_with_purposes = list(gages_dam_datamodel.gage_main_dam_purpose.keys())
        gages_input_new = GagesModel.update_data_model(gages_dam_datamodel.gages_input.data_source.data_config,
                                                       gages_dam_datamodel.gages_input,
                                                       sites_id_update=sites_id_with_purposes)
        return gages_input_new
    sites_id = []
    for key, value in gages_dam_datamodel.gage_main_dam_purpose.items():
        if value == purpose:
            sites_id.append(key)
    assert (all(x < y for x, y in zip(sites_id, sites_id[1:])))
    gages_input_new = GagesModel.update_data_model(gages_dam_datamodel.gages_input.data_source.data_config,
                                                   gages_dam_datamodel.gages_input, sites_id_update=sites_id)
    return gages_input_new


class GagesDamDataModel(object):
    def __init__(self, gages_input, nid_input, care_1purpose=False, *args):
        self.gages_input = gages_input
        self.nid_input = nid_input
        if len(args) == 0:
            self.gage_main_dam_purpose = self.spatial_join_dam(care_1purpose)
        else:
            self.gage_main_dam_purpose = args[0]
        # self.update_attr()

    def spatial_join_dam(self, care_1purpose):
        # ALL_PURPOSES = ['C', 'D', 'F', 'G', 'H', 'I', 'N', 'O', 'P', 'R', 'S', 'T', 'X']
        gage_region_dir = self.gages_input.data_source.all_configs.get("gage_region_dir")
        region_shapefiles = self.gages_input.data_source.all_configs.get("regions")
        # read sites from shapefile of region, get id from it.
        shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                      region_shapefiles]
        dam_dict = {}
        for shapefile in shapefiles:
            polys = gpd.read_file(shapefile)
            points = self.nid_input.nid_data
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
            u1 = self.no_clear_diff_between_nid_gages(u1, spatial_dam)
            main_purposes = []
            for u1_i in u1:
                purposes = []
                storages = []
                for index_i in range(gages_id_dam.shape[0]):
                    if gages_id_dam[index_i] == u1_i:
                        now_purpose = spatial_dam["PURPOSES"].iloc[index_i]
                        if type(now_purpose) == float:  # if purpose is nan, then set it to X
                            now_purpose = 'X'
                        purposes.append(now_purpose)
                        # storages.append(spatial_dam["NID_STORAGE"].iloc[index_i])
                        # NOR STORAGE
                        storages.append(spatial_dam["NORMAL_STORAGE"].iloc[index_i])
                # main_purpose = which_is_main_purpose(purposes, storages, care_1purpose=care_1purpose)
                main_purpose = only_one_main_purpose(purposes, storages)
                main_purposes.append(main_purpose)
            d = dict(zip(u1.tolist(), main_purposes))
            dam_dict = {**dam_dict, **d}
        # sorted by keys(gages_id)
        dam_dict_sorted = {}
        for key in sorted(dam_dict.keys()):
            dam_dict_sorted[key] = dam_dict[key]
        return dam_dict_sorted

    def update_attr(self):
        dam_dict = self.gage_main_dam_purpose
        attr_lst = self.gages_input.data_source.all_configs.get("attr_chosen")
        data_attr = self.gages_input.data_attr
        stat_dict = self.gages_input.stat_dict
        f_dict = self.gages_input.f_dict
        var_dict = self.gages_input.var_dict
        # update attr_lst, var_dict, f_dict, data_attr
        var_dam = 'GAGE_MAIN_DAM_PURPOSE'
        attr_lst.append(var_dam)
        dam_keys = dam_dict.keys()
        site_dam_purpose = []
        for site_id in self.gages_input.t_s_dict['sites_id']:
            if site_id in dam_keys:
                site_dam_purpose.append(dam_dict[site_id])
            else:
                site_dam_purpose.append(None)
        site_dam_purpose_int, uniques = pd.factorize(site_dam_purpose)
        site_dam_purpose_int = np.array(site_dam_purpose_int).reshape(len(site_dam_purpose_int), 1)
        self.gages_input.data_attr = np.append(data_attr, site_dam_purpose_int, axis=1)
        stat_dict[var_dam] = cal_stat(self.gages_input.data_attr[:, -1])
        # update f_dict and var_dict
        print("update f_dict and var_dict")
        var_dict['dam_purpose'] = var_dam
        f_dict[var_dam] = uniques.tolist()

    def no_clear_diff_between_nid_gages(self, u1, spatial_dam):
        """if there is clear diff for some basins in dam number and (dor value not considered now) between NID dataset and GAGES-II dataset,
        these basins will be excluded in the analysis"""
        print("excluede some basins")
        # there are some basins from shapefile which are not in CONUS, that will cause some bug, so do an intersection before search the attibutes. Also decrease the number needed to be calculated
        usgs_id = np.intersect1d(u1, self.gages_input.t_s_dict["sites_id"])
        attr_lst_dam_num = ["NDAMS_2009"]
        data_gages_dam_num, var_dict, f_dict = self.gages_input.data_source.read_attr(usgs_id, attr_lst_dam_num)
        u2 = [usgs_id[i] for i in range(len(usgs_id)) if data_gages_dam_num[i] > 0]
        return np.array(u2)

        # gages_id_dam = spatial_dam['GAGE_ID'].values
        # storage_nid = []
        # for u2_i in u2:
        #     storages = []
        #     for index_i in range(gages_id_dam.shape[0]):
        #         if gages_id_dam[index_i] == u2_i:
        #             # NOR STORAGE
        #             storages.append(spatial_dam["NORMAL_STORAGE"].iloc[index_i])
        #     storage_nid.append(np.sum(storages))
        #
        # #  unit of Normal storage in NID is acre-feet, 1 acre-feet = 1233.48 m^3
        #
        # # mm/year 1-km grid,  megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
        # # attr_lst = ["RUNAVE7100", "STOR_NID_2009"]
        # attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
        # data_attr, var_dict, f_dict = self.read_attr(usgs_id, attr_lst)
        # run_avg = data_attr[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
        # nor_storage = data_attr[:, 1] * 1000  # m^3
        # dors = nor_storage / run_avg
