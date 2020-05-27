import os
import unittest

import torch
import pandas as pd

from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import GagesModels, load_dataconfig_case_exp
from explore.stat import statError
from hydroDL.master import *
import definitions
import numpy as np

from utils.dataset_format import subset_of_dict
from visual.plot_model import plot_we_need, plot_gages_map_and_ts
from visual.plot_stat import plot_diff_boxes


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = definitions.CONFIG_DIR
        # exp11 and exp17 are used for ensemble test code
        # self.config_file = os.path.join(config_dir, "basic/config_exp11.ini")
        # self.subdir = r"basic/exp11"
        # self.config_file = os.path.join(config_dir, "basic/config_exp17.ini")
        # self.subdir = r"basic/exp17"

        self.config_file = os.path.join(config_dir, "basic/config_exp12.ini")
        self.subdir = r"basic/exp12"
        self.random_seed = 1234
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def test_gages_data_model(self):
        gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False)
        save_datamodel(gages_model.data_model_train, data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model.data_model_test, data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        print("read and save data model")

    def test_train_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        with torch.cuda.device(1):
            # pre_trained_model_epoch = 170
            master_train(data_model)
            # master_train(data_model, pre_trained_model_epoch=pre_trained_model_epoch)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(1):
            pred, obs = master_test(data_model, epoch=self.test_epoch)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)
            plot_we_need(data_model, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

    def test_ensemble_results(self):
        preds = []
        obss = []
        # cases_exps = ["basic_exp11", "basic_exp17"]
        cases_exps = ["basic_exp12", "basic_exp13", "basic_exp14", "basic_exp15", "basic_exp16", "basic_exp18"]
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], self.test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            preds.append(pred_i)
            obss.append(obs_i)
        preds_np = np.array(preds)
        obss_np = np.array(obss)
        pred_mean = np.mean(preds_np, axis=0)
        obs_mean = np.mean(obss_np, axis=0)
        inds = statError(obs_mean, pred_mean)
        inds_df = pd.DataFrame(inds)

        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')

        # plot map ts
        show_ind_key = 'NSE'
        idx_lst = np.arange(len(data_model.t_s_dict["sites_id"])).tolist()

        # nse_range = [0.5, 1]
        nse_range = [0, 1]
        # nse_range = [-10000, 1]
        # nse_range = [-10000, 0]
        idx_lst_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        plot_gages_map_and_ts(data_model, obs_mean, pred_mean, inds_df, show_ind_key, idx_lst_nse,
                              pertile_range=[0, 100])

    def test_ensemble_results_plot_box(self):
        preds = []
        obss = []
        # cases_exps = ["basic_exp11", "basic_exp17"]
        cases_exps = ["basic_exp12", "basic_exp13", "basic_exp14", "basic_exp15", "basic_exp16", "basic_exp18"]
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], self.test_epoch)
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
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(inds, keys)
        box_fig = plot_diff_boxes(inds_test)


if __name__ == '__main__':
    unittest.main()
