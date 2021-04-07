import copy
import itertools
import os
import shutil
import unittest

import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data import *
from data.data_input import GagesModel, _basin_norm, save_result, load_result, save_datamodel
from data.gages_input_dataset import generate_gages_models, GagesModels, load_ensemble_result
from explore.stat import statError
from hydroDL.master.master import master_test_with_pretrained_model, master_test
from utils.dataset_format import subset_of_dict
from utils.hydro_util import hydro_logger
from visual.plot_model import plot_ts_obs_pred, plot_map, plot_we_need
from visual.plot_stat import plot_ecdf, plot_diff_boxes
from data.config import cfg, update_cfg, cmd


class TestForecastCase(unittest.TestCase):
    def setUp(self) -> None:
        """use model trained in one to test another"""
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="basic/exp1", train_period=["1990-01-01", "2000-01-01"], train_mode=0,
                   test_period=["2010-01-01", "2020-01-01"], quick_data=0, cache_state=1,
                   flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
                   gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        # args = cmd(sub="basic/exp46", train_period=["1980-01-01", "1990-01-01"], train_mode=0,
        #            test_period=["2000-01-01", "2010-01-01"], quick_data=0, cache_state=1,
        #            flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
        #            gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        # args = cmd(sub="basic/exp47", train_period=["1980-01-01", "1990-01-01"], train_mode=0,
        #            test_period=["2010-01-01", "2020-01-01"], quick_data=0, cache_state=1,
        #            flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
        #            gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        # args = cmd(sub="basic/exp48", train_period=["2000-01-01", "2010-01-01"], train_mode=0,
        #            test_period=["1980-01-01", "1990-01-01"], quick_data=0, cache_state=1,
        #            flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
        #            gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        # args = cmd(sub="basic/exp49", train_period=["2010-01-01", "2020-01-01"], train_mode=0,
        #            test_period=["1980-01-01", "1990-01-01"], quick_data=0, cache_state=1,
        #            flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
        #            gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        update_cfg(self.config_file, args)
        self.config_data = GagesConfig(self.config_file)

        config4pretrained = copy.deepcopy(cfg)
        args_pre = cmd(sub="basic/exp37")
        # args_pre = cmd(sub="basic/exp6")
        # args_pre = cmd(sub="basic/exp8")
        # args_pre = cmd(sub="basic/exp9")
        update_cfg(config4pretrained, args_pre)
        self.config_data4pretrained = GagesConfig(config4pretrained)

        pretrained_model_name = self.config_data4pretrained.data_path["Out"].split("/")[-1] + "_pretrained_model"
        self.pretrained_model_dir = os.path.join(self.config_data.data_path["Out"], pretrained_model_name)
        if not os.path.isdir(self.pretrained_model_dir):
            os.makedirs(self.pretrained_model_dir)
        self.pretrianed_file_name = os.path.join(self.pretrained_model_dir, "model_Ep300.pt")
        if not os.path.isfile(self.pretrianed_file_name):
            pretrianed_source = os.path.join(self.config_data4pretrained.data_path["Out"], "model_Ep300.pt")
            shutil.copy(pretrianed_source, self.pretrianed_file_name)

    def test_read_data(self):
        config_file = self.config_file
        if config_file.CACHE.QUICK_DATA:
            data_dir = config_file.CACHE.DATA_DIR
            gages_model_train, gages_model_test = generate_gages_models(self.config_data, data_dir, t_range=[
                self.config_data.model_dict["data"]["tRangeTrain"], self.config_data.model_dict["data"]["tRangeTest"]],
                                                                        screen_basin_area_huc4=False)
        else:
            gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False)
            gages_model_train = gages_model.data_model_train
            gages_model_test = gages_model.data_model_test
        if config_file.CACHE.STATE:
            save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                           stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                           forcing_file_name='test_forcing', attr_file_name='test_attr',
                           f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                           t_s_dict_file_name='test_dictTimeSpace.json')

    def test_show_multi_exps_results(self):
        periods = [["1980-01-01", "1990-01-01"], ["1990-01-01", "2000-01-01"], ["2000-01-01", "2010-01-01"],
                   ["2010-01-01", "2020-01-01"]]
        train_test_period_pairs = list(itertools.permutations(periods, 2))
        sub_lst = ["basic/exp6", "basic/exp46", "basic/exp47", "basic/exp7", "basic/exp37", "basic/exp1",
                   "basic/exp48", "basic/exp8", "basic/exp5", "basic/exp49", "basic/exp50", "basic/exp9"]
        exp_lst = [["basic_exp6"], ["basic_exp46"], ["basic_exp47"], ["basic_exp7"], ["basic_exp37"], ["basic_exp1"],
                   ["basic_exp48"], ["basic_exp8"], ["basic_exp5"], ["basic_exp49"], ["basic_exp50"], ["basic_exp9"]]
        for i in range(len(exp_lst)):
            config_file = copy.deepcopy(cfg)
            args = cmd(sub=sub_lst[i], train_period=train_test_period_pairs[i][0], train_mode=0,
                       test_period=train_test_period_pairs[i][1], quick_data=0, cache_state=1,
                       flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1}, te=300,
                       gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
            update_cfg(config_file, args)
            config_data = GagesConfig(config_file)
            test_epoch = config_data.config_file.TEST_EPOCH
            inds_df, pred_mean, obs_mean = load_ensemble_result(config_file, exp_lst[i], test_epoch, return_value=True)
            hydro_logger.info("the median NSE of %s is %s", sub_lst[i], inds_df["NSE"].median())

    def test_pretrained_model_test(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        test_epoch = self.config_data.config_file.TEST_EPOCH
        with torch.cuda.device(0):
            pretrained_model_file = self.pretrianed_file_name
            pretrained_model_name = self.pretrained_model_dir
            pred, obs = master_test_with_pretrained_model(data_model, pretrained_model_file, pretrained_model_name)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_dir = data_model.data_source.data_config.data_path['Temp']
            save_result(save_dir, test_epoch, pred, obs)

    def test_plot_pretrained_model_test(self):
        data_model_test = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        save_dir = data_model_test.data_source.data_config.data_path['Temp']
        pred, obs = load_result(save_dir, self.config_data.config_file.TEST_EPOCH)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(obs, pred)
        keys = ["Bias", "RMSE", "NSE"]
        t_s_dict = data_model_test.t_s_dict
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        # plot nse ecdf
        sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds[keys[2]]})
        plot_ecdf(sites_df_nse, keys[2])
        # plot map
        gauge_dict = data_model_test.data_source.gage_dict
        plot_map(gauge_dict, sites_df_nse, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")
        # plot box，使用seaborn库
        inds_test = subset_of_dict(inds, keys)
        box_fig = plot_diff_boxes(inds_test)
        # plot ts
        show_me_num = 5
        ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
        plt.show()


if __name__ == '__main__':
    unittest.main()
