import os
import unittest

import torch
import pandas as pd
import numpy as np
import definitions
from data import *
from data.data_input import GagesModel, _basin_norm, save_result, load_result
from explore.stat import statError
from hydroDL.master.master import master_test_with_pretrained_model
from utils.dataset_format import subset_of_dict
from visual.plot_model import plot_ts_obs_pred, plot_map
from visual.plot_stat import plot_ecdf, plot_diff_boxes


class TestForecastCase(unittest.TestCase):
    def setUp(self) -> None:
        """use model trained in no-major-dam regions to test with-major-dam regions """
        config_dir = definitions.CONFIG_DIR
        # no major dam regions
        self.nomajordam_config_file = os.path.join(config_dir, "nodam/config_exp3.ini")
        self.nomajordam_subdir = r"nodam/exp3"
        self.nomajordam_config_data = GagesConfig.set_subdir(self.nomajordam_config_file, self.nomajordam_subdir)
        # all ref regions
        self.majordam_config_file = os.path.join(config_dir, "majordam/config_exp2.ini")
        self.majordam_subdir = r"majordam/exp2"
        self.majordam_config_data = GagesConfig.set_subdir(self.majordam_config_file, self.majordam_subdir)

        test_epoch_lst = [100, 200, 220, 250, 280, 290, 295, 300, 305, 310, 320]
        # self.test_epoch = test_epoch_lst[0]
        # self.test_epoch = test_epoch_lst[1]
        # self.test_epoch = test_epoch_lst[2]
        # self.test_epoch = test_epoch_lst[3]
        # self.test_epoch = test_epoch_lst[4]
        # self.test_epoch = test_epoch_lst[5]
        # self.test_epoch = test_epoch_lst[6]
        self.test_epoch = test_epoch_lst[7]
        # self.test_epoch = test_epoch_lst[8]
        # self.test_epoch = test_epoch_lst[9]
        # self.test_epoch = test_epoch_lst[10]

    def test_pretrained_model_test(self):
        data_model = GagesModel.load_datamodel(self.majordam_config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(1):
            pretrained_model_file = os.path.join(self.nomajordam_config_data.data_path["Out"],
                                                 "model_Ep" + str(self.test_epoch) + ".pt")
            pretrained_model_name = "nodam" + self.nomajordam_subdir.split("/")[1] + "_pretrained_model"
            pred, obs = master_test_with_pretrained_model(data_model, pretrained_model_file, pretrained_model_name)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_dir = os.path.join(data_model.data_source.data_config.data_path['Out'], pretrained_model_name)
            save_result(save_dir, self.test_epoch, pred, obs)

    def test_plot_pretrained_model_test(self):
        data_model_test = GagesModel.load_datamodel(self.majordam_config_data.data_path["Temp"],
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        pretrained_model_name = "nodam" + self.nomajordam_subdir.split("/")[1] + "_pretrained_model"
        save_dir = os.path.join(data_model_test.data_source.data_config.data_path['Out'], pretrained_model_name)
        pred, obs = load_result(save_dir, self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(obs, pred)
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(inds, keys)
        box_fig = plot_diff_boxes(inds_test)
        box_fig.savefig(os.path.join(save_dir, "box_fig.png"))
        # plot ts
        show_me_num = 5
        t_s_dict = data_model_test.t_s_dict
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
        ts_fig.savefig(os.path.join(save_dir, "ts_fig.png"))
        # plot nse ecdf
        sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
        plot_ecdf(sites_df_nse, keys[2])
        # plot map
        gauge_dict = data_model_test.data_source.gage_dict
        plot_map(gauge_dict, sites_df_nse, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
