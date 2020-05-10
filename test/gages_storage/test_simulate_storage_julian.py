import copy
import unittest

import torch

import definitions
from data import GagesConfig, GagesSource
from data.data_config import add_model_param
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesSimInvDataModel, GagesSimDataModel, GagesJulianDataModel
from explore.stat import statError
from hydroDL.master.master import train_lstm_siminv, test_lstm_siminv, master_train_natural_flow, \
    master_test_natural_flow
import numpy as np
import os
import pandas as pd
from utils import serialize_numpy, unserialize_numpy
from utils.dataset_format import subset_of_dict
from visual import plot_ts_obs_pred
from visual.plot_model import plot_map
from visual.plot_stat import plot_ecdf, plot_diff_boxes


class MyTestCaseSimulateAndInv(unittest.TestCase):
    def setUp(self) -> None:
        """before all of these, natural flow model need to be generated by config.ini of gages dataset, and it need
        to be moved to right dir manually """
        config_dir = definitions.CONFIG_DIR
        self.config_file_natflow = os.path.join(config_dir, "storage/config1_exp1.ini")
        self.config_file_lstm = os.path.join(config_dir, "storage/config2_exp1.ini")
        self.subdir = r"storage/exp1"
        self.config_data_natflow = GagesConfig.set_subdir(self.config_file_natflow, self.subdir)
        self.config_data_lstm = GagesConfig.set_subdir(self.config_file_lstm, self.subdir)
        add_model_param(self.config_data_lstm, "model", seqLength=1)
        test_epoch_lst = [100, 200, 220, 250, 270, 280, 290, 300, 310, 320, 400, 500]
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
        # self.test_epoch = test_epoch_lst[11]

    def test_siminv_data_temp(self):
        quick_data_dir = os.path.join(self.config_data_natflow.data_path["DB"], "quickdata")
        # data_dir = os.path.join(quick_data_dir, "conus-all_85-05_nan-0.1_00-1.0")
        data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
        data_model_8595 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
        data_model_9505 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        conus_sites_id = data_model_8595.t_s_dict["sites_id"]
        nomajordam_source_data = GagesSource.choose_some_basins(self.config_data_natflow,
                                                                self.config_data_natflow.model_dict["data"][
                                                                    "tRangeTrain"],
                                                                screen_basin_area_huc4=False, major_dam_num=0)
        nomajordam_sites_id = nomajordam_source_data.all_configs['flow_screen_gage_id']
        nomajordam_in_conus = np.intersect1d(conus_sites_id, nomajordam_sites_id)
        majordam_source_data = GagesSource.choose_some_basins(self.config_data_natflow,
                                                              self.config_data_natflow.model_dict["data"][
                                                                  "tRangeTrain"],
                                                              screen_basin_area_huc4=False, major_dam_num=[1, 2000])
        majordam_sites_id = majordam_source_data.all_configs['flow_screen_gage_id']
        majordam_in_conus = np.intersect1d(conus_sites_id, majordam_sites_id)

        gages_model_train_natflow = GagesModel.update_data_model(self.config_data_natflow, data_model_8595,
                                                                 sites_id_update=nomajordam_in_conus,
                                                                 data_attr_update=True, screen_basin_area_huc4=False)
        gages_model_test_natflow = GagesModel.update_data_model(self.config_data_natflow, data_model_9505,
                                                                sites_id_update=nomajordam_in_conus,
                                                                data_attr_update=True,
                                                                train_stat_dict=gages_model_train_natflow.stat_dict,
                                                                screen_basin_area_huc4=False)

        gages_model_train_lstm = GagesModel.update_data_model(self.config_data_lstm, data_model_8595,
                                                              sites_id_update=majordam_in_conus, data_attr_update=True,
                                                              screen_basin_area_huc4=False)

        gages_model_test_lstm = GagesModel.update_data_model(self.config_data_lstm, data_model_9505,
                                                             sites_id_update=majordam_in_conus, data_attr_update=True,
                                                             train_stat_dict=gages_model_train_lstm.stat_dict,
                                                             screen_basin_area_huc4=False)

        save_datamodel(gages_model_train_natflow, "1", data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test_natflow, "1", data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        save_datamodel(gages_model_train_lstm, "2", data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test_lstm, "2", data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        print("read and save data model")

    def test_train_julian(self):
        with torch.cuda.device(1):
            data_model1 = GagesModel.load_datamodel(self.config_data_natflow.data_path["Temp"], "1",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model1.update_model_param('train', nEpoch=300)
            data_model2 = GagesModel.load_datamodel(self.config_data_lstm.data_path["Temp"], "2",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model = GagesJulianDataModel(data_model1, data_model2)
            # pre_trained_model_epoch = 150
            # master_train_natural_flow(data_model, pre_trained_model_epoch)
            master_train_natural_flow(data_model)

    def test_siminv_test(self):
        with torch.cuda.device(1):
            df1 = GagesModel.load_datamodel(self.config_data_natflow.data_path["Temp"], "1",
                                            data_source_file_name='test_data_source.txt',
                                            stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                            forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                            f_dict_file_name='test_dictFactorize.json',
                                            var_dict_file_name='test_dictAttribute.json',
                                            t_s_dict_file_name='test_dictTimeSpace.json')
            df1.update_model_param('train', nEpoch=300)
            df2 = GagesModel.load_datamodel(self.config_data_lstm.data_path["Temp"], "2",
                                            data_source_file_name='test_data_source.txt',
                                            stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                            forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                            f_dict_file_name='test_dictFactorize.json',
                                            var_dict_file_name='test_dictAttribute.json',
                                            t_s_dict_file_name='test_dictTimeSpace.json')
            data_model = GagesJulianDataModel(df1, df2)
            test_epoch = self.test_epoch
            pred, obs = master_test_natural_flow(data_model, epoch=test_epoch)
            basin_area = df2.data_source.read_attr(df2.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
            mean_prep = df2.data_source.read_attr(df2.t_s_dict["sites_id"], ['PPTAVG_BASIN'], is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(df2.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)

    def test_siminv_plot(self):
        data_model = GagesModel.load_datamodel(self.config_data_lstm.data_path["Temp"], "2",
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        test_epoch = self.test_epoch
        flow_pred_file = os.path.join(data_model.data_source.data_config.data_path['Temp'],
                                      'epoch' + str(test_epoch) + 'flow_pred.npy')
        flow_obs_file = os.path.join(data_model.data_source.data_config.data_path['Temp'],
                                     'epoch' + str(test_epoch) + 'flow_obs.npy')
        pred = unserialize_numpy(flow_pred_file)
        obs = unserialize_numpy(flow_obs_file)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])

        inds = statError(obs, pred)
        inds['STAID'] = data_model.t_s_dict["sites_id"]
        inds_df = pd.DataFrame(inds)
        inds_df.to_csv(os.path.join(self.config_data_lstm.data_path["Out"], 'data_df.csv'))
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(inds, keys)
        box_fig = plot_diff_boxes(inds_test)
        box_fig.savefig(os.path.join(self.config_data_lstm.data_path["Out"], "box_fig.png"))
        # plot ts
        show_me_num = 5
        t_s_dict = data_model.t_s_dict
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        time_seq_length = self.config_data_lstm.model_dict["model"]["seqLength"]
        time_start = np.datetime64(t_range[0]) + np.timedelta64(time_seq_length - 1, 'D')
        t_range[0] = np.datetime_as_string(time_start, unit='D')
        ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
        ts_fig.savefig(os.path.join(self.config_data_lstm.data_path["Out"], "ts_fig.png"))

        # plot nse ecdf
        sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
        plot_ecdf(sites_df_nse, keys[2])
        # plot map
        gauge_dict = data_model.data_source.gage_dict
        plot_map(gauge_dict, sites_df_nse, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
