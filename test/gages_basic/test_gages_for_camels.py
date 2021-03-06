import os
import unittest

import torch
import pandas as pd
import numpy as np
from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from explore.stat import statError
from hydroDL.master import *
import definitions
from utils import unserialize_numpy
from visual.plot_model import plot_we_need


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = definitions.CONFIG_DIR
        # self.config_file = os.path.join(config_dir, "basic/config_exp31.ini")
        # self.subdir = r"basic/exp31"
        # self.random_seed = 1234
        self.config_file = os.path.join(config_dir, "basic/config_exp34.ini")
        self.subdir = r"basic/exp34"
        self.random_seed = 111
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def test_gages_data_model(self):
        camels531_gageid_file = os.path.join(self.config_data.data_path["DB"], "camels531", "CAMELS531.txt")
        gauge_df = pd.read_csv(camels531_gageid_file, dtype={"GaugeID": str})
        gauge_list = gauge_df["GaugeID"].values
        all_sites_camels_531 = np.sort([str(gauge).zfill(8) for gauge in gauge_list])
        gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False,
                                  sites_id=all_sites_camels_531.tolist())
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
        with torch.cuda.device(2):
            # pre_trained_model_epoch = 230
            master_train(data_model, random_seed=self.random_seed)
            # master_train(data_model, pre_trained_model_epoch=pre_trained_model_epoch,  random_seed=self.random_seed)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(2):
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

    def test_export_result(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        flow_pred_file = os.path.join(data_model.data_source.data_config.data_path['Temp'], 'flow_pred.npy')
        flow_obs_file = os.path.join(data_model.data_source.data_config.data_path['Temp'], 'flow_obs.npy')
        pred = unserialize_numpy(flow_pred_file)
        obs = unserialize_numpy(flow_obs_file)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
        inds = statError(obs, pred)
        inds['STAID'] = data_model.t_s_dict["sites_id"]
        inds_df = pd.DataFrame(inds)

        inds_df.to_csv(os.path.join(self.config_data.data_path["Out"], 'data_df.csv'))


if __name__ == '__main__':
    unittest.main()
