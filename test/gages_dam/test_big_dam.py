import copy
import os
import unittest
import torch

from data import GagesConfig, GagesSource
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
import numpy as np

from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # python gages_dor_analysis.py --sub dam/exp12 --ctx 0 --attr_screen {\"DOR\":0.003}
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="dam/exp12", cache_state=1, attr_screen={"DOR": 0.003})
        update_cfg(self.config_file, args)
        self.config_data = GagesConfig(self.config_file)
        self.random_seed = self.config_file.RANDOM_SEED
        self.test_epoch = self.config_file.TEST_EPOCH
        self.gpu_num = self.config_file.CTX
        self.train_mode = self.config_file.TRAIN_MODE
        self.dor = self.config_file.GAGES.attrScreenParams.DOR
        self.cache = self.config_file.CACHE.STATE
        print("train and test in large-dor basins: \n")

    def test_some_reservoirs(self):
        config_data = self.config_data
        gages_model = GagesModels(config_data, screen_basin_area_huc4=False, DOR=self.dor)
        gages_model_train = gages_model.data_model_train
        gages_model_test = gages_model.data_model_test
        if self.cache:
            save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                           stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                           forcing_file_name='test_forcing', attr_file_name='test_attr',
                           f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                           t_s_dict_file_name='test_dictTimeSpace.json')

    def test_train_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        with torch.cuda.device(0):
            master_train(data_model)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(0):
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


if __name__ == '__main__':
    unittest.main()
