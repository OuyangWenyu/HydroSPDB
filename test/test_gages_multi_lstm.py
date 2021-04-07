import copy
import os
import unittest

import torch

from data import GagesConfig
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm
from data.gages_input_dataset import GagesModels, generate_gages_models
from hydroDL.master.master import master_train_batch1st_lstm, master_test_batch1st_lstm
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """Here the dimension of tensors in each batch is [batch, sequence, var], which means batch first
         Since CudnnLstm is sequence first, here multi-lstm doesn't include it.
         if we wanna train the model in sequence-first mode, we need set batch_first=False"""
        config_file = copy.deepcopy(cfg)
        args = cmd(sub="basic/exp51", quick_data=1, cache_state=1, model_name="EasyLstm")
        update_cfg(config_file, args)
        self.config_data = GagesConfig(config_file)

    def test_datamodel(self):
        if self.config_data.config_file.CACHE.QUICK_DATA:
            data_dir = self.config_data.config_file.CACHE.DATA_DIR
            gages_model_train, gages_model_test = generate_gages_models(self.config_data, data_dir,
                                                                        screen_basin_area_huc4=False)
        else:
            gages_model = GagesModels(self.config_data)
            gages_model_train = gages_model.data_model_train
            gages_model_test = gages_model.data_model_test
        if self.config_data.config_file.CACHE.STATE:
            save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
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
        with torch.cuda.device(0):
            master_train_batch1st_lstm(data_model)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(0):
            pred, obs = master_test_batch1st_lstm(data_model, load_epoch=295)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            plot_we_need(data_model, obs, pred)


if __name__ == '__main__':
    unittest.main()
