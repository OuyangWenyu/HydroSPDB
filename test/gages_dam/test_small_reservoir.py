import os
import unittest
import numpy as np
import torch

import definitions
from data import GagesConfig, GagesSource, DataModel
from data.data_input import save_datamodel, GagesModel
from data.gages_input_dataset import GagesModels
from explore.stat import statError
from hydroDL import master_train, master_test
from utils.dataset_format import subset_of_dict
from visual import plot_ts_obs_pred
from visual.plot_model import plot_boxes_inds


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """choose basins with small DOR """
        config_dir = definitions.CONFIG_DIR
        # self.config_file = os.path.join(config_dir, "smallres/config_smallres_ex4.ini")
        # self.subdir = r"smallres/exp4"
        # self.config_file = os.path.join(config_dir, "smallres/config_smallres_ex5.ini")
        # self.subdir = r"smallres/exp5"
        # self.config_file = os.path.join(config_dir, "smallres/config_smallres_ex6.ini")
        # self.subdir = r"smallres/exp6"

        self.config_file = os.path.join(config_dir, "dam/config_exp1.ini")
        self.subdir = r"dam/exp1"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)

    def test_gages_data_model(self):
        dor = 0.02
        gages_model = GagesModels(self.config_data, DOR=dor)
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

    def test_some_reservoirs(self):
        """choose some small reservoirs to train and test"""
        # 读取模型配置文件
        config_data = self.config_data
        # according to paper "High-resolution mapping of the world's reservoirs and dams for sustainable river-flow management"
        dor = 0.02
        source_data = GagesSource.choose_some_basins(config_data, config_data.model_dict["data"]["tRangeTrain"],
                                                     DOR=dor)
        sites_id = source_data.all_configs['flow_screen_gage_id']
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "allnonref_85-05_nan-0.1_00-1.0")
        data_model_train = GagesModel.load_datamodel(data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')
        data_model_test = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')

        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train, sites_id_update=sites_id)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test, sites_id_update=sites_id,
                                                        train_stat_dict=gages_model_train.stat_dict)
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
        with torch.cuda.device(2):
            # pre_trained_model_epoch = 240
            master_train(data_model)
            # master_train(data_model, pre_trained_model_epoch=pre_trained_model_epoch)


if __name__ == '__main__':
    unittest.main()
