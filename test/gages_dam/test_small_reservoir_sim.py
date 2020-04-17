import os
import unittest
import numpy as np
import torch

import definitions
from data import GagesConfig, GagesSource, DataModel
from data.data_config import add_model_param
from data.data_input import save_datamodel, GagesModel
from data.gages_input_dataset import GagesModels, GagesSimDataModel
from explore.stat import statError
from hydroDL import master_train, master_test
from hydroDL.master.master import master_train_natural_flow
from utils.dataset_format import subset_of_dict
from visual import plot_ts_obs_pred
from visual.plot_model import plot_boxes_inds


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """choose basins with small DOR """
        config_dir = definitions.CONFIG_DIR
        self.sim_config_file = os.path.join(config_dir, "dam/config1_exp2.ini")
        self.config_file = os.path.join(config_dir, "dam/config2_exp2.ini")
        self.subdir = "dam/exp2"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.sim_config_data = GagesConfig.set_subdir(self.sim_config_file, self.subdir)
        add_model_param(self.config_data, "model", seqLength=1)
        # choose some small basins, unit: SQKM
        # self.basin_area_screen = 100
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

    def test_some_reservoirs(self):
        """choose some small reservoirs for 2nd lstm not for simulate"""
        # 读取模型配置文件
        config_data = self.config_data
        # according to paper "High-resolution mapping of the world's reservoirs and dams for sustainable river-flow management"
        dor = 0.02
        source_data = GagesSource.choose_some_basins(config_data, config_data.model_dict["data"]["tRangeTrain"],
                                                     DOR=dor)
        sites_id = source_data.all_configs['flow_screen_gage_id']

        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        sim_data_dir = os.path.join(quick_data_dir, "allref_85-05_nan-0.1_00-1.0")
        data_dir = os.path.join(quick_data_dir, "allnonref_85-05_nan-0.1_00-1.0")
        data_model_sim8595 = GagesModel.load_datamodel(sim_data_dir,
                                                       data_source_file_name='data_source.txt',
                                                       stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                       forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                       f_dict_file_name='dictFactorize.json',
                                                       var_dict_file_name='dictAttribute.json',
                                                       t_s_dict_file_name='dictTimeSpace.json')
        data_model_8595 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
        data_model_sim9505 = GagesModel.load_datamodel(sim_data_dir,
                                                       data_source_file_name='test_data_source.txt',
                                                       stat_file_name='test_Statistics.json',
                                                       flow_file_name='test_flow.npy',
                                                       forcing_file_name='test_forcing.npy',
                                                       attr_file_name='test_attr.npy',
                                                       f_dict_file_name='test_dictFactorize.json',
                                                       var_dict_file_name='test_dictAttribute.json',
                                                       t_s_dict_file_name='test_dictTimeSpace.json')
        data_model_9505 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')

        sim_gages_model_train = GagesModel.update_data_model(self.sim_config_data, data_model_sim8595,
                                                             data_attr_update=True)
        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_8595, sites_id_update=sites_id,
                                                         data_attr_update=True)
        sim_gages_model_test = GagesModel.update_data_model(self.sim_config_data, data_model_sim9505,
                                                            data_attr_update=True,
                                                            train_stat_dict=sim_gages_model_train.stat_dict)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_9505, sites_id_update=sites_id,
                                                        data_attr_update=True,
                                                        train_stat_dict=gages_model_train.stat_dict)
        save_datamodel(sim_gages_model_train, "1", data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(sim_gages_model_test, "1", data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        save_datamodel(gages_model_train, "2", data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test, "2", data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        print("read and save data model")

    def test_train_gages_sim(self):
        with torch.cuda.device(1):
            # load model from npy data and then update some params for the test func
            data_model1 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "1",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model1.update_model_param('train', nEpoch=300)
            data_model2 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "2",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model = GagesSimDataModel(data_model1, data_model2)
            # pre_trained_model_epoch = 150
            # master_train_natural_flow(data_model, pre_trained_model_epoch)
            master_train_natural_flow(data_model)


if __name__ == '__main__':
    unittest.main()
