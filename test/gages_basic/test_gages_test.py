import os
import unittest

import torch

import hydroDL
import definitions
from data import *
from data.data_input import GagesModel, _basin_norm
from hydroDL.master.master import master_test_with_pretrained_model


class TestForecastCase(unittest.TestCase):
    def setUp(self) -> None:
        """use model trained in allref regions to test nonref regions """
        config_dir = definitions.CONFIG_DIR
        # camels regions
        self.camels_config_file = os.path.join(config_dir, "basic/config_exp1.ini")
        self.camels_subdir = r"basic/exp1"
        self.camels_config_data = GagesConfig.set_subdir(self.camels_config_file, self.camels_subdir)
        # all ref regions
        self.ref_config_file = os.path.join(config_dir, "basic/config_exp2.ini")
        self.ref_subdir = r"basic/exp2"
        self.ref_config_data = GagesConfig.set_subdir(self.ref_config_file, self.ref_subdir)
        # all nonref regions
        self.nonref_config_file = os.path.join(config_dir, "basic/config_exp18.ini")
        self.nonref_subdir = r"basic/exp18"
        self.nonref_config_data = GagesConfig.set_subdir(self.nonref_config_file, self.nonref_subdir)

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
        data_model = GagesModel.load_datamodel(self.nonref_config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(1):
            pretrained_model_file = os.path.join(self.ref_config_data.data_path["Out"],
                                                 "model_Ep" + str(self.test_epoch) + ".pt")
            pretrained_model_name = self.ref_subdir.split("/")[1] + "_pretrained_model"
            pred, obs = master_test_with_pretrained_model(data_model, pretrained_model_file, pretrained_model_name)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            # save_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)
            # plot_we_need(data_model, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
