import os
import unittest
import torch

import definitions
from data import GagesConfig, GagesSource
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """choose basins with small DOR """
        config_dir = definitions.CONFIG_DIR
        # self.config_file = os.path.join(config_dir, "dam/config_exp4.ini")
        # self.subdir = r"dam/exp4"
        self.config_file = os.path.join(config_dir, "dam/config_exp18.ini")
        self.subdir = r"dam/exp18"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def test_some_reservoirs(self):
        """choose some small reservoirs to train and test"""
        # 读取模型配置文件
        config_data = self.config_data
        # according to paper "High-resolution mapping of the world's reservoirs and dams for sustainable river-flow management"
        dor = 0.02
        source_data = GagesSource.choose_some_basins(config_data, config_data.model_dict["data"]["tRangeTrain"],
                                                     screen_basin_area_huc4=False, DOR=dor)
        sites_id = source_data.all_configs['flow_screen_gage_id']
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
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

        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train, sites_id_update=sites_id,
                                                         screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test, sites_id_update=sites_id,
                                                        train_stat_dict=gages_model_train.stat_dict,
                                                        screen_basin_area_huc4=False)
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
            pre_trained_model_epoch = 120
            # master_train(data_model)
            master_train(data_model, pre_trained_model_epoch=pre_trained_model_epoch)

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


if __name__ == '__main__':
    unittest.main()
