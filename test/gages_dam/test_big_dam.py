import os
import unittest
import torch

import definitions
from data import GagesConfig, GagesSource
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
import numpy as np
from hydroDL import master_train, master_test
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """choose basins with big storage """
        config_dir = definitions.CONFIG_DIR
        self.config_file = os.path.join(config_dir, "dam/config_exp12.ini")
        self.subdir = r"dam/exp12"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def test_some_reservoirs(self):
        # a control group for simulate/exp4
        dor = 0.02
        source_data = GagesSource.choose_some_basins(self.config_data,
                                                     self.config_data.model_dict["data"]["tRangeTrain"],
                                                     screen_basin_area_huc4=False, DOR=dor)
        sites_id_dor = source_data.all_configs['flow_screen_gage_id']

        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
        data_model_9000 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
        data_model_0010 = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        conus_sites_id_all = data_model_9000.t_s_dict["sites_id"]
        nomajordam_source_data = GagesSource.choose_some_basins(self.config_data,
                                                                self.config_data.model_dict["data"]["tRangeTrain"],
                                                                screen_basin_area_huc4=False, major_dam_num=0)
        nomajordam_sites_id = nomajordam_source_data.all_configs['flow_screen_gage_id']
        # In no major dam case, all sites are chosen as natural flow generator
        nomajordam_in_conus = np.intersect1d(conus_sites_id_all, nomajordam_sites_id)

        conus_sites_id_dor = np.intersect1d(conus_sites_id_all, sites_id_dor)
        majordam_source_data = GagesSource.choose_some_basins(self.config_data,
                                                              self.config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False, major_dam_num=[1, 2000])
        majordam_sites_id = majordam_source_data.all_configs['flow_screen_gage_id']
        majordam_in_conus = np.intersect1d(conus_sites_id_dor, majordam_sites_id)

        chosen_sites_id = np.sort(np.append(nomajordam_in_conus, majordam_in_conus))

        gages_model_train_lstm = GagesModel.update_data_model(self.config_data, data_model_9000,
                                                              sites_id_update=chosen_sites_id, data_attr_update=True,
                                                              screen_basin_area_huc4=False)

        gages_model_test_lstm = GagesModel.update_data_model(self.config_data, data_model_0010,
                                                             sites_id_update=chosen_sites_id, data_attr_update=True,
                                                             train_stat_dict=gages_model_train_lstm.stat_dict,
                                                             screen_basin_area_huc4=False)

        save_datamodel(gages_model_train_lstm, data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test_lstm, data_source_file_name='test_data_source.txt',
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
            pre_trained_model_epoch = 150
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
