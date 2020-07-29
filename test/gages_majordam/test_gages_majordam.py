import unittest

import torch
from data.config import cfg
from data import GagesConfig, GagesSource
from data.data_input import GagesModel, _basin_norm, save_datamodel, save_result
from data.gages_input_dataset import GagesModels
from hydroDL.master.master import master_train, master_test
import os
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    """data pre-process and post-process"""

    def setUp(self) -> None:
        """update some parameters"""
        self.config_data = GagesConfig(cfg)

    def test_gages_data_model(self):
        config_data = self.config_data
        major_dam_num = [1, 200]  # max major dam num is 155
        if cfg.CACHE.QUICK_DATA:
            source_data = GagesSource.choose_some_basins(config_data, config_data.model_dict["data"]["tRangeTrain"],
                                                         screen_basin_area_huc4=False, major_dam_num=major_dam_num)
            sites_id = source_data.all_configs['flow_screen_gage_id']
            print("The binary data has exsited")
            quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
            # data_dir = os.path.join(quick_data_dir, "conus-all_85-05_nan-0.1_00-1.0")
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
            gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train,
                                                             sites_id_update=sites_id,
                                                             screen_basin_area_huc4=False)
            gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test, sites_id_update=sites_id,
                                                            train_stat_dict=gages_model_train.stat_dict,
                                                            screen_basin_area_huc4=False)
        else:
            gages_model = GagesModels(config_data, screen_basin_area_huc4=False, major_dam_num=major_dam_num)
            gages_model_train = gages_model.data_model_train
            gages_model_test = gages_model.data_model_test
        if cfg.CACHE.STATE:
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

    def test_dam_train(self):
        with torch.cuda.device(0):
            gages_model_train = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                                          data_source_file_name='data_source.txt',
                                                          stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                          forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                          f_dict_file_name='dictFactorize.json',
                                                          var_dict_file_name='dictAttribute.json',
                                                          t_s_dict_file_name='dictTimeSpace.json')
            master_train(gages_model_train)
            # pre_trained_model_epoch = 130
            # master_train(gages_model_train, pre_trained_model_epoch=pre_trained_model_epoch)

    def test_dam_test(self):
        with torch.cuda.device(0):
            gages_input = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
            pred, obs = master_test(gages_input, epoch=cfg.TEST_EPOCH)
            basin_area = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                           is_return_dict=False)
            mean_prep = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                          is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_input.data_source.data_config.data_path['Temp'], cfg.TEST_EPOCH, pred, obs)
            plot_we_need(gages_input, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
