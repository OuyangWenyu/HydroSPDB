import copy
import os
import unittest
import torch

import numpy as np
from data import GagesConfig, GagesSource
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """choose basins with small DOR """
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="nodam/exp13", cache_state=1, dam_plan=2, attr_screen={"DOR": 0.003})
        update_cfg(self.config_file, args)
        self.config_data = GagesConfig(self.config_file)
        self.random_seed = self.config_file.RANDOM_SEED
        self.test_epoch = self.config_file.TEST_EPOCH
        self.gpu_num = self.config_file.CTX
        self.train_mode = self.config_file.TRAIN_MODE
        self.dam_plan = self.config_file.DAM_PLAN
        self.cache = self.config_file.CACHE.STATE

    def test_some_reservoirs(self):
        print("train and test in basins with different combination: \n")
        dam_plan = self.dam_plan
        config_data = self.config_data
        test_epoch = self.test_epoch
        if dam_plan == 2:
            dam_num = 0
            dor = self.config_file.GAGES.attrScreenParams.DOR
            source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              DOR=dor)
            # basins with dams
            source_data_withoutdams = GagesSource.choose_some_basins(config_data,
                                                                     config_data.model_dict["data"]["tRangeTrain"],
                                                                     screen_basin_area_huc4=False,
                                                                     dam_num=dam_num)

            sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
            sites_id_withoutdams = source_data_withoutdams.all_configs['flow_screen_gage_id']
            sites_id_chosen = np.sort(np.union1d(np.array(sites_id_dor1), np.array(sites_id_withoutdams))).tolist()
        elif dam_plan == 3:
            dam_num = [1, 100000]
            # basins with dams
            source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                                  config_data.model_dict["data"]["tRangeTrain"],
                                                                  screen_basin_area_huc4=False,
                                                                  dam_num=dam_num)
            sites_id_chosen = source_data_withdams.all_configs['flow_screen_gage_id']
        else:
            print("wrong choice")
            sites_id_chosen = None
        gages_model = GagesModels(config_data, screen_basin_area_huc4=False, sites_id=sites_id_chosen)
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
        with torch.cuda.device(self.gpu_num):
            if self.train_mode:
                master_train(gages_model_train, random_seed=self.random_seed)
            pred, obs = master_test(gages_model_test, epoch=test_epoch)
            basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                                is_return_dict=False)
            mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                               is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_model_test.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)


if __name__ == '__main__':
    unittest.main()
