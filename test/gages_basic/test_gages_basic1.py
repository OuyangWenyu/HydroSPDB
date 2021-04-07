import copy
import os
import unittest

import torch
import pandas as pd

from data import *
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from explore.stat import statError
from hydroDL.master import *
from utils import unserialize_numpy
from visual.plot_model import plot_we_need


class MyTestCaseGages(unittest.TestCase):
    # python gages_conus_analysis.py --sub basic/exp5 --quick_data 0 --cache_state 1 --train_period 2000-01-01 2010-01-01 --test_period 2010-01-01 2020-01-01 --gage_id_file /mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv --flow_screen "{\"missing_data_ratio\":1,\"zero_value_ratio\":1}"
    # python gages_conus_analysis.py --sub basic/exp6 --quick_data 0 --cache_state 1 --train_period 1980-01-01 1990-01-01 --test_period 1990-01-01 2000-01-01 --gage_id_file /mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv --flow_screen "{\"missing_data_ratio\":1,\"zero_value_ratio\":1}"
    # python gages_conus_analysis.py --sub basic/exp7 --quick_data 0 --cache_state 1 --train_period 1990-01-01 2000-01-01 --test_period 1980-01-01 1990-01-01 --gage_id_file /mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv --flow_screen "{\"missing_data_ratio\":1,\"zero_value_ratio\":1}"
    # python gages_conus_analysis.py --sub basic/exp8 --quick_data 0 --cache_state 1 --train_period 2000-01-01 2010-01-01 --test_period 1990-01-01 2000-01-01 --gage_id_file /mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv --flow_screen "{\"missing_data_ratio\":1,\"zero_value_ratio\":1}"
    # python gages_conus_analysis.py --sub basic/exp9 --quick_data 0 --cache_state 1 --train_period 2010-01-01 2020-01-01 --test_period 2000-01-01 2010-01-01 --gage_id_file /mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv --flow_screen "{\"missing_data_ratio\":1,\"zero_value_ratio\":1}"

    def setUp(self) -> None:
        config_file = copy.deepcopy(cfg)
        args = cmd(sub="basic/exp7", train_period=["1990-01-01", "2000-01-01"],
                   test_period=["1980-01-01", "1990-01-01"], quick_data=0, cache_state=1,
                   flow_screen={'missing_data_ratio': 1, 'zero_value_ratio': 1},
                   gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
        update_cfg(config_file, args)
        self.config_data = GagesConfig(config_file)
        self.test_epoch = 300

    def test_gages_data_model(self):
        gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False)
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

    def test_gages_data_model_quickdata(self):
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

        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train, data_attr_update=True,
                                                         screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test, data_attr_update=True,
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
