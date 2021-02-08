import copy
import pprint
import unittest

import torch
from easydict import EasyDict

from data import *
from data.config import cfg, update_cfg_item
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.water_balance_input import GagesWaterBalanceDataModel
from hydroDL.master import *
from hydroDL.master.pgml import master_train_reservoir_water_balance
from visual.plot_model import plot_we_need
import numpy as np


class TestDemo4Equation(unittest.TestCase):
    """a demo test for how to combine NN with equation"""

    def setUp(self) -> None:
        # before the startup, we need to know how to calculate the derivative of Neural Network
        pprint.pprint(cfg)
        test_case = EasyDict(sub="resequ/exp1")
        update_cfg_item(cfg, test_case)
        main_layer = EasyDict(CTX=0)
        cfg.update(main_layer)
        model_layer = EasyDict(miniBatch=[100, 30], nEpoch=20)
        cfg.MODEL.update(model_layer)

        self.config_file_inflow = copy.deepcopy(cfg)
        self.config_data_inflow = GagesConfig(self.config_file_inflow)

        self.config_file = copy.deepcopy(cfg)
        gages_layer = EasyDict(
            gageIdScreen=["01407500", "05017500", "06020600", "06036650", "06089000", "06101500", "06108000",
                          "06126500", "06130500", "06225500"])
        self.config_file.GAGES.update(gages_layer)
        pgml_model_layer = EasyDict(miniBatch=[5, 30], nEpoch=20)
        self.config_file.MODEL.update(pgml_model_layer)
        self.config_data = GagesConfig(self.config_file)
        self.test_epoch = 20

    def test_gages_data_model_readquickdata(self):
        data_dir = cfg.CACHE.DATA_DIR
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
                                                         sites_id_update=self.config_file.GAGES.gageIdScreen,
                                                         data_attr_update=True,
                                                         screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test,
                                                        sites_id_update=self.config_file.GAGES.gageIdScreen,
                                                        data_attr_update=True,
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
        with torch.cuda.device(cfg.CTX):
            # load model from npy data and then update some params for the test func
            data_model1 = GagesModel.load_datamodel(self.config_data_inflow.data_path["Temp"], "1",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model2 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "2",
                                                    data_source_file_name='data_source.txt',
                                                    stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                    forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                    f_dict_file_name='dictFactorize.json',
                                                    var_dict_file_name='dictAttribute.json',
                                                    t_s_dict_file_name='dictTimeSpace.json')
            data_model = GagesWaterBalanceDataModel(data_model1, data_model2)
            master_train_reservoir_water_balance(data_model)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(1):
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
