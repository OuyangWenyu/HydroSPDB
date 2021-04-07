import copy
import unittest

import torch

import definitions
from data import GagesConfig
from data.config import cfg, cmd, update_cfg
from data.data_input import GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesDamDataModel, choose_which_purpose
from data.nid_input import NidModel
from hydroDL.master.master import master_train, master_test
import os

from utils import unserialize_json
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    """python gages_w-wo-dam_analysis.py --sub dam/exp40 --ctx 0 --dam_plan 3 --rs 1234
    See the performance when adding reservoirs-related attributes as input for the LSTM model.
    """

    def setUp(self) -> None:
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="dam/exp40", cache_state=1, dam_plan=3)
        update_cfg(self.config_file, args)
        self.config_data = GagesConfig(self.config_file)
        # self.nid_file = 'PA_U.xlsx'
        # self.nid_file = 'OH_U.xlsx'
        self.nid_file = 'NID2018_U.xlsx'
        self.test_epoch = 300

    def test_dam_train(self):
        with torch.cuda.device(0):
            quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
            data_dir = os.path.join(quick_data_dir, "allnonref_85-05_nan-0.1_00-1.0")
            data_model_8595 = GagesModel.load_datamodel(data_dir,
                                                        data_source_file_name='data_source.txt',
                                                        stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                        forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                        f_dict_file_name='dictFactorize.json',
                                                        var_dict_file_name='dictAttribute.json',
                                                        t_s_dict_file_name='dictTimeSpace.json')

            gages_model_train = GagesModel.update_data_model(self.config_data, data_model_8595)
            nid_dir = os.path.join("/".join(self.config_data.data_path["DB"].split("/")[:-1]), "nid", "quickdata")
            nid_input = NidModel.load_nidmodel(nid_dir, nid_file=self.nid_file,
                                               nid_source_file_name='nid_source.txt', nid_data_file_name='nid_data.shp')
            gage_main_dam_purpose = unserialize_json(os.path.join(nid_dir, "dam_main_purpose_dict.json"))
            data_input = GagesDamDataModel(gages_model_train, nid_input, True, gage_main_dam_purpose)
            gages_input = choose_which_purpose(data_input)
            master_train(gages_input)

    def test_dam_test(self):
        with torch.cuda.device(1):
            quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
            data_dir = os.path.join(quick_data_dir, "allnonref-dam_95-05_nan-0.1_00-1.0")
            data_model_test = GagesModel.load_datamodel(data_dir,
                                                        data_source_file_name='test_data_source.txt',
                                                        stat_file_name='test_Statistics.json',
                                                        flow_file_name='test_flow.npy',
                                                        forcing_file_name='test_forcing.npy',
                                                        attr_file_name='test_attr.npy',
                                                        f_dict_file_name='test_dictFactorize.json',
                                                        var_dict_file_name='test_dictAttribute.json',
                                                        t_s_dict_file_name='test_dictTimeSpace.json')
            gages_input = GagesModel.update_data_model(self.config_data, data_model_test)
            pred, obs = master_test(gages_input, epoch=self.test_epoch)
            basin_area = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                           is_return_dict=False)
            mean_prep = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                          is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_input.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)
            plot_we_need(gages_input, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
