import copy
import os
import shutil
import unittest

import torch
import pandas as pd

from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from explore.stat import statError
from hydroDL.master import *
import definitions
from utils import unserialize_numpy, unserialize_json_ordered
from visual.plot_model import plot_we_need
from data.config import cfg, update_cfg, cmd


class MyTestCaseGages(unittest.TestCase):
    # python gages_conus_analysis.py --sub basic/exp4 --quick_data 0 --cache_state 1 --train_period 2000-01-01 2010-01-01 --test_period 2010-01-01 2020-01-01

    def setUp(self) -> None:
        config_file = copy.deepcopy(cfg)
        args = cmd(sub="basic/exp4", train_period=["2000-01-01", "2010-01-01"],
                   test_period=["2010-01-01", "2020-01-01"], quick_data=0, cache_state=1)
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


if __name__ == '__main__':
    unittest.main()
