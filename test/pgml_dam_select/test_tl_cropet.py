import shutil
import unittest

import copy
import os
import pandas as pd
import torch

from data import GagesConfig
from data.data_input import GagesModel, save_datamodel, _basin_norm, save_result
from data.gages_input_dataset import GagesEtDataModel, GagesModels, generate_gages_models
from data.gridmet_input import GridmetConfig, GridmetSource, GridmetModel, save_gridmet_datamodel
from hydroDL.master.master import master_tl_train_gridmet, master_tl_test_gridmet

import definitions
from utils import unserialize_json_ordered
from data.config import cfg
from visual.plot_model import plot_we_need


class MyTestCase(unittest.TestCase):
    """use pretrained model in 3557 then retrain in 328 with cet data"""

    def setUp(self) -> None:
        # Firstly, choose the dammed basins with irrigation as the main purpose of reservoirs
        # prerequisite: app/streamflow/gages_conus_result_section2.py has been run
        nid_dir = os.path.join(cfg.NID.NID_DIR, "test")
        main_purpose_file = os.path.join(nid_dir, "dam_main_purpose_dict.json")
        all_sites_purposes_dict = unserialize_json_ordered(main_purpose_file)
        all_sites_purposes = pd.Series(all_sites_purposes_dict)
        include_irr = all_sites_purposes.apply(lambda x: "I" in x)
        self.irri_basins_id = all_sites_purposes[include_irr == True].index.tolist()
        df = pd.DataFrame({"GAGE_ID": self.irri_basins_id})

        OUTPUT_IRRI_GAGE_ID = False
        if OUTPUT_IRRI_GAGE_ID:
            df.to_csv("irrigation_gage_id.csv", index=None)

        self.t_range_train = ["2008-01-01", "2013-01-01"]
        self.t_range_test = ["2013-01-01", "2018-01-01"]

        config4gridmet = copy.deepcopy(cfg)

        config4gridmet.SUBSET = "gridmet"
        config4gridmet.SUB_EXP = "exp7"
        config4gridmet.TEMP_PATH = os.path.join(config4gridmet.ROOT_DIR, 'temp', config4gridmet.DATASET,
                                                config4gridmet.SUBSET, config4gridmet.SUB_EXP)
        if not os.path.exists(config4gridmet.TEMP_PATH):
            os.makedirs(config4gridmet.TEMP_PATH)
        config4gridmet.OUT_PATH = os.path.join(config4gridmet.ROOT_DIR, 'output', config4gridmet.DATASET,
                                               config4gridmet.SUBSET, config4gridmet.SUB_EXP)
        if not os.path.exists(config4gridmet.OUT_PATH):
            os.makedirs(config4gridmet.OUT_PATH)

        config4gridmet.MODEL.tRangeTrain = self.t_range_train
        config4gridmet.MODEL.tRangeTest = self.t_range_test
        config4gridmet.MODEL.name = "CudnnLstmModel_FT_comb"
        config4gridmet.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
        config4gridmet.CACHE.QUICK_DATA = False
        config4gridmet.CACHE.GEN_QUICK_DATA = True
        self.config_data = GagesConfig(config4gridmet)

        # copy the traiend model file to the dir
        config4pretrained = copy.deepcopy(cfg)
        config4pretrained.SUBSET = "gridmet"
        config4pretrained.SUB_EXP = "exp6"
        config4pretrained.TEMP_PATH = os.path.join(config4pretrained.ROOT_DIR, 'temp', config4pretrained.DATASET,
                                                   config4pretrained.SUBSET, config4pretrained.SUB_EXP)
        config4pretrained.OUT_PATH = os.path.join(config4pretrained.ROOT_DIR, 'output', config4pretrained.DATASET,
                                                  config4pretrained.SUBSET, config4pretrained.SUB_EXP)

        config4pretrained.MODEL.tRangeTrain = self.t_range_train
        config4pretrained.MODEL.tRangeTest = self.t_range_test
        config4pretrained.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
        config4pretrained.CACHE.QUICK_DATA = False
        config4pretrained.CACHE.GEN_QUICK_DATA = True
        config_data4pretrained = GagesConfig(config4pretrained)
        pretrained_model_dir = os.path.join(self.config_data.data_path["Temp"], "pretrained")
        if not os.path.isdir(pretrained_model_dir):
            os.makedirs(pretrained_model_dir)
        self.pretrianed_file_name = os.path.join(pretrained_model_dir, "model_Ep300.pt")
        if not os.path.isfile(self.pretrianed_file_name):
            pretrianed_source = os.path.join(config_data4pretrained.data_path["Out"], "model_Ep300.pt")
            shutil.copy(pretrianed_source, self.pretrianed_file_name)
        # set the size
        model_tmp = torch.load(self.pretrianed_file_name, map_location="cuda:0")
        self.trained_x_size = model_tmp.nx

    def test_gages_data_model(self):
        gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False, sites_id=self.irri_basins_id)
        gages_model_train = gages_model.data_model_train
        gages_model_test = gages_model.data_model_test
        save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')

    def test_gridmet_data_model(self):
        CROP_ET_ZIP_DIR = os.path.join(definitions.ROOT_DIR, "example", "data", "gridmet")
        gridmet_config = GridmetConfig(CROP_ET_ZIP_DIR, et_dir_name="irrigation328",
                                       et_shp_file_name="some_from_irrigation")
        gridmet_source = GridmetSource(gridmet_config, self.irri_basins_id)
        gridmet_data_model_train = GridmetModel(gridmet_source, self.t_range_train)
        dir_temp = self.config_data.data_path["Temp"]
        save_gridmet_datamodel(dir_temp, gridmet_data_model_train, gridmet_source_file_name='gridmet_source.txt',
                               gridmet_stat_cet_file_name='gridmet_stat_cet.json',
                               gridmet_stat_forcing_file_name='gridmet_stat_forcing.json',
                               gridmet_forcing_file_name='gridmet_forcing', gridmet_cet_file_name='gridmet_cet',
                               gridmet_time_range_file_name='gridmet_time_range.txt')
        gridmet_data_model_test = GridmetModel(gridmet_source, self.t_range_test, is_test=True,
                                               stat_train=gridmet_data_model_train.stat_forcing_dict,
                                               stat_cet_train=gridmet_data_model_train.stat_cet_dict)
        save_gridmet_datamodel(dir_temp, gridmet_data_model_test, gridmet_source_file_name='test_gridmet_source.txt',
                               gridmet_stat_cet_file_name='test_gridmet_stat_cet.json',
                               gridmet_stat_forcing_file_name='test_gridmet_stat_forcing.json',
                               gridmet_forcing_file_name='test_gridmet_forcing',
                               gridmet_cet_file_name='test_gridmet_cet',
                               gridmet_time_range_file_name='test_gridmet_time_range.txt')

    def test_dam_tl_train(self):
        # train by transfer learning
        data_dir = self.config_data.data_path["Temp"]
        data_model_train = GagesModel.load_datamodel(data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')
        gridmet_data_model_train = GridmetModel.load_gridmet_datamodel(data_dir,
                                                                       gridmet_source_file_name='gridmet_source.txt',
                                                                       gridmet_stat_cet_file_name='gridmet_stat_cet.json',
                                                                       gridmet_stat_forcing_file_name='gridmet_stat_forcing.json',
                                                                       gridmet_forcing_file_name='gridmet_forcing.npy',
                                                                       gridmet_cet_file_name='gridmet_cet.npy',
                                                                       gridmet_time_range_file_name='gridmet_time_range.txt')
        with torch.cuda.device(0):
            data_et_model = GagesEtDataModel(data_model_train, gridmet_data_model_train, True)
            master_tl_train_gridmet(data_et_model, self.pretrianed_file_name, self.trained_x_size)

    def test_dam_tl_test(self):
        with torch.cuda.device(0):
            data_dir = self.config_data.data_path["Temp"]
            gages_model_test = GagesModel.load_datamodel(data_dir,
                                                         data_source_file_name='test_data_source.txt',
                                                         stat_file_name='test_Statistics.json',
                                                         flow_file_name='test_flow.npy',
                                                         forcing_file_name='test_forcing.npy',
                                                         attr_file_name='test_attr.npy',
                                                         f_dict_file_name='test_dictFactorize.json',
                                                         var_dict_file_name='test_dictAttribute.json',
                                                         t_s_dict_file_name='test_dictTimeSpace.json')
            gridmet_data_model_test = GridmetModel.load_gridmet_datamodel(data_dir,
                                                                          gridmet_source_file_name='test_gridmet_source.txt',
                                                                          gridmet_stat_cet_file_name='test_gridmet_stat_cet.json',
                                                                          gridmet_stat_forcing_file_name='test_gridmet_stat_forcing.json',
                                                                          gridmet_forcing_file_name='test_gridmet_forcing.npy',
                                                                          gridmet_cet_file_name='test_gridmet_cet.npy',
                                                                          gridmet_time_range_file_name='test_gridmet_time_range.txt')
            data_et_model = GagesEtDataModel(gages_model_test, gridmet_data_model_test, True)
            pred, obs = master_tl_test_gridmet(data_et_model, self.pretrianed_file_name, self.trained_x_size,
                                               epoch=cfg.TEST_EPOCH)
            basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                                is_return_dict=False)
            mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                               is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_model_test.data_source.data_config.data_path['Temp'], cfg.TEST_EPOCH, pred, obs)
            plot_we_need(gages_model_test, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
