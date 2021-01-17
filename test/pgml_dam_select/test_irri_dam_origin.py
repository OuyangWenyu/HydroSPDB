import unittest

import copy
import os
import pandas as pd
import torch

from data import GagesConfig
from data.data_input import GagesModel, save_datamodel, _basin_norm, save_result
from data.gages_input_dataset import GagesEtDataModel, GagesModels, generate_gages_models
from data.gridmet_input import GridmetConfig, GridmetSource, GridmetModel
from hydroDL.master.master import master_train_gridmet, master_test_gridmet

import definitions
from utils import unserialize_json_ordered
from data.config import cfg
from visual.plot_model import plot_we_need


class TestOriginGagesIrri(unittest.TestCase):
    """data pre-process and post-process"""

    def setUp(self) -> None:
        """update some parameters"""
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
        config4gridmet.SUB_EXP = "exp2"
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
        config4gridmet.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
        config4gridmet.CACHE.QUICK_DATA = False
        config4gridmet.CACHE.GEN_QUICK_DATA = True
        self.config_data = GagesConfig(config4gridmet)

    def test_gages_data_model(self):
        # the major procedures for only 300+ irrigation basins:
        # 1. weighted average value of all crops in a basin for ETc
        # 2. use the forcing data from gridmet as the input to LSTM
        # 3. add ETo as additional input
        # 4. use ETo and ETc as additional inputs
        # 5. compare the CDFs of all these 3 cases
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

    def test_dam_train(self):
        data_dir = self.config_data.data_path["Temp"]
        data_model_train = GagesModel.load_datamodel(data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')

        CROP_ET_ZIP_DIR = os.path.join(definitions.ROOT_DIR, "example", "data", "gridmet")
        gridmet_config = GridmetConfig(CROP_ET_ZIP_DIR)
        gridmet_source = GridmetSource(gridmet_config, self.irri_basins_id)
        gridmet_data_model_train = GridmetModel(gridmet_source, self.t_range_train)

        with torch.cuda.device(0):
            data_et_model = GagesEtDataModel(data_model_train, gridmet_data_model_train)
            master_train_gridmet(data_et_model)

    def test_dam_test(self):
        with torch.cuda.device(0):
            data_dir = self.config_data.config_file.CACHE.DATA_DIR
            gages_model_test = GagesModel.load_datamodel(data_dir,
                                                         data_source_file_name='test_data_source.txt',
                                                         stat_file_name='test_Statistics.json',
                                                         flow_file_name='test_flow.npy',
                                                         forcing_file_name='test_forcing.npy',
                                                         attr_file_name='test_attr.npy',
                                                         f_dict_file_name='test_dictFactorize.json',
                                                         var_dict_file_name='test_dictAttribute.json',
                                                         t_s_dict_file_name='test_dictTimeSpace.json')
            CROP_ET_ZIP_DIR = os.path.join(definitions.ROOT_DIR, "example", "data", "gridmet")
            gridmet_config = GridmetConfig(CROP_ET_ZIP_DIR)
            gridmet_source = GridmetSource(gridmet_config, self.irri_basins_id)
            gridmet_data_model_train = GridmetModel(gridmet_source, self.t_range_train)
            gridmet_data_model_test = GridmetModel(gridmet_source, self.t_range_test, is_test=True,
                                                   stat_train=gridmet_data_model_train.stat_forcing_dict)
            data_et_model = GagesEtDataModel(gages_model_test, gridmet_data_model_test, True)
            pred, obs = master_test_gridmet(data_et_model, epoch=cfg.TEST_EPOCH)
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
