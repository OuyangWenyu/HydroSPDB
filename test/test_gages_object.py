"""模块测试"""
import copy

import definitions
import unittest
from data import *
import os

from data.data_input import save_datamodel
from data.gages_input_dataset import GagesModels
from utils import *
from data.config import cfg, cmd, update_cfg


class TestDataClassCase(unittest.TestCase):
    config_file = copy.deepcopy(cfg)
    args = cmd(sub="basic/exp7", train_period=["2010-01-01", "2020-01-01"],
               test_period=["1995-01-01", "1996-10-01"], quick_data=0, cache_state=1,
               flow_screen={'missing_data_ratio': 0, 'zero_value_ratio': 1},
               gage_id=["01011000", "01013500", "01015800", "01017000", "01019000", "01022500", "01030500", "01031500",
                        "01034000", "01034500", "01038000", "01042500", "01046500", "01047000", "01048000", "01049000",
                        "01049500", "01052500", "01053500", "01054000", ])
    # gage_id_file="/mnt/data/owen411/code/hydro-anthropogenic-lstm/example/output/gages/basic/exp37/3557basins_ID_NSE_DOR.csv")
    update_cfg(config_file, args)
    data_source_dump = os.path.join(config_file.TEMP_PATH, 'data_source.txt')

    def setUp(self):
        self.config_data = GagesConfig(self.config_file)
        print('setUp...')

    def test_data_source(self):
        # serialize_json(self.config_data.model_dict, self.model_dict_file)
        source_data = GagesSource(self.config_data, self.config_data.model_dict["data"]["tRangeTrain"],
                                  screen_basin_area_huc4=False)
        dir_temp = source_data.all_configs["temp_dir"]
        if not os.path.isdir(dir_temp):
            os.mkdir(dir_temp)
        my_file = os.path.join(dir_temp, 'data_source.txt')
        serialize_pickle(source_data, my_file)

    def test_data_model(self):
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

    def test_usgs_screen_streamflow(self):
        source_data = unserialize_pickle(self.data_source_dump)
        data_flow = source_data.read_usgs()
        usgs_id = source_data.all_configs["flow_screen_gage_id"]
        data_flow, usgs_id, t_range_list = source_data.usgs_screen_streamflow(data_flow, usgs_ids=usgs_id)
        print(data_flow)
        print(usgs_id)
        print(t_range_list)

    def test_read_forcing(self):
        source_data = unserialize_pickle(self.data_source_dump)
        usgs_id = ["0208925200"]
        # usgs_id = ["05481650"] + ["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
        #                           "01055000", "01057000", "01073000"] + ['03144816', '03145000', '03156000', '03157000',
        #                                                                  '03157500', '03219500', '03220000', '03221000',
        #                                                                  '03223000', '03224500', '03225500', '03226800',
        #                                                                  '02383000', '02383500', '02384500', '02385170',
        #                                                                  '02385500', '02385800', '02387000', '02387500',
        #                                                                  '02387600', '02388300', '02388320', '02388350',
        #                                                                  '02388500']
        usgs_id.sort()
        t_range_list = hydro_time.t_range_days(["2000-01-01", "2020-01-01"])
        forcing_data = source_data.read_forcing(usgs_id, t_range_list)
        print(forcing_data)

    def test_read_usgs_gage(self):
        source_data = unserialize_pickle(self.data_source_dump)
        t_range_list = hydro_time.t_range_days(["1995-10-01", "2000-10-01"])
        source_data.read_usge_gage("11", '07311600', t_range_list)

    def test_read_usgs(self):
        source_data = unserialize_pickle(self.data_source_dump)
        source_data.read_usgs()

    def test_read_attr(self):
        config_dir = definitions.CONFIG_DIR
        config_file = os.path.join(config_dir, "landuse/config_exp2.ini")
        subdir = r"landuse/exp2"
        config_data = GagesConfig.set_subdir(config_file, subdir)
        data_source_dump = os.path.join(config_data.data_path["Temp"], 'data_source.txt')
        source_data = unserialize_pickle(data_source_dump)
        usgs_id_lst = source_data.all_configs['flow_screen_gage_id']
        print(all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:])))
        var_lst = source_data.all_configs['attr_chosen']
        source_data.read_attr(usgs_id_lst, var_lst, is_return_dict=True)


if __name__ == '__main__':
    unittest.main()
