import os
import unittest
import numpy as np

import definitions
from data import GagesConfig, GagesSource
from utils import serialize_pickle, unserialize_pickle
from utils.dataset_format import trans_daymet_to_camels


class MyTestCase(unittest.TestCase):
    def setUp(self):
        config_dir = definitions.CONFIG_DIR
        config_file = os.path.join(config_dir, "transdata/config_exp1.ini")
        subdir = r"transdata/exp1"
        self.config_data = GagesConfig.set_subdir(config_file, subdir)

    def test_data_source(self):
        source_data = GagesSource(self.config_data, self.config_data.model_dict["data"]["tRangeTrain"],
                                  screen_basin_area_huc4=False)
        my_file = os.path.join(self.config_data.data_path["Temp"], 'data_source.txt')
        serialize_pickle(source_data, my_file)

    def test_trans_all_forcing_file_to_camels(self):
        """the function need to be run region by region"""
        data_source_dump = os.path.join(self.config_data.data_path["Temp"], 'data_source.txt')
        source_data = unserialize_pickle(data_source_dump)
        output_dir = os.path.join(self.config_data.data_path["DB"], "basin_mean_forcing", "daymet")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        region_names = [region_temp.split("_")[-1] for region_temp in source_data.all_configs['regions']]
        # forcing data file generated is named as "allref", so rename the "all"
        region_names = ["allref" if r == "all" else r for r in region_names]
        year_start = int(source_data.t_range[0].split("-")[0])
        year_end = int(source_data.t_range[1].split("-")[0])
        years = np.arange(year_start, year_end)
        assert (all(x < y for x, y in zip(source_data.gage_dict['STAID'], source_data.gage_dict['STAID'][1:])))
        for year in years:
            trans_daymet_to_camels(source_data.all_configs["forcing_dir"], output_dir, source_data.gage_dict,
                                   region_names[0], year)


if __name__ == '__main__':
    unittest.main()
