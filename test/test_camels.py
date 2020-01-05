import collections
import os
import unittest

from data import *

import geopandas as gpd

from utils import spatial_join


class TestDataFuncCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    root = os.path.expanduser('~')
    project_dir = 'Documents/Code/hydro-anthropogenic-lstm'
    dataset = 'camels'
    dir_db = os.path.join(root, project_dir, 'example/data', dataset)
    dir_out = os.path.join(root, project_dir, 'example/output', dataset)
    dir_temp = os.path.join(root, project_dir, 'example/temp', dataset)

    t_train = ['1995-01-01', '1997-01-01']
    t_test = ['1997-01-01', '1999-01-01']

    def setUp(self):
        self.config_data = CamelsConfig(self.config_file)
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def test_init_path(self):
        test_data = collections.OrderedDict(DB=self.dir_db, Out=self.dir_out, Temp=self.dir_temp)
        self.assertEqual(self.config_data.data_path, test_data)

    def test_init_data_param(self):
        opt_data = self.config_data.init_data_param()
        test_data = collections.OrderedDict(varT=['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                                            forcingDir='basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing',
                                            forcingType='nldas',
                                            forcingUrl='https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip',
                                            varC=['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                                                  'lai_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                                                  'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                                                  'max_water_content', 'geol_1st_class', 'geol_2nd_class',
                                                  'geol_porostiy',
                                                  'geol_permeability'],
                                            attrDir='camels_attributes_v2.0/camels_attributes_v2.0',
                                            attrUrl="https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip",
                                            streamflowDir='basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow',
                                            gageIdScreen=['01013500', '01022500', '01030500', '01031500', '01047000',
                                                          '01052500'])
        self.assertEqual(test_data, opt_data)

    def test_read_gages_config(self):
        gages_data = self.config_data.read_data_config()
        dir_db_ = self.dir_db
        test_data = collections.OrderedDict(root_dir=dir_db_, out_dir=self.dir_out, temp_dir=self.dir_temp,
                                            flow_dir=os.path.join(dir_db_,
                                                                  'basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow'),
                                            flow_screen_gage_id=['01013500', '01022500', '01030500', '01031500',
                                                                 '01047000', '01052500'],
                                            forcing_chosen=['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                                            forcing_dir=os.path.join(dir_db_,
                                                                     'basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing',
                                                                     'nldas'),
                                            forcing_type='nldas',
                                            forcing_url="https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip",
                                            attr_url='https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip',
                                            attr_chosen=['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
                                                         'lai_max',
                                                         'lai_diff', 'dom_land_cover_frac', 'dom_land_cover',
                                                         'root_depth_50',
                                                         'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                                                         'max_water_content', 'geol_1st_class', 'geol_2nd_class',
                                                         'geol_porostiy',
                                                         'geol_permeability'
                                                         ],
                                            attr_dir=os.path.join(dir_db_,
                                                                  'camels_attributes_v2.0/camels_attributes_v2.0'),
                                            gauge_id_file=os.path.join(dir_db_,
                                                                       'basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata',
                                                                       'gauge_information.txt'))
        self.assertEqual(test_data, gages_data)


if __name__ == '__main__':
    unittest.main()
