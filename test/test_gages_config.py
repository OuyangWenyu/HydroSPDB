import collections
import copy
import os
import unittest
import definitions
from data import *

import geopandas as gpd

from data.download_data import download_google_drive, download_small_zip
from utils import spatial_join
from data.config import cfg


class TestDataFuncCase(unittest.TestCase):
    config_file = copy.deepcopy(cfg)
    config_file.SUBSET = "basic"
    config_file.SUB_EXP = "exp5"
    config_file.TEMP_PATH = os.path.join(config_file.ROOT_DIR, 'temp', config_file.DATASET,
                                         config_file.SUBSET, config_file.SUB_EXP)
    if not os.path.exists(config_file.TEMP_PATH):
        os.makedirs(config_file.TEMP_PATH)
    config_file.OUT_PATH = os.path.join(config_file.ROOT_DIR, 'output', config_file.DATASET,
                                        config_file.SUBSET, config_file.SUB_EXP)
    if not os.path.exists(config_file.OUT_PATH):
        os.makedirs(config_file.OUT_PATH)

    t_range_train = ["2000-01-01", "2010-01-01"]
    t_range_test = ["2010-01-01", "2020-01-01"]
    config_file.MODEL.tRangeTrain = t_range_train
    config_file.MODEL.tRangeTest = t_range_test
    config_file.GAGES.streamflowScreenParams = {'missing_data_ratio': 1, 'zero_value_ratio': 1}
    config_file.CACHE.QUICK_DATA = False
    config_file.CACHE.GEN_QUICK_DATA = True
    project_dir = definitions.ROOT_DIR
    dataset = 'gages'
    # dataset = 'camels'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)

    def setUp(self):
        self.config_data = GagesConfig(self.config_file)
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def test_init_path(self):
        test_data = collections.OrderedDict(DB=self.dir_db, Out=os.path.join(self.dir_out, "basic", "exp5"),
                                            Temp=os.path.join(self.dir_temp, "basic", "exp5"))
        self.assertEqual(self.config_data.data_path, test_data)

    def test_download_small_zip(self):
        dir_db_ = self.dir_db
        data_url = 'https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip'
        download_small_zip(data_url, dir_db_)

    def test_read_gpd_file(self):
        dir_db_ = self.dir_db
        gage_region_dir = os.path.join(dir_db_, 'boundaries_shapefiles_by_aggeco', 'boundaries-shapefiles-by-aggeco')
        shapefile = os.path.join(gage_region_dir, 'bas_nonref_CntlPlains.shp')
        shape_data = gpd.read_file(shapefile)
        print(shape_data.columns)
        gages_id = shape_data['GAGE_ID'].values
        print(gages_id)

    def test_spatial_join(self):
        dir_db_ = self.dir_db
        points_file = os.path.join(dir_db_, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp")
        polygons_file = os.path.join(dir_db_, "huc4", "HUC4.shp")
        spatial_join(points_file, polygons_file)

    def test_download_google_drive(self):
        dir_db_ = self.dir_db
        google_drive_dir_name = "daymet"
        download_dir = os.path.join(dir_db_, 'gagesII_forcing', 'daymet')
        client_secrets_file = os.path.join(dir_db_, "mycreds.txt")
        download_google_drive(client_secrets_file, google_drive_dir_name, download_dir)


if __name__ == '__main__':
    unittest.main()
