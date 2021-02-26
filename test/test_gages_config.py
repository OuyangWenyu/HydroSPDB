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
    project_dir = definitions.ROOT_DIR
    dataset = 'gages'
    # dataset = 'camels'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)
    gages_screen_ids = None
    t_train = ['1995-10-01', '2000-10-01']
    t_test = ['2000-10-01', '2005-10-01']
    hydroDams = ['NDAMS_2009', 'DDENS_2009',
                 'STOR_NID_2009', 'STOR_NOR_2009', 'MAJ_NDAMS_2009', 'MAJ_DDENS_2009',
                 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
                 'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS']
    # screen_params={'missing_data_ratio': 0.1, 'zero_value_ratio': 0.005}
    screen_params = {'missing_data_ratio': 0.01, 'zero_value_ratio': 0.005}
    # regions = ['bas_ref_all']
    regions = ['bas_nonref_CntlPlains', 'bas_nonref_EastHghlnds']

    # t_train = ['1995-10-01', '1997-10-01']
    # t_test = ['1997-10-01', '1999-01-01']

    def setUp(self):
        self.config_data = GagesConfig(self.config_file)
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def test_init_path(self):
        test_data = collections.OrderedDict(DB=self.dir_db, Out=self.dir_out, Temp=self.dir_temp)
        self.assertEqual(self.config_data.data_path, test_data)

    def test_read_gages_config(self):
        gages_data = self.config_data.read_data_config()
        dir_db_ = self.dir_db
        test_data = collections.OrderedDict(root_dir=dir_db_, out_dir=self.dir_out, temp_dir=self.dir_temp,
                                            regions=self.regions,
                                            flow_dir=os.path.join(dir_db_, 'gages_streamflow'),
                                            flow_url='https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb'
                                                     '&site_no={}&referred_module=sw&period=&begin_date={}-{}-{'
                                                     '}&end_date={}-{}-{}',
                                            flow_screen_gage_id=self.gages_screen_ids,  # self.gages_screen_ids,
                                            flow_screen_param=self.screen_params,
                                            forcing_chosen=['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                                            forcing_dir=os.path.join(dir_db_, 'gagesII_forcing', 'daymet'),
                                            forcing_type='daymet',
                                            forcing_url=None,
                                            attr_chosen=['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM', 'FORESTNLCD06',
                                                         'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
                                                         'MIXEDFORNLCD06',
                                                         'SHRUBNLCD06',
                                                         'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06',
                                                         'ROCKDEPAVE',
                                                         'AWCAVE', 'PERMAVE', 'RFACT', 'GEOL_REEDBUSH_DOM',
                                                         'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE',
                                                         'STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY',
                                                         'REACHCODE', 'ARTIFPATH_PCT',
                                                         'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE',
                                                         'PERDUN',
                                                         'PERHOR', 'TOPWET', 'CONTACT'] + self.hydroDams,
                                            attr_dir=os.path.join(dir_db_, 'basinchar_and_report_sept_2011'),
                                            attr_url=[
                                                "https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
                                                "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
                                                "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
                                                "https://water.usgs.gov/GIS/dsdl/mainstem_line_covers.zip"],
                                            gage_files_dir=os.path.join(dir_db_, 'basinchar_and_report_sept_2011',
                                                                        'spreadsheets-in-csv-format'),
                                            gage_id_file=os.path.join(dir_db_, 'basinchar_and_report_sept_2011',
                                                                      'spreadsheets-in-csv-format',
                                                                      'conterm_basinid.txt'),
                                            gage_region_dir=os.path.join(dir_db_, 'boundaries_shapefiles_by_aggeco',
                                                                         'boundaries-shapefiles-by-aggeco'),
                                            gage_point_file=os.path.join(dir_db_, "gagesII_9322_point_shapefile",
                                                                         "gagesII_9322_sept30_2011.shp"),
                                            huc4_shp_file=os.path.join(dir_db_, "huc4", "HUC4.shp"),
                                            t_range_all=['1980-01-01', '2015-01-01'])
        self.assertEqual(test_data, gages_data)

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
