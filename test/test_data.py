import collections
import os
import unittest

from data import *

import geopandas as gpd

from utils import spatial_join


class DataFuncCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    root = os.path.expanduser('~')
    dir_db = os.path.join(root, 'Documents/Code/hydro-anthropogenic-lstm/example/data/gages')
    dir_out = os.path.join(root, 'Documents/Code/hydro-anthropogenic-lstm/example/output/gages')
    dir_temp = os.path.join(root, 'Documents/Code/hydro-anthropogenic-lstm/example/temp/gages')

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def test_init_path(self):
        path_data = init_path(self.config_file)
        root_ = self.root
        test_data = collections.OrderedDict(
            DB=os.path.join(root_, 'Documents/Code/hydro-anthropogenic-lstm/example/data/gages'),
            Out=os.path.join(root_, 'Documents/Code/hydro-anthropogenic-lstm/example/output/gages'),
            Temp=os.path.join(root_, 'Documents/Code/hydro-anthropogenic-lstm/example/temp/gages'))
        self.assertEqual(path_data, test_data)

    def test_init_data_param(self):
        opt_data = init_data_param(self.config_file)
        test_data = collections.OrderedDict(varT=['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                                            forcingDir='gagesII_forcing',
                                            forcingType='daymet',
                                            forcingUrl=None,
                                            varC=['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM', 'FORESTNLCD06',
                                                  'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06',
                                                  'SHRUBNLCD06',
                                                  'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06', 'ROCKDEPAVE',
                                                  'AWCAVE', 'PERMAVE', 'RFACT', 'GEOL_REEDBUSH_DOM',
                                                  'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE',
                                                  'STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY',
                                                  'REACHCODE', 'ARTIFPATH_PCT',
                                                  'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE', 'PERDUN',
                                                  'PERHOR', 'TOPWET', 'CONTACT', 'NDAMS_2009', 'DDENS_2009',
                                                  'STOR_NID_2009', 'STOR_NOR_2009', 'MAJ_NDAMS_2009', 'MAJ_DDENS_2009',
                                                  'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
                                                  'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS'
                                                  ],
                                            attrDir='basinchar_and_report_sept_2011',
                                            attrUrl='https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip',
                                            streamflowDir='gages_streamflow',
                                            streamflowUrl='https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb'
                                                          '&site_no={}&referred_module=sw&period=&begin_date={}-{}-{'
                                                          '}&end_date={}-{}-{}',
                                            gageIdScreen=['03144816', '03145000', '03156000', '03157000', '03157500',
                                                          '03219500', '03220000', '03221000', '03223000', '03224500',
                                                          '03225500', '03226800'],
                                            tRangeScreen=['1995-01-01', '1997-01-01'],
                                            streamflowScreenParam={'missing_data_ratio': 0.1, 'zero_value_ratio': 0.005,
                                                                   'basin_area_ceil': 'HUC4'},
                                            tRangeTrain=['1995-01-01', '2000-01-01'],
                                            tRangeTest=['2000-01-01', '2005-01-01'],
                                            regions=['bas_nonref_CntlPlains'],
                                            doNorm=[True, True],
                                            rmNan=[True, False],
                                            daObs=0,
                                            tRangeAll=['1980-01-01', '2015-01-01'])
        self.assertEqual(test_data, opt_data)

    def test_download_kaggle_file(self):
        dir_db_ = self.dir_db
        kaggle_json = os.path.join(dir_db_, 'kaggle.json')
        name_of_dataset = "owenyy/wbdhu4-a-us-september2019-shpfile"
        path_download = os.path.join(dir_db_, "huc4")
        file_download = os.path.join(path_download, "HUC4.shp")
        download_kaggle_file(kaggle_json, name_of_dataset, path_download, file_download)

    def test_read_gages_config(self):
        gages_data = read_gages_config(self.config_file)
        dir_db_ = self.dir_db
        test_data = collections.OrderedDict(root_dir=dir_db_, out_dir=self.dir_out, temp_dir=self.dir_temp,
                                            t_range_train=['1995-01-01', '2000-01-01'],
                                            t_range_test=['2000-01-01', '2005-01-01'],
                                            regions=['bas_nonref_CntlPlains'],
                                            flow_dir=os.path.join(dir_db_, 'gages_streamflow'),
                                            flow_url='https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb'
                                                     '&site_no={}&referred_module=sw&period=&begin_date={}-{}-{'
                                                     '}&end_date={}-{}-{}',
                                            flow_screen_gage_id=['03144816', '03145000', '03156000', '03157000',
                                                                 '03157500',
                                                                 '03219500', '03220000', '03221000', '03223000',
                                                                 '03224500',
                                                                 '03225500', '03226800'],
                                            flow_screen_t_range=['1995-01-01', '1997-01-01'],
                                            flow_screen_param={'missing_data_ratio': 0.1, 'zero_value_ratio': 0.005,
                                                               'basin_area_ceil': 'HUC4'},
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
                                                         'PERHOR', 'TOPWET', 'CONTACT', 'NDAMS_2009', 'DDENS_2009',
                                                         'STOR_NID_2009', 'STOR_NOR_2009', 'MAJ_NDAMS_2009',
                                                         'MAJ_DDENS_2009',
                                                         'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
                                                         'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS'
                                                         ],
                                            attr_dir=os.path.join(dir_db_, 'basinchar_and_report_sept_2011'),
                                            attr_url='https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip',
                                            gage_files_dir=os.path.join(dir_db_, 'basinchar_and_report_sept_2011',
                                                                        'spreadsheets-in-csv-format'),
                                            gage_id_file=os.path.join(dir_db_, 'basinchar_and_report_sept_2011',
                                                                      'spreadsheets-in-csv-format',
                                                                      'conterm_basinid.txt'),
                                            gage_region_dir=os.path.join(dir_db_, 'boundaries-shapefiles-by-aggeco'),
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
        gage_region_dir = os.path.join(dir_db_, 'boundaries-shapefiles-by-aggeco')
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
