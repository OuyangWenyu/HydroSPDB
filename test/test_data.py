import collections
import os
import unittest

from data import *


class MyTestCase(unittest.TestCase):
    config_file = r"../data/config.ini"
    root = os.path.expanduser('~')
    dir_db = os.path.join(root, 'Documents/Code/hydro-anthropogenic-lstm/example/data/gages')

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def test_init_path(self):
        path_data = init_path(self.config_file)
        root_ = self.root
        test_data = collections.OrderedDict(
            DB=os.path.join(root_, 'Documents/Code/hydro-anthropogenic-lstm/example/data/gages'),
            Out=os.path.join(root_, 'Documents/Code/hydro-anthropogenic-lstm/example/output/gages'))
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
                                            streamflowUrl='https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}',
                                            tRangeTrain=['1995-01-01', '2000-01-01'],
                                            tRangeTest=['2000-01-01', '2005-01-01'],
                                            regions=['bas_nonref_CntlPlains'],
                                            doNorm=[True, True],
                                            rmNan=[True, False],
                                            daObs=0)
        self.assertEqual(opt_data, test_data)

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
        test_data = collections.OrderedDict(root_dir=dir_db_,
                                            t_range_train=['1995-01-01', '2000-01-01'],
                                            t_range_test=['2000-01-01', '2005-01-01'],
                                            regions=['bas_nonref_CntlPlains'],
                                            flow_dir=os.path.join(dir_db_, 'gages_streamflow'),
                                            flow_url='https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}',
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
                                            gage_id_file=os.path.join(dir_db_, 'basinchar_and_report_sept_2011',
                                                                      'spreadsheets-in-csv-format',
                                                                      'conterm_basinid.txt'),
                                            gage_region_dir=os.path.join(dir_db_, 'boundaries-shapefiles-by-aggeco'),
                                            gage_point_file=os.path.join(dir_db_, "gagesII_9322_point_shapefile",
                                                                         "gagesII_9322_sept30_2011.shp"),
                                            huc4_shp_file=os.path.join(dir_db_, "huc4", "HUC4.shp")
                                            )
        self.assertEqual(gages_data, test_data)

    def test_download_small_zip(self):
        dir_db_ = self.dir_db
        data_url = 'https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip'
        download_small_zip(data_url, dir_db_)


if __name__ == '__main__':
    unittest.main()
