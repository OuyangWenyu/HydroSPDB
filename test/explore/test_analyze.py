import unittest
import numpy as np
import pandas as pd
import definitions
from data import *
import os
from data.data_input import GagesModel, load_result
from explore.stat import statError
from visual.plot_model import plot_gages_map_and_ts, plot_gages_attrs_boxes
from visual.plot_stat import plot_diff_boxes
import matplotlib.pyplot as plt
import seaborn as sns


class TestExploreCase(unittest.TestCase):
    def setUp(self):
        """analyze result of model"""
        config_dir = definitions.CONFIG_DIR
        self.ref_config_file = os.path.join(config_dir, "basic/config_exp2.ini")
        self.ref_subdir = r"basic/exp2"
        self.ref_config_data = GagesConfig.set_subdir(self.ref_config_file, self.ref_subdir)
        self.config_file = os.path.join(config_dir, "basic/config_exp18.ini")
        self.subdir = r"basic/exp18"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

        self.data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')

        self.ref_data_model = GagesModel.load_datamodel(self.ref_config_data.data_path["Temp"],
                                                        data_source_file_name='test_data_source.txt',
                                                        stat_file_name='test_Statistics.json',
                                                        flow_file_name='test_flow.npy',
                                                        forcing_file_name='test_forcing.npy',
                                                        attr_file_name='test_attr.npy',
                                                        f_dict_file_name='test_dictFactorize.json',
                                                        var_dict_file_name='test_dictAttribute.json',
                                                        t_s_dict_file_name='test_dictTimeSpace.json')

        attrBasin = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM']
        attrLandcover = ['FORESTNLCD06', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06', 'MIXEDFORNLCD06', 'SHRUBNLCD06',
                         'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attrSoil = ['ROCKDEPAVE', 'AWCAVE', 'PERMAVE', 'RFACT']
        attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT', 'GEOL_REEDBUSH_SITE']
        attrHydro = ['STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
                     'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'BFI_AVE', 'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT']
        attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_AVG_DIS_ALL_MAJ_DAMS']
        attrHydroModOther = ['CANALS_PCT', 'RAW_AVG_DIS_ALLCANALS',
                             'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES', 'FRESHW_WITHDRAWAL',
                             'PCT_IRRIG_AG', 'POWER_SUM_MW']
        attrLandscapePat = ['FRAGUN_BASIN']
        attrLC06Basin = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06']
        attrPopInfrastr = ['ROADS_KM_SQ_KM']
        attrProtAreas = ['PADCAT1_PCT_BASIN', 'PADCAT2_PCT_BASIN']
        self.attr_lst = attrLandscapePat + attrLC06Basin + attrPopInfrastr + attrProtAreas
        # attr_lst = attrHydroModOther

    def tearDown(self):
        print('tearDown...')

    def test_analyze_isref(self):
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        sites_ref = self.ref_data_model.t_s_dict["sites_id"]
        attrs_nonref = self.data_model.data_source.read_attr(sites_nonref, self.attr_lst, is_return_dict=False)
        attrs_ref = self.data_model.data_source.read_attr(sites_ref, self.attr_lst, is_return_dict=False)
        plot_gages_attrs_boxes(sites_nonref, sites_ref, self.attr_lst, attrs_nonref, attrs_ref, diff_str="IS_REF",
                               row_and_col=[2, 4])

    def test_analyze_isnsegood(self):
        # plot is_nse_good
        pred, obs = load_result(self.data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(obs, pred)
        inds_df = pd.DataFrame(inds)
        nse_below = 0.5
        show_ind_key = 'NSE'
        idx_lst_small_nse = inds_df[(inds_df[show_ind_key] < nse_below)].index.tolist()
        sites_small_nse = np.array(self.data_model.t_s_dict['sites_id'])[idx_lst_small_nse]
        assert (all(x < y for x, y in zip(sites_small_nse, sites_small_nse[1:])))
        idx_lst_big_nse = inds_df[(inds_df[show_ind_key] >= nse_below)].index.tolist()
        sites_big_nse = np.array(self.data_model.t_s_dict['sites_id'])[idx_lst_big_nse]
        assert (all(x < y for x, y in zip(sites_big_nse, sites_big_nse[1:])))

        attrs_small = self.data_model.data_source.read_attr(sites_small_nse, self.attr_lst, is_return_dict=False)
        attrs_big = self.data_model.data_source.read_attr(sites_big_nse, self.attr_lst, is_return_dict=False)
        plot_gages_attrs_boxes(sites_small_nse, sites_big_nse, self.attr_lst, attrs_small, attrs_big,
                               diff_str="IS_NSE_GOOD", row_and_col=[2, 4])

    def test_map_ts(self):
        # plot map ts
        pred, obs = load_result(self.data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(obs, pred)
        inds_df = pd.DataFrame(inds)
        show_ind_key = 'NSE'
        idx_lst = np.arange(len(self.data_model.t_s_dict["sites_id"])).tolist()
        plot_gages_map_and_ts(self.data_model, obs, pred, inds_df, show_ind_key, idx_lst, pertile_range=[5, 100])


if __name__ == '__main__':
    unittest.main()
