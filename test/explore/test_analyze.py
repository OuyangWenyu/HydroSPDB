import unittest
import numpy as np
import pandas as pd
import definitions
from data import *
import os
from data.data_input import GagesModel, load_result
from explore.stat import statError
from visual.plot_model import plot_gages_map_and_ts, plot_gages_attrs_boxes, plot_scatter_multi_attrs
from visual.plot_stat import plot_diff_boxes, plot_scatter_xyc
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
        # self.attr_lst = attrLandscapePat + attrLC06Basin + attrPopInfrastr + attrProtAreas
        self.attr_lst = attrHydroModDams

        # plot is_nse_good
        pred, obs = load_result(self.data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        self.pred = pred.reshape(pred.shape[0], pred.shape[1])
        self.obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(self.obs, self.pred)
        self.inds_df = pd.DataFrame(inds)

    def test_analyze_isref(self):
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        sites_ref = self.ref_data_model.t_s_dict["sites_id"]
        attrs_nonref = self.data_model.data_source.read_attr(sites_nonref, self.attr_lst, is_return_dict=False)
        attrs_ref = self.data_model.data_source.read_attr(sites_ref, self.attr_lst, is_return_dict=False)
        plot_gages_attrs_boxes(sites_nonref, sites_ref, self.attr_lst, attrs_nonref, attrs_ref, diff_str="IS_REF",
                               row_and_col=[2, 4])

    def test_analyze_isnsegood(self):
        inds_df = self.inds_df
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
        inds_df = self.inds_df
        show_ind_key = 'NSE'
        idx_lst = np.arange(len(self.data_model.t_s_dict["sites_id"])).tolist()

        nse_range = [-10, 0]
        idx_lst_small_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        plot_gages_map_and_ts(self.data_model, self.obs, self.pred, inds_df, show_ind_key, idx_lst_small_nse,
                              pertile_range=[0, 100])

    def test_x_y_scatter(self):
        elev_var = "ELEV_MEAN_M_BASIN"
        attr_elev_lst = [elev_var]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_elev = self.data_model.data_source.read_attr(sites_nonref, attr_elev_lst, is_return_dict=False)
        inds_df_now = self.inds_df
        nse_range = [0, 1]
        show_ind_key = 'NSE'
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        nse_values = self.inds_df["NSE"].values[idx_lst_nse_range]
        df = pd.DataFrame({elev_var: attrs_elev[idx_lst_nse_range, 0], show_ind_key: nse_values})
        sns.jointplot(x=elev_var, y=show_ind_key, data=df, kind="reg")
        plt.show()

    def test_x_y_color_scatter(self):
        elev_var = "ELEV_MEAN_M_BASIN"
        stor_var = "STOR_NOR_2009"
        attr_elev_stor_lst = [elev_var, stor_var]
        sites_nonref = self.data_model.t_s_dict["sites_id"]
        attrs_elev_stor = self.data_model.data_source.read_attr(sites_nonref, attr_elev_stor_lst, is_return_dict=False)

        inds_df_now = self.inds_df
        nse_range = [0, 1]
        show_ind_key = 'NSE'
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        nse_values = self.inds_df["NSE"].values[idx_lst_nse_range]

        plot_scatter_xyc(elev_var, attrs_elev_stor[idx_lst_nse_range, 0], stor_var,
                         attrs_elev_stor[idx_lst_nse_range, 1], c_label=show_ind_key, c=nse_values, is_reg=True,
                         ylim=[0, 2000])
        # plt.xlabel(elev_var + '(m)')
        # plt.ylabel(stor_var + '(1000 m^3/sq km)')

    def test_scatters(self):
        attr_lst = self.attr_lst
        show_ind_key = 'NSE'
        y_var_lst = [show_ind_key]
        inds_df_now = self.inds_df
        nse_range = [0, 1]
        # idx_lst_nse_range = inds_df_now.index.tolist()
        idx_lst_nse_range = inds_df_now[
            (inds_df_now[show_ind_key] >= nse_range[0]) & (inds_df_now[show_ind_key] < nse_range[1])].index.tolist()
        plot_scatter_multi_attrs(self.data_model, self.inds_df, idx_lst_nse_range, attr_lst, y_var_lst)


if __name__ == '__main__':
    unittest.main()
