import copy
import os
import unittest

from data import GagesConfig, GagesSource
from data.config import cfg, cmd, update_cfg
from data.data_input import GagesModel
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp
from explore.stat import ecdf
from matplotlib import pyplot as plt, gridspec
import numpy as np
import seaborn as sns
import pandas as pd

from utils.hydro_util import hydro_logger
from visual.plot_model import plot_gages_map_and_scatter
from visual.plot_stat import plot_ecdfs_matplot, plot_ecdfs


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """test results in different dor value settings"""
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="basic/exp37")
        # args = cmd(sub="dam/exp12", cache_state=1, attr_screen={"DOR": 0.003})
        self.dor = 0.1
        update_cfg(self.config_file, args)
        self.test_epoch = 300
        self.FIGURE_DPI = 600
        self.config_data = GagesConfig(self.config_file)
        self.exp_lst = ["basic_exp37", "basic_exp39", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43"]

    def test_zero_small_dor_basins_locations(self):
        conus_exps = self.exp_lst
        test_epoch = self.test_epoch
        inds_df, pred, obs = load_ensemble_result(self.config_file, conus_exps, test_epoch, return_value=True)
        conus_config_data = load_dataconfig_case_exp(self.config_file, conus_exps[0])
        conus_data_model = GagesModel.load_datamodel(conus_config_data.data_path["Temp"],
                                                     data_source_file_name='test_data_source.txt',
                                                     stat_file_name='test_Statistics.json',
                                                     flow_file_name='test_flow.npy',
                                                     forcing_file_name='test_forcing.npy',
                                                     attr_file_name='test_attr.npy',
                                                     f_dict_file_name='test_dictFactorize.json',
                                                     var_dict_file_name='test_dictAttribute.json',
                                                     t_s_dict_file_name='test_dictTimeSpace.json')
        conus_sites = conus_data_model.t_s_dict["sites_id"]

        all_lat = conus_data_model.data_source.gage_dict["LAT_GAGE"]
        all_lon = conus_data_model.data_source.gage_dict["LNG_GAGE"]
        show_ind_key = "NSE"
        attr_lst = ["SLOPE_PCT", "ELEV_MEAN_M_BASIN"]
        attrs = conus_data_model.data_source.read_attr(conus_sites, attr_lst, is_return_dict=False)

        western_lon_idx = [i for i in range(all_lon.size) if all_lon[i] < -100]

        nse_range = [0, 1]
        idx_lst_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        idx_lst_nse = np.intersect1d(western_lon_idx, idx_lst_nse)

        # small dor
        source_data_dor1 = GagesSource.choose_some_basins(conus_config_data,
                                                          conus_config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=-self.dor)

        # basins with dams
        source_data_withdams = GagesSource.choose_some_basins(conus_config_data,
                                                              conus_config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              dam_num=[1, 10000])
        # basins without dams
        source_data_withoutdams = GagesSource.choose_some_basins(conus_config_data,
                                                                 conus_config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False,
                                                                 dam_num=0)

        sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
        sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']

        sites_id_nodam = source_data_withoutdams.all_configs['flow_screen_gage_id']
        sites_id_smalldam = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()

        idx_lst_nodam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_nodam]
        idx_lst_smalldam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_smalldam]

        type_1_index_lst = np.intersect1d(idx_lst_nodam_in_conus, idx_lst_nse).tolist()
        type_2_index_lst = np.intersect1d(idx_lst_smalldam_in_conus, idx_lst_nse).tolist()
        pd.DataFrame({"GAGE_ID": np.array(conus_sites)[type_1_index_lst]}).to_csv(
            os.path.join(conus_config_data.data_path["Out"], "western-zero-dor-sites.csv"))
        pd.DataFrame({"GAGE_ID": np.array(conus_sites)[type_2_index_lst]}).to_csv(
            os.path.join(conus_config_data.data_path["Out"], "western-small-dor-sites.csv"))
        frame = []
        df_type1 = pd.DataFrame({"type": np.full(len(type_1_index_lst), "zero-dor"),
                                 show_ind_key: inds_df[show_ind_key].values[type_1_index_lst],
                                 "lat": all_lat[type_1_index_lst],
                                 "lon": all_lon[type_1_index_lst],
                                 "slope": attrs[type_1_index_lst, 0],
                                 "elevation": attrs[type_1_index_lst, 1]})
        frame.append(df_type1)
        df_type2 = pd.DataFrame({"type": np.full(len(type_2_index_lst), "small-dor"),
                                 show_ind_key: inds_df[show_ind_key].values[type_2_index_lst],
                                 "lat": all_lat[type_2_index_lst],
                                 "lon": all_lon[type_2_index_lst],
                                 "slope": attrs[type_2_index_lst, 0],
                                 "elevation": attrs[type_2_index_lst, 1]})
        frame.append(df_type2)
        data_df = pd.concat(frame)
        idx_lst = [np.arange(len(type_1_index_lst)),
                   np.arange(len(type_1_index_lst), len(type_1_index_lst) + len(type_2_index_lst))]
        plot_gages_map_and_scatter(data_df, [show_ind_key, "lat", "lon", "slope"], idx_lst, cmap_strs=["Reds", "Blues"],
                                   labels=["zero-dor", "small-dor"], scatter_label=[attr_lst[0], show_ind_key],
                                   wspace=2, hspace=1.5, legend_y=.8, sub_fig_ratio=[6, 4, 1])
        plt.tight_layout()
        plt.show()

    def test_diff_dor_fig2_in_the_paper(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        config_data = self.config_data
        config_file = self.config_file
        test_epoch = self.test_epoch
        exp_lst = self.exp_lst
        figure_dpi = self.FIGURE_DPI
        inds_df, pred_mean, obs_mean = load_ensemble_result(config_file, exp_lst, test_epoch, return_value=True)
        diversion_yes = True
        diversion_no = False
        source_data_diversion = GagesSource.choose_some_basins(config_data,
                                                               config_data.model_dict["data"]["tRangeTrain"],
                                                               screen_basin_area_huc4=False,
                                                               diversion=diversion_yes)
        source_data_nodivert = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              diversion=diversion_no)
        sites_id_nodivert = source_data_nodivert.all_configs['flow_screen_gage_id']
        sites_id_diversion = source_data_diversion.all_configs['flow_screen_gage_id']

        dor_1 = - self.dor
        dor_2 = self.dor
        source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                          config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor_1)
        source_data_dor2 = GagesSource.choose_some_basins(config_data,
                                                          config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor_2)
        sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
        sites_id_dor2 = source_data_dor2.all_configs['flow_screen_gage_id']

        # basins with dams
        source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              dam_num=[1, 100000])
        sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']
        sites_id_dor1 = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()

        no_divert_small_dor = np.intersect1d(sites_id_nodivert, sites_id_dor1)
        no_divert_large_dor = np.intersect1d(sites_id_nodivert, sites_id_dor2)
        diversion_small_dor = np.intersect1d(sites_id_diversion, sites_id_dor1)
        diversion_large_dor = np.intersect1d(sites_id_diversion, sites_id_dor2)

        all_sites = data_model.t_s_dict["sites_id"]
        idx_lst_nodivert_smalldor = [i for i in range(len(all_sites)) if all_sites[i] in no_divert_small_dor]
        idx_lst_nodivert_largedor = [i for i in range(len(all_sites)) if all_sites[i] in no_divert_large_dor]
        idx_lst_diversion_smalldor = [i for i in range(len(all_sites)) if all_sites[i] in diversion_small_dor]
        idx_lst_diversion_largedor = [i for i in range(len(all_sites)) if all_sites[i] in diversion_large_dor]

        keys_nse = "NSE"
        xs = []
        ys = []
        cases_exps_legends_together = ["not_diverted_small_dor", "not_diverted_large_dor", "diversion_small_dor",
                                       "diversion_large_dor", "CONUS"]

        x1, y1 = ecdf(inds_df[keys_nse].iloc[idx_lst_nodivert_smalldor])
        xs.append(x1)
        ys.append(y1)

        x2, y2 = ecdf(inds_df[keys_nse].iloc[idx_lst_nodivert_largedor])
        xs.append(x2)
        ys.append(y2)

        x3, y3 = ecdf(inds_df[keys_nse].iloc[idx_lst_diversion_smalldor])
        xs.append(x3)
        ys.append(y3)

        x4, y4 = ecdf(inds_df[keys_nse].iloc[idx_lst_diversion_largedor])
        xs.append(x4)
        ys.append(y4)

        x_conus, y_conus = ecdf(inds_df[keys_nse])
        xs.append(x_conus)
        ys.append(y_conus)
        hydro_logger.info("The median NSEs of all five curves (%s) are \n %.2f, %.2f, %.2f, %.2f, %.2f",
                          cases_exps_legends_together, np.median(x1), np.median(x2), np.median(x3), np.median(x4),
                          np.median(x_conus))
        # plot_ecdfs_matplot(xs, ys, cases_exps_legends_together,
        #                    colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "grey"],
        #                    dash_lines=[False, False, False, False, True], x_str="NSE", y_str="CDF")
        # plot using two linestyles and two colors for dor and diversion.
        # plot_ecdfs(xs, ys, cases_exps_legends_together, x_str="NSE", y_str="CDF")
        # define color scheme and line style
        colors = ["#1f77b4", "#d62728"]
        linestyles = ['-', "--"]
        markers = ["", "."]

        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # for i, marker in enumerate(markers):
        for i, linestyle in enumerate(linestyles):
            for j, color in enumerate(colors):
                plt.plot(xs[i * 2 + j], ys[i * 2 + j], color=color, ls=linestyle,  # marker=marker,
                         label=cases_exps_legends_together[i * 2 + j])
        line_i, = axes.plot(x_conus, y_conus, color="grey", label=cases_exps_legends_together[4])
        line_i.set_dashes([2, 2, 10, 2])

        x_str = "NSE"
        y_str = "CDF"
        x_lim = (0, 1)
        y_lim = (0, 1)
        x_interval = 0.1
        y_interval = 0.1
        plt.xlabel(x_str, fontsize=18)
        plt.ylabel(y_str, fontsize=18)
        axes.set_xlim(x_lim[0], x_lim[1])
        axes.set_ylim(y_lim[0], y_lim[1])
        # set x y number font size
        plt.xticks(np.arange(x_lim[0], x_lim[1] + x_lim[1] / 100, x_interval), fontsize=16)
        plt.yticks(np.arange(y_lim[0], y_lim[1] + y_lim[1] / 100, y_interval), fontsize=16)
        plt.grid()
        # Hide the right and top spines
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.legend()
        plt.legend(prop={'size': 16})
        plt.savefig(os.path.join(config_data.data_path["Out"], 'new_dor_divert_comp_matplotlib.png'), dpi=figure_dpi,
                    bbox_inches="tight")
        plt.show()

    def test_diff_dor(self):
        dor_1 = - self.dor
        dor_2 = self.dor
        test_epoch = self.test_epoch
        config_file = self.config_file

        conus_exps = ["basic_exp37"]
        pair1_exps = ["dam_exp1"]
        pair2_exps = ["nodam_exp7"]
        pair3_exps = ["dam_exp27"]
        nodam_exp_lst = ["nodam_exp1"]
        smalldam_exp_lst = ["dam_exp17"]  # -0.003["dam_exp11"] -0.08["dam_exp17"] -1["dam_exp32"]
        largedam_exp_lst = ["dam_exp4"]  # 0.003["dam_exp12"] 0.08["dam_exp18"] 1["dam_exp33"]
        pair1_config_data = load_dataconfig_case_exp(config_file, pair1_exps[0])
        pair2_config_data = load_dataconfig_case_exp(config_file, pair2_exps[0])
        pair3_config_data = load_dataconfig_case_exp(config_file, pair3_exps[0])
        conus_config_data = load_dataconfig_case_exp(config_file, conus_exps[0])

        conus_data_model = GagesModel.load_datamodel(conus_config_data.data_path["Temp"],
                                                     data_source_file_name='test_data_source.txt',
                                                     stat_file_name='test_Statistics.json',
                                                     flow_file_name='test_flow.npy',
                                                     forcing_file_name='test_forcing.npy',
                                                     attr_file_name='test_attr.npy',
                                                     f_dict_file_name='test_dictFactorize.json',
                                                     var_dict_file_name='test_dictAttribute.json',
                                                     t_s_dict_file_name='test_dictTimeSpace.json')
        conus_sites = conus_data_model.t_s_dict["sites_id"]

        source_data_dor1 = GagesSource.choose_some_basins(conus_config_data,
                                                          conus_config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor_1)
        source_data_dor2 = GagesSource.choose_some_basins(conus_config_data,
                                                          conus_config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor_2)
        # basins with dams
        source_data_withdams = GagesSource.choose_some_basins(conus_config_data,
                                                              conus_config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              dam_num=[1, 10000])
        # basins without dams
        source_data_withoutdams = GagesSource.choose_some_basins(conus_config_data,
                                                                 conus_config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False,
                                                                 dam_num=0)

        sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
        sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']

        sites_id_nodam = source_data_withoutdams.all_configs['flow_screen_gage_id']
        sites_id_smalldam = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()
        sites_id_largedam = source_data_dor2.all_configs['flow_screen_gage_id']

        # sites_id_nolargedam = np.sort(np.union1d(np.array(sites_id_nodam), np.array(sites_id_largedam))).tolist()
        # pair1_sites = np.sort(np.intersect1d(np.array(sites_id_dor1), np.array(conus_sites))).tolist()
        # pair2_sites = np.sort(np.intersect1d(np.array(sites_id_nolargedam), np.array(conus_sites))).tolist()
        # pair3_sites = np.sort(np.intersect1d(np.array(sites_id_withdams), np.array(conus_sites))).tolist()

        pair1_data_model = GagesModel.load_datamodel(pair1_config_data.data_path["Temp"],
                                                     data_source_file_name='test_data_source.txt',
                                                     stat_file_name='test_Statistics.json',
                                                     flow_file_name='test_flow.npy',
                                                     forcing_file_name='test_forcing.npy',
                                                     attr_file_name='test_attr.npy',
                                                     f_dict_file_name='test_dictFactorize.json',
                                                     var_dict_file_name='test_dictAttribute.json',
                                                     t_s_dict_file_name='test_dictTimeSpace.json')
        pair1_sites = pair1_data_model.t_s_dict["sites_id"]
        pair2_data_model = GagesModel.load_datamodel(pair2_config_data.data_path["Temp"],
                                                     data_source_file_name='test_data_source.txt',
                                                     stat_file_name='test_Statistics.json',
                                                     flow_file_name='test_flow.npy',
                                                     forcing_file_name='test_forcing.npy',
                                                     attr_file_name='test_attr.npy',
                                                     f_dict_file_name='test_dictFactorize.json',
                                                     var_dict_file_name='test_dictAttribute.json',
                                                     t_s_dict_file_name='test_dictTimeSpace.json')
        pair2_sites = pair2_data_model.t_s_dict["sites_id"]
        pair3_data_model = GagesModel.load_datamodel(pair3_config_data.data_path["Temp"],
                                                     data_source_file_name='test_data_source.txt',
                                                     stat_file_name='test_Statistics.json',
                                                     flow_file_name='test_flow.npy',
                                                     forcing_file_name='test_forcing.npy',
                                                     attr_file_name='test_attr.npy',
                                                     f_dict_file_name='test_dictFactorize.json',
                                                     var_dict_file_name='test_dictAttribute.json',
                                                     t_s_dict_file_name='test_dictTimeSpace.json')
        pair3_sites = pair3_data_model.t_s_dict["sites_id"]

        idx_lst_nodam_in_pair1 = [i for i in range(len(pair1_sites)) if pair1_sites[i] in sites_id_nodam]
        idx_lst_nodam_in_pair2 = [i for i in range(len(pair2_sites)) if pair2_sites[i] in sites_id_nodam]
        idx_lst_nodam_in_pair3 = [i for i in range(len(pair3_sites)) if pair3_sites[i] in sites_id_nodam]
        idx_lst_nodam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_nodam]

        idx_lst_smalldam_in_pair1 = [i for i in range(len(pair1_sites)) if pair1_sites[i] in sites_id_smalldam]
        idx_lst_smalldam_in_pair2 = [i for i in range(len(pair2_sites)) if pair2_sites[i] in sites_id_smalldam]
        idx_lst_smalldam_in_pair3 = [i for i in range(len(pair3_sites)) if pair3_sites[i] in sites_id_smalldam]
        idx_lst_smalldam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_smalldam]

        idx_lst_largedam_in_pair1 = [i for i in range(len(pair1_sites)) if pair1_sites[i] in sites_id_largedam]
        idx_lst_largedam_in_pair2 = [i for i in range(len(pair2_sites)) if pair2_sites[i] in sites_id_largedam]
        idx_lst_largedam_in_pair3 = [i for i in range(len(pair3_sites)) if pair3_sites[i] in sites_id_largedam]
        idx_lst_largedam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_largedam]

        print("multi box")
        inds_df_pair1 = load_ensemble_result(config_file, pair1_exps, test_epoch)
        inds_df_pair2 = load_ensemble_result(config_file, pair2_exps, test_epoch)
        inds_df_pair3 = load_ensemble_result(config_file, pair3_exps, test_epoch)
        inds_df_conus = load_ensemble_result(config_file, conus_exps, test_epoch)

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3)
        keys_nse = "NSE"
        color_chosen = ["Greens", "Blues", "Reds"]
        median_loc = 0.015
        decimal_places = 2
        sns.despine()
        sns.set(font_scale=1.5)

        attr_nodam = "zero_dor"
        cases_exps_legends_nodam = ["LSTM-Z", "LSTM-ZS", "LSTM-ZL", "LSTM-CONUS"]
        frames_nodam = []
        inds_df_nodam = load_ensemble_result(config_file, nodam_exp_lst, test_epoch)
        df_nodam_alone = pd.DataFrame({attr_nodam: np.full([inds_df_nodam.shape[0]], cases_exps_legends_nodam[0]),
                                       keys_nse: inds_df_nodam[keys_nse]})
        frames_nodam.append(df_nodam_alone)

        df_nodam_in_pair1 = pd.DataFrame(
            {attr_nodam: np.full([inds_df_pair1[keys_nse].iloc[idx_lst_nodam_in_pair1].shape[0]],
                                 cases_exps_legends_nodam[1]),
             keys_nse: inds_df_pair1[keys_nse].iloc[idx_lst_nodam_in_pair1]})
        frames_nodam.append(df_nodam_in_pair1)

        df_nodam_in_pair2 = pd.DataFrame(
            {attr_nodam: np.full([inds_df_pair2[keys_nse].iloc[idx_lst_nodam_in_pair2].shape[0]],
                                 cases_exps_legends_nodam[2]),
             keys_nse: inds_df_pair2[keys_nse].iloc[idx_lst_nodam_in_pair2]})
        frames_nodam.append(df_nodam_in_pair2)

        df_nodam_in_conus = pd.DataFrame(
            {attr_nodam: np.full([inds_df_conus[keys_nse].iloc[idx_lst_nodam_in_conus].shape[0]],
                                 cases_exps_legends_nodam[3]),
             keys_nse: inds_df_conus[keys_nse].iloc[idx_lst_nodam_in_conus]})
        frames_nodam.append(df_nodam_in_conus)
        result_nodam = pd.concat(frames_nodam)
        ax1 = plt.subplot(gs[0])
        # ax1.set_title("(a)")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
        ax1.set_ylim([0, 1])
        sns.boxplot(ax=ax1, x=attr_nodam, y=keys_nse, data=result_nodam, showfliers=False, palette=color_chosen[0])
        medians_nodam = result_nodam.groupby([attr_nodam], sort=False)[keys_nse].median().values
        median_labels_nodam = [str(np.round(s, decimal_places)) for s in medians_nodam]
        pos1 = range(len(medians_nodam))
        for tick, label in zip(pos1, ax1.get_xticklabels()):
            ax1.text(pos1[tick], medians_nodam[tick] + median_loc, median_labels_nodam[tick],
                     horizontalalignment='center', size='x-small', weight='semibold')

        attr_smalldam = "small_dor"
        cases_exps_legends_smalldam = ["LSTM-S", "LSTM-ZS", "LSTM-SL", "LSTM-CONUS"]
        frames_smalldam = []
        inds_df_smalldam = load_ensemble_result(config_file, smalldam_exp_lst, test_epoch)
        df_smalldam_alone = pd.DataFrame(
            {attr_smalldam: np.full([inds_df_smalldam.shape[0]], cases_exps_legends_smalldam[0]),
             keys_nse: inds_df_smalldam[keys_nse]})
        frames_smalldam.append(df_smalldam_alone)

        df_smalldam_in_pair1 = pd.DataFrame(
            {attr_smalldam: np.full([inds_df_pair1[keys_nse].iloc[idx_lst_smalldam_in_pair1].shape[0]],
                                    cases_exps_legends_smalldam[1]),
             keys_nse: inds_df_pair1[keys_nse].iloc[idx_lst_smalldam_in_pair1]})
        frames_smalldam.append(df_smalldam_in_pair1)

        df_smalldam_in_pair3 = pd.DataFrame(
            {attr_smalldam: np.full([inds_df_pair3[keys_nse].iloc[idx_lst_smalldam_in_pair3].shape[0]],
                                    cases_exps_legends_smalldam[2]),
             keys_nse: inds_df_pair3[keys_nse].iloc[idx_lst_smalldam_in_pair3]})
        frames_smalldam.append(df_smalldam_in_pair3)

        df_smalldam_in_conus = pd.DataFrame(
            {attr_smalldam: np.full([inds_df_conus[keys_nse].iloc[idx_lst_smalldam_in_conus].shape[0]],
                                    cases_exps_legends_smalldam[3]),
             keys_nse: inds_df_conus[keys_nse].iloc[idx_lst_smalldam_in_conus]})
        frames_smalldam.append(df_smalldam_in_conus)
        result_smalldam = pd.concat(frames_smalldam)
        ax2 = plt.subplot(gs[1])
        # ax2.set_title("(b)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
        ax2.set_ylim([0, 1])
        ax2.set(ylabel=None)
        sns.boxplot(ax=ax2, x=attr_smalldam, y=keys_nse, data=result_smalldam, showfliers=False,
                    palette=color_chosen[1])
        medians_smalldam = result_smalldam.groupby([attr_smalldam], sort=False)[keys_nse].median().values
        median_labels_smalldam = [str(np.round(s, decimal_places)) for s in medians_smalldam]
        pos2 = range(len(medians_smalldam))
        for tick, label in zip(pos2, ax2.get_xticklabels()):
            ax2.text(pos2[tick], medians_smalldam[tick] + median_loc, median_labels_smalldam[tick],
                     horizontalalignment='center', size='x-small', weight='semibold')

        attr_largedam = "large_dor"
        cases_exps_legends_largedam = ["LSTM-L", "LSTM-ZL", "LSTM-SL", "LSTM-CONUS"]
        frames_largedam = []
        inds_df_largedam = load_ensemble_result(config_file, largedam_exp_lst, test_epoch)
        df_largedam_alone = pd.DataFrame(
            {attr_largedam: np.full([inds_df_largedam.shape[0]], cases_exps_legends_largedam[0]),
             keys_nse: inds_df_largedam[keys_nse]})
        frames_largedam.append(df_largedam_alone)

        df_largedam_in_pair2 = pd.DataFrame(
            {attr_largedam: np.full([inds_df_pair2[keys_nse].iloc[idx_lst_largedam_in_pair2].shape[0]],
                                    cases_exps_legends_largedam[1]),
             keys_nse: inds_df_pair2[keys_nse].iloc[idx_lst_largedam_in_pair2]})
        frames_largedam.append(df_largedam_in_pair2)

        df_largedam_in_pair3 = pd.DataFrame(
            {attr_largedam: np.full([inds_df_pair3[keys_nse].iloc[idx_lst_largedam_in_pair3].shape[0]],
                                    cases_exps_legends_largedam[2]),
             keys_nse: inds_df_pair3[keys_nse].iloc[idx_lst_largedam_in_pair3]})
        frames_largedam.append(df_largedam_in_pair3)

        df_largedam_in_conus = pd.DataFrame(
            {attr_largedam: np.full([inds_df_conus[keys_nse].iloc[idx_lst_largedam_in_conus].shape[0]],
                                    cases_exps_legends_largedam[3]),
             keys_nse: inds_df_conus[keys_nse].iloc[idx_lst_largedam_in_conus]})
        frames_largedam.append(df_largedam_in_conus)
        result_largedam = pd.concat(frames_largedam)
        ax3 = plt.subplot(gs[2])
        # ax3.set_title("(c)")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30)
        ax3.set_ylim([0, 1])
        ax3.set(ylabel=None)
        sns.boxplot(ax=ax3, x=attr_largedam, y=keys_nse, data=result_largedam, showfliers=False,
                    palette=color_chosen[2])
        medians_largedam = result_largedam.groupby([attr_largedam], sort=False)[keys_nse].median().values
        median_labels_largedam = [str(np.round(s, decimal_places)) for s in medians_largedam]
        pos3 = range(len(medians_largedam))
        for tick, label in zip(pos3, ax3.get_xticklabels()):
            ax3.text(pos3[tick], medians_largedam[tick] + median_loc, median_labels_largedam[tick],
                     horizontalalignment='center', size='x-small', weight='semibold')
        # sns.despine()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
