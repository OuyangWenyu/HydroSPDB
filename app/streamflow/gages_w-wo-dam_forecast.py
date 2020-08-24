import sys

from matplotlib import gridspec
import matplotlib.pyplot as plt
from data import GagesSource
from data.data_input import GagesModel
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp
from explore.stat import ecdf
from visual.plot_model import plot_gages_map_and_box, plot_gages_map_and_ts, plot_gages_map_and_scatter
from visual.plot_stat import plot_ecdfs
import numpy as np
import seaborn as sns
import pandas as pd

sys.path.append("../..")
import os
from data.config import cfg, update_cfg, cmd

conus_exps = ["basic_exp37", "basic_exp39", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43"]
pair1_exps = ["dam_exp1", "dam_exp2", "dam_exp3", "dam_exp7", "dam_exp8", "dam_exp9"]
pair2_exps = ["nodam_exp7", "nodam_exp8", "nodam_exp9", "nodam_exp10", "nodam_exp11", "nodam_exp12"]
pair3_exps = ["dam_exp27", "dam_exp26", "dam_exp28", "dam_exp29", "dam_exp30", "dam_exp31"]
nodam_exp_lst = ["nodam_exp1", "nodam_exp2", "nodam_exp3", "nodam_exp4", "nodam_exp5", "nodam_exp6"]
smalldam_exp_lst = ["dam_exp20", "dam_exp21", "dam_exp22", "dam_exp23", "dam_exp24", "dam_exp25"]
largedam_exp_lst = ["dam_exp4", "dam_exp5", "dam_exp6", "dam_exp13", "dam_exp16", "dam_exp19"]

test_epoch = 300

# nodam_config_data = load_dataconfig_case_exp(nodam_exp_lst[0])
# smalldam_config_data = load_dataconfig_case_exp(smalldam_exp_lst[0])
# largedam_config_data = load_dataconfig_case_exp(largedam_exp_lst[0])
pair1_config_data = load_dataconfig_case_exp(cfg, pair1_exps[0])
pair2_config_data = load_dataconfig_case_exp(cfg, pair2_exps[0])
pair3_config_data = load_dataconfig_case_exp(cfg, pair3_exps[0])
conus_config_data = load_dataconfig_case_exp(cfg, conus_exps[0])

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

dor_1 = - 0.02
dor_2 = 0.02
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

compare_item = 0
if compare_item == 0:
    inds_df, pred, obs = load_ensemble_result(cfg, conus_exps, test_epoch, return_value=True)

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

    type_1_index_lst = np.intersect1d(idx_lst_nodam_in_conus, idx_lst_nse).tolist()
    type_2_index_lst = np.intersect1d(idx_lst_smalldam_in_conus, idx_lst_nse).tolist()
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
                               labels=["zero-dor", "small-dor"], scatter_label=[attr_lst[0], show_ind_key], wspace=2,
                               hspace=1.5, legend_y=.8, sub_fig_ratio=[6, 4, 1])
    plt.tight_layout()
    # plt.savefig(os.path.join(conus_config_data.data_path["Out"], 'zero-small-dor_map_comp.png'), dpi=100,
    #             bbox_inches="tight")
    plt.savefig(os.path.join(conus_config_data.data_path["Out"], 'zero-small-dor_western_map_comp.png'), dpi=140,
                bbox_inches="tight")
elif compare_item == 3:
    all_lst = [nodam_exp_lst, smalldam_exp_lst, largedam_exp_lst, pair1_exps, pair2_exps, pair3_exps]
    for an_exp_lst in all_lst:
        an_exp_median_nse_lst = []
        for i in range(len(an_exp_lst)):
            inds_df_an_exp_i = load_ensemble_result(cfg, [an_exp_lst[i]], test_epoch)
            an_exp_i_median = inds_df_an_exp_i.median()
            an_exp_median_nse_lst.append(an_exp_i_median["NSE"])
        print(an_exp_median_nse_lst)
        inds_df_an_exp = load_ensemble_result(cfg, an_exp_lst, test_epoch)
        inds_df_an_exp_median = inds_df_an_exp.median()
        print(inds_df_an_exp_median["NSE"])
elif compare_item == 2:
    print("multi box")
    inds_df_pair1 = load_ensemble_result(cfg, pair1_exps, test_epoch)
    inds_df_pair2 = load_ensemble_result(cfg, pair2_exps, test_epoch)
    inds_df_pair3 = load_ensemble_result(cfg, pair3_exps, test_epoch)
    inds_df_conus = load_ensemble_result(cfg, conus_exps, test_epoch)

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 3)
    keys_nse = "NSE"
    color_chosen = ["Greens", "Blues", "Reds"]
    median_loc = 0.015
    sns.despine()
    sns.set(font_scale=1.5)

    attr_nodam = "zero_dor"
    cases_exps_legends_nodam = ["LSTM-Z", "LSTM-ZS", "LSTM-ZL", "LSTM-CONUS"]
    frames_nodam = []
    inds_df_nodam = load_ensemble_result(cfg, nodam_exp_lst, test_epoch)
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
    ax1.set_title("(a)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    ax1.set_ylim([0, 1])
    sns.boxplot(ax=ax1, x=attr_nodam, y=keys_nse, data=result_nodam, showfliers=False, palette=color_chosen[0])
    medians_nodam = result_nodam.groupby([attr_nodam], sort=False)[keys_nse].median().values
    median_labels_nodam = [str(np.round(s, 3)) for s in medians_nodam]
    pos1 = range(len(medians_nodam))
    for tick, label in zip(pos1, ax1.get_xticklabels()):
        ax1.text(pos1[tick], medians_nodam[tick] + median_loc, median_labels_nodam[tick],
                 horizontalalignment='center', size='x-small', weight='semibold')

    attr_smalldam = "small_dor"
    cases_exps_legends_smalldam = ["LSTM-S", "LSTM-ZS", "LSTM-SL", "LSTM-CONUS"]
    frames_smalldam = []
    inds_df_smalldam = load_ensemble_result(cfg, smalldam_exp_lst, test_epoch)
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
    ax2.set_title("(b)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    ax2.set_ylim([0, 1])
    ax2.set(ylabel=None)
    sns.boxplot(ax=ax2, x=attr_smalldam, y=keys_nse, data=result_smalldam, showfliers=False, palette=color_chosen[1])
    medians_smalldam = result_smalldam.groupby([attr_smalldam], sort=False)[keys_nse].median().values
    median_labels_smalldam = [str(np.round(s, 3)) for s in medians_smalldam]
    pos2 = range(len(medians_smalldam))
    for tick, label in zip(pos2, ax2.get_xticklabels()):
        ax2.text(pos2[tick], medians_smalldam[tick] + median_loc, median_labels_smalldam[tick],
                 horizontalalignment='center', size='x-small', weight='semibold')

    attr_largedam = "large_dor"
    cases_exps_legends_largedam = ["LSTM-L", "LSTM-ZL", "LSTM-SL", "LSTM-CONUS"]
    frames_largedam = []
    inds_df_largedam = load_ensemble_result(cfg, largedam_exp_lst, test_epoch)
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
    ax3.set_title("(c)")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30)
    ax3.set_ylim([0, 1])
    ax3.set(ylabel=None)
    sns.boxplot(ax=ax3, x=attr_largedam, y=keys_nse, data=result_largedam, showfliers=False, palette=color_chosen[2])
    medians_largedam = result_largedam.groupby([attr_largedam], sort=False)[keys_nse].median().values
    median_labels_largedam = [str(np.round(s, 3)) for s in medians_largedam]
    pos3 = range(len(medians_largedam))
    for tick, label in zip(pos3, ax3.get_xticklabels()):
        ax3.text(pos3[tick], medians_largedam[tick] + median_loc, median_labels_largedam[tick],
                 horizontalalignment='center', size='x-small', weight='semibold')
    # sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(conus_config_data.data_path["Out"], '3exps_data_synergy.png'), dpi=300,
                bbox_inches="tight")

elif compare_item == 1:  # ecdf
    print("multi plots")
    inds_df_pair1 = load_ensemble_result(cfg, pair1_exps, test_epoch)
    inds_df_pair2 = load_ensemble_result(cfg, pair2_exps, test_epoch)
    inds_df_pair3 = load_ensemble_result(cfg, pair3_exps, test_epoch)
    inds_df_conus = load_ensemble_result(cfg, conus_exps, test_epoch)

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)
    keys_nse = "NSE"

    xs_nodam = []
    ys_nodam = []
    cases_exps_legends_nodam = ["no_dam_alone", "no_dam_in_pair1", "no_dam_in_pair2", "no_dam_in_conus"]
    inds_df_nodam = load_ensemble_result(cfg, nodam_exp_lst, test_epoch)
    x_nodam_solo, y_nodam_solo = ecdf(inds_df_nodam[keys_nse])
    xs_nodam.append(x_nodam_solo)
    ys_nodam.append(y_nodam_solo)
    x_nodam_pair1, y_nodam_pair1 = ecdf(inds_df_pair1[keys_nse].iloc[idx_lst_nodam_in_pair1])
    xs_nodam.append(x_nodam_pair1)
    ys_nodam.append(y_nodam_pair1)
    x_nodam_pair2, y_nodam_pair2 = ecdf(inds_df_pair2[keys_nse].iloc[idx_lst_nodam_in_pair2])
    xs_nodam.append(x_nodam_pair2)
    ys_nodam.append(y_nodam_pair2)
    x_nodam_conus, y_nodam_conus = ecdf(inds_df_conus[keys_nse].iloc[idx_lst_nodam_in_conus])
    xs_nodam.append(x_nodam_conus)
    ys_nodam.append(y_nodam_conus)
    ax1 = plt.subplot(gs[0])
    plot_ecdfs(xs_nodam, ys_nodam, cases_exps_legends_nodam, x_str="NSE", y_str="CDF", ax_as_subplot=ax1)

    xs_smalldam = []
    ys_smalldam = []
    cases_exps_legends_smalldam = ["small_dam_alone", "small_dam_in_pair1", "small_dam_in_pair3", "small_dam_in_conus"]
    inds_df_smalldam = load_ensemble_result(cfg, smalldam_exp_lst, test_epoch)
    x_smalldam_solo, y_smalldam_solo = ecdf(inds_df_smalldam[keys_nse])
    xs_smalldam.append(x_smalldam_solo)
    ys_smalldam.append(y_smalldam_solo)
    x_smalldam_pair1, y_smalldam_pair1 = ecdf(inds_df_pair1[keys_nse].iloc[idx_lst_smalldam_in_pair1])
    xs_smalldam.append(x_smalldam_pair1)
    ys_smalldam.append(y_smalldam_pair1)
    x_smalldam_pair3, y_smalldam_pair3 = ecdf(inds_df_pair3[keys_nse].iloc[idx_lst_smalldam_in_pair3])
    xs_smalldam.append(x_smalldam_pair3)
    ys_smalldam.append(y_smalldam_pair3)
    x_smalldam_conus, y_smalldam_conus = ecdf(inds_df_conus[keys_nse].iloc[idx_lst_smalldam_in_conus])
    xs_smalldam.append(x_smalldam_conus)
    ys_smalldam.append(y_smalldam_conus)
    ax2 = plt.subplot(gs[1])
    plot_ecdfs(xs_smalldam, ys_smalldam, cases_exps_legends_smalldam, x_str="NSE", y_str="CDF", ax_as_subplot=ax2)

    xs_largedam = []
    ys_largedam = []
    cases_exps_legends_largedam = ["large_dam_alone", "large_dam_in_pair2", "large_dam_in_pair3", "large_dam_in_conus"]
    inds_df_largedam = load_ensemble_result(cfg, largedam_exp_lst, test_epoch)
    x_largedam_solo, y_largedam_solo = ecdf(inds_df_largedam[keys_nse])
    xs_largedam.append(x_largedam_solo)
    ys_largedam.append(y_largedam_solo)
    x_largedam_pair2, y_largedam_pair2 = ecdf(inds_df_pair2[keys_nse].iloc[idx_lst_largedam_in_pair2])
    xs_largedam.append(x_largedam_pair2)
    ys_largedam.append(y_largedam_pair2)
    x_largedam_pair3, y_largedam_pair3 = ecdf(inds_df_pair3[keys_nse].iloc[idx_lst_largedam_in_pair3])
    xs_largedam.append(x_largedam_pair3)
    ys_largedam.append(y_largedam_pair3)
    x_largedam_conus, y_largedam_conus = ecdf(inds_df_conus[keys_nse].iloc[idx_lst_largedam_in_conus])
    xs_largedam.append(x_largedam_conus)
    ys_largedam.append(y_largedam_conus)
    ax3 = plt.subplot(gs[2])
    plot_ecdfs(xs_largedam, ys_largedam, cases_exps_legends_largedam, x_str="NSE", y_str="CDF", ax_as_subplot=ax3)
else:
    print("End")
