import sys

import torch

from data import GagesSource
from data.data_input import GagesModel, _basin_norm, save_result
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp, GagesDamDataModel
from data.nid_input import NidModel, save_nidinput
from explore.gages_stat import split_results_to_regions
from explore.stat import ecdf
from hydroDL import master_test
from utils import unserialize_json, serialize_json
from utils.dataset_format import subset_of_dict
from utils.hydro_math import is_any_elem_in_a_lst
from utils.hydro_util import hydro_logger
from visual.plot_model import plot_we_need, plot_gages_map_and_box, plot_gages_map
from visual.plot_stat import plot_ecdfs, plot_ecdfs_matplot, plot_boxs, plot_diff_boxes

sys.path.append("../..")
import os
from data.config import cfg, update_cfg, cmd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

exp_lst = ["basic_exp37", "basic_exp39", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43"]
exp_attr_lst = ["basic_exp2", "basic_exp3", "basic_exp13", "basic_exp19", "basic_exp20", "basic_exp25"]
gpu_lst = [0, 0, 0, 0, 0, 0]
gpu_attr_lst = [0, 0, 0, 0, 0, 0]
dor_cutoff = 0.1
# exp_lst = ["basic_exp37"]
# exp_attr_lst = ["basic_exp2"]
# gpu_lst = [0]
# gpu_attr_lst = [0]
doLst = list()
# doLst.append('train')
# doLst.append('test')
doLst.append('post')
test_epoch = 300
FIGURE_DPI = 600
# test
if 'test' in doLst:
    for i in range(len(exp_lst)):
        config_data = load_dataconfig_case_exp(cfg, exp_lst[i])
        quick_data_dir = os.path.join(config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
        data_model_train = GagesModel.load_datamodel(data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')
        data_model_test = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')

        # new_data_source = GagesSource(config_data, config_data.model_dict["data"]["tRangeTrain"],
        #                               screen_basin_area_huc4=False)
        # sites_ids = new_data_source.gage_dict["STAID"].tolist()
        # gages_model_train = GagesModel.update_data_model(config_data, data_model_train, sites_id_update=sites_ids,
        #                                                  data_attr_update=True, screen_basin_area_huc4=False)
        # gages_model_test = GagesModel.update_data_model(config_data, data_model_test, sites_id_update=sites_ids,
        #                                                 data_attr_update=True,
        #                                                 train_stat_dict=gages_model_train.stat_dict,
        #                                                 screen_basin_area_huc4=False)

        gages_model_train = GagesModel.update_data_model(config_data, data_model_train, data_attr_update=True,
                                                         screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(config_data, data_model_test, data_attr_update=True,
                                                        train_stat_dict=gages_model_train.stat_dict,
                                                        screen_basin_area_huc4=False)
        with torch.cuda.device(gpu_lst[i]):
            pred, obs = master_test(gages_model_test, epoch=test_epoch)
            basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                                is_return_dict=False)
            mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                               is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_model_test.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)
            plot_we_need(gages_model_test, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

# plot box - latency
if 'post' in doLst:
    config_data = load_dataconfig_case_exp(cfg, exp_lst[0])
    quick_data_dir = os.path.join(config_data.data_path["DB"], "quickdata")
    data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
    data_model_train = GagesModel.load_datamodel(data_dir,
                                                 data_source_file_name='data_source.txt',
                                                 stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                 forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                 f_dict_file_name='dictFactorize.json',
                                                 var_dict_file_name='dictAttribute.json',
                                                 t_s_dict_file_name='dictTimeSpace.json')
    data_model_test = GagesModel.load_datamodel(data_dir,
                                                data_source_file_name='test_data_source.txt',
                                                stat_file_name='test_Statistics.json',
                                                flow_file_name='test_flow.npy',
                                                forcing_file_name='test_forcing.npy',
                                                attr_file_name='test_attr.npy',
                                                f_dict_file_name='test_dictFactorize.json',
                                                var_dict_file_name='test_dictAttribute.json',
                                                t_s_dict_file_name='test_dictTimeSpace.json')
    gages_model_train = GagesModel.update_data_model(config_data, data_model_train, data_attr_update=True,
                                                     screen_basin_area_huc4=False)
    data_model = GagesModel.update_data_model(config_data, data_model_test, data_attr_update=True,
                                              train_stat_dict=gages_model_train.stat_dict,
                                              screen_basin_area_huc4=False)

    inds_df, pred_mean, obs_mean = load_ensemble_result(cfg, exp_lst, test_epoch, return_value=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    ########################### plot diversion dor ecdf  ###########################
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

    dor_1 = - dor_cutoff
    dor_2 = dor_cutoff
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
    plot_ecdfs_matplot(xs, ys, cases_exps_legends_together, colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "grey"],
                       dash_lines=[False, False, False, False, True], x_str="NSE", y_str="CDF")
    plt.savefig(os.path.join(config_data.data_path["Out"], 'dor_divert_comp_matplotlib.png'), dpi=FIGURE_DPI,
                bbox_inches="tight")

    ############################ plot map  ###########################
    show_ind_NSE = 'NSE'
    idx_lst = np.arange(len(data_model.t_s_dict["sites_id"])).tolist()
    # nse_range = [0.5, 1]
    nse_range = [0, 1]
    # nse_range = [-10000, 1]
    # nse_range = [-10000, 0]
    idx_lstl_nse = inds_df[
        (inds_df[show_ind_NSE] >= nse_range[0]) & (inds_df[show_ind_NSE] <= nse_range[1])].index.tolist()
    plot_gages_map(data_model, inds_df, show_ind_NSE, idx_lstl_nse, cbar_font_size=14)

    plt.savefig(os.path.join(config_data.data_path["Out"], 'map_NSE.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    # plot box，使用seaborn库
    keys = ["Bias", "NSE", "FHV", "FLV"]
    inds_test = subset_of_dict(inds_df, keys)
    plot_diff_boxes(inds_test)
    plt.savefig(os.path.join(config_data.data_path["Out"], 'boxes.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    ############################ plot map box   ###########################
    # plot NSE
    nse_range = [0, 1]
    # nse_range = [-10000, 1]
    # nse_range = [-10000, 0]
    idx_lstl_nse = inds_df[
        (inds_df[show_ind_NSE] >= nse_range[0]) & (inds_df[show_ind_NSE] <= nse_range[1])].index.tolist()
    plot_gages_map_and_box(data_model, inds_df, show_ind_NSE, idx_lstl_nse, titles=["NSE map", "NSE boxplot"],
                           wh_ratio=[1, 5], adjust_xy=(0, 0.04))
    plt.savefig(os.path.join(config_data.data_path["Out"], 'map_box_NSE.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    # plot %BiasFLV (the percentage of bias of FDC Low-segment Volume)
    show_ind_FLV = 'FLV'
    percentile_range_FLV = [2, 98]
    plot_gages_map_and_box(data_model, inds_df, show_ind_FLV, pertile_range=percentile_range_FLV,
                           titles=["FLV map", "FLV boxplot"], wh_ratio=[1, 5], adjust_xy=(0, 0.04))
    plt.savefig(os.path.join(config_data.data_path["Out"], 'map_box_FLV.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    # plot %BiasFHV (the percentage of bias of FDC High-segment Volume)
    show_ind_FHV = 'FHV'
    percentile_range_FHV = [2, 98]
    plot_gages_map_and_box(data_model, inds_df, show_ind_FHV, pertile_range=percentile_range_FHV,
                           titles=["FHV map", "FHV boxplot"], wh_ratio=[1, 5], adjust_xy=(0, 0.04))
    plt.savefig(os.path.join(config_data.data_path["Out"], 'map_box_FHV.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    # plot Bias
    show_ind_bias = 'Bias'
    percent_range_bias = [2, 98]
    plot_gages_map_and_box(data_model, inds_df, show_ind_bias, pertile_range=percent_range_bias,
                           titles=["Bias map", "Bias boxplot"], wh_ratio=[1, 5], adjust_xy=(0, 0.04))
    plt.savefig(os.path.join(config_data.data_path["Out"], 'map_box_bias.png'), dpi=FIGURE_DPI, bbox_inches="tight")
    # plt.figure()

    ###################### plot map and box between LSTM with and without anthropogenic attrs####################
    comp_version = 1
    inds_df_attr, pred_mean_attr, obs_mean_attr = load_ensemble_result(cfg, exp_attr_lst, test_epoch, return_value=True)
    if comp_version == 0:
        show_ind_NSE = "NSE"
        attr = "attributes_combination"
        cases_exps_legends = ["attr_comb_1", "attr_comb_2"]
        inds_df_lst = []
        inds_df_lst.append(inds_df)
        inds_df_lst.append(inds_df_attr)
        frames = []
        for i in range(len(cases_exps_legends)):
            df_i = pd.DataFrame({attr: np.full([inds_df_lst[i].shape[0]], cases_exps_legends[i]),
                                 show_ind_NSE: inds_df_lst[i][show_ind_NSE]})
            frames.append(df_i)
        result = pd.concat(frames)

        sns.boxplot(x=attr, y=show_ind_NSE, data=result, showfliers=False)
        # sns.despine(offset=10, trim=True)
    else:
        delta_nse = inds_df - inds_df_attr
        delta_range = [-0.1, 0.1]
        idx_lst_delta = delta_nse[
            (delta_nse[show_ind_NSE] >= delta_range[0]) & (delta_nse[show_ind_NSE] < delta_range[1])].index.tolist()
        fig = plot_gages_map_and_box(data_model, delta_nse, show_ind_NSE, idx_lst=idx_lst_delta,
                                     titles=["NSE delta map", "NSE delta boxplot"], wh_ratio=[1, 5],
                                     adjust_xy=(0, 0.04))
        plt.savefig(os.path.join(config_data.data_path["Out"], 'w-wo-attr_delta_map_box.png'), dpi=FIGURE_DPI,
                    bbox_inches="tight")


    ############################ plot three factors  ###########################
    attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
    usgs_id = data_model.t_s_dict["sites_id"]
    attrs_runavg_stor = data_model.data_source.read_attr(usgs_id, attr_lst, is_return_dict=False)
    run_avg = attrs_runavg_stor[:, 0] * (10 ** (-3)) * (10 ** 6)  # m^3 per year
    nor_storage = attrs_runavg_stor[:, 1] * 1000  # m^3
    dors_value = nor_storage / run_avg
    dors = np.full(len(usgs_id), "dor<" + str(dor_cutoff))
    for i in range(len(usgs_id)):
        if dors_value[i] >= dor_cutoff:
            dors[i] = "dor≥" + str(dor_cutoff)

    diversions = np.full(len(usgs_id), "no ")
    diversion_strs = ["diversion", "divert"]
    attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
    data_attr = data_model.data_source.read_attr_origin(usgs_id, attr_lst)
    diversion_strs_lower = [elem.lower() for elem in diversion_strs]
    data_attr0_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[0]])
    data_attr1_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[1]])
    data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
    for i in range(len(usgs_id)):
        if is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True):
            diversions[i] = "yes"

    nid_gene_file = os.path.join(cfg.NID.NID_DIR, "test", "dam_main_purpose_dict.json")
    if not os.path.isfile(nid_gene_file):
        df = GagesModel.load_datamodel(cfg.CACHE.DATA_DIR,
                                       data_source_file_name='data_source.txt',
                                       stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                       forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                       f_dict_file_name='dictFactorize.json',
                                       var_dict_file_name='dictAttribute.json',
                                       t_s_dict_file_name='dictTimeSpace.json')
        nid_input = NidModel(cfg)
        nid_dir = os.path.join(cfg.NID.NID_DIR, "test")
        save_nidinput(nid_input, nid_dir, nid_source_file_name='nid_source.txt', nid_data_file_name='nid_data.shp')
        data_input = GagesDamDataModel(df, nid_input)
        serialize_json(data_input.gage_main_dam_purpose, os.path.join(nid_dir, "dam_main_purpose_dict.json"))
    gage_main_dam_purpose = unserialize_json(nid_gene_file)
    gage_main_dam_purpose_lst = list(gage_main_dam_purpose.values())
    gage_main_dam_purpose_lst_merge = "".join(gage_main_dam_purpose_lst)
    gage_main_dam_purpose_unique = np.unique(list(gage_main_dam_purpose_lst_merge))
    # gage_main_dam_purpose_unique = np.unique(gage_main_dam_purpose_lst)
    purpose_regions = {}
    for i in range(gage_main_dam_purpose_unique.size):
        sites_id = []
        for key, value in gage_main_dam_purpose.items():
            if gage_main_dam_purpose_unique[i] in value:
                sites_id.append(key)
        assert (all(x < y for x, y in zip(sites_id, sites_id[1:])))
        purpose_regions[gage_main_dam_purpose_unique[i]] = sites_id
    id_regions_idx = []
    id_regions_sites_ids = []
    regions_name = []
    show_min_num = 10
    df_id_region = np.array(data_model.t_s_dict["sites_id"])
    for key, value in purpose_regions.items():
        gages_id = value
        c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
        if c.size < show_min_num:
            continue
        assert (all(x < y for x, y in zip(ind1, ind1[1:])))
        assert (all(x < y for x, y in zip(c, c[1:])))
        id_regions_idx.append(ind1)
        id_regions_sites_ids.append(c)
        regions_name.append(key)
    preds, obss, inds_dfs = split_results_to_regions(data_model, test_epoch, id_regions_idx,
                                                     id_regions_sites_ids)
    frames = []
    x_name = "purposes"
    y_name = "NSE"
    hue_name = "DOR"
    col_name = "diversion"
    for i in range(len(id_regions_idx)):
        # plot box with seaborn
        keys = ["NSE"]
        inds_test = subset_of_dict(inds_dfs[i], keys)
        inds_test = inds_test[keys[0]].values
        df_dict_i = {}
        str_i = regions_name[i]
        df_dict_i[x_name] = np.full([inds_test.size], str_i)
        df_dict_i[y_name] = inds_test
        df_dict_i[hue_name] = dors[id_regions_idx[i]]
        df_dict_i[col_name] = diversions[id_regions_idx[i]]
        # df_dict_i[hue_name] = nor_storage[id_regions_idx[i]]
        df_i = pd.DataFrame(df_dict_i)
        frames.append(df_i)
    result = pd.concat(frames)
    plt.figure()
    plot_boxs(result, x_name, y_name, ylim=[-0.4, 1.0], rotation=0)
    plt.savefig(os.path.join(config_data.data_path["Out"], 'purpose_distribution.png'), dpi=FIGURE_DPI,
                bbox_inches="tight")
    # g = sns.catplot(x=x_name, y=y_name, hue=hue_name, col=col_name,
    #                 data=result, kind="swarm",
    #                 height=4, aspect=.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.set(font="serif", font_scale=1.5, color_codes=True)
    g = sns.catplot(ax=ax, x=x_name, y=y_name,
                    hue=hue_name, col=col_name,
                    data=result, palette="Set1", hue_order=["dor≥" + str(dor_cutoff), "dor<" + str(dor_cutoff)],
                    kind="box", dodge=True, showfliers=False)
    # g.set(ylim=(-1, 1))
    plt.savefig(os.path.join(config_data.data_path["Out"], '3factors_distribution.png'), dpi=FIGURE_DPI,
                bbox_inches="tight")

    output_excel_df = pd.DataFrame(
        {"GAUGE ID": usgs_id, "NSE": round(inds_df["NSE"], 2), "DOR": np.round(dors_value, 2)})
    import csv

    output_excel_df.to_csv(os.path.join(data_model.data_source.all_configs["out_dir"], '3557basins_ID_NSE_DOR.csv'),
                           quoting=csv.QUOTE_NONNUMERIC, index=None)

    ########################### plot new-version diversion dor ecdf  ###########################
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
    # Hide the right and top spines
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.legend()
    plt.legend(prop={'size': 16})
    plt.grid()
    plt.savefig(os.path.join(config_data.data_path["Out"], 'new_dor_divert_comp_matplotlib.png'), dpi=FIGURE_DPI,
                bbox_inches="tight")
