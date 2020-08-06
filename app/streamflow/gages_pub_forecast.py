import sys

import matplotlib.pyplot as plt
import torch
from functools import reduce

from matplotlib import gridspec

from data import GagesSource
from data.data_input import GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp, load_pub_ensemble_result
from explore.stat import statError
from hydroDL.master.master import master_test_with_pretrained_model

sys.path.append("../..")
import os
from data.config import cfg, update_cfg, cmd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# camels_exp_lst = ["basic_exp31", "basic_exp32", "basic_exp33", "basic_exp34", "basic_exp49", "basic_exp36"]
camels_exp_lst = ["basic_exp31"]
camels_pub_on_diff_dor_exp_lst = ["basic_exp12", "basic_exp14", "basic_exp15"]

# exp_lst = [["ecoregion_exp1", "ecoregion_exp6"], ["ecoregion_exp2", "ecoregion_exp5"],
#            ["ecoregion_exp3", "ecoregion_exp7"], camels_pub_on_diff_dor_exp_lst]
exp_lst = [["ecoregion_exp9", "ecoregion_exp12"], ["ecoregion_exp10", "ecoregion_exp13"],
           ["ecoregion_exp11", "ecoregion_exp14"], camels_pub_on_diff_dor_exp_lst]
# ["ecoregion_exp4", "ecoregion_exp8"],
# train_data_name_lst = [["LSTM-z", "LSTM-zs"], ["LSTM-z", "LSTM-zl"], ["LSTM-s", "LSTM-sl"]]
train_data_name_lst = [["Train-z", "Train-zs"], ["Train-z", "Train-zl"], ["Train-s", "Train-sl"],
                       ["Train-c"]]  # ["Train-c", "Train-cn"]
test_data_name_lst = [["Train-z", "PUB-z", "PUB-s"], ["Train-z", "PUB-z", "PUB-l"], ["Train-s", "PUB-s", "PUB-l"],
                      ["Train-c", "PUB-z", "PUB-s", "PUB-l"]]  # ["Train-c", "PUB-c", "PUB-n"]
# test_epoch = 300
test_epoch = 20
split_num = 2

# test
doLst = list()
# doLst.append('train')
doLst.append('test')
# doLst.append('post')

if 'test' in doLst:
    zerodor_config_data = load_dataconfig_case_exp(cfg, camels_pub_on_diff_dor_exp_lst[0])
    quick_data_dir = os.path.join(zerodor_config_data.data_path["DB"], "quickdata")
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

    camels531_gageid_file = os.path.join(zerodor_config_data.data_path["DB"], "camels531", "camels531.txt")
    gauge_df = pd.read_csv(camels531_gageid_file, dtype={"GaugeID": str})
    gauge_list = gauge_df["GaugeID"].values
    all_sites_camels_531 = np.sort([str(gauge).zfill(8) for gauge in gauge_list])

    # basins without dams
    source_data_withoutdams = GagesSource.choose_some_basins(zerodor_config_data,
                                                             zerodor_config_data.model_dict["data"]["tRangeTrain"],
                                                             screen_basin_area_huc4=False,
                                                             dam_num=0)
    sites_id_zerodor = source_data_withoutdams.all_configs['flow_screen_gage_id']
    sites_zero_dor_not_in_camels = [sites_id_zerodor[i] for i in range(len(sites_id_zerodor)) if
                                    sites_id_zerodor[i] not in all_sites_camels_531]
    gages_model_zerodor_train = GagesModel.update_data_model(zerodor_config_data, data_model_train,
                                                             sites_id_update=sites_zero_dor_not_in_camels,
                                                             data_attr_update=True, screen_basin_area_huc4=False)
    gages_model_zerodor_test = GagesModel.update_data_model(zerodor_config_data, data_model_test,
                                                            sites_id_update=sites_zero_dor_not_in_camels,
                                                            data_attr_update=True,
                                                            train_stat_dict=gages_model_zerodor_train.stat_dict,
                                                            screen_basin_area_huc4=False)

    smalldor_config_data = load_dataconfig_case_exp(cfg, camels_pub_on_diff_dor_exp_lst[1])
    source_data_dor1 = GagesSource.choose_some_basins(smalldor_config_data,
                                                      smalldor_config_data.model_dict["data"]["tRangeTrain"],
                                                      screen_basin_area_huc4=False,
                                                      DOR=-0.02)
    # basins with dams
    source_data_withdams = GagesSource.choose_some_basins(smalldor_config_data,
                                                          smalldor_config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          dam_num=[1, 100000])
    sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
    sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']
    sites_id_smalldor = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()
    sites_small_dor_not_in_camels = [sites_id_smalldor[i] for i in range(len(sites_id_smalldor)) if
                                     sites_id_smalldor[i] not in all_sites_camels_531]
    gages_model_smalldor_train = GagesModel.update_data_model(smalldor_config_data, data_model_train,
                                                              sites_id_update=sites_small_dor_not_in_camels,
                                                              data_attr_update=True, screen_basin_area_huc4=False)
    gages_model_smalldor_test = GagesModel.update_data_model(smalldor_config_data, data_model_test,
                                                             sites_id_update=sites_small_dor_not_in_camels,
                                                             data_attr_update=True,
                                                             train_stat_dict=gages_model_smalldor_train.stat_dict,
                                                             screen_basin_area_huc4=False)

    largedor_config_data = load_dataconfig_case_exp(cfg, camels_pub_on_diff_dor_exp_lst[2])
    source_data_large_dor = GagesSource.choose_some_basins(largedor_config_data,
                                                           largedor_config_data.model_dict["data"]["tRangeTrain"],
                                                           screen_basin_area_huc4=False,
                                                           DOR=0.02)
    sites_id_largedor = source_data_large_dor.all_configs['flow_screen_gage_id']
    sites_large_dor_not_in_camels = [sites_id_largedor[i] for i in range(len(sites_id_largedor)) if
                                     sites_id_largedor[i] not in all_sites_camels_531]
    gages_model_largedor_train = GagesModel.update_data_model(largedor_config_data, data_model_train,
                                                              sites_id_update=sites_large_dor_not_in_camels,
                                                              data_attr_update=True, screen_basin_area_huc4=False)
    gages_model_largedor_test = GagesModel.update_data_model(largedor_config_data, data_model_test,
                                                             sites_id_update=sites_large_dor_not_in_camels,
                                                             data_attr_update=True,
                                                             train_stat_dict=gages_model_largedor_train.stat_dict,
                                                             screen_basin_area_huc4=False)

    gages_models_test = [gages_model_zerodor_test, gages_model_smalldor_test, gages_model_largedor_test]
    for gages_model_test in gages_models_test:
        for i in range(len(camels_exp_lst)):
            ref_config_data = load_dataconfig_case_exp(cfg, camels_exp_lst[i])
            with torch.cuda.device(0):
                pretrained_model_file = os.path.join(ref_config_data.data_path["Out"],
                                                     "model_Ep" + str(test_epoch) + ".pt")
                pretrained_model_name = camels_exp_lst[i] + "_pretrained_model"
                pred, obs = master_test_with_pretrained_model(gages_model_test, pretrained_model_file,
                                                              pretrained_model_name)
                basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"],
                                                                    ['DRAIN_SQKM'],
                                                                    is_return_dict=False)
                mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"],
                                                                   ['PPTAVG_BASIN'],
                                                                   is_return_dict=False)
                mean_prep = mean_prep / 365 * 10
                pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
                obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
                save_dir = os.path.join(gages_model_test.data_source.data_config.data_path['Out'],
                                        pretrained_model_name)
                save_result(save_dir, test_epoch, pred, obs)

train_set = "training"
test_set = "testing"
show_ind_key = "NSE"

# fig = plt.figure(figsize=(12, 4))
# gs = gridspec.GridSpec(1, 11)
fig = plt.figure(figsize=(8, 9))
gs = gridspec.GridSpec(2, 2)
titles = ["(a)", "(b)", "(c)", "(d)"]

colors = ["Greens", "Blues", "Reds", "Greys"]
sns.set(font_scale=1.2)

for k in range(len(exp_lst)):
    if k == len(exp_lst) - 1:
        print("camels pub")
        # ax_k = plt.subplot(gs[k * 3: k * 3 + 2])
        ax_k = plt.subplot(gs[1, 1])
        ax_k.set_title(titles[k])
        frames_camels_pub = []
        inds_df_camels, pred_mean, obs_mean = load_ensemble_result(cfg, camels_exp_lst, test_epoch, return_value=True)
        df_camels_pub = pd.DataFrame({train_set: np.full([inds_df_camels.shape[0]], train_data_name_lst[k][0]),
                                      test_set: np.full([inds_df_camels.shape[0]], test_data_name_lst[k][0]),
                                      show_ind_key: inds_df_camels[show_ind_key]})
        frames_camels_pub.append(df_camels_pub)
        for j in range(len(exp_lst[k])):
            inds_df_camels, pred_mean, obs_mean = load_pub_ensemble_result(cfg, exp_lst[k][j], camels_exp_lst,
                                                                           test_epoch,
                                                                           return_value=True)
            df_camels_pub = pd.DataFrame({train_set: np.full([inds_df_camels.shape[0]], train_data_name_lst[k][0]),
                                          test_set: np.full([inds_df_camels.shape[0]], test_data_name_lst[k][j + 1]),
                                          show_ind_key: inds_df_camels[show_ind_key]})
            frames_camels_pub.append(df_camels_pub)

        result_camels_pub = pd.concat(frames_camels_pub)
        ax_k.set_ylim([-1, 1])
        ax_k.set_yticks(np.arange(-1, 1, 0.2))
        sns_box = sns.boxplot(ax=ax_k, x=train_set, y=show_ind_key, hue=test_set, data=result_camels_pub,
                              showfliers=False, palette=colors[k])
        # plt.subplots_adjust(hspace=0.8)
        # sns.boxplot(ax=ax_k, x=test_set, y=show_ind_key, data=result_camels_pub, showfliers=False, palette=colors[k])
        continue
    # ax_k = plt.subplot(gs[k * 3:(k + 1) * 3])
    ax_k = plt.subplot(gs[int(k / 2), k % 2])
    ax_k.set_title(titles[k])
    frames = []
    for j in range(len(exp_lst[k])):
        config_data = load_dataconfig_case_exp(cfg, exp_lst[k][j])
        preds = []
        obss = []
        preds2 = []
        obss2 = []
        predsbase = []
        obssbase = []
        for i in range(split_num):
            data_model_base = GagesModel.load_datamodel(config_data.data_path["Temp"], str(i),
                                                        data_source_file_name='test_data_source_base.txt',
                                                        stat_file_name='test_Statistics_base.json',
                                                        flow_file_name='test_flow_base.npy',
                                                        forcing_file_name='test_forcing_base.npy',
                                                        attr_file_name='test_attr_base.npy',
                                                        f_dict_file_name='test_dictFactorize_base.json',
                                                        var_dict_file_name='test_dictAttribute_base.json',
                                                        t_s_dict_file_name='test_dictTimeSpace_base.json')
            data_model = GagesModel.load_datamodel(config_data.data_path["Temp"], str(i),
                                                   data_source_file_name='test_data_source.txt',
                                                   stat_file_name='test_Statistics.json',
                                                   flow_file_name='test_flow.npy',
                                                   forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                   f_dict_file_name='test_dictFactorize.json',
                                                   var_dict_file_name='test_dictAttribute.json',
                                                   t_s_dict_file_name='test_dictTimeSpace.json')
            data_model_2 = GagesModel.load_datamodel(config_data.data_path["Temp"], str(i),
                                                     data_source_file_name='test_data_source_2.txt',
                                                     stat_file_name='test_Statistics_2.json',
                                                     flow_file_name='test_flow_2.npy',
                                                     forcing_file_name='test_forcing_2.npy',
                                                     attr_file_name='test_attr_2.npy',
                                                     f_dict_file_name='test_dictFactorize_2.json',
                                                     var_dict_file_name='test_dictAttribute_2.json',
                                                     t_s_dict_file_name='test_dictTimeSpace_2.json')
            pred_base, obs_base = load_result(data_model_base.data_source.data_config.data_path['Temp'],
                                              test_epoch, pred_name='flow_pred_base',
                                              obs_name='flow_obs_base')
            pred_base = pred_base.reshape(pred_base.shape[0], pred_base.shape[1])
            obs_base = obs_base.reshape(obs_base.shape[0], obs_base.shape[1])
            print("the size of", str(k), str(j), str(i), "Train-base", str(pred_base.shape[0]))
            predsbase.append(pred_base)
            obssbase.append(obs_base)

            pred_i, obs_i = load_result(data_model.data_source.data_config.data_path['Temp'], test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            print("the size of", str(k), str(j), str(i), "PUB-1", str(pred_i.shape[0]))
            preds.append(pred_i)
            obss.append(obs_i)

            pred_2, obs_2 = load_result(data_model_2.data_source.data_config.data_path['Temp'],
                                        test_epoch, pred_name='flow_pred_2',
                                        obs_name='flow_obs_2')
            pred_2 = pred_2.reshape(pred_2.shape[0], pred_2.shape[1])
            obs_2 = obs_2.reshape(obs_2.shape[0], obs_2.shape[1])
            print("the size of", str(k), str(j), str(i), "PUB-2", str(pred_2.shape[0]))
            preds2.append(pred_2)
            obss2.append(obs_2)

        predsbase_np = reduce(lambda a, b: np.vstack((a, b)), predsbase)
        obssbase_np = reduce(lambda a, b: np.vstack((a, b)), obssbase)
        indsbase = statError(obssbase_np, predsbase_np)
        inds_df_abase = pd.DataFrame(indsbase)

        preds_np = reduce(lambda a, b: np.vstack((a, b)), preds)
        obss_np = reduce(lambda a, b: np.vstack((a, b)), obss)
        inds = statError(obss_np, preds_np)
        inds_df_a = pd.DataFrame(inds)

        preds2_np = reduce(lambda a, b: np.vstack((a, b)), preds2)
        obss2_np = reduce(lambda a, b: np.vstack((a, b)), obss2)
        inds2 = statError(obss2_np, preds2_np)
        inds_df_a2 = pd.DataFrame(inds2)

        if j == 0:
            df_abase = pd.DataFrame({train_set: np.full([inds_df_abase.shape[0]], train_data_name_lst[k][j]),
                                     test_set: np.full([inds_df_abase.shape[0]], test_data_name_lst[k][0]),
                                     # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][1]),
                                     show_ind_key: inds_df_abase[show_ind_key]})
        df_a = pd.DataFrame({train_set: np.full([inds_df_a.shape[0]], train_data_name_lst[k][j]),
                             test_set: np.full([inds_df_a.shape[0]], test_data_name_lst[k][1]),
                             # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][0])
                             show_ind_key: inds_df_a[show_ind_key]})
        df_a2 = pd.DataFrame({train_set: np.full([inds_df_a2.shape[0]], train_data_name_lst[k][j]),
                              test_set: np.full([inds_df_a2.shape[0]], test_data_name_lst[k][2]),
                              # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][1]),
                              show_ind_key: inds_df_a2[show_ind_key]})
        if j == 0:
            frames.append(df_abase)
        frames.append(df_a)
        frames.append(df_a2)
        # keys_nse = "NSE"
        # xs = []
        # ys = []
        # cases_exps_legends_together = ["PUB_test_in_basins_1", "PUB_test_in_basins_2"]
        #
        # x1, y1 = ecdf(inds_df_a[keys_nse])
        # xs.append(x1)
        # ys.append(y1)
        #
        # x2, y2 = ecdf(inds_df_a2[keys_nse])
        # xs.append(x2)
        # ys.append(y2)
        #
        # plot_ecdfs(xs, ys, cases_exps_legends_together)

    result = pd.concat(frames)
    sns_box = sns.boxplot(ax=ax_k, x=train_set, y=show_ind_key, hue=test_set, data=result, showfliers=False,
                          palette=colors[k])  # , width=0.8
    # plt.subplots_adjust(hspace=0.8)
    ax_k.set_ylim([-1, 1])
    ax_k.set_yticks(np.arange(-1, 1, 0.2))
    medians = result.groupby([train_set, test_set], sort=False)[show_ind_key].median().values
    print(medians)
    # median_labels = [str(np.round(s, 3)) for s in medians]
    # pos = range(len(medians))
    # for tick, label in zip(pos, ax_k.get_xticklabels()):
    #     ax_k.text(pos[tick], medians[tick] + 0.02, median_labels[tick],
    #               horizontalalignment='center', size='x-small', weight='semibold')

sns.despine()
plt.tight_layout()
show_config_data = load_dataconfig_case_exp(cfg, exp_lst[0][0])
plt.savefig(os.path.join(show_config_data.data_path["Out"], '4exps_pub.png'), dpi=300, bbox_inches="tight")
