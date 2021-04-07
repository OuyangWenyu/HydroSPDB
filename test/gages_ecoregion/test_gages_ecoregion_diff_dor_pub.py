import os
import unittest
from functools import reduce

import pandas as pd
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import load_dataconfig_case_exp, load_pub_ensemble_result
from explore.stat import statError, ecdf
from matplotlib import pyplot as plt, gridspec
import numpy as np
import seaborn as sns

from utils.hydro_util import hydro_logger
from visual.plot_stat import plot_ecdfs, create_median_labels


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        # plot the results
        self.exp_lst = ["ecoregion_exp9", "ecoregion_exp12"]
        self.train_data_name_lst = ["Train-z", "Train-zs"]
        self.test_data_name_lst = ["Train-z", "PUB-z", "PUB-s"]

        # self.symmetric_exp_lst = ["ecoregion_exp15", "ecoregion_exp16", "ecoregion_exp17"]
        # self.symmetric_train_data_name_lst = ["Train-z", "Train-s", "Train-zs"]
        # self.symmetric_test_data_name_lst = ["Train-z", "Train-s", "PUB-z", "PUB-s"]

        # self.symmetric_exp_lst = ["ecoregion_exp18", "ecoregion_exp19", "ecoregion_exp20"]
        # self.symmetric_train_data_name_lst = ["Train-z", "Train-l", "Train-zl"]
        # self.symmetric_test_data_name_lst = ["Train-z", "Train-l", "PUB-z", "PUB-l"]

        self.symmetric_exp_lst = ["ecoregion_exp21", "ecoregion_exp22", "ecoregion_exp23"]
        self.symmetric_train_data_name_lst = ["Train-s", "Train-l", "Train-sl"]
        self.symmetric_test_data_name_lst = ["Train-s", "Train-l", "PUB-s", "PUB-l"]

        self.train_set = "training"
        self.test_set = "testing"
        self.show_ind_key = "NSE"
        self.test_epoch = 300
        self.split_num = 2

    def test_plot_each_symmetric_exp(self):
        train_set = self.train_set
        test_set = self.test_set
        show_ind_key = self.show_ind_key
        test_epoch = self.test_epoch
        split_num = self.split_num
        exp_lst = self.symmetric_exp_lst
        train_data_name_lst = self.symmetric_train_data_name_lst
        test_data_name_lst = self.symmetric_test_data_name_lst

        colors = "Greens"
        sns.set(font_scale=1)
        fig = plt.figure()
        ax_k = fig.add_axes()
        frames = []
        for j in range(len(exp_lst)):
            config_data = load_dataconfig_case_exp(cfg, exp_lst[j])
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
                                                       forcing_file_name='test_forcing.npy',
                                                       attr_file_name='test_attr.npy',
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
                hydro_logger.info("the size of %s %s Train-base %s", j, i, pred_base.shape[0])
                predsbase.append(pred_base)
                obssbase.append(obs_base)

                pred_i, obs_i = load_result(data_model.data_source.data_config.data_path['Temp'], test_epoch)
                pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
                obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
                hydro_logger.info("the size of %s %s PUB-1 %s", j, i, pred_i.shape[0])
                preds.append(pred_i)
                obss.append(obs_i)

                pred_2, obs_2 = load_result(data_model_2.data_source.data_config.data_path['Temp'],
                                            test_epoch, pred_name='flow_pred_2',
                                            obs_name='flow_obs_2')
                pred_2 = pred_2.reshape(pred_2.shape[0], pred_2.shape[1])
                obs_2 = obs_2.reshape(obs_2.shape[0], obs_2.shape[1])
                hydro_logger.info("the size of %s %s PUB-2 %s", j, i, pred_2.shape[0])
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

            if j == 0 or j == 1:
                df_abase = pd.DataFrame({train_set: np.full([inds_df_abase.shape[0]], train_data_name_lst[j]),
                                         test_set: np.full([inds_df_abase.shape[0]], test_data_name_lst[j]),
                                         show_ind_key: inds_df_abase[show_ind_key]})
                frames.append(df_abase)
            if j == 1:
                df_a = pd.DataFrame({train_set: np.full([inds_df_a.shape[0]], train_data_name_lst[j]),
                                     test_set: np.full([inds_df_a.shape[0]], test_data_name_lst[3]),
                                     show_ind_key: inds_df_a[show_ind_key]})
                df_a2 = pd.DataFrame({train_set: np.full([inds_df_a2.shape[0]], train_data_name_lst[j]),
                                      test_set: np.full([inds_df_a2.shape[0]], test_data_name_lst[2]),
                                      show_ind_key: inds_df_a2[show_ind_key]})
            else:
                df_a = pd.DataFrame({train_set: np.full([inds_df_a.shape[0]], train_data_name_lst[j]),
                                     test_set: np.full([inds_df_a.shape[0]], test_data_name_lst[2]),
                                     show_ind_key: inds_df_a[show_ind_key]})
                df_a2 = pd.DataFrame({train_set: np.full([inds_df_a2.shape[0]], train_data_name_lst[j]),
                                      test_set: np.full([inds_df_a2.shape[0]], test_data_name_lst[3]),
                                      show_ind_key: inds_df_a2[show_ind_key]})
            frames.append(df_a)
            frames.append(df_a2)

        result = pd.concat(frames)
        sns_box = sns.boxplot(ax=ax_k, x=train_set, y=show_ind_key, hue=test_set,  # hue_order=test_data_name_lst,
                              data=result, showfliers=False, palette=colors)  # , width=0.8
        medians = result.groupby([train_set, test_set], sort=False)[show_ind_key].median().values
        hydro_logger.info(medians)
        create_median_labels(sns_box.axes, has_fliers=False)

        sns.despine()
        plt.tight_layout()
        plt.show()
        hydro_logger.debug("plot successfully")

    def test_plot_each_exp(self):
        train_set = "training"
        test_set = "testing"
        show_ind_key = "NSE"
        test_epoch = 300
        split_num = 2
        exp_lst = self.exp_lst
        train_data_name_lst = self.train_data_name_lst
        test_data_name_lst = self.test_data_name_lst

        colors = "Greens"
        sns.set(font_scale=1)
        fig = plt.figure()
        ax_k = fig.add_axes()
        frames = []
        for j in range(len(exp_lst)):
            config_data = load_dataconfig_case_exp(cfg, exp_lst[j])
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
                                                       forcing_file_name='test_forcing.npy',
                                                       attr_file_name='test_attr.npy',
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
                hydro_logger.info("the size of %s %s Train-base %s", j, i, pred_base.shape[0])
                predsbase.append(pred_base)
                obssbase.append(obs_base)

                pred_i, obs_i = load_result(data_model.data_source.data_config.data_path['Temp'], test_epoch)
                pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
                obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
                hydro_logger.info("the size of %s %s PUB-1 %s", j, i, pred_i.shape[0])
                preds.append(pred_i)
                obss.append(obs_i)

                pred_2, obs_2 = load_result(data_model_2.data_source.data_config.data_path['Temp'],
                                            test_epoch, pred_name='flow_pred_2',
                                            obs_name='flow_obs_2')
                pred_2 = pred_2.reshape(pred_2.shape[0], pred_2.shape[1])
                obs_2 = obs_2.reshape(obs_2.shape[0], obs_2.shape[1])
                hydro_logger.info("the size of %s %s PUB-2 %s", j, i, pred_2.shape[0])
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
                df_abase = pd.DataFrame({train_set: np.full([inds_df_abase.shape[0]], train_data_name_lst[j]),
                                         test_set: np.full([inds_df_abase.shape[0]], test_data_name_lst[0]),
                                         # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][1]),
                                         show_ind_key: inds_df_abase[show_ind_key]})
            df_a = pd.DataFrame({train_set: np.full([inds_df_a.shape[0]], train_data_name_lst[j]),
                                 test_set: np.full([inds_df_a.shape[0]], test_data_name_lst[1]),
                                 # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][0])
                                 show_ind_key: inds_df_a[show_ind_key]})
            df_a2 = pd.DataFrame({train_set: np.full([inds_df_a2.shape[0]], train_data_name_lst[j]),
                                  test_set: np.full([inds_df_a2.shape[0]], test_data_name_lst[2]),
                                  # train_data_name_lst[k][j] + "-" + test_data_name_lst[k][1]),
                                  show_ind_key: inds_df_a2[show_ind_key]})
            if j == 0:
                frames.append(df_abase)
            frames.append(df_a)
            frames.append(df_a2)

        result = pd.concat(frames)
        sns_box = sns.boxplot(ax=ax_k, x=train_set, y=show_ind_key, hue=test_set, data=result, showfliers=False,
                              palette=colors)  # , width=0.8
        # plt.subplots_adjust(hspace=0.8)
        # ax_k.set_ylim([-1, 1])
        # ax_k.set_yticks(np.arange(-1, 1, 0.2))
        medians = result.groupby([train_set, test_set], sort=False)[show_ind_key].median().values
        hydro_logger.info(medians)
        create_median_labels(sns_box.axes, has_fliers=False)

        sns.despine()
        plt.tight_layout()
        plt.show()
        # show_config_data = load_dataconfig_case_exp(cfg, exp_lst[0][0])
        # plt.savefig(os.path.join(show_config_data.data_path["Out"], '4exps_pub.png'), dpi=600,
        #             bbox_inches="tight")
        hydro_logger.debug("plot successfully")


if __name__ == '__main__':
    unittest.main()
