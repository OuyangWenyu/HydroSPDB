import os
import unittest
from functools import reduce

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import GagesModels, load_dataconfig_case_exp
from explore.stat import statError, ecdf
from hydroDL.master import *
import definitions
from utils import serialize_numpy, unserialize_numpy
from utils.hydro_math import random_choice_no_return
from visual.plot_model import plot_we_need
import numpy as np
from matplotlib import pyplot

from visual.plot_stat import plot_ecdfs


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = definitions.CONFIG_DIR
        self.config_file = os.path.join(config_dir, "ecoregion/config_exp3.ini")
        self.subdir = r"ecoregion/exp3"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        test_epoch_lst = [100, 150, 200, 220, 250, 280, 290, 300, 310, 320, 350]
        # self.test_epoch = test_epoch_lst[0]
        # self.test_epoch = test_epoch_lst[1]
        # self.test_epoch = test_epoch_lst[2]
        # self.test_epoch = test_epoch_lst[3]
        # self.test_epoch = test_epoch_lst[4]
        # self.test_epoch = test_epoch_lst[5]
        # self.test_epoch = test_epoch_lst[6]
        self.test_epoch = test_epoch_lst[7]
        # self.test_epoch = test_epoch_lst[8]
        # self.test_epoch = test_epoch_lst[9]
        # self.test_epoch = test_epoch_lst[10]
        self.eco_names = [("ECO2_CODE", 5.2), ("ECO2_CODE", 5.3), ("ECO2_CODE", 6.2), ("ECO2_CODE", 7.1),
                          ("ECO2_CODE", 8.1), ("ECO2_CODE", 8.2), ("ECO2_CODE", 8.3), ("ECO2_CODE", 8.4),
                          ("ECO2_CODE", 8.5), ("ECO2_CODE", 9.2), ("ECO2_CODE", 9.3), ("ECO2_CODE", 9.4),
                          ("ECO2_CODE", 9.5), ("ECO2_CODE", 9.6), ("ECO2_CODE", 10.1), ("ECO2_CODE", 10.2),
                          ("ECO2_CODE", 10.4), ("ECO2_CODE", 11.1), ("ECO2_CODE", 12.1), ("ECO2_CODE", 13.1)]
        self.split_num = 3

    def test_split_nomajordam_ecoregion(self):
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        # data_dir = os.path.join(quick_data_dir, "conus-all_85-05_nan-0.1_00-1.0")
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
        conus_sites_id = data_model_train.t_s_dict["sites_id"]
        nomajordam_source_data = GagesSource.choose_some_basins(self.config_data,
                                                                self.config_data.model_dict["data"]["tRangeTrain"],
                                                                screen_basin_area_huc4=False, major_dam_num=0)
        nomajordam_sites_id = nomajordam_source_data.all_configs['flow_screen_gage_id']
        nomajordam_in_conus = np.intersect1d(conus_sites_id, nomajordam_sites_id)
        majordam_source_data = GagesSource.choose_some_basins(self.config_data,
                                                              self.config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False, major_dam_num=[1, 2000])
        majordam_sites_id = majordam_source_data.all_configs['flow_screen_gage_id']
        majordam_in_conus = np.intersect1d(conus_sites_id, majordam_sites_id)

        sites_lst_train = []
        sites_lst_test_nomajordam = []
        sites_lst_test_majordam = []

        random_seed = 1
        np.random.seed(random_seed)
        kf = KFold(n_splits=self.split_num, shuffle=True, random_state=random_seed)
        eco_name_chosen = []
        for eco_name in self.eco_names:
            eco_source_data = GagesSource.choose_some_basins(self.config_data,
                                                             self.config_data.model_dict["data"]["tRangeTrain"],
                                                             screen_basin_area_huc4=False, ecoregion=eco_name)
            eco_sites_id = eco_source_data.all_configs['flow_screen_gage_id']
            nomajordam_sites_id_inter = np.intersect1d(nomajordam_in_conus, eco_sites_id)
            majordam_sites_id_inter = np.intersect1d(majordam_in_conus, eco_sites_id)

            if nomajordam_sites_id_inter.size < majordam_sites_id_inter.size:
                if nomajordam_sites_id_inter.size < self.split_num:
                    continue
                for train, test in kf.split(nomajordam_sites_id_inter):
                    sites_lst_train_nomajordam = nomajordam_sites_id_inter[train]
                    sites_lst_test_nomajordam.append(nomajordam_sites_id_inter[test])

                    majordam_chosen_lst = random_choice_no_return(majordam_sites_id_inter, [train.size, test.size])
                    sites_lst_train_majordam = majordam_chosen_lst[0]
                    sites_lst_test_majordam.append(majordam_chosen_lst[1])

                    sites_lst_train.append(np.sort(np.append(sites_lst_train_nomajordam, sites_lst_train_majordam)))

            else:
                if majordam_sites_id_inter.size < self.split_num:
                    continue
                for train, test in kf.split(majordam_sites_id_inter):
                    sites_lst_train_majordam = majordam_sites_id_inter[train]
                    sites_lst_test_majordam.append(majordam_sites_id_inter[test])

                    nomajordam_chosen_lst = random_choice_no_return(nomajordam_sites_id_inter,
                                                                        [train.size, test.size])
                    sites_lst_train_nomajordam = nomajordam_chosen_lst[0]
                    sites_lst_test_nomajordam.append(nomajordam_chosen_lst[1])

                    sites_lst_train.append(np.sort(np.append(sites_lst_train_nomajordam, sites_lst_train_majordam)))

            eco_name_chosen.append(eco_name)
        for i in range(self.split_num):
            sites_ids_train_ilst = [sites_lst_train[j] for j in range(len(sites_lst_train)) if j % self.split_num == i]
            sites_ids_train_i = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst))
            sites_ids_test_ilst = [sites_lst_test_nomajordam[j] for j in range(len(sites_lst_test_nomajordam)) if
                                   j % self.split_num == i]
            sites_ids_test_i = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst))
            sites_ids_test_majordam_ilst = [sites_lst_test_majordam[j] for j in range(len(sites_lst_test_majordam)) if
                                            j % self.split_num == i]
            sites_ids_test_majordam_i = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_majordam_ilst))
            subdir_i = os.path.join(self.subdir, str(i))
            config_data_i = GagesConfig.set_subdir(self.config_file, subdir_i)
            gages_model_train_i = GagesModel.update_data_model(config_data_i, data_model_train,
                                                               sites_id_update=sites_ids_train_i,
                                                               data_attr_update=True, screen_basin_area_huc4=False)
            gages_model_test_i = GagesModel.update_data_model(config_data_i, data_model_test,
                                                              sites_id_update=sites_ids_test_i,
                                                              data_attr_update=True,
                                                              train_stat_dict=gages_model_train_i.stat_dict,
                                                              screen_basin_area_huc4=False)
            gages_model_test_majordam_i = GagesModel.update_data_model(config_data_i, data_model_test,
                                                                       sites_id_update=sites_ids_test_majordam_i,
                                                                       data_attr_update=True,
                                                                       train_stat_dict=gages_model_train_i.stat_dict,
                                                                       screen_basin_area_huc4=False)
            save_datamodel(gages_model_train_i, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test_i, data_source_file_name='test_data_source.txt',
                           stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                           forcing_file_name='test_forcing', attr_file_name='test_attr',
                           f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                           t_s_dict_file_name='test_dictTimeSpace.json')
            save_datamodel(gages_model_test_majordam_i, data_source_file_name='test_data_source_majordam.txt',
                           stat_file_name='test_Statistics_majordam.json', flow_file_name='test_flow_majordam',
                           forcing_file_name='test_forcing_majordam', attr_file_name='test_attr_majordam',
                           f_dict_file_name='test_dictFactorize_majordam.json',
                           var_dict_file_name='test_dictAttribute_majordam.json',
                           t_s_dict_file_name='test_dictTimeSpace_majordam.json')
            print("save ecoregion " + str(i) + " data model")

    def test_train_pub_ecoregion(self):
        with torch.cuda.device(1):
            for i in range(0, self.split_num):
                data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                       data_source_file_name='data_source.txt',
                                                       stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                       forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                       f_dict_file_name='dictFactorize.json',
                                                       var_dict_file_name='dictAttribute.json',
                                                       t_s_dict_file_name='dictTimeSpace.json')
                master_train(data_model)

    def test_test_pub_ecoregion(self):
        for i in range(self.split_num):
            data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                   data_source_file_name='test_data_source.txt',
                                                   stat_file_name='test_Statistics.json',
                                                   flow_file_name='test_flow.npy',
                                                   forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                   f_dict_file_name='test_dictFactorize.json',
                                                   var_dict_file_name='test_dictAttribute.json',
                                                   t_s_dict_file_name='test_dictTimeSpace.json')
            data_model_majordam = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                            data_source_file_name='test_data_source_majordam.txt',
                                                            stat_file_name='test_Statistics_majordam.json',
                                                            flow_file_name='test_flow_majordam.npy',
                                                            forcing_file_name='test_forcing_majordam.npy',
                                                            attr_file_name='test_attr_majordam.npy',
                                                            f_dict_file_name='test_dictFactorize_majordam.json',
                                                            var_dict_file_name='test_dictAttribute_majordam.json',
                                                            t_s_dict_file_name='test_dictTimeSpace_majordam.json')
            with torch.cuda.device(1):
                pred, obs = master_test(data_model, epoch=self.test_epoch)
                basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                              is_return_dict=False)
                mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                             is_return_dict=False)
                mean_prep = mean_prep / 365 * 10
                pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
                obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
                save_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)

                pred_majordam, obs_majordam = master_test(data_model_majordam, epoch=self.test_epoch,
                                                          save_file_suffix="majordam")
                basin_area_majordam = data_model_majordam.data_source.read_attr(
                    data_model_majordam.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
                mean_prep_majordam = data_model_majordam.data_source.read_attr(data_model_majordam.t_s_dict["sites_id"],
                                                                               ['PPTAVG_BASIN'],
                                                                               is_return_dict=False)
                mean_prep_majordam = mean_prep_majordam / 365 * 10
                pred_majordam = _basin_norm(pred_majordam, basin_area_majordam, mean_prep_majordam, to_norm=False)
                obs_majordam = _basin_norm(obs_majordam, basin_area_majordam, mean_prep_majordam, to_norm=False)
                save_result(data_model_majordam.data_source.data_config.data_path['Temp'], self.test_epoch,
                            pred_majordam, obs_majordam, pred_name='flow_pred_majordam', obs_name='flow_obs_majordam')

    def test_comp_result(self):
        for i in range(self.split_num):
            data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                   data_source_file_name='test_data_source.txt',
                                                   stat_file_name='test_Statistics.json',
                                                   flow_file_name='test_flow.npy',
                                                   forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                   f_dict_file_name='test_dictFactorize.json',
                                                   var_dict_file_name='test_dictAttribute.json',
                                                   t_s_dict_file_name='test_dictTimeSpace.json')
            data_model_majordam = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                            data_source_file_name='test_data_source_majordam.txt',
                                                            stat_file_name='test_Statistics_majordam.json',
                                                            flow_file_name='test_flow_majordam.npy',
                                                            forcing_file_name='test_forcing_majordam.npy',
                                                            attr_file_name='test_attr_majordam.npy',
                                                            f_dict_file_name='test_dictFactorize_majordam.json',
                                                            var_dict_file_name='test_dictAttribute_majordam.json',
                                                            t_s_dict_file_name='test_dictTimeSpace_majordam.json')
            pred, obs = load_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
            pred = pred.reshape(pred.shape[0], pred.shape[1])
            obs = obs.reshape(obs.shape[0], obs.shape[1])
            inds = statError(obs, pred)
            inds['STAID'] = data_model.t_s_dict["sites_id"]
            inds_df = pd.DataFrame(inds)

            pred_majordam, obs_majordam = load_result(data_model_majordam.data_source.data_config.data_path['Temp'],
                                                      self.test_epoch, pred_name='flow_pred_majordam',
                                                      obs_name='flow_obs_majordam')
            pred_majordam = pred_majordam.reshape(pred_majordam.shape[0], pred_majordam.shape[1])
            obs_majordam = obs_majordam.reshape(obs_majordam.shape[0], obs_majordam.shape[1])
            inds_majordam = statError(obs_majordam, pred_majordam)
            inds_majordam['STAID'] = data_model_majordam.t_s_dict["sites_id"]
            inds_majordam_df = pd.DataFrame(inds_majordam)

            keys_nse = "NSE"
            xs = []
            ys = []
            cases_exps_legends_together = ["PUB_test_in_no-major-dam_basins", "PUB_test_in_major-dam_basins"]

            x1, y1 = ecdf(inds_df[keys_nse])
            xs.append(x1)
            ys.append(y1)

            x2, y2 = ecdf(inds_majordam_df[keys_nse])
            xs.append(x2)
            ys.append(y2)

            plot_ecdfs(xs, ys, cases_exps_legends_together)


if __name__ == '__main__':
    unittest.main()
