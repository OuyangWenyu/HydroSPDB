import copy
import os
import unittest

import torch
import pandas as pd

from data import *
from data.config import cfg, cmd, update_cfg
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from hydroDL.master import *
from sklearn.model_selection import KFold
import numpy as np
from functools import reduce

from utils.hydro_math import random_choice_no_return


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        self.config_file = copy.deepcopy(cfg)
        args = cmd(sub="ecoregion/exp23", cache_state=1, pub_plan=3, plus=1, split_num=2)
        update_cfg(self.config_file, args)
        self.random_seed = self.config_file.RANDOM_SEED
        self.test_epoch = self.config_file.TEST_EPOCH
        self.gpu_num = self.config_file.CTX
        train_mode = self.config_file.TRAIN_MODE
        cache = self.config_file.CACHE.STATE
        self.pub_plan = self.config_file.PUB_PLAN
        self.plus = self.config_file.PLUS
        print("train and test for PUB: \n")
        self.split_num = self.config_file.SPLIT_NUM
        self.eco_names = [("ECO2_CODE", 5.2), ("ECO2_CODE", 5.3), ("ECO2_CODE", 6.2), ("ECO2_CODE", 7.1),
                          ("ECO2_CODE", 8.1), ("ECO2_CODE", 8.2), ("ECO2_CODE", 8.3), ("ECO2_CODE", 8.4),
                          ("ECO2_CODE", 8.5), ("ECO2_CODE", 9.2), ("ECO2_CODE", 9.3), ("ECO2_CODE", 9.4),
                          ("ECO2_CODE", 9.5), ("ECO2_CODE", 9.6), ("ECO2_CODE", 10.1), ("ECO2_CODE", 10.2),
                          ("ECO2_CODE", 10.4), ("ECO2_CODE", 11.1), ("ECO2_CODE", 12.1), ("ECO2_CODE", 13.1)]
        self.config_data = GagesConfig(self.config_file)

    def test_gages_data_model_readquickdata(self):
        pub_plan = self.pub_plan
        config_file = self.config_file
        config_data = self.config_data
        plus = self.plus
        random_seed = self.random_seed
        split_num = self.split_num
        eco_names = self.eco_names
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
        conus_sites_id = data_model_train.t_s_dict["sites_id"]
        if pub_plan == 0:
            """do a pub test like freddy's"""
            camels531_gageid_file = os.path.join(config_data.data_path["DB"], "camels531", "camels531.txt")
            gauge_df = pd.read_csv(camels531_gageid_file, dtype={"GaugeID": str})
            gauge_list = gauge_df["GaugeID"].values
            all_sites_camels_531 = np.sort([str(gauge).zfill(8) for gauge in gauge_list])
            sites_id_train = np.intersect1d(conus_sites_id, all_sites_camels_531)
            # basins not in CAMELS
            sites_id_test = [a_temp_site for a_temp_site in conus_sites_id if a_temp_site not in all_sites_camels_531]
            assert (all(x < y for x, y in zip(sites_id_test, sites_id_test[1:])))
        elif pub_plan == 1 or pub_plan == 4:
            source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              DOR=-0.02)
            # basins with dams
            source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                                  config_data.model_dict["data"]["tRangeTrain"],
                                                                  screen_basin_area_huc4=False,
                                                                  dam_num=[1, 100000])
            # basins without dams
            source_data_withoutdams = GagesSource.choose_some_basins(config_data,
                                                                     config_data.model_dict["data"]["tRangeTrain"],
                                                                     screen_basin_area_huc4=False,
                                                                     dam_num=0)

            sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
            sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']

            sites_id_train = source_data_withoutdams.all_configs['flow_screen_gage_id']
            sites_id_test = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()

        elif pub_plan == 2 or pub_plan == 5:
            source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              DOR=0.02)
            # basins without dams
            source_data_withoutdams = GagesSource.choose_some_basins(config_data,
                                                                     config_data.model_dict["data"]["tRangeTrain"],
                                                                     screen_basin_area_huc4=False,
                                                                     dam_num=0)

            sites_id_train = source_data_withoutdams.all_configs['flow_screen_gage_id']
            sites_id_test = source_data_dor1.all_configs['flow_screen_gage_id']

        elif pub_plan == 3 or pub_plan == 6:
            dor_1 = - 0.02
            dor_2 = 0.02
            source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              DOR=dor_1)
            # basins with dams
            source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                                  config_data.model_dict["data"]["tRangeTrain"],
                                                                  screen_basin_area_huc4=False,
                                                                  dam_num=[1, 100000])
            sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
            sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']

            source_data_dor2 = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              DOR=dor_2)

            sites_id_train = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()
            sites_id_test = source_data_dor2.all_configs['flow_screen_gage_id']

        else:
            print("wrong plan")
            sites_id_train = None
            sites_id_test = None

        # former has a less number than latter
        former_sites_in_conus = np.intersect1d(conus_sites_id, sites_id_train)
        latter_sites_in_conus = np.intersect1d(conus_sites_id, sites_id_test)
        assert len(former_sites_in_conus) <= len(latter_sites_in_conus)

        former_sites_ecoregions = []
        latter_sites_ecoregions = []
        for eco_name in eco_names:
            eco_source_data = GagesSource.choose_some_basins(config_data,
                                                             config_data.model_dict["data"]["tRangeTrain"],
                                                             screen_basin_area_huc4=False, ecoregion=eco_name)
            eco_sites_id = eco_source_data.all_configs['flow_screen_gage_id']
            former_sites_id_inter = np.intersect1d(former_sites_in_conus, eco_sites_id)
            latter_sites_id_inter = np.intersect1d(latter_sites_in_conus, eco_sites_id)
            if len(former_sites_id_inter) > len(latter_sites_id_inter):
                a_chosen_idx = np.random.choice(former_sites_id_inter.size, latter_sites_id_inter.size, replace=False)
                former_chosen = former_sites_id_inter[a_chosen_idx]
                former_sites_ecoregions.append(former_chosen)
                latter_sites_ecoregions.append(latter_sites_id_inter)
            else:
                a_chosen_idx = np.random.choice(latter_sites_id_inter.size, former_sites_id_inter.size, replace=False)
                latter_chosen = latter_sites_id_inter[a_chosen_idx]
                former_sites_ecoregions.append(former_sites_id_inter)
                latter_sites_ecoregions.append(latter_chosen)
        sites_ids_former = np.sort(reduce(lambda x, y: np.hstack((x, y)), former_sites_ecoregions))
        sites_ids_latter = np.sort(reduce(lambda x, y: np.hstack((x, y)), latter_sites_ecoregions))
        if pub_plan in [0, 1, 2, 3]:
            train_sites_in_conus = sites_ids_former
            test_sites_in_conus = sites_ids_latter
        elif pub_plan in [4, 5, 6]:
            train_sites_in_conus = sites_ids_latter
            test_sites_in_conus = sites_ids_former
        else:
            print("wrong plan")
            train_sites_in_conus = None
            test_sites_in_conus = None

        if plus == 0:
            all_index_lst_train_1 = []
            # all sites come from train1 dataset
            sites_lst_train = []
            all_index_lst_test_1 = []
            sites_lst_test_1 = []
            all_index_lst_test_2 = []
            sites_lst_test_2 = []
            np.random.seed(random_seed)
            kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
            eco_name_chosen = []
            for eco_name in eco_names:
                eco_source_data = GagesSource.choose_some_basins(config_data,
                                                                 config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False, ecoregion=eco_name)
                eco_sites_id = eco_source_data.all_configs['flow_screen_gage_id']
                train_sites_id_inter = np.intersect1d(train_sites_in_conus, eco_sites_id)
                test_sites_id_inter = np.intersect1d(test_sites_in_conus, eco_sites_id)
                if train_sites_id_inter.size < split_num or test_sites_id_inter.size < 1:
                    continue
                for train, test in kf.split(train_sites_id_inter):
                    all_index_lst_train_1.append(train)
                    sites_lst_train.append(train_sites_id_inter[train])
                    all_index_lst_test_1.append(test)
                    sites_lst_test_1.append(train_sites_id_inter[test])
                    if test_sites_id_inter.size < test.size:
                        all_index_lst_test_2.append(np.arange(test_sites_id_inter.size))
                        sites_lst_test_2.append(test_sites_id_inter)
                    else:
                        test2_chosen_idx = np.random.choice(test_sites_id_inter.size, test.size, replace=False)
                        all_index_lst_test_2.append(test2_chosen_idx)
                        sites_lst_test_2.append(test_sites_id_inter[test2_chosen_idx])
                eco_name_chosen.append(eco_name)
        elif plus == -1:
            print("camels pub, only do pub on the camels basins")
            all_index_lst_train_1 = []
            # all sites come from train1 dataset
            sites_lst_train = []
            all_index_lst_test_1 = []
            sites_lst_test_1 = []
            np.random.seed(random_seed)
            kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
            eco_name_chosen = []
            for eco_name in eco_names:
                eco_source_data = GagesSource.choose_some_basins(config_data,
                                                                 config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False, ecoregion=eco_name)
                eco_sites_id = eco_source_data.all_configs['flow_screen_gage_id']
                train_sites_id_inter = np.intersect1d(train_sites_in_conus, eco_sites_id)
                if train_sites_id_inter.size < split_num:
                    continue
                for train, test in kf.split(train_sites_id_inter):
                    all_index_lst_train_1.append(train)
                    sites_lst_train.append(train_sites_id_inter[train])
                    all_index_lst_test_1.append(test)
                    sites_lst_test_1.append(train_sites_id_inter[test])
                eco_name_chosen.append(eco_name)
        elif plus == -2:
            print("camels pub, only do pub on the camels basins, same with freddy's split method")
            all_index_lst_train_1 = []
            # all sites come from train1 dataset
            sites_lst_train = []
            all_index_lst_test_1 = []
            sites_lst_test_1 = []
            np.random.seed(random_seed)
            kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)

            for train, test in kf.split(train_sites_in_conus):
                all_index_lst_train_1.append(train)
                sites_lst_train.append(train_sites_in_conus[train])
                all_index_lst_test_1.append(test)
                sites_lst_test_1.append(train_sites_in_conus[test])
        else:
            sites_lst_train = []
            sites_lst_test_1 = []
            sites_lst_test_2 = []

            np.random.seed(random_seed)
            kf = KFold(n_splits=split_num, shuffle=True, random_state=random_seed)
            eco_name_chosen = []
            for eco_name in eco_names:
                eco_source_data = GagesSource.choose_some_basins(config_data,
                                                                 config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False, ecoregion=eco_name)
                eco_sites_id = eco_source_data.all_configs['flow_screen_gage_id']
                sites_id_inter_1 = np.intersect1d(train_sites_in_conus, eco_sites_id)
                sites_id_inter_2 = np.intersect1d(test_sites_in_conus, eco_sites_id)

                if sites_id_inter_1.size <= sites_id_inter_2.size:
                    if sites_id_inter_1.size < split_num:
                        continue
                    for train, test in kf.split(sites_id_inter_1):
                        sites_lst_train_1 = sites_id_inter_1[train]
                        sites_lst_test_1.append(sites_id_inter_1[test])

                        chosen_lst_2 = random_choice_no_return(sites_id_inter_2, [train.size, test.size])
                        sites_lst_train_2 = chosen_lst_2[0]
                        sites_lst_test_2.append(chosen_lst_2[1])

                        sites_lst_train.append(np.sort(np.append(sites_lst_train_1, sites_lst_train_2)))

                else:
                    if sites_id_inter_2.size < split_num:
                        continue
                    for train, test in kf.split(sites_id_inter_2):
                        sites_lst_train_2 = sites_id_inter_2[train]
                        sites_lst_test_2.append(sites_id_inter_2[test])

                        chosen_lst_1 = random_choice_no_return(sites_id_inter_1, [train.size, test.size])
                        sites_lst_train_1 = chosen_lst_1[0]
                        sites_lst_test_1.append(chosen_lst_1[1])

                        sites_lst_train.append(np.sort(np.append(sites_lst_train_1, sites_lst_train_2)))

                eco_name_chosen.append(eco_name)
        for i in range(split_num):
            sites_ids_train_ilst = [sites_lst_train[j] for j in range(len(sites_lst_train)) if
                                    j % split_num == i]
            sites_ids_train_i = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_train_ilst))
            sites_ids_test_ilst_1 = [sites_lst_test_1[j] for j in range(len(sites_lst_test_1)) if
                                     j % split_num == i]
            sites_ids_test_i_1 = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_1))

            if plus >= 0:
                sites_ids_test_ilst_2 = [sites_lst_test_2[j] for j in range(len(sites_lst_test_2)) if
                                         j % split_num == i]
                sites_ids_test_i_2 = np.sort(reduce(lambda x, y: np.hstack((x, y)), sites_ids_test_ilst_2))
            config_data_i = GagesConfig.set_subdir(config_file, str(i))

            gages_model_train_i = GagesModel.update_data_model(config_data_i, data_model_train,
                                                               sites_id_update=sites_ids_train_i,
                                                               data_attr_update=True, screen_basin_area_huc4=False)
            gages_model_test_baseline_i = GagesModel.update_data_model(config_data_i, data_model_test,
                                                                       sites_id_update=sites_ids_train_i,
                                                                       data_attr_update=True,
                                                                       train_stat_dict=gages_model_train_i.stat_dict,
                                                                       screen_basin_area_huc4=False)
            gages_model_test_i_1 = GagesModel.update_data_model(config_data_i, data_model_test,
                                                                sites_id_update=sites_ids_test_i_1,
                                                                data_attr_update=True,
                                                                train_stat_dict=gages_model_train_i.stat_dict,
                                                                screen_basin_area_huc4=False)
            if plus >= 0:
                gages_model_test_i_2 = GagesModel.update_data_model(config_data_i, data_model_test,
                                                                    sites_id_update=sites_ids_test_i_2,
                                                                    data_attr_update=True,
                                                                    train_stat_dict=gages_model_train_i.stat_dict,
                                                                    screen_basin_area_huc4=False)
            save_datamodel(gages_model_train_i, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test_baseline_i, data_source_file_name='test_data_source_base.txt',
                           stat_file_name='test_Statistics_base.json', flow_file_name='test_flow_base',
                           forcing_file_name='test_forcing_base', attr_file_name='test_attr_base',
                           f_dict_file_name='test_dictFactorize_base.json',
                           var_dict_file_name='test_dictAttribute_base.json',
                           t_s_dict_file_name='test_dictTimeSpace_base.json')
            save_datamodel(gages_model_test_i_1, data_source_file_name='test_data_source.txt',
                           stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                           forcing_file_name='test_forcing', attr_file_name='test_attr',
                           f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                           t_s_dict_file_name='test_dictTimeSpace.json')
            if plus >= 0:
                save_datamodel(gages_model_test_i_2, data_source_file_name='test_data_source_2.txt',
                               stat_file_name='test_Statistics_2.json', flow_file_name='test_flow_2',
                               forcing_file_name='test_forcing_2', attr_file_name='test_attr_2',
                               f_dict_file_name='test_dictFactorize_2.json',
                               var_dict_file_name='test_dictAttribute_2.json',
                               t_s_dict_file_name='test_dictTimeSpace_2.json')
            print("save ecoregion " + str(i) + " data model")

    def test_train_gages(self):
        with torch.cuda.device(self.gpu_num):
            for i in range(self.split_num):
                data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                       data_source_file_name='data_source.txt',
                                                       stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                       forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                       f_dict_file_name='dictFactorize.json',
                                                       var_dict_file_name='dictAttribute.json',
                                                       t_s_dict_file_name='dictTimeSpace.json')
                master_train(data_model, random_seed=self.random_seed)

    def test_test_gages(self):
        with torch.cuda.device(self.gpu_num):
            for i in range(self.split_num):
                data_model_baseline = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                                data_source_file_name='test_data_source_base.txt',
                                                                stat_file_name='test_Statistics_base.json',
                                                                flow_file_name='test_flow_base.npy',
                                                                forcing_file_name='test_forcing_base.npy',
                                                                attr_file_name='test_attr_base.npy',
                                                                f_dict_file_name='test_dictFactorize_base.json',
                                                                var_dict_file_name='test_dictAttribute_base.json',
                                                                t_s_dict_file_name='test_dictTimeSpace_base.json')
                data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                       data_source_file_name='test_data_source.txt',
                                                       stat_file_name='test_Statistics.json',
                                                       flow_file_name='test_flow.npy',
                                                       forcing_file_name='test_forcing.npy',
                                                       attr_file_name='test_attr.npy',
                                                       f_dict_file_name='test_dictFactorize.json',
                                                       var_dict_file_name='test_dictAttribute.json',
                                                       t_s_dict_file_name='test_dictTimeSpace.json')
                if self.plus >= 0:
                    data_model_2 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(i),
                                                             data_source_file_name='test_data_source_2.txt',
                                                             stat_file_name='test_Statistics_2.json',
                                                             flow_file_name='test_flow_2.npy',
                                                             forcing_file_name='test_forcing_2.npy',
                                                             attr_file_name='test_attr_2.npy',
                                                             f_dict_file_name='test_dictFactorize_2.json',
                                                             var_dict_file_name='test_dictAttribute_2.json',
                                                             t_s_dict_file_name='test_dictTimeSpace_2.json')
                pred_baseline, obs_baseline = master_test(data_model_baseline, epoch=self.test_epoch,
                                                          save_file_suffix="base")
                basin_area_baseline = data_model_baseline.data_source.read_attr(
                    data_model_baseline.t_s_dict["sites_id"], ['DRAIN_SQKM'], is_return_dict=False)
                mean_prep_baseline = data_model_baseline.data_source.read_attr(data_model_baseline.t_s_dict["sites_id"],
                                                                               ['PPTAVG_BASIN'], is_return_dict=False)
                mean_prep_baseline = mean_prep_baseline / 365 * 10
                pred_baseline = _basin_norm(pred_baseline, basin_area_baseline, mean_prep_baseline, to_norm=False)
                obs_baseline = _basin_norm(obs_baseline, basin_area_baseline, mean_prep_baseline, to_norm=False)
                save_result(data_model_baseline.data_source.data_config.data_path['Temp'], self.test_epoch,
                            pred_baseline, obs_baseline, pred_name='flow_pred_base', obs_name='flow_obs_base')

                pred, obs = master_test(data_model, epoch=self.test_epoch)
                basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                              is_return_dict=False)
                mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                             is_return_dict=False)
                mean_prep = mean_prep / 365 * 10
                pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
                obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
                save_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)
                if self.plus >= 0:
                    pred_2, obs_2 = master_test(data_model_2, epoch=self.test_epoch, save_file_suffix="2")
                    basin_area_2 = data_model_2.data_source.read_attr(data_model_2.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                                      is_return_dict=False)
                    mean_prep_2 = data_model_2.data_source.read_attr(data_model_2.t_s_dict["sites_id"],
                                                                     ['PPTAVG_BASIN'], is_return_dict=False)
                    mean_prep_2 = mean_prep_2 / 365 * 10
                    pred_2 = _basin_norm(pred_2, basin_area_2, mean_prep_2, to_norm=False)
                    obs_2 = _basin_norm(obs_2, basin_area_2, mean_prep_2, to_norm=False)
                    save_result(data_model_2.data_source.data_config.data_path['Temp'], self.test_epoch, pred_2, obs_2,
                                pred_name='flow_pred_2', obs_name='flow_obs_2')


if __name__ == '__main__':
    unittest.main()
