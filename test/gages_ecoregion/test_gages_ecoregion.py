import os
import unittest
from functools import reduce

import torch
import pandas as pd

from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import GagesModels, load_dataconfig_case_exp
from explore.stat import statError
from hydroDL.master import *
import definitions
from utils import serialize_numpy, unserialize_numpy
from visual.plot_model import plot_we_need
import numpy as np
from matplotlib import pyplot


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        """before all of these, natural flow model need to be generated by config.ini of gages dataset, and it need
        to be moved to right dir manually """
        config_dir = definitions.CONFIG_DIR
        self.config_file = os.path.join(config_dir, "ecoregion/config_exp1.ini")
        self.subdir = r"ecoregion/exp1"
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

    def test_gages_data_model_readquickdata(self):
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

        for eco_name in self.eco_names:
            source_data = GagesSource.choose_some_basins(self.config_data,
                                                         self.config_data.model_dict["data"]["tRangeTrain"],
                                                         screen_basin_area_huc4=False, ecoregion=eco_name)
            sites_id = source_data.all_configs['flow_screen_gage_id']
            sites_id_inter = np.intersect1d(data_model_train.t_s_dict["sites_id"], sites_id)
            if sites_id_inter.size < 1:
                continue
            subdir = os.path.join(self.subdir, str(eco_name[1]))
            config_data = GagesConfig.set_subdir(self.config_file, subdir)
            gages_model_train = GagesModel.update_data_model(config_data, data_model_train,
                                                             sites_id_update=sites_id,
                                                             data_attr_update=True, screen_basin_area_huc4=False)
            gages_model_test = GagesModel.update_data_model(config_data, data_model_test, sites_id_update=sites_id,
                                                            data_attr_update=True,
                                                            train_stat_dict=gages_model_train.stat_dict,
                                                            screen_basin_area_huc4=False)
            save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                           stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                           attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                           var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
            save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                           stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                           forcing_file_name='test_forcing', attr_file_name='test_attr',
                           f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                           t_s_dict_file_name='test_dictTimeSpace.json')
            print("save ecoregion " + str(eco_name[1]) + " data model")

    def test_train_gages(self):
        with torch.cuda.device(1):
            for i in range(7, len(self.eco_names)):
                data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"], str(self.eco_names[i][1]),
                                                       data_source_file_name='data_source.txt',
                                                       stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                       forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                       f_dict_file_name='dictFactorize.json',
                                                       var_dict_file_name='dictAttribute.json',
                                                       t_s_dict_file_name='dictTimeSpace.json')

                master_train(data_model)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
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
            plot_we_need(data_model, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

    def test_export_result(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        pred, obs = load_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
        inds = statError(obs, pred)
        inds['STAID'] = data_model.t_s_dict["sites_id"]
        inds_df = pd.DataFrame(inds)

        inds_df.to_csv(os.path.join(self.config_data.data_path["Out"], 'data_df.csv'))

    def test_multi_cases_stat_together(self):
        cases_exps = ["basic_exp14", "basic_exp14", "basic_exp2", "basic_exp3", "basic_exp4",
                      "basic_exp5", "basic_exp6", "basic_exp7", "basic_exp8", "basic_exp9"]
        inds_medians = []
        inds_means = []
        pred = []
        obs = []
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], self.test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            pred.append(pred_i)
            obs.append(obs_i)
            inds_i = statError(obs_i, pred_i)
            inds_df_i = pd.DataFrame(inds_i)
            inds_medians.append(inds_df_i.median(axis=0))
            inds_means.append(inds_df_i.mean(axis=0))
        print(pd.DataFrame(inds_medians)["NSE"])
        print(pd.DataFrame(inds_means)["NSE"])
        pred_stack = reduce(lambda a, b: np.vstack((a, b)),
                            list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), pred)))
        obs_stack = reduce(lambda a, b: np.vstack((a, b)),
                           list(map(lambda x: x.reshape(x.shape[0], x.shape[1]), obs)))
        inds = statError(obs_stack, pred_stack)
        inds_df = pd.DataFrame(inds)
        print(inds_df.median(axis=0))
        print(inds_df.mean(axis=0))

    def test_explore_gages_prcp_log(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        i = np.random.randint(data_model.data_forcing.shape[0], size=1)
        print(i)
        a = data_model.data_forcing[i, :, 1].flatten()
        series = a[~np.isnan(a)]
        series = series[np.where(series >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.plot(series)
        pyplot.show()
        # histogram
        pyplot.hist(series)
        pyplot.show()
        # sqrt transform
        transform = np.sqrt(series)
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()
        transform = np.log(series + 1)
        # transform = np.log(series + 0.1)
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()
        # transform = stats.boxcox(series, lmbda=0.0)
        # pyplot.figure(1)
        # # line plot
        # pyplot.subplot(211)
        # pyplot.plot(transform)
        # # histogram
        # pyplot.subplot(212)
        # pyplot.hist(transform)
        # pyplot.show()
        # for j in range(data_model.data_forcing.shape[2]):
        #     x_explore_j = data_model.data_forcing[i, :, j].flatten()
        #     plot_dist(x_explore_j)

    def test_explore_gages_prcp_basin(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                      is_return_dict=False)
        mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                     is_return_dict=False)
        flow = data_model.data_flow
        temparea = np.tile(basin_area, (1, flow.shape[1]))
        tempprep = np.tile(mean_prep / 365 * 10, (1, flow.shape[1]))
        flowua = (flow * 0.0283168 * 3600 * 24) / (
                (temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # unit (m^3/day)/(m^3/day)
        i = np.random.randint(data_model.data_forcing.shape[0], size=1)
        a = flow[i].flatten()
        series = a[~np.isnan(a)]
        # series = series[np.where(series >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(series)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(series)
        pyplot.show()

        b = flowua[i].flatten()
        transform = b[~np.isnan(b)]
        # transform = series[np.where(transform >= 0)]
        # series = series[np.where(series > 0)]
        pyplot.figure(1)
        # line plot
        pyplot.subplot(211)
        pyplot.plot(transform)
        # histogram
        pyplot.subplot(212)
        pyplot.hist(transform)
        pyplot.show()


if __name__ == '__main__':
    unittest.main()
