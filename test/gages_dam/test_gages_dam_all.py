import unittest

import torch

import definitions
from data import GagesConfig, GagesSource, DataModel
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import GagesDamDataModel, GagesModels, choose_which_purpose
from data.nid_input import NidModel, save_nidinput
from explore.gages_stat import split_results_to_regions
from explore.stat import statError
from hydroDL.master.master import master_train, master_test
import numpy as np
import os
import pandas as pd

from utils import serialize_json, unserialize_json
from utils.dataset_format import subset_of_dict
from visual import plot_ts_obs_pred
from visual.plot_model import plot_ind_map, plot_we_need, plot_map
from visual.plot_stat import plot_ecdf, plot_diff_boxes


class MyTestCase(unittest.TestCase):
    """data pre-process and post-process"""

    def setUp(self) -> None:
        """before all of these, natural flow model need to be generated by config.ini of gages dataset, and it need
        to be moved to right dir manually """
        config_dir = definitions.CONFIG_DIR
        self.config_file = os.path.join(config_dir, "dam/config_exp13.ini")
        self.subdir = r"dam/exp13"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        # self.nid_file = 'PA_U.xlsx'
        # self.nid_file = 'OH_U.xlsx'
        self.nid_file = 'NID2018_U.xlsx'
        self.test_epoch = 300

    def test_dam_train(self):
        with torch.cuda.device(1):
            quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
            data_dir = os.path.join(quick_data_dir, "allnonref_85-05_nan-0.1_00-1.0")
            data_model_8595 = GagesModel.load_datamodel(data_dir,
                                                        data_source_file_name='data_source.txt',
                                                        stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                        forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                        f_dict_file_name='dictFactorize.json',
                                                        var_dict_file_name='dictAttribute.json',
                                                        t_s_dict_file_name='dictTimeSpace.json')

            gages_model_train = GagesModel.update_data_model(self.config_data, data_model_8595)
            nid_dir = os.path.join("/".join(self.config_data.data_path["DB"].split("/")[:-1]), "nid", "quickdata")
            nid_input = NidModel.load_nidmodel(nid_dir, nid_file=self.nid_file,
                                               nid_source_file_name='nid_source.txt', nid_data_file_name='nid_data.shp')
            gage_main_dam_purpose = unserialize_json(os.path.join(nid_dir, "dam_main_purpose_dict.json"))
            data_input = GagesDamDataModel(gages_model_train, nid_input, True, gage_main_dam_purpose)
            gages_input = choose_which_purpose(data_input)
            # master_train(gages_input)
            pre_trained_model_epoch = 80
            master_train(gages_input, pre_trained_model_epoch=pre_trained_model_epoch)

    def test_dam_test(self):
        with torch.cuda.device(1):
            quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
            data_dir = os.path.join(quick_data_dir, "allnonref-dam_95-05_nan-0.1_00-1.0")
            data_model_test = GagesModel.load_datamodel(data_dir,
                                                        data_source_file_name='test_data_source.txt',
                                                        stat_file_name='test_Statistics.json',
                                                        flow_file_name='test_flow.npy',
                                                        forcing_file_name='test_forcing.npy',
                                                        attr_file_name='test_attr.npy',
                                                        f_dict_file_name='test_dictFactorize.json',
                                                        var_dict_file_name='test_dictAttribute.json',
                                                        t_s_dict_file_name='test_dictTimeSpace.json')
            gages_input = GagesModel.update_data_model(self.config_data, data_model_test)
            pred, obs = master_test(gages_input, epoch=self.test_epoch)
            basin_area = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                           is_return_dict=False)
            mean_prep = gages_input.data_source.read_attr(gages_input.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                          is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_input.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)
            plot_we_need(gages_input, obs, pred, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

    def test_purposes_seperate(self):
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "allnonref-dam_95-05_nan-0.1_00-1.0")
        data_model_test = GagesModel.load_datamodel(data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        data_model = GagesModel.update_data_model(self.config_data, data_model_test)
        nid_dir = os.path.join("/".join(self.config_data.data_path["DB"].split("/")[:-1]), "nid", "quickdata")
        gage_main_dam_purpose = unserialize_json(os.path.join(nid_dir, "dam_main_purpose_dict.json"))
        gage_main_dam_purpose_lst = list(gage_main_dam_purpose.values())
        gage_main_dam_purpose_unique = np.unique(gage_main_dam_purpose_lst)
        purpose_regions = {}
        for i in range(gage_main_dam_purpose_unique.size):
            sites_id = []
            for key, value in gage_main_dam_purpose.items():
                if value == gage_main_dam_purpose_unique[i]:
                    sites_id.append(key)
            assert (all(x < y for x, y in zip(sites_id, sites_id[1:])))
            purpose_regions[gage_main_dam_purpose_unique[i]] = sites_id
        id_regions_idx = []
        id_regions_sites_ids = []
        df_id_region = np.array(data_model.t_s_dict["sites_id"])
        for key, value in purpose_regions.items():
            gages_id = value
            c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
            assert (all(x < y for x, y in zip(ind1, ind1[1:])))
            assert (all(x < y for x, y in zip(c, c[1:])))
            id_regions_idx.append(ind1)
            id_regions_sites_ids.append(c)
        pred_all, obs_all = load_result(self.config_data.data_path["Temp"], self.test_epoch)
        pred_all = pred_all.reshape(pred_all.shape[0], pred_all.shape[1])
        obs_all = obs_all.reshape(obs_all.shape[0], obs_all.shape[1])
        for i in range(9, len(gage_main_dam_purpose_unique)):
            pred = pred_all[id_regions_idx[i], :]
            obs = obs_all[id_regions_idx[i], :]
            inds = statError(obs, pred)
            inds['STAID'] = id_regions_sites_ids[i]
            inds_df = pd.DataFrame(inds)
            inds_df.to_csv(os.path.join(self.config_data.data_path["Out"],
                                        gage_main_dam_purpose_unique[i] + "epoch" + str(
                                            self.test_epoch) + 'data_df.csv'))
            # plot box，使用seaborn库
            keys = ["Bias", "RMSE", "NSE"]
            inds_test = subset_of_dict(inds, keys)
            box_fig = plot_diff_boxes(inds_test)
            box_fig.savefig(os.path.join(self.config_data.data_path["Out"],
                                         gage_main_dam_purpose_unique[i] + "epoch" + str(
                                             self.test_epoch) + "box_fig.png"))
            # plot ts
            sites = np.array(df_id_region[id_regions_idx[i]])
            t_range = np.array(data_model.t_s_dict["t_final_range"])
            show_me_num = 1
            ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
            ts_fig.savefig(os.path.join(self.config_data.data_path["Out"],
                                        gage_main_dam_purpose_unique[i] + "epoch" + str(
                                            self.test_epoch) + "ts_fig.png"))
            # plot nse ecdf
            sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
            plot_ecdf(sites_df_nse, keys[2], os.path.join(self.config_data.data_path["Out"],
                                                          gage_main_dam_purpose_unique[i] + "epoch" + str(
                                                              self.test_epoch) + "ecdf_fig.png"))
            # plot map
            gauge_dict = data_model.data_source.gage_dict
            save_map_file = os.path.join(self.config_data.data_path["Out"],
                                         gage_main_dam_purpose_unique[i] + "epoch" + str(
                                             self.test_epoch) + "map_fig.png")
            plot_map(gauge_dict, sites_df_nse, save_file=save_map_file, id_col="STAID", lon_col="LNG_GAGE",
                     lat_col="LAT_GAGE")

    def test_purposes_inds(self):
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
        data_dir = os.path.join(quick_data_dir, "allnonref-dam_95-05_nan-0.1_00-1.0")
        data_model = GagesModel.load_datamodel(data_dir,
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json',
                                               flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy',
                                               attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        gages_data_model = GagesModel.update_data_model(self.config_data, data_model)
        nid_dir = os.path.join("/".join(self.config_data.data_path["DB"].split("/")[:-1]), "nid", "quickdata")
        gage_main_dam_purpose = unserialize_json(os.path.join(nid_dir, "dam_main_purpose_dict.json"))
        gage_main_dam_purpose_lst = list(gage_main_dam_purpose.values())
        gage_main_dam_purpose_unique = np.unique(gage_main_dam_purpose_lst)
        purpose_regions = {}
        for i in range(gage_main_dam_purpose_unique.size):
            sites_id = []
            for key, value in gage_main_dam_purpose.items():
                if value == gage_main_dam_purpose_unique[i]:
                    sites_id.append(key)
            assert (all(x < y for x, y in zip(sites_id, sites_id[1:])))
            purpose_regions[gage_main_dam_purpose_unique[i]] = sites_id
        id_regions_idx = []
        id_regions_sites_ids = []
        df_id_region = np.array(gages_data_model.t_s_dict["sites_id"])
        for key, value in purpose_regions.items():
            gages_id = value
            c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
            assert (all(x < y for x, y in zip(ind1, ind1[1:])))
            assert (all(x < y for x, y in zip(c, c[1:])))
            id_regions_idx.append(ind1)
            id_regions_sites_ids.append(c)
        preds, obss, inds_dfs = split_results_to_regions(gages_data_model, self.test_epoch, id_regions_idx,
                                                         id_regions_sites_ids)
        region_names = list(purpose_regions.keys())
        inds_medians = []
        inds_means = []
        for i in range(len(region_names)):
            inds_medians.append(inds_dfs[i].median(axis=0))
            inds_means.append(inds_dfs[i].mean(axis=0))
        print(inds_medians)
        print(inds_means)


if __name__ == '__main__':
    unittest.main()