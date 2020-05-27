import os
import unittest
import pandas as pd
import torch
import geopandas as gpd
from data import *
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from explore.gages_stat import stat_every_region
from explore.stat import statError
from hydroDL.master import *
import definitions
from utils import unserialize_numpy
import numpy as np
from matplotlib import pyplot


class MyTestCaseGages(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = definitions.CONFIG_DIR
        # self.config_file = os.path.join(config_dir, "basic/config_exp4.ini")
        # self.subdir = r"basic/exp4"
        # self.random_seed = 1234
        # self.config_file = os.path.join(config_dir, "basic/config_exp8.ini")
        # self.subdir = r"basic/exp8"
        # self.random_seed = 1234
        self.config_file = os.path.join(config_dir, "basic/config_exp29.ini")
        self.subdir = r"basic/exp29"
        self.random_seed = 1111

        # self.config_file = os.path.join(config_dir, "basic/config_exp12.ini")
        # self.subdir = r"basic/exp12"
        # self.random_seed = 1234
        # self.config_file = os.path.join(config_dir, "basic/config_exp15.ini")
        # self.subdir = r"basic/exp15"
        # self.random_seed = 1111
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def test_gages_data_model(self):
        gages_model = GagesModels(self.config_data, screen_basin_area_huc4=False)
        save_datamodel(gages_model.data_model_train, data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model.data_model_test, data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
        print("read and save data model")

    def test_gages_data_model_quickdata(self):
        quick_data_dir = os.path.join(self.config_data.data_path["DB"], "quickdata")
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

        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train, data_attr_update=True,
                                                         screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(self.config_data, data_model_test, data_attr_update=True,
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
        print("read and save data model")

    def test_train_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='data_source.txt',
                                               stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                               forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                               f_dict_file_name='dictFactorize.json',
                                               var_dict_file_name='dictAttribute.json',
                                               t_s_dict_file_name='dictTimeSpace.json')
        with torch.cuda.device(2):
            # pre_trained_model_epoch = 210
            master_train(data_model, random_seed=self.random_seed)
            # master_train(data_model, pre_trained_model_epoch=pre_trained_model_epoch, random_seed=self.random_seed)

    def test_test_gages(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        with torch.cuda.device(2):
            pred, obs = master_test(data_model, epoch=self.test_epoch)
            basin_area = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                          is_return_dict=False)
            mean_prep = data_model.data_source.read_attr(data_model.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                         is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch, pred, obs)

    def test_regions_stat(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        inds_medians, inds_means = stat_every_region(data_model, self.test_epoch)
        print(pd.DataFrame(inds_medians)["NSE"])
        print(pd.DataFrame(inds_means)["NSE"])

    def test_regions_seperate(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        gage_region_dir = data_model.data_source.all_configs.get("gage_region_dir")
        region_shapefiles = data_model.data_source.all_configs.get("regions")
        shapefiles = [os.path.join(gage_region_dir, region_shapefile + '.shp') for region_shapefile in
                      region_shapefiles]
        df_id_region = np.array(data_model.t_s_dict["sites_id"])
        assert (all(x < y for x, y in zip(df_id_region, df_id_region[1:])))
        id_regions_idx = []
        id_regions_sites_ids = []
        for shapefile in shapefiles:
            shape_data = gpd.read_file(shapefile)
            gages_id = shape_data['GAGE_ID'].values
            c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
            assert (all(x < y for x, y in zip(ind1, ind1[1:])))
            assert (all(x < y for x, y in zip(c, c[1:])))
            id_regions_idx.append(ind1)
            id_regions_sites_ids.append(c)
        flow_pred_file = os.path.join(data_model.data_source.data_config.data_path['Temp'], 'flow_pred.npy')
        flow_obs_file = os.path.join(data_model.data_source.data_config.data_path['Temp'], 'flow_obs.npy')
        pred_all = unserialize_numpy(flow_pred_file)
        obs_all = unserialize_numpy(flow_obs_file)
        pred_all = pred_all.reshape(pred_all.shape[0], pred_all.shape[1])
        obs_all = obs_all.reshape(obs_all.shape[0], obs_all.shape[1])
        for i in range(len(id_regions_idx)):
            pred = pred_all[id_regions_idx[i], :]
            obs = obs_all[id_regions_idx[i], :]
            inds = statError(obs, pred)
            inds['STAID'] = id_regions_sites_ids[i]
            inds_df = pd.DataFrame(inds)
            inds_df.to_csv(os.path.join(self.config_data.data_path["Out"],
                                        region_shapefiles[i] + "epoch" + str(self.test_epoch) + 'data_df.csv'))
            # plot box，使用seaborn库
            # keys = ["Bias", "RMSE", "NSE"]
            # inds_test = subset_of_dict(inds, keys)
            # box_fig = plot_boxes_inds(inds_test)
            # box_fig.savefig(os.path.join(self.config_data.data_path["Out"],
            #                              region_shapefiles[i] + "epoch" + str(self.test_epoch) + "box_fig.png"))
            # # plot ts
            # sites = np.array(df_id_region[id_regions_idx[i]])
            # t_range = np.array(data_model.t_s_dict["t_final_range"])
            # show_me_num = 5
            # ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
            # ts_fig.savefig(os.path.join(self.config_data.data_path["Out"],
            #                             region_shapefiles[i] + "epoch" + str(self.test_epoch) + "ts_fig.png"))
            # # plot nse ecdf
            # sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
            # plot_ecdf(sites_df_nse, keys[2], os.path.join(self.config_data.data_path["Out"],
            #                                               region_shapefiles[i] + "epoch" + str(
            #                                                   self.test_epoch) + "ecdf_fig.png"))
            # # plot map
            # gauge_dict = data_model.data_source.gage_dict
            # save_map_file = os.path.join(self.config_data.data_path["Out"], region_shapefiles[i] + "epoch" + str(
            #     self.test_epoch) + "map_fig.png")
            # plot_map(gauge_dict, sites_df_nse, save_file=save_map_file, id_col="STAID", lon_col="LNG_GAGE",
            #          lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
