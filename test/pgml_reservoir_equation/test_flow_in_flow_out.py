"""Get some output from a designed-rule of reservoir operation"""
import copy
import os
import unittest
import numpy as np
import torch
from easydict import EasyDict

from data import GagesConfig, GagesSource
from data.config import cfg, update_cfg_item
from data.data_input import save_datamodel, GagesModel, _basin_norm, save_result, load_result
from data.gages_input_dataset import GagesSimDataModel
from explore.stat import statError
from hydroDL.master.master import master_train_natural_flow, master_test_natural_flow
from hydroDL.master.pgml import generate_fake_outflow
from utils import serialize_numpy, unserialize_numpy
from utils.dataset_format import subset_of_dict
from utils.hydro_math import choose_continuous_largest, index_of_continuous_largest
from utils.hydro_time import t_range_to_julian
from visual import plot_ts_obs_pred
from visual.plot_model import plot_map
import pandas as pd

from visual.plot_stat import plot_ecdf, plot_diff_boxes


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.sim_config_file = copy.deepcopy(cfg)
        test_case = EasyDict(sub="simulate/exp10")
        update_cfg_item(self.sim_config_file, test_case)
        main_layer = EasyDict(CTX=1)
        self.sim_config_file.update(main_layer)
        model_layer = EasyDict(miniBatch=[100, 30], nEpoch=20)
        self.sim_config_file.MODEL.update(model_layer)
        self.sim_config_data = GagesConfig(self.sim_config_file)

        self.config_file = copy.deepcopy(cfg)
        gages_layer = EasyDict(
            gageIdScreen=["01407500", "05017500", "06020600", "06036650", "06089000", "06101500", "06108000",
                          "06126500", "06130500", "06225500"])
        self.config_file.GAGES.update(gages_layer)
        pgml_model_layer = EasyDict(miniBatch=[5, 30], nEpoch=20)
        self.config_file.MODEL.update(pgml_model_layer)
        self.config_data = GagesConfig(self.config_file)
        self.test_epoch = 300

    def test_choose_the_initial_stages_day(self):
        """firstly we need to define the s^0, we should found the Continuous strongest precipitation in one year,
        then use the last day as the initial stage for the reservoir"""
        sim_data_dir = cfg.CACHE.DATA_DIR
        data_model_train = GagesModel.load_datamodel(sim_data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')
        data_model_test = GagesModel.load_datamodel(sim_data_dir,
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
        forcing_types = data_model_train.data_source.all_configs['forcing_chosen']
        index = forcing_types.index('prcp')
        train_prep = data_model_train.data_forcing[:, :, index]
        train_years = data_model_train.t_s_dict["t_final_range"]
        train_days = t_range_to_julian(train_years)
        time_length_chosen = 7
        days_chosen_train = index_of_continuous_largest(train_prep, train_days, time_length_chosen)
        test_prep = data_model_test.data_forcing[:, :, index]
        test_years = data_model_test.t_s_dict["t_final_range"]
        test_days = t_range_to_julian(test_years)
        days_chosen_test = index_of_continuous_largest(test_prep, test_days, time_length_chosen)
        days_chosen = np.concatenate((np.array(days_chosen_train), np.array(days_chosen_test)), axis=1)
        my_file = os.path.join(cfg.CACHE.QUICK_DATA_DIR, "init_stor_time_idx")
        serialize_numpy(days_chosen, my_file)
        print(days_chosen)
        print("read and save data model")

    def test_generate_outflow(self):
        """generate fake outflow by using operation rules"""
        my_file = os.path.join(cfg.CACHE.QUICK_DATA_DIR, "init_stor_time_idx.npy")
        initial_storages_time_idx = unserialize_numpy(my_file)
        data_dir = cfg.CACHE.DATA_DIR
        data_model_train = GagesModel.load_datamodel(data_dir,
                                                     data_source_file_name='data_source.txt',
                                                     stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                     forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                     f_dict_file_name='dictFactorize.json',
                                                     var_dict_file_name='dictAttribute.json',
                                                     t_s_dict_file_name='dictTimeSpace.json')
        conus_sites_id = data_model_train.t_s_dict["sites_id"]
        nolargedam_source_data = GagesSource.choose_some_basins(self.sim_config_data,
                                                                self.sim_config_data.model_dict["data"][
                                                                    "tRangeTrain"],
                                                                screen_basin_area_huc4=False, DOR=-0.02)
        nolargedam_sites_id = nolargedam_source_data.all_configs['flow_screen_gage_id']
        nolargedam_in_conus = np.intersect1d(conus_sites_id, nolargedam_sites_id)

        gages_model_train_inflow = GagesModel.update_data_model(self.sim_config_data, data_model_train,
                                                                sites_id_update=nolargedam_in_conus,
                                                                data_attr_update=True, screen_basin_area_huc4=False)
        gages_model_train = GagesModel.update_data_model(self.config_data, data_model_train,
                                                         sites_id_update=self.config_file.GAGES.gageIdScreen,
                                                         data_attr_update=True,
                                                         screen_basin_area_huc4=False)
        data_model = GagesSimDataModel(gages_model_train_inflow, gages_model_train)
        generate_fake_outflow(data_model, initial_storages_time_idx)

    def test_test_gages_sim(self):
        with torch.cuda.device(1):
            data_model1 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "1",
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
            data_model1.update_model_param('train', nEpoch=300)
            data_model2 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "2",
                                                    data_source_file_name='test_data_source.txt',
                                                    stat_file_name='test_Statistics.json',
                                                    flow_file_name='test_flow.npy',
                                                    forcing_file_name='test_forcing.npy',
                                                    attr_file_name='test_attr.npy',
                                                    f_dict_file_name='test_dictFactorize.json',
                                                    var_dict_file_name='test_dictAttribute.json',
                                                    t_s_dict_file_name='test_dictTimeSpace.json')
            model_input = GagesSimDataModel(data_model1, data_model2)
            pred, obs = master_test_natural_flow(model_input, epoch=self.test_epoch)
            basin_area = model_input.data_model2.data_source.read_attr(model_input.data_model2.t_s_dict["sites_id"],
                                                                       ['DRAIN_SQKM'], is_return_dict=False)
            mean_prep = model_input.data_model2.data_source.read_attr(model_input.data_model2.t_s_dict["sites_id"],
                                                                      ['PPTAVG_BASIN'], is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(model_input.data_model2.data_source.data_config.data_path['Temp'], str(self.test_epoch), pred,
                        obs)

    def test_sim_plot(self):
        data_model2 = GagesModel.load_datamodel(self.config_data.data_path["Temp"], "2",
                                                data_source_file_name='test_data_source.txt',
                                                stat_file_name='test_Statistics.json',
                                                flow_file_name='test_flow.npy',
                                                forcing_file_name='test_forcing.npy',
                                                attr_file_name='test_attr.npy',
                                                f_dict_file_name='test_dictFactorize.json',
                                                var_dict_file_name='test_dictAttribute.json',
                                                t_s_dict_file_name='test_dictTimeSpace.json')
        pred, obs = load_result(data_model2.data_source.data_config.data_path['Temp'], self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(obs.shape[0], obs.shape[1])
        inds = statError(obs, pred)
        inds['STAID'] = data_model2.t_s_dict["sites_id"]
        inds_df = pd.DataFrame(inds)
        inds_df.to_csv(os.path.join(self.config_data.data_path["Out"], 'data_df.csv'))
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(inds, keys)
        box_fig = plot_diff_boxes(inds_test)
        box_fig.savefig(os.path.join(self.config_data.data_path["Out"], "box_fig.png"))
        # plot ts
        show_me_num = 5
        t_s_dict = data_model2.t_s_dict
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        time_seq_length = data_model2.data_source.data_config.model_dict['model']['seqLength']
        time_start = np.datetime64(t_range[0]) + np.timedelta64(time_seq_length - 1, 'D')
        t_range[0] = np.datetime_as_string(time_start, unit='D')
        ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
        ts_fig.savefig(os.path.join(self.config_data.data_path["Out"], "ts_fig.png"))

        # plot nse ecdf
        sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
        plot_ecdf(sites_df_nse, keys[2])
        # plot map
        gauge_dict = data_model2.data_source.gage_dict
        plot_map(gauge_dict, sites_df_nse, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")


if __name__ == '__main__':
    unittest.main()
