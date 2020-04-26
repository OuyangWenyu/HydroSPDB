import unittest
import pandas as pd
import definitions
from data import *
import os
import numpy as np
from data.data_input import GagesModel, load_result
from explore.stat import statError
from utils import hydro_time
from visual.plot_stat import plot_ts_map


class TestExploreCase(unittest.TestCase):
    def setUp(self):
        """analyze result of model"""
        config_dir = definitions.CONFIG_DIR
        self.config_file = os.path.join(config_dir, "basic/config_exp18.ini")
        self.subdir = r"basic/exp18"
        self.config_data = GagesConfig.set_subdir(self.config_file, self.subdir)
        self.test_epoch = 300

    def tearDown(self):
        print('tearDown...')

    def test_analyze(self):
        data_model = GagesModel.load_datamodel(self.config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        pred, obs = load_result(data_model.data_source.data_config.data_path['Temp'], self.test_epoch)
        pred = pred.reshape(pred.shape[0], pred.shape[1])
        obs = obs.reshape(pred.shape[0], pred.shape[1])
        inds = statError(obs, pred)
        inds_df = pd.DataFrame(inds)
        idx_lst = inds_df[(inds_df['NSE'] < 0.6)].index.tolist()
        data_map = inds_df[(inds_df['NSE'] < 0.6)]['NSE'].values
        all_lat = data_model.data_source.gage_dict["LAT_GAGE"]
        all_lon = data_model.data_source.gage_dict["LNG_GAGE"]
        all_sites_id = data_model.data_source.gage_dict["STAID"]
        sites = np.array(data_model.t_s_dict['sites_id'])[idx_lst]
        sites_index = np.array([np.where(all_sites_id == i) for i in sites]).flatten()
        lat = all_lat[sites_index]
        lon = all_lon[sites_index]
        data_ts_obs_np = obs[idx_lst, :]
        data_ts_pred_np = pred[idx_lst, :]
        data_ts = [[data_ts_obs_np[i], data_ts_pred_np[i]] for i in range(data_ts_obs_np.shape[0])]
        t = hydro_time.t_range_days(data_model.t_s_dict["t_final_range"]).tolist()
        plot_ts_map(data_map.tolist(), data_ts, lat, lon, t, sites.tolist(), pertile_range=[5, 100])


if __name__ == '__main__':
    unittest.main()
