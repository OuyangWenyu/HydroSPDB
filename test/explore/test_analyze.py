import unittest
import numpy as np
import pandas as pd
import definitions
from data import *
import os
from data.data_input import GagesModel, load_result
from explore.stat import statError
from visual.plot_model import plot_gages_map_and_ts
from visual.plot_stat import plot_diff_boxes
import matplotlib.pyplot as plt
import seaborn as sns


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
        nse_below = 0.5
        show_ind_key = 'NSE'
        idx_lst_small_nse = inds_df[(inds_df[show_ind_key] < nse_below)].index.tolist()
        sites_small_nse = np.array(data_model.t_s_dict['sites_id'])[idx_lst_small_nse]
        assert (all(x < y for x, y in zip(sites_small_nse, sites_small_nse[1:])))
        idx_lst_big_nse = inds_df[(inds_df[show_ind_key] >= nse_below)].index.tolist()
        sites_big_nse = np.array(data_model.t_s_dict['sites_id'])[idx_lst_big_nse]
        assert (all(x < y for x, y in zip(sites_big_nse, sites_big_nse[1:])))

        attr_lst = ["DRAIN_SQKM", "NDAMS_2009", "STOR_NID_2009"]
        attrs_small = data_model.data_source.read_attr(sites_small_nse, attr_lst, is_return_dict=False)
        attrs_big = data_model.data_source.read_attr(sites_big_nse, attr_lst, is_return_dict=False)
        bad_nse = np.tile(0, sites_small_nse.size).reshape(sites_small_nse.size, 1)
        good_nse = np.tile(1, sites_big_nse.size).reshape(sites_big_nse.size, 1)
        attrs_bad = np.concatenate((attrs_small, bad_nse), axis=1)
        attrs_good = np.concatenate((attrs_big, good_nse), axis=1)
        is_nse_good = ["IS_NSE_GOOD"]
        df_small = pd.DataFrame(attrs_bad, columns=attr_lst + is_nse_good)
        df_big = pd.DataFrame(attrs_good, columns=attr_lst + is_nse_good)
        result = pd.concat([df_small, df_big])
        plot_diff_boxes(result, row_and_col=[2, 2], y_col=[0, 1, 2], x_col=3)
        # plot_gages_map_and_ts(data_model, obs, pred, inds_df, show_ind_key, idx_lst_small_nse, pertile_range=[5, 100])


if __name__ == '__main__':
    unittest.main()
