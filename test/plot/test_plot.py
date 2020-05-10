import os
import unittest
import numpy as np
import pandas as pd
import definitions
from data import GagesConfig
from data.data_input import CamelsModel, GagesModel, load_result
from data.gages_input_dataset import load_dataconfig_case_exp
from explore.stat import statError, ecdf
from utils import unserialize_numpy, unserialize_json
from utils.dataset_format import subset_of_dict
from visual import plot_box_inds, plot_ts_obs_pred
from visual.plot import plotCDF
from visual.plot_model import plot_ind_map, plot_map
from visual.plot_stat import plot_pdf_cdf, plot_ecdf, plot_ts_map, plot_ecdfs, plot_diff_boxes


def test_stat():
    print("main")
    arrays = {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    print(arrays)
    a_percentile_keys = [key for key in arrays.keys()]
    a_percentile_values = [np.percentile(value, [0, 25, 50, 75, 100]) for value in arrays.values()]
    a_percentile = {a_percentile_keys[k]: a_percentile_values[k] for k in range(len(a_percentile_keys))}
    print(a_percentile)

    b_percentile = {key: np.percentile(value, [0, 25, 50, 75, 100]) for key, value in arrays.items()}
    print(b_percentile)

    # 取一个数组小于30%分位数的所有值
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    a = np.array([2, 6, 3, 8, 4, 1, 5, 7, 9])
    print(np.where(a < np.percentile(a, 30)))
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    c = np.array([[[3], [1]], [[2], [4]]])
    print(c.squeeze().shape)
    print(c.shape)
    print(c)
    print(c.reshape(c.shape[0], c.shape[1]).shape)

    a = np.array([[2, 6, 3, 8, 4, 1, 5, 7, 9], [2, 3, 4, np.nan, 6, 7, 8, 9, 10]])
    print(a.shape)
    a[np.where(np.isnan(a))] = 0
    b = a[np.where(a < np.percentile(a, 30))]
    print(b)

    a_index = [np.where(a_i < np.percentile(a_i, 30)) for a_i in a]
    print(a_index)
    b = [a[i][a_index[i]].tolist() for i in range(a.shape[0])]
    print(type(b))
    c = np.array(b)
    print(c.shape)


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    # dataset = 'gages'
    # dataset_exp = dataset + '/basic/exp11'
    dataset = 'camels'
    dataset_exp = dataset + '/basic/exp5'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset_exp)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset_exp)
    flow_pred_file = os.path.join(dir_temp, 'flow_pred.npy')
    flow_obs_file = os.path.join(dir_temp, 'flow_obs.npy')
    t_s_dict_file = os.path.join(dir_temp, 'test_dictTimeSpace.json')
    # gages point shp file
    gage_point_file = os.path.join(dir_db, 'gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.shp')

    def setUp(self):
        # pred = unserialize_numpy(self.flow_pred_file)
        # obs = unserialize_numpy(self.flow_obs_file)
        # self.pred = pred.reshape(pred.shape[0], pred.shape[1])
        # self.obs = obs.reshape(pred.shape[0], pred.shape[1])
        # # 统计性能指标
        # self.inds = statError(self.obs, self.pred)
        # t_s_dict = unserialize_json(self.t_s_dict_file)
        # sites = np.array(t_s_dict["sites_id"])
        self.keys = ["NSE"]
        # self.inds_test = subset_of_dict(self.inds, self.keys)

        self.test_epoch = 300

    def test_plot_ecdf_together(self):
        xs = []
        ys = []
        # cases_exps = ["basic_exp18", "simulate_exp10", "inv_exp1", "siminv_exp1"]
        # cases_exps = ["basic_exp18", "simulate_exp1"]
        # cases_exps = ["basic_exp18", "gagests_exp18"]
        # cases_exps = ["basic_exp18", "basic_exp19"]
        cases_exps = ["nodam_exp3", "majordam_exp2"]
        cases_exps_legends = ["no_major_dam", "with_major_dam"]
        # cases_exps = ["dam_exp4", "dam_exp5", "dam_exp6"]
        # cases_exps = ["dam_exp1", "dam_exp2", "dam_exp3"]
        # cases_exps_legends = ["dam-lstm", "dam-with-natural-flow", "dam-with-kernel"]
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], self.test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            inds_i = statError(obs_i, pred_i)
            x, y = ecdf(inds_i[self.keys[0]])
            xs.append(x)
            ys.append(y)
        plot_ecdfs(xs, ys, cases_exps_legends)

    def test_plot_box(self):
        """测试可视化代码"""
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        plot_diff_boxes(inds_test)

    def test_plot_ts(self):
        """测试可视化代码"""
        # plot time series
        show_me_num = 5
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        plot_ts_obs_pred(self.obs, self.pred, sites, t_range, show_me_num)

    def test_plot_ind_map(self):
        """plot nse value on map"""
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        keys = ["NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        # concat sites and inds
        sites_df = pd.DataFrame({"sites": sites, keys[0]: inds_test[keys[0]]})
        plot_ind_map(self.gage_point_file, sites_df)

    def test_plot_map(self):
        data_model = CamelsModel.load_datamodel(self.dir_temp,
                                                data_source_file_name='test_data_source.txt',
                                                stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                                forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                f_dict_file_name='test_dictFactorize.json',
                                                var_dict_file_name='test_dictAttribute.json',
                                                t_s_dict_file_name='test_dictTimeSpace.json')
        gauge_dict = data_model.data_source.gage_dict
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        keys = ["NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        sites_df = pd.DataFrame({"sites": sites, keys[0]: inds_test[keys[0]]})
        plot_map(gauge_dict, sites_df, id_col="id", lon_col="lon", lat_col="lat")
        # plot_map(gauge_dict, sites_df, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

    def test_plot_kuai_cdf(self):
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        keys = ["NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        plotCDF([inds_test[keys[0]]], ref=None, legendLst=["LSTM"], linespec=['-', '-', ':', ':', ':'])

    def test_plot_pdf_cdf(self):
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        keys = ["NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        x = pd.DataFrame(inds_test)
        # x = inds_test[keys[0]]
        # plot_dist(x)
        plot_pdf_cdf(x, keys[0])

    def test_plot_ecdf(self):
        plot_ecdf(self.inds_test, self.keys[0])

    def test_plot_map_ts(self):
        data_map = np.arange(5).tolist()
        lat = [24, 30, 40, 50, 50.5]
        lon = [-120, -110, -100, -90, -70]
        data_ts_obs_np = np.arange(30).reshape(5, 6)
        data_ts_pred_np = np.arange(30, 60).reshape(5, 6)
        data_ts = [[data_ts_obs_np[i], data_ts_pred_np[i]] for i in range(data_ts_obs_np.shape[0])]
        print(data_ts)
        t = np.arange(6).tolist()
        sites_id = ["01", "02", "03", "04", "05"]
        plot_ts_map(data_map, data_ts, lat, lon, t, sites_id)

    def test_click(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        print('Plotting Data...')

        fig, ax = plt.subplots()
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        i = [0]

        def f1():
            ax.plot([1, 5, 10, 20], [1, 5, 10, 20])

        def f2():
            ax.plot([1, 5, 10, 20], [2, 10, 20, 40])

        def f3():
            ax.plot([1, 5, 10, 20], [5, 9, 17, 28])

        def update(event=None):
            if i[0] == 0: f1()
            if i[0] == 1: f2()
            if i[0] == 2: f3()
            fig.canvas.draw_idle()
            i[0] += 1
            print('Step {} done..., Press a key to continue'.format(i[0]))

        fig.canvas.mpl_connect("key_press_event", update)

        update()
        plt.show()


if __name__ == '__main__':
    unittest.main()
