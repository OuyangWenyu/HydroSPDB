import os
import unittest
import numpy as np
import definitions
from explore.stat import statError, statError1d
from utils import unserialize_numpy, unserialize_json
from utils.dataset_format import subset_of_dict
from visual import plot_box_inds, plot_ts_obs_pred
from visual.plot_model import plot_boxes_inds, plot_ind_map


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

    print(statError1d(np.array([1, 2, 3]), np.array([4, 5, 6])))


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = 'gages'
    # dataset = 'camels'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)
    flow_pred_file = os.path.join(dir_temp, 'flow_pred.npy')
    flow_obs_file = os.path.join(dir_temp, 'flow_obs.npy')
    t_s_dict_file = os.path.join(dir_temp, 'dictTimeSpace_test.json')

    def setUp(self):
        pred = unserialize_numpy(self.flow_pred_file)
        obs = unserialize_numpy(self.flow_obs_file)
        self.pred = pred.reshape(pred.shape[0], pred.shape[1])
        self.obs = obs.reshape(pred.shape[0], pred.shape[1])
        # 统计性能指标
        self.inds = statError(self.obs, self.pred)

    def test_plot_box(self):
        """测试可视化代码"""
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        plot_boxes_inds(inds_test)

    def test_plot_ts(self):
        """测试可视化代码"""
        # plot time series
        show_me_num = 5
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        t_range = np.array(t_s_dict["t_final_range"])
        plot_ts_obs_pred(self.obs, self.pred, sites, t_range, show_me_num)

    def test_plot_map(self):
        """plot nse value on map"""
        t_s_dict = unserialize_json(self.t_s_dict_file)
        sites = np.array(t_s_dict["sites_id"])
        keys = ["NSE"]
        inds_test = subset_of_dict(self.inds, keys)
        plot_ind_map(
            "/mnt/sdc/wvo5024/hydro-anthropogenic-lstm/example/data/gages/gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.shp",
            inds_test["NSE"], sites)


if __name__ == '__main__':
    unittest.main()
