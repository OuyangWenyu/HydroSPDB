import os
import unittest
import numpy as np
import pandas as pd
from cartopy.feature import NaturalEarthFeature
import cartopy.crs as ccrs
from os import listdir
from PIL import Image
import definitions
import geopandas as gpd

from data import GagesSource
from data.data_input import load_result, GagesModel
from data.gages_input_dataset import load_dataconfig_case_exp, load_ensemble_result
from explore.stat import statError, ecdf
from utils import unserialize_json, unserialize_numpy
from utils.dataset_format import subset_of_dict
from visual import plot_ts_obs_pred
from visual.plot import plotCDF
from visual.plot_model import plot_ind_map, plot_map, plot_gages_map_and_ts, plot_gages_map_and_box, \
    plot_gages_map_and_scatter
from visual.plot_stat import plot_pdf_cdf, plot_ecdf, plot_ts_map, plot_ecdfs, plot_diff_boxes, plot_ecdfs_matplot
import matplotlib.pyplot as plt


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
    dataset = 'gages'
    dataset_exp = dataset + '/basic/exp11'
    # dataset = 'camels'
    # dataset_exp = dataset + '/basic/exp5'
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset_exp)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset_exp)
    t_s_dict_file = os.path.join(dir_temp, 'test_dictTimeSpace.json')
    # gages point shp file
    gage_point_file = os.path.join(dir_db, 'gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.shp')

    def setUp(self):
        self.test_epoch = 20
        flow_pred_file = os.path.join(self.dir_temp, "epoch" + str(self.test_epoch) + 'flow_pred.npy')
        flow_obs_file = os.path.join(self.dir_temp, "epoch" + str(self.test_epoch) + 'flow_obs.npy')
        pred = unserialize_numpy(flow_pred_file)
        obs = unserialize_numpy(flow_obs_file)
        self.pred = pred.reshape(pred.shape[0], pred.shape[1])
        self.obs = obs.reshape(pred.shape[0], pred.shape[1])
        # # 统计性能指标
        self.inds = statError(self.obs, self.pred)
        # t_s_dict = unserialize_json(self.t_s_dict_file)
        # sites = np.array(t_s_dict["sites_id"])
        self.keys = ["NSE"]
        self.inds_test = subset_of_dict(self.inds, self.keys)

    def test_plot_ecdf_together(self):
        xs = []
        ys = []
        # cases_exps = ["basic_exp18", "simulate_exp10", "inv_exp1", "siminv_exp1"]
        # cases_exps = ["basic_exp18", "simulate_exp1"]
        # cases_exps = ["basic_exp18", "gagests_exp18"]
        cases_exps = ["basic_exp37", "basic_exp38"]
        # cases_exps = ["nodam_exp3", "majordam_exp2"]
        # cases_exps_legends = ["no_major_dam", "with_major_dam"]
        cases_exps_legends = ["random_1234", "random_12345"]
        # cases_exps = ["dam_exp4", "dam_exp5", "dam_exp6"]
        # cases_exps = ["dam_exp1", "dam_exp2", "dam_exp3"]
        # cases_exps_legends = ["dam-lstm", "dam-with-natural-flow", "dam-with-kernel"]
        test_epoch = 300
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            inds_i = statError(obs_i, pred_i)
            x, y = ecdf(inds_i[self.keys[0]])
            xs.append(x)
            ys.append(y)
        plot_ecdfs(xs, ys, cases_exps_legends, x_str="NSE", y_str="CDF")
        cases_exps_addition = ["basic_exp39"]
        xs_addition = []
        ys_addition = []
        for case_exp in cases_exps_addition:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            inds_i = statError(obs_i, pred_i)
            x, y = ecdf(inds_i[self.keys[0]])
            xs_addition.append(x)
            ys_addition.append(y)
        plot_ecdfs(xs_addition, ys_addition, ["new"], x_str="NSE", y_str="CDF")
        # plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
        plt.show()

    def test_plot_ecdf_matplotlib(self):
        xs = []
        ys = []
        cases_exps = ["basic_exp37", "basic_exp38", "basic_exp39", "basic_exp40", "basic_exp41"]
        cases_exps_legends = ["random_1234", "random_123", "random_12345", "random_111", "random_1111"]
        test_epoch = 300
        for case_exp in cases_exps:
            config_data_i = load_dataconfig_case_exp(case_exp)
            pred_i, obs_i = load_result(config_data_i.data_path['Temp'], test_epoch)
            pred_i = pred_i.reshape(pred_i.shape[0], pred_i.shape[1])
            obs_i = obs_i.reshape(obs_i.shape[0], obs_i.shape[1])
            inds_i = statError(obs_i, pred_i)
            x, y = ecdf(inds_i[self.keys[0]])
            xs.append(x)
            ys.append(y)
        dash_lines = [False, False, False, False, True]
        plot_ecdfs_matplot(xs, ys, cases_exps_legends, colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "grey"],
                           dash_lines=dash_lines, x_str="NSE", y_str="CDF")
        plt.show()

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
        data_model = GagesModel.load_datamodel(self.dir_temp,
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
        # plot_map(gauge_dict, sites_df, id_col="id", lon_col="lon", lat_col="lat")
        plot_map(gauge_dict, sites_df, id_col="STAID", lon_col="LNG_GAGE", lat_col="LAT_GAGE")

    def test_plot_map_cartopy(self):
        data_model = GagesModel.load_datamodel(self.dir_temp,
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        show_ind_key = "NSE"
        inds_df = pd.DataFrame(self.inds)
        # nse_range = [-10000, 0]
        nse_range = [0, 1]
        idx_lst_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        plot_gages_map_and_ts(data_model, self.obs, self.pred, inds_df, show_ind_key, idx_lst_nse,
                              pertile_range=[0, 100], plot_ts=False, fig_size=(8, 4), cmap_str="jet")
        plt.show()

    def test_plot_loss_from_log(self):
        conus_exps = ["basic_exp50"]
        config_data = load_dataconfig_case_exp(conus_exps[0])
        log_file = os.path.join(config_data.data_path["Out"], "340epoch_run.csv")
        df_log = pd.read_csv(log_file, header=None)
        log_time_lst = np.array([float(log_i.split(" ")[-1]) for log_i in df_log.iloc[:, 0].values])
        print("time: ", str(np.sum(log_time_lst[340:]) / 60), " mins")

    def test_plot_map_cartopy_multi_vars(self):
        conus_exps = ["basic_exp37"]
        config_data = load_dataconfig_case_exp(conus_exps[0])

        dor_1 = - 0.02
        source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                          config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor_1)
        # basins with dams
        source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              dam_num=[1, 10000])
        # basins without dams
        source_data_withoutdams = GagesSource.choose_some_basins(config_data,
                                                                 config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False,
                                                                 dam_num=0)

        sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
        sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']

        sites_id_nodam = source_data_withoutdams.all_configs['flow_screen_gage_id']
        sites_id_smalldam = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()

        data_model = GagesModel.load_datamodel(config_data.data_path["Temp"],
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        all_lat = data_model.data_source.gage_dict["LAT_GAGE"]
        all_lon = data_model.data_source.gage_dict["LNG_GAGE"]

        conus_sites = data_model.t_s_dict["sites_id"]
        idx_lst_nodam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_nodam]
        idx_lst_smalldam_in_conus = [i for i in range(len(conus_sites)) if conus_sites[i] in sites_id_smalldam]

        attr_lst = ["SLOPE_PCT", "ELEV_MEAN_M_BASIN"]
        attrs = data_model.data_source.read_attr(conus_sites, attr_lst, is_return_dict=False)

        test_epoch = 300
        inds_df, pred, obs = load_ensemble_result(conus_exps, test_epoch, return_value=True)
        show_ind_key = "NSE"
        nse_range = [0, 1]
        idx_lst_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()

        type_1_index_lst = np.intersect1d(idx_lst_nodam_in_conus, idx_lst_nse).tolist()
        type_2_index_lst = np.intersect1d(idx_lst_smalldam_in_conus, idx_lst_nse).tolist()
        frame = []
        df_type1 = pd.DataFrame({"type": np.full(len(type_1_index_lst), "zero-dor"),
                                 show_ind_key: inds_df[show_ind_key].values[type_1_index_lst],
                                 "lat": all_lat[type_1_index_lst],
                                 "lon": all_lon[type_1_index_lst],
                                 "slope": attrs[type_1_index_lst, 0],
                                 "elevation": attrs[type_1_index_lst, 1]})
        frame.append(df_type1)
        df_type2 = pd.DataFrame({"type": np.full(len(type_2_index_lst), "small-dor"),
                                 show_ind_key: inds_df[show_ind_key].values[type_2_index_lst],
                                 "lat": all_lat[type_2_index_lst],
                                 "lon": all_lon[type_2_index_lst],
                                 "slope": attrs[type_2_index_lst, 0],
                                 "elevation": attrs[type_2_index_lst, 1]})
        frame.append(df_type2)
        data_df = pd.concat(frame)
        idx_lst = [np.arange(len(type_1_index_lst)),
                   np.arange(len(type_1_index_lst), len(type_1_index_lst) + len(type_2_index_lst))]
        plot_gages_map_and_scatter(data_df, [show_ind_key, "lat", "lon", "elevation"], idx_lst,
                                   cmap_strs=["Reds", "Blues"],
                                   labels=["zero-dor", "small-dor"], scatter_label=[attr_lst[1], show_ind_key])
        # matplotlib.rcParams.update({'font.size': 12})
        plt.tight_layout()
        plt.show()

    def test_plot_map_dam(self):
        data_model = GagesModel.load_datamodel(self.dir_temp,
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        usgs_id = data_model.t_s_dict["sites_id"]
        assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
        # attr_dam_lst = ["NDAMS_2009"]
        attr_dam_lst = ["STOR_NOR_2009"]
        data_attr, var_dict, f_dict = data_model.data_source.read_attr(usgs_id, attr_dam_lst)
        show_ind_key_dam = attr_dam_lst[0]
        inds_df_dam_num = pd.DataFrame({show_ind_key_dam: data_attr[:, 0]})
        dam_num_range = [1, 500]
        idx_lst_dam_num = inds_df_dam_num[
            (inds_df_dam_num[show_ind_key_dam] >= dam_num_range[0]) & (
                    inds_df_dam_num[show_ind_key_dam] < dam_num_range[1])].index.tolist()
        fig = plot_gages_map_and_box(data_model, inds_df_dam_num, show_ind_key_dam, idx_lst_dam_num,
                                     titles=["dam map", "dam boxplot"],
                                     wh_ratio=[1, 5], adjust_xy=(0, 0.04))
        plt.show()

    def test_plot_map_and_box(self):
        data_model = GagesModel.load_datamodel(self.dir_temp,
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        show_ind_key = "NSE"
        inds_df = pd.DataFrame(self.inds)
        # nse_range = [-10000, 0]
        nse_range = [0, 1]
        idx_lst_nse = inds_df[
            (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
        fig = plot_gages_map_and_box(data_model, inds_df, show_ind_key, idx_lst_nse, titles=["NSE map", "NSE boxplot"],
                                     wh_ratio=[1, 5], adjust_xy=(0, 0.04))
        plt.show()

    def test_plot_delta_map_and_box(self):
        data_model = GagesModel.load_datamodel(self.dir_temp,
                                               data_source_file_name='test_data_source.txt',
                                               stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                               forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                               f_dict_file_name='test_dictFactorize.json',
                                               var_dict_file_name='test_dictAttribute.json',
                                               t_s_dict_file_name='test_dictTimeSpace.json')
        show_ind_key = "NSE"
        inds_df = pd.DataFrame(self.inds)
        inds_df1 = pd.DataFrame(self.inds)
        inds_delta = inds_df - inds_df1
        print(inds_delta)
        inds_df = pd.DataFrame(self.inds)[show_ind_key]
        inds_df_fake = inds_df.copy()
        temp = np.random.uniform(-1, 1, inds_df_fake.size)
        comp_df = inds_df_fake + temp
        delta_nse = (comp_df - inds_df).to_frame()
        delta_range = [-0.9, 0.9]
        idx_lst_delta = delta_nse[
            (delta_nse[show_ind_key] >= delta_range[0]) & (delta_nse[show_ind_key] < delta_range[1])].index.tolist()
        fig = plot_gages_map_and_box(data_model, delta_nse, show_ind_key, idx_lst=idx_lst_delta,
                                     titles=["NSE map", "NSE boxplot"], wh_ratio=[1, 5], adjust_xy=(0, 0.04))
        # save figure without padding
        # plt.savefig('testmapbox.png', dpi=500, bbox_inches="tight")
        plt.show()

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

    def test_ecoregion(self):
        na_ecoregion_path = "../../example/data/map/ecoregion/NA_CEC_Eco_Level2.shp"
        na_ecoregion = gpd.read_file(na_ecoregion_path)
        print(na_ecoregion)
        country_boundary_us_path = "../../example/data/map/usa/usa-boundary-dissolved.shp"
        country_boundary_us = gpd.read_file(country_boundary_us_path)

        state_boundary_us_path = "../../example/data/map/usa/usa-states-census-2014.shp"
        state_boundary_us = gpd.read_file(state_boundary_us_path)

        # Are both layers in the same CRS?
        if (na_ecoregion.crs == country_boundary_us.crs):
            print("Both layers are in the same crs!", na_ecoregion.crs, country_boundary_us.crs)
        else:
            na_ecoregion = na_ecoregion.to_crs(country_boundary_us.crs)
            print("new proj", na_ecoregion.crs, country_boundary_us.crs)

        # Clip data
        na_ecoregion_clip = gpd.clip(na_ecoregion, country_boundary_us)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        na_ecoregion.plot(ax=ax1)
        na_ecoregion_clip.plot(ax=ax2)

        ax1.set_title("Unclipped ecoregions")
        ax2.set_title("Clipped ecoregions")

        ax1.set_axis_off()
        ax2.set_axis_off()

        plt.axis('equal')
        plt.show()

    def test_plot_a_fig_and_map_together(self):
        inds_df = pd.DataFrame(self.inds)

        def custom_plot1(ax=None):
            if ax is None:
                fig, ax = plt.subplots()
            x1 = np.linspace(0.0, 5.0)
            y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
            ax.plot(x1, y1, 'ko-')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('Damped oscillation')

        def custom_plot2(ax=None):
            # plot map ts
            data_model = GagesModel.load_datamodel(self.dir_temp,
                                                   data_source_file_name='test_data_source.txt',
                                                   stat_file_name='test_Statistics.json',
                                                   flow_file_name='test_flow.npy',
                                                   forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                   f_dict_file_name='test_dictFactorize.json',
                                                   var_dict_file_name='test_dictAttribute.json',
                                                   t_s_dict_file_name='test_dictTimeSpace.json')
            show_ind_key = 'NSE'
            # nse_range = [0.5, 1]
            nse_range = [0, 1]
            # nse_range = [-10000, 1]
            # nse_range = [-10000, 0]
            idx_lst = inds_df[
                (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()

            data_map = (inds_df.loc[idx_lst])[show_ind_key].values
            all_lat = data_model.data_source.gage_dict["LAT_GAGE"]
            all_lon = data_model.data_source.gage_dict["LNG_GAGE"]
            all_sites_id = data_model.data_source.gage_dict["STAID"]
            sites = np.array(data_model.t_s_dict['sites_id'])[idx_lst]
            sites_index = np.array([np.where(all_sites_id == i) for i in sites]).flatten()
            lat = all_lat[sites_index]
            lon = all_lon[sites_index]
            pertile_range = [0, 100]
            cmap_str = "jet"
            temp = data_map

            assert 0 <= pertile_range[0] < pertile_range[1] <= 100
            vmin = np.percentile(temp, pertile_range[0])
            vmax = np.percentile(temp, pertile_range[1])
            llcrnrlat = np.min(lat),
            urcrnrlat = np.max(lat),
            llcrnrlon = np.min(lon),
            urcrnrlon = np.max(lon),
            extent = [llcrnrlon[0], urcrnrlon[0], llcrnrlat[0], urcrnrlat[0]]

            ax.set_extent(extent)
            states = NaturalEarthFeature(category="cultural", scale="50m",
                                         facecolor="none",
                                         name="admin_1_states_provinces_shp")
            ax.add_feature(states, linewidth=.5, edgecolor="black")
            ax.coastlines('50m', linewidth=0.8)
            # auto projection
            scat = plt.scatter(lon, lat, c=temp, s=10, cmap=cmap_str, vmin=vmin, vmax=vmax)

            # get size and extent of axes:
            axpos = ax.get_position()
            pos_x = axpos.x0 + axpos.width + 0.01  # + 0.25*axpos.width
            pos_y = axpos.y0
            cax_width = 0.02
            cax_height = axpos.height
            # create new axes where the colorbar should go.
            # it should be next to the original axes and have the same height!
            pos_cax = fig.add_axes([pos_x, pos_y, cax_width, cax_height])
            plt.colorbar(ax=ax, cax=pos_cax)

        # 1. Plot in same line, this would work
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        custom_plot1(ax1)
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        custom_plot2(ax2)

        # 2. Plot in different line, default option
        # custom_plot1()
        # custom_plot2()
        plt.show()

    def test_concat_two_figs(self):
        # 获取当前文件夹中所有JPG图像
        im_list = [Image.open(fn) for fn in listdir() if fn.endswith('.png')]

        # 图片转化为相同的尺寸
        ims = []
        for i in im_list:
            new_img = i.resize((1280, 1280), Image.BILINEAR)
            ims.append(new_img)

            # 单幅图像尺寸
            width, height = ims[0].size

            # 创建空白长图
            result = Image.new(ims[0].mode, (width * len(ims), height))

            # 拼接图片
            for i, im in enumerate(ims):
                result.paste(im, box=(i * width, 0))

            # 保存图片
            result.save('res1.png')


if __name__ == '__main__':
    unittest.main()
