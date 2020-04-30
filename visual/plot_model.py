"""本项目调用可视化函数进行可视化的一些函数"""
import os

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from explore.stat import statError
from utils import hydro_time
from utils.dataset_format import subset_of_dict
from visual.plot_stat import plot_ts, plot_boxs, plot_diff_boxes, plot_point_map, plot_ecdf, plot_ts_map


def plot_scatter_multi_attrs(data_model, inds_df, idx_lst_nse_range, attr_lst, y_var_lst):
    """scatter plot: there are many independent vars and  dependent var"""
    sites_all = data_model.t_s_dict["sites_id"]
    attrs_ = data_model.data_source.read_attr(sites_all, attr_lst, is_return_dict=False)
    x_var_values = [attrs_[idx_lst_nse_range, i] for i in range(len(attr_lst))]
    y_var_values = [inds_df[y_var].values[idx_lst_nse_range] for y_var in y_var_lst]
    xy_values = np.array(x_var_values + y_var_values).T
    df = pd.DataFrame(xy_values, columns=attr_lst + y_var_lst)
    g = sns.pairplot(df, x_vars=attr_lst, y_vars=y_var_lst, height=5, aspect=.8, kind="reg")
    #  plot_kws=dict(s=5, edgecolor="b", linewidth=1)
    g.axes[0, 0].set_ylim((0, 1))
    plt.show()


def plot_region_seperately(gages_data_model, epoch, id_regions_idx, preds, obss, inds_dfs):
    df_id_region = np.array(gages_data_model.t_s_dict["sites_id"])
    regions_name = gages_data_model.data_source.all_configs.get("regions")
    for i in range(len(id_regions_idx)):
        # plot box，使用seaborn库
        keys = ["Bias", "RMSE", "NSE"]
        inds_test = subset_of_dict(inds_dfs[i], keys)
        box_fig = plot_diff_boxes(inds_test)
        box_fig.savefig(os.path.join(gages_data_model.data_source.data_config.data_path["Out"],
                                     regions_name[i] + "epoch" + str(epoch) + "box_fig.png"))
        # plot ts
        sites = np.array(df_id_region[id_regions_idx[i]])
        t_range = np.array(gages_data_model.t_s_dict["t_final_range"])
        show_me_num = 5
        ts_fig = plot_ts_obs_pred(obss[i], preds[i], sites, t_range, show_me_num)
        ts_fig.savefig(os.path.join(gages_data_model.data_source.data_config.data_path["Out"],
                                    regions_name[i] + "epoch" + str(epoch) + "ts_fig.png"))
        # plot nse ecdf
        sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
        plot_ecdf(sites_df_nse, keys[2], os.path.join(gages_data_model.data_source.data_config.data_path["Out"],
                                                      regions_name[i] + "epoch" + str(epoch) + "ecdf_fig.png"))
        # plot map
        gauge_dict = gages_data_model.data_source.gage_dict
        save_map_file = os.path.join(gages_data_model.data_source.data_config.data_path["Out"],
                                     regions_name[i] + "epoch" + str(epoch) + "map_fig.png")
        plot_map(gauge_dict, sites_df_nse, save_file=save_map_file, id_col="STAID", lon_col="LNG_GAGE",
                 lat_col="LAT_GAGE")


def plot_we_need(data_model_test, obs, pred, show_me_num=5, point_file=None, **kwargs):
    pred = pred.reshape(pred.shape[0], pred.shape[1])
    obs = obs.reshape(pred.shape[0], pred.shape[1])
    inds = statError(obs, pred)
    # plot box，使用seaborn库
    keys = ["Bias", "RMSE", "NSE"]
    inds_test = subset_of_dict(inds, keys)
    box_fig = plot_diff_boxes(inds_test)
    box_fig.savefig(os.path.join(data_model_test.data_source.data_config.data_path["Out"], "box_fig.png"))
    # plot ts
    t_s_dict = data_model_test.t_s_dict
    sites = np.array(t_s_dict["sites_id"])
    t_range = np.array(t_s_dict["t_final_range"])
    ts_fig = plot_ts_obs_pred(obs, pred, sites, t_range, show_me_num)
    ts_fig.savefig(os.path.join(data_model_test.data_source.data_config.data_path["Out"], "ts_fig.png"))
    # plot nse ecdf
    sites_df_nse = pd.DataFrame({"sites": sites, keys[2]: inds_test[keys[2]]})
    plot_ecdf(sites_df_nse, keys[2])
    # plot map
    if point_file is None:
        gauge_dict = data_model_test.data_source.gage_dict
        plot_map(gauge_dict, sites_df_nse, **kwargs)
    else:
        plot_ind_map(point_file, sites_df_nse, percentile=25)


def plot_box_inds(indicators):
    """plot boxplots in one coordination"""
    data = pd.DataFrame(indicators)
    # 将数据转换为tidy data格式，首先，增加一列名称列，然后剩下的所有值重组到var_name和value_name两列中
    indict_name = "indicator"
    indicts = pd.Series(data.columns.values, name=indict_name)
    data_t = pd.DataFrame(data.values.T)
    data_t = pd.concat([indicts, data_t], axis=1)
    formatted_data = pd.melt(data_t, [indict_name])
    formatted_data = formatted_data.sort_values(by=[indict_name])
    box_fig = plot_boxs(formatted_data, x_name=indict_name, y_name='value')
    return box_fig


def plot_gages_attrs_boxes(sites1, sites2, attr_lst, attrs1, attrs2, diff_str, row_and_col=None):
    """plot boxplots of GAGES model results"""
    if type(sites1) == list or type(sites2) == list:
        sites1 = np.array(sites1)
        sites2 = np.array(sites2)
    sites1diff = np.tile(0, sites1.size).reshape(sites1.size, 1)
    site2diff = np.tile(1, sites2.size).reshape(sites2.size, 1)
    attrs1_new = np.concatenate((attrs1, sites1diff), axis=1)
    attrs2_new = np.concatenate((attrs2, site2diff), axis=1)
    diff_str_lst = [diff_str]
    df1 = pd.DataFrame(attrs1_new, columns=attr_lst + diff_str_lst)
    df2 = pd.DataFrame(attrs2_new, columns=attr_lst + diff_str_lst)
    result = pd.concat([df1, df2])
    box_fig = plot_diff_boxes(result, row_and_col=row_and_col, y_col=np.arange(len(attr_lst)).tolist(),
                              x_col=len(attr_lst))
    return box_fig


def plot_ts_obs_pred(obs, pred, sites, t_range, num):
    """绘制观测值和预测值比较的时间序列图
    :parameter
        obs, pred: 都是二维序列变量，第一维是站点，第二维是值，
        sites: 所有站点的编号
        num:随机抽选num个并列到两个图上比较
    """
    num_lst = np.sort(np.random.choice(obs.shape[0], num, replace=False))
    # 首先把随机抽到的两个变量的几个站点数据整合到一个dataframe中，时间序列也要整合到该dataframe中
    sites_lst = pd.Series(sites[num_lst])
    obs_value = pd.DataFrame(obs[num_lst].T, columns=sites_lst)
    pred_value = pd.DataFrame(pred[num_lst].T, columns=sites_lst)
    tag_column = 'tag'
    time_column = 'time'
    sites_column = "sites"
    flow_column = "flow"
    tag_obs = 'obs'
    tag_pred = 'pred'
    t_rng_lst = pd.DataFrame({time_column: pd.date_range(str(t_range[0]), periods=obs_value.shape[0], freq='D')})
    obs_df = pd.concat([t_rng_lst, obs_value], axis=1)
    pred_df = pd.concat([t_rng_lst, pred_value], axis=1)
    obs_format = pd.melt(obs_df, id_vars=[time_column], value_vars=sites_lst, var_name=sites_column,
                         value_name=flow_column)
    pred_format = pd.melt(pred_df, id_vars=[time_column], value_vars=sites_lst, var_name=sites_column,
                          value_name=flow_column)
    obs_tag = pd.DataFrame({tag_column: np.full([obs_format.shape[0]], tag_obs)})
    obs_formatted = pd.concat([obs_tag, obs_format], axis=1)
    pred_tag = pd.DataFrame({tag_column: np.full([pred_format.shape[0]], tag_pred)})
    pred_formatted = pd.concat([pred_tag, pred_format], axis=1)
    tidy_data = pd.concat([obs_formatted, pred_formatted])
    ts_fig = plot_ts(tidy_data, sites_column, tag_column, time_column, flow_column)
    return ts_fig


def plot_ind_map(all_points_file, df_ind_value, percentile=0):
    """plot ind values on a map"""
    all_points = gpd.read_file(all_points_file)
    print(all_points.head())
    print(all_points.crs)
    # Here transform coordination to WGS84
    crs_wgs84 = CRS.from_epsg(4326)
    print(crs_wgs84)
    if not all_points.crs == crs_wgs84:
        all_points = all_points.to_crs(crs_wgs84)
    print(all_points.head())
    print(all_points.crs)
    sites = df_ind_value['sites'].values
    index = np.array([np.where(all_points["STAID"] == i) for i in sites]).flatten()
    newdata = gpd.GeoDataFrame(df_ind_value, crs=all_points.crs)

    newdata['geometry'] = None
    for idx in range(len(index)):
        # copy the point object to the geometry column on this row:
        newdata.at[idx, 'geometry'] = all_points.at[index[idx], 'geometry']

    plot_point_map(newdata, percentile=percentile)


def plot_map(gauge_dict, df_ind_value, save_file=None, proj_epsg=4269, percentile=10, **kwargs):
    """plot ind values on a map, epsg num of NAD83 is 4269"""
    sites = df_ind_value['sites'].values
    index = np.array([np.where(gauge_dict[kwargs["id_col"]] == i) for i in sites]).flatten()
    assert (all(x < y for x, y in zip(index, index[1:])))
    # index = [i for i in range(gauge_dict[id_col_name].size) if gauge_dict[id_col_name][i] in sites]
    newdata = gpd.GeoDataFrame(df_ind_value, crs=CRS.from_epsg(proj_epsg).to_wkt())
    newdata['geometry'] = None
    for idx in range(len(index)):
        # copy the point object to the geometry column on this row:
        point = Point(gauge_dict[kwargs["lon_col"]][index[idx]], gauge_dict[kwargs["lat_col"]][index[idx]])
        newdata.at[idx, 'geometry'] = point

    plot_point_map(newdata, percentile=percentile, save_file=save_file)


def plot_gages_map_and_ts(data_model, obs, pred, inds_df, show_ind_key, idx_lst, pertile_range):
    data_map = (inds_df.loc[idx_lst])[show_ind_key].values
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
    plot_ts_map(data_map.tolist(), data_ts, lat, lon, t, sites.tolist(), pertile_range=pertile_range)
