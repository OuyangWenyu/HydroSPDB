"""本项目调用可视化函数进行可视化的一些函数"""
import pandas as pd
import numpy as np
import geopandas as gpd

from visual.plot_stat import plot_ts, plot_boxs, plot_diff_boxes


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


def plot_boxes_inds(indicators):
    """plot boxplots in different coordination"""
    data = pd.DataFrame(indicators)
    box_fig = plot_diff_boxes(data)
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
    t_rng_lst = pd.DataFrame({time_column: pd.date_range(t_range[0], periods=obs_value.shape[0], freq='D')})
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


def plot_ind_map(all_points_file, ind_value, sites):
    """plot ind values on a map"""
    all_points = gpd.read_file(all_points_file)
    index = [i for i in range(all_points["STAID"].size) if all_points["STAID"].values[i] in sites]
    all_points_chosen = all_points.iloc[index]
    all_points_chosen.loc['ind'] = ind_value

