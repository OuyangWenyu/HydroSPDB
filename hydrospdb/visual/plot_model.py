"""some plot functions based on plot.py and plot_stat.py"""
import os
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hydrospdb.data.source.data_camels import Camels

from hydrospdb.utils.hydro_utils import t_range_days
from hydrospdb.visual.plot_stat import (
    plot_boxs,
    plot_diff_boxes,
    plot_ts_map,
    plot_map_carto,
)


def plot_scatter_multi_attrs(
    config_data, inds_df, idx_lst_nse_range, attr_lst, y_var_lst
):
    """scatter plot: there are many independent vars and dependent var"""
    sites_all = config_data.t_s_dict["sites_id"]
    attrs_ = config_data.data_source.read_attr(sites_all, attr_lst, is_return_dict=False)
    x_var_values = [attrs_[idx_lst_nse_range, i] for i in range(len(attr_lst))]
    y_var_values = [inds_df[y_var].values[idx_lst_nse_range] for y_var in y_var_lst]
    xy_values = np.array(x_var_values + y_var_values).T
    df = pd.DataFrame(xy_values, columns=attr_lst + y_var_lst)
    g = sns.pairplot(
        df, x_vars=attr_lst, y_vars=y_var_lst, height=5, aspect=0.8, kind="reg"
    )
    #  plot_kws=dict(s=5, edgecolor="b", linewidth=1)
    g.axes[0, 0].set_ylim((0, 1))
    plt.show()


def plot_box_inds(indicators):
    """plot boxplots in one coordination"""
    data = pd.DataFrame(indicators)
    # transform data to "tidy data". Firstly add a column，then assign all other values to "var_name" and "value_name" columns
    indict_name = "indicator"
    indicts = pd.Series(data.columns.values, name=indict_name)
    data_t = pd.DataFrame(data.values.T)
    data_t = pd.concat([indicts, data_t], axis=1)
    formatted_data = pd.melt(data_t, [indict_name])
    formatted_data = formatted_data.sort_values(by=[indict_name])
    return plot_boxs(formatted_data, x_name=indict_name, y_name="value")


def plot_gages_attrs_boxes(
    sites1, sites2, attr_lst, attrs1, attrs2, diff_str, row_and_col=None
):
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
    return plot_diff_boxes(
        result,
        row_and_col=row_and_col,
        y_col=np.arange(len(attr_lst)).tolist(),
        x_col=len(attr_lst),
    )


def plot_gages_map_and_ts(
    data_source: Camels,
    data_params: dict,
    obs,
    pred,
    inds_df,
    show_ind_key,
    pertile_range,
    plot_ts=True,
    fig_size=(8, 8),
    cmap_str="viridis",
):
    """
    Plot indicators map and show time series

    Parameters
    ----------
    data_source : Camels
        _description_
    data_params : dict
        _description_
    obs : _type_
        _description_
    pred : _type_
        _description_
    inds_df : _type_
        _description_
    show_ind_key : _type_
        _description_
    pertile_range : _type_
        _description_
    plot_ts : bool, optional
        _description_, by default True
    fig_size : tuple, optional
        _description_, by default (8, 8)
    cmap_str : str, optional
        _description_, by default "viridis"

    Returns
    -------
    _type_
        _description_
    """
    data_map = inds_df[show_ind_key].values
    sites = np.array(data_params["object_ids"])
    camels_sites = data_source.camels_sites
    lat = camels_sites[camels_sites["gage_id"].isin(sites)]["LAT_GAGE"]
    lon = camels_sites[camels_sites["gage_id"].isin(sites)]["LON_GAGE"]
    data_ts = [[obs[i], pred[i]] for i in range(obs.shape[0])]
    warmup_length = data_params["warmup_length"]
    t = t_range_days(data_params["t_range_test"]).tolist()[warmup_length:]
    if plot_ts:
        plot_ts_map(
            data_map.tolist(),
            data_ts,
            lat,
            lon,
            t,
            sites.tolist(),
            pertile_range=pertile_range,
        )
    else:
        return plot_map_carto(
            data_map,
            lat=lat,
            lon=lon,
            pertile_range=pertile_range,
            fig_size=(fig_size[0], fig_size[1] - 2),
            cmap_str=cmap_str,
        )


def plot_gages_map_and_box(
    chosen_sites_df,
    inds_df,
    show_ind_key,
    idx_lst=None,
    pertile_range=None,
    is_all_data_shown_in_box=True,
    fig_size=(12, 7),
    cmap_str="jet",
    titles=None,
    wh_ratio=None,
    adjust_xy=(0, 0.05),
):
    if pertile_range is None:
        pertile_range = [0, 100]
    if titles is None:
        titles = ["title1", "title2"]
    if wh_ratio is None:
        wh_ratio = [1, 4]
    if idx_lst is None:
        data_map = inds_df[show_ind_key].values
        lat = chosen_sites_df["LAT_GAGE"]
        lon = chosen_sites_df["LNG_GAGE"]
    else:
        assert pertile_range == [0, 100]
        data_map = (inds_df.loc[idx_lst])[show_ind_key].values
        all_lat = chosen_sites_df["LAT_GAGE"].values
        all_lon = chosen_sites_df["LNG_GAGE"].values
        lat = all_lat[idx_lst]
        lon = all_lon[idx_lst]
    assert len(data_map) == len(lat)
    # Figure
    fig = plt.figure(figsize=fig_size)
    # first ax for plotting map
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.set(title=titles[0])
    plot_map_carto(
        data_map,
        lat=lat,
        lon=lon,
        fig=fig,
        ax=ax1,
        fig_size=(fig_size[0], fig_size[1] - 2),
        pertile_range=pertile_range,
        cmap_str=cmap_str,
    )

    ax2 = plt.subplot(2, 1, 2)
    ax2.set(title=titles[1])
    if is_all_data_shown_in_box:
        sns.boxplot(
            data=inds_df[show_ind_key].values,
            orient="h",
            linewidth=3,
            ax=ax2,
            showfliers=False,
        )
    elif pertile_range == [0, 100]:
        sns.boxplot(data=data_map, orient="h", linewidth=3, ax=ax2, showfliers=False)

    else:
        vmin = np.nanpercentile(data_map, pertile_range[0])
        vmax = np.nanpercentile(data_map, pertile_range[1])
        data_shown_in_box = np.array([i for i in data_map if vmin <= i <= vmax])
        sns.boxplot(
            data=data_shown_in_box,
            orient="h",
            linewidth=3,
            ax=ax2,
            showfliers=False,
        )
    # adjust location
    pos1 = ax1.get_position()  # get the original position
    pos2 = ax2.get_position()  # get the original position
    pos2_ = [
        pos1.x0 + adjust_xy[0],
        pos1.y0 - pos2.height / wh_ratio[1] - adjust_xy[1],
        pos1.width / wh_ratio[0],
        pos2.height / wh_ratio[1],
    ]
    ax2.set_position(pos2_)  # set a new position
    return fig


def plot_gages_map_and_scatter(
    inds_df,
    items,
    idx_lst,
    cmap_strs=None,
    markers=None,
    labels=None,
    scatter_label=None,
    hist_bins=50,
    wspace=3,
    hspace=1.2,
    legend_x=0.01,
    legend_y=0.7,
    sub_fig_ratio=None,
):
    """inds_df: Lat,lon, slope/elavation, dor range, NSE"""
    if cmap_strs is None:
        cmap_strs = ["Reds", "Blues"]
    if markers is None:
        markers = ["o", "x"]
    if labels is None:
        labels = ["zero-dor", "small-dor"]
    if scatter_label is None:
        scatter_label = ["SLOPE_PCT", "NSE"]
    if sub_fig_ratio is None:
        sub_fig_ratio = [8, 4, 1]
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig_width = np.sum(sub_fig_ratio)
    fig_length = sub_fig_ratio[1] + sub_fig_ratio[2]
    fig = plt.figure(figsize=(fig_width, fig_length))
    grid = plt.GridSpec(fig_length, fig_width, wspace=wspace, hspace=hspace)

    ax1 = plt.subplot(
        grid[0 : sub_fig_ratio[1], 0 : sub_fig_ratio[0]], projection=ccrs.PlateCarree()
    )
    data = inds_df[items[0]].values
    lat = inds_df[items[1]].values
    lon = inds_df[items[2]].values
    plot_map_carto(
        data,
        lat,
        lon,
        fig=grid,
        ax=ax1,
        cmap_str=cmap_strs,
        idx_lst=idx_lst,
        markers=markers,
        need_colorbar=False,
    )

    ax2 = plt.subplot(
        grid[
            0 : sub_fig_ratio[1], sub_fig_ratio[0] : sub_fig_ratio[0] + sub_fig_ratio[1]
        ]
    )
    attr = inds_df[items[3]].values
    ax2.scatter(
        attr[idx_lst[0]],
        data[idx_lst[0]],
        color="red",
        marker=markers[0],
        alpha=0.5,
        label=labels[0],
    )
    ax2.scatter(
        attr[idx_lst[1]],
        data[idx_lst[1]],
        color="blue",
        marker=markers[1],
        alpha=0.5,
        label=labels[1],
    )
    ax2.set_xlabel(scatter_label[0])
    ax2.set_ylabel(scatter_label[1])
    handles, labels = ax2.get_legend_handles_labels()

    y_hist = plt.subplot(
        grid[
            0 : sub_fig_ratio[1],
            (sub_fig_ratio[0] + sub_fig_ratio[1]) : np.sum(sub_fig_ratio),
        ],
        xticklabels=[],
        sharey=ax2,
    )
    plt.hist(
        data[idx_lst[0]], hist_bins, orientation="horizontal", color="red", alpha=0.5
    )
    plt.hist(
        data[idx_lst[1]], hist_bins, orientation="horizontal", color="blue", alpha=0.5
    )

    x_hist = plt.subplot(
        grid[
            sub_fig_ratio[1] : sub_fig_ratio[1] + sub_fig_ratio[2],
            sub_fig_ratio[0] : (sub_fig_ratio[0] + sub_fig_ratio[1]),
        ],
        yticklabels=[],
        sharex=ax2,
    )
    plt.hist(
        attr[idx_lst[0]], hist_bins, orientation="vertical", color="red", alpha=0.5
    )
    plt.hist(
        attr[idx_lst[1]], hist_bins, orientation="vertical", color="blue", alpha=0.5
    )
    x_hist.invert_yaxis()  # invert y ax

    x_value = legend_x  # Offset by eye
    y_value = legend_y
    axbox = ax1.get_position()
    fig.legend(handles, labels, loc=(axbox.x0 - x_value, axbox.y1 - y_value))


def plot_sites_and_attr(
    all_sites_id,
    all_lon,
    all_lat,
    sites_lst1,
    sites_lst2,
    data_attr,
    pertile_range=None,
    is_discrete=False,
    cmap_str="viridis",
    sites_names=None,
    fig_size=(11, 4),
    markers=None,
    marker_sizes=None,
    colors=None,
    cbar_font_size=None,
    legend_font_size=None,
):
    """plot a map for all 3557 sites and all camels ones, and show one attributes"""
    if sites_names is None:
        sites_names = ["CAMELS", "Non_CAMELS"]
    if markers is None:
        markers = ["o", "x"]
    if marker_sizes is None:
        marker_sizes = [1, 3]
    if colors is None:
        colors = ["r", "b"]
    type_1_index_lst = [
        i for i in range(len(all_sites_id)) if all_sites_id[i] in sites_lst1
    ]
    type_2_index_lst = [
        i for i in range(len(all_sites_id)) if all_sites_id[i] in sites_lst2
    ]

    idx_lst = [type_1_index_lst, type_2_index_lst]

    plot_map_carto(
        data_attr,
        all_lat,
        all_lon,
        fig_size=fig_size,
        pertile_range=pertile_range,
        cmap_str=cmap_str,
        idx_lst=idx_lst,
        markers=markers,
        marker_size=marker_sizes,
        is_discrete=is_discrete,
        category_names=sites_names,
        colors=colors,
        legend_font_size=legend_font_size,
        colorbar_font_size=cbar_font_size,
    )
