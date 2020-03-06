"""使用seaborn库绘制各类统计相关的图形"""
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from matplotlib import gridspec

from explore.stat import ecdf
from utils.hydro_math import flat_data


def plot_boxs(data, x_name, y_name):
    """绘制箱型图"""
    sns.set(style="ticks", palette="pastel")

    # Draw a nested boxplot to show bills by day and time
    sns_box = sns.boxplot(x=x_name, y=y_name, data=data, showfliers=False)
    sns.despine(offset=10, trim=True)
    plt.show()

    return sns_box.get_figure()


def plot_diff_boxes(data):
    """绘制箱型图 in one row and different cols"""
    subplot_num = data.shape[1]
    f, axes = plt.subplots(1, subplot_num)
    for i in range(subplot_num):
        sns.boxplot(y=data.columns.values[i], data=data, orient='v', ax=axes[i], showfliers=False)
    plt.show()
    return f


def plot_ts(data, row_name, col_name, x_name, y_name):
    """绘制时间序列对比图"""
    sns.set(style="whitegrid")
    g = sns.FacetGrid(data, row=row_name, col=col_name, margin_titles=True)
    g.map(plt.plot, x_name, y_name, color="steelblue")

    plt.show()
    return g


def plot_point_map(gpd_gdf, percentile=0):
    """plot point data on a map"""
    # Choose points in which NSE value are bigger than the 25% quartile value range
    percentile_data = np.percentile(gpd_gdf['NSE'].values, percentile).astype(float)
    # the result of query is a tuple with one element, but it's right for plotting
    data_chosen = gpd_gdf.query("NSE > " + str(percentile_data))
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    proj = gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5)
    polyplot_kwargs = {'facecolor': (0.9, 0.9, 0.9), 'linewidth': 0}
    pointplot_kwargs = {'hue': 'NSE', 'legend': True, 'linewidth': 0.01}
    # ax = gplt.polyplot(contiguous_usa.geometry, projection=proj, **polyplot_kwargs)
    ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())
    gplt.pointplot(data_chosen, ax=ax, **pointplot_kwargs)
    ax.set_title("NSE " + "Map")
    plt.show()
    # plt.savefig("NSE-usa.png", bbox_inches='tight', pad_inches=0.1)


def plot_ecdf(mydataframe, mycolumn):
    """Empirical cumulative distribution function"""
    x, y = ecdf(mydataframe[mycolumn])
    df = pd.DataFrame({"x": x, "y": y})
    sns.set_style("ticks", {'axes.grid': True})
    ax = sns.lineplot(x="x", y="y", data=df, estimator=None).set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
                                                                 yticks=np.arange(0, 1, 0.05))
    plt.show()
    return ax


def plot_pdf_cdf(mydataframe, mycolumn):
    # settings
    f, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=320)
    axes[0].set_ylabel('fraction (PDF)')
    axes[1].set_ylabel('fraction (CDF)')

    # left plot (PDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
    sns.distplot(
        mydataframe[mycolumn], kde=True, axlabel=mycolumn,
        hist_kws={"density": True}, ax=axes[0]
    ).set(xlim=(0, 1))

    # right plot (CDF) # REMEMBER TO CHANGE bins, xlim PROPERLY!!
    sns.distplot(
        mydataframe[mycolumn], kde=False, axlabel=mycolumn,
        hist_kws={"density": True, "cumulative": True, "histtype": "step", "linewidth": 4}, ax=axes[1],
    ).set(xlim=(0, 1), ylim=(0, 1))
    plt.show()


def plot_loss_early_stop(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    max_loss = max(np.amax(np.array(train_loss)), np.amax(np.array(valid_loss)))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max_loss + 0.05)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_map_carto(data, lat, lon, ax=None, pertile_range=None):
    temp = flat_data(data)
    vmin = np.percentile(temp, 5)
    vmax = np.percentile(temp, 95)
    llcrnrlat = np.min(lat),
    urcrnrlat = np.max(lat),
    llcrnrlon = np.min(lon),
    urcrnrlon = np.max(lon),
    extent = [llcrnrlon[0], urcrnrlon[0], llcrnrlat[0], urcrnrlat[0]]
    # Figure
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                 facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)
    # auto projection
    pcm = ax.scatter(lon, lat, c=temp, s=10, cmap='viridis', vmin=vmin, vmax=vmax)
    # colorbar
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.3, 0.04, 0.4])
    cbar = fig.colorbar(pcm, cax=cbar_ax, extend='both', orientation='vertical')
    plt.show()


def plot_ts_matplot(t, y, color='r', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    ax.plot(t, y, color=color)
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_ts_map(dataMap, dataTs, lat, lon, t):
    assert type(dataMap) == list
    assert type(dataTs) == list
    # setup axes
    fig = plt.figure()
    # plot maps
    ax1 = plt.subplot(211, projection=ccrs.PlateCarree())
    plot_map_carto(dataMap, lat=lat, lon=lon, ax=ax1)
    # line plot
    ax2 = plt.subplot(212)

    # plot ts
    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon) ** 2 + (yClick - lat) ** 2)
        ind = np.argmin(d)
        titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        tsLst = dataTs[ind]
        plot_ts_matplot(t, tsLst, ax=ax2)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
