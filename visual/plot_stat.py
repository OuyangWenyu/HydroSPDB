"""使用seaborn库绘制各类统计相关的图形"""
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import pandas as pd

from explore.stat import ecdf


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
