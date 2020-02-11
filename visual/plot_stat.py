"""使用seaborn库绘制各类统计相关的图形"""
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc


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


def plot_point_map(gpd_gdf):
    """plot point data on a map"""
    gpd_gdf = gpd_gdf.query('STATE not in ["AK", "HI", "PR"]')
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    scheme = mc.Quantiles(gpd_gdf['POP_2010'], k=5)

    ax = gplt.polyplot(
        contiguous_usa,
        zorder=-1,
        linewidth=1,
        projection=gcrs.AlbersEqualArea(),
        edgecolor='white',
        facecolor='lightgray',
        figsize=(8, 12)
    )
    gplt.pointplot(
        gpd_gdf,
        scale='POP_2010',
        limits=(2, 30),
        hue='POP_2010',
        cmap='Blues',
        scheme=scheme,
        legend=True,
        legend_var='scale',
        legend_values=[8000000, 2000000, 1000000, 100000],
        legend_labels=['8 million', '2 million', '1 million', '100 thousand'],
        legend_kwargs={'frameon': False, 'loc': 'lower right'},
        ax=ax
    )

    plt.title("Large cities in the contiguous United States, 2010")
    plt.savefig("largest-cities-usa.png", bbox_inches='tight', pad_inches=0.1)
