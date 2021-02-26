"""basic plot functions for statistics, using cartopy, geoplot, matplotlib, and seaborn"""
import matplotlib
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
from matplotlib.cm import ScalarMappable
from explore.stat import ecdf
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.hydro_util import hydro_logger


def swarmplot_without_legend(x, y, hue, vmin, vmax, cmap, **kwargs):
    fig = plt.gcf()
    ax = sns.swarmplot(x, y, hue, **kwargs)
    # remove the legend, because we want to set a colorbar instead
    ax.legend().remove()
    norm = plt.Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    return fig


def plot_scatter_xyc(x_label, x, y_label, y, c_label=None, c=None, is_reg=False, xlim=None, ylim=None):
    """scatter plot: x-y relationship with c as colorbar"""
    if c is None:
        df = pd.DataFrame({x_label: x, y_label: y})
        points = plt.scatter(df[x_label], df[y_label])
    else:
        df = pd.DataFrame({x_label: x, y_label: y, c_label: c})
        points = plt.scatter(df[x_label], df[y_label], c=df[c_label], s=20, cmap="Spectral")  # set style options
        # add a color bar
        plt.colorbar(points)

    # set limits
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    # build the regression plot
    if is_reg:
        plot = sns.regplot(x_label, y_label, data=df, scatter=False)  # , color=".1"
        plot = plot.set(xlabel=x_label, ylabel=y_label)  # add labels
    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    plt.show()


def swarmplot_with_cbar(cmap_str, cbar_label, ylim, *args, **kwargs):
    fig = plt.gcf()
    ax = sns.swarmplot(*args, **kwargs)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # remove the legend, because we want to set a colorbar instead
    ax.legend().remove()
    # create colorbar
    cmap = plt.get_cmap(cmap_str)
    for key in kwargs:
        # if any criteria is not matched, we can filter this site
        if key == 'hue':
            hue_name = kwargs[key]
        if key == 'data':
            df = kwargs[key]
        if key == 'palette':
            color_str = kwargs[key]
    assert color_str == cmap_str
    norm = plt.Normalize(df[hue_name].min(), df[hue_name].max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label, labelpad=10)
    return fig


def plot_boxs(data, x_name, y_name, uniform_color=None, swarm_plot=False, hue=None, colormap=False, xlim=None,
              ylim=None):
    sns.set(style="ticks", palette="pastel")
    # Draw a nested boxplot to show bills by day and time
    if uniform_color is not None:
        sns_box = sns.boxplot(x=x_name, y=y_name, data=data, color=uniform_color, showfliers=False)
    else:
        sns_box = sns.boxplot(x=x_name, y=y_name, data=data, showfliers=False)
    if swarm_plot:
        if hue is not None:
            if colormap:
                # Create a matplotlib colormap from the sns seagreen color palette
                cmap = sns.light_palette("seagreen", reverse=False, as_cmap=True)
                # Normalize to the range of possible values from df["c"]
                norm = matplotlib.colors.Normalize(vmin=data[hue].min(), vmax=data[hue].max())
                # create a color dictionary (value in c : color from colormap)
                colors = {}
                for cval in data[hue]:
                    colors.update({cval: cmap(norm(cval))})

                # plot the swarmplot with the colors dictionary as palette, s=2 means size is 2
                sns_box = sns.swarmplot(x=x_name, y=y_name, hue=hue, s=2, data=data, palette=colors)
                # remove the legend, because we want to set a colorbar instead
                plt.gca().legend_.remove()
                # create colorbar
                divider = make_axes_locatable(plt.gca())
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig = sns_box.get_figure()
                fig.add_axes(ax_cb)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                                       norm=norm,
                                                       orientation='vertical')
                cb1.set_label('Some Units')
            else:
                palette = sns.light_palette("seagreen", reverse=False, n_colors=10)
                sns_box = sns.swarmplot(x=x_name, y=y_name, hue=hue, s=2, data=data, palette=palette)
        else:
            sns_box = sns.swarmplot(x=x_name, y=y_name, data=data, color=".2")
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    sns.despine()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    # plt.show()
    return sns_box.get_figure()


def plot_diff_boxes(data, row_and_col=None, y_col=None, x_col=None, hspace=0.3, wspace=1, title_str=None,
                    title_font_size=14):
    """plot boxplots in rows and cols"""
    # matplotlib.use('TkAgg')
    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    if y_col is None:
        subplot_num = data.shape[1]
    else:
        subplot_num = len(y_col)
    if row_and_col is None:
        row_num = 1
        col_num = subplot_num
        f, axes = plt.subplots(row_num, col_num)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
    else:
        assert subplot_num <= row_and_col[0] * row_and_col[1]
        row_num = row_and_col[0]
        col_num = row_and_col[1]
        f, axes = plt.subplots(row_num, col_num)
        f.tight_layout()
    for i in range(subplot_num):
        if y_col is None:
            if row_num == 1 or col_num == 1:
                sns.boxplot(y=data.columns.values[i], data=data, width=0.5, orient='v', ax=axes[i],
                            showfliers=False).set(xlabel=data.columns.values[i], ylabel='')
            else:
                row_idx = int(i / col_num)
                col_idx = i % col_num
                sns.boxplot(y=data.columns.values[i], data=data, orient='v', ax=axes[row_idx, col_idx],
                            showfliers=False)
        else:
            assert x_col is not None
            if row_num == 1 or col_num == 1:
                sns.boxplot(x=data.columns.values[x_col], y=data.columns.values[y_col[i]], data=data, orient='v',
                            ax=axes[i], showfliers=False)
            else:
                row_idx = int(i / col_num)
                col_idx = i % col_num
                sns.boxplot(x=data.columns.values[x_col], y=data.columns.values[y_col[i]],
                            data=data, orient='v', ax=axes[row_idx, col_idx], showfliers=False)
    if title_str is not None:
        f.suptitle(title_str, fontsize=title_font_size)
    return f


def plot_ts(data, row_name, col_name, x_name, y_name):
    sns.set(style="whitegrid")
    g = sns.FacetGrid(data, row=row_name, col=col_name, margin_titles=True)
    g.map(plt.plot, x_name, y_name, color="steelblue")

    plt.show()
    return g


def plot_point_map(gpd_gdf, percentile=0, save_file=None):
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
    if save_file is not None:
        plt.savefig(save_file)
        # plt.savefig("NSE-usa.png", bbox_inches='tight', pad_inches=0.1)


def plot_ecdfs(xs, ys, legends=None, style=None, case_str="case", event_str="event", x_str="x", y_str="y",
               ax_as_subplot=None, interval=0.1):
    """Empirical cumulative distribution function"""
    assert type(xs) == type(ys) == list
    assert len(xs) == len(ys)
    if legends is not None:
        assert type(legends) == list
        assert len(ys) == len(legends)
    if style is not None:
        assert type(style) == list
        assert len(ys) == len(style)
    for y in ys:
        assert (all(xi < yi for xi, yi in zip(y, y[1:])))
    frames = []
    for i in range(len(xs)):
        df_dict_i = {}
        if legends is None:
            str_i = x_str + str(i)
        else:
            str_i = legends[i]
        assert (all(xi < yi for xi, yi in zip(xs[i], xs[i][1:])))
        df_dict_i[x_str] = xs[i]
        df_dict_i[y_str] = ys[i]
        df_dict_i[case_str] = np.full([xs[i].size], str_i)
        if style is not None:
            df_dict_i[event_str] = np.full([xs[i].size], style[i])
        df_i = pd.DataFrame(df_dict_i)
        frames.append(df_i)
    df = pd.concat(frames)
    sns.set_style("ticks", {'axes.grid': True})
    if style is None:
        if ax_as_subplot is None:
            ecdfplt = sns.lineplot(x=x_str, y=y_str, hue=case_str, data=df, estimator=None).set(
                xlim=(0, 1), xticks=np.arange(0, 1, interval), yticks=np.arange(0, 1, interval))
        else:
            ecdfplt = sns.lineplot(ax=ax_as_subplot, x=x_str, y=y_str, hue=case_str, data=df, estimator=None).set(
                xlim=(0, 1), xticks=np.arange(0, 1, interval), yticks=np.arange(0, 1, interval))

    else:
        if ax_as_subplot is None:
            ecdfplt = sns.lineplot(x=x_str, y=y_str, hue=case_str, style=event_str, data=df, estimator=None).set(
                xlim=(0, 1), xticks=np.arange(0, 1, interval), yticks=np.arange(0, 1, interval))
        else:
            ecdfplt = sns.lineplot(ax=ax_as_subplot, x=x_str, y=y_str, hue=case_str, style=event_str, data=df,
                                   estimator=None).set(xlim=(0, 1), xticks=np.arange(0, 1, interval),
                                                       yticks=np.arange(0, 1, interval))
    return ecdfplt


def plot_ecdf(mydataframe, mycolumn, save_file=None):
    """Empirical cumulative distribution function"""
    x, y = ecdf(mydataframe[mycolumn])
    df = pd.DataFrame({"x": x, "y": y})
    sns.set_style("ticks", {'axes.grid': True})
    sns.lineplot(x="x", y="y", data=df, estimator=None).set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
                                                            yticks=np.arange(0, 1, 0.05))
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def plot_ecdfs_matplot(xs, ys, legends=None, colors=None, dash_lines=None, x_str="x", y_str="y", x_interval=0.1,
                       y_interval=0.1, x_lim=(0, 1), y_lim=(0, 1), show_legend=True, legend_font_size=16):
    """Empirical cumulative distribution function with matplotlib"""
    assert type(xs) == type(ys) == list
    assert len(xs) == len(ys)
    if legends is not None:
        assert type(legends) == list
        assert len(ys) == len(legends)
    if colors is not None:
        assert type(colors) == list
        assert len(ys) == len(colors)
    if dash_lines is not None:
        assert type(dash_lines) == list
        assert len(ys) == len(dash_lines)
    else:
        dash_lines = np.full(len(xs), False).tolist()
    for y in ys:
        assert (all(xi < yi for xi, yi in zip(y, y[1:])))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if colors is None:
        for i in range(len(xs)):
            assert (all(xi < yi for xi, yi in zip(xs[i], xs[i][1:])))
            line_i, = ax.plot(xs[i], ys[i], label=legends[i])
            if dash_lines[i]:
                line_i.set_dashes([2, 2, 10, 2])
    else:
        for i, color in enumerate(colors):
            if np.nanmax(np.array(xs[i])) == np.inf or np.nanmin(np.array(xs[i])) == -np.inf:
                assert (all(xi <= yi for xi, yi in zip(xs[i], xs[i][1:])))
            else:
                assert (all(xi < yi for xi, yi in zip(xs[i], xs[i][1:])))
            line_i, = ax.plot(xs[i], ys[i], color=color, label=legends[i])
            if dash_lines[i]:
                line_i.set_dashes([2, 2, 10, 2])

    plt.xlabel(x_str, fontsize=18)
    plt.ylabel(y_str, fontsize=18)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    # set x y number font size
    plt.xticks(np.arange(x_lim[0], x_lim[1] + x_lim[1] / 100, x_interval), fontsize=16)
    plt.yticks(np.arange(y_lim[0], y_lim[1] + y_lim[1] / 100, y_interval), fontsize=16)
    if show_legend:
        ax.legend()
        plt.legend(prop={'size': legend_font_size})
    plt.grid()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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


def plot_map_carto(data, lat, lon, fig=None, ax=None, pertile_range=None, fig_size=(8, 8), need_colorbar=True,
                   colorbar_size=None, cmap_str="jet", idx_lst=None, markers=None, marker_size=20, is_discrete=False,
                   colors=["r", "b"], category_names=None, legend_font_size=None, colorbar_font_size=None):
    # Here we ignore these inf values, transform them to nan, and then recalculate the min and max values
    try:
        data[data == np.inf] = np.nan
        data[data == -np.inf] = np.nan
    except ValueError as e:
        hydro_logger.warning(e)
    if pertile_range is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
    else:
        assert 0 <= pertile_range[0] < pertile_range[1] <= 100
        vmin = np.nanpercentile(data, pertile_range[0])
        vmax = np.nanpercentile(data, pertile_range[1])
    llcrnrlat = np.min(lat),
    urcrnrlat = np.max(lat),
    llcrnrlon = np.min(lon),
    urcrnrlon = np.max(lon),
    extent = [llcrnrlon[0], urcrnrlon[0], llcrnrlat[0], urcrnrlat[0]]
    # Figure
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent)
    states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)
    if idx_lst is not None:
        if type(marker_size) != list:
            marker_size = np.full(len(idx_lst), marker_size).tolist()
        else:
            assert len(marker_size) == len(idx_lst)
        if type(markers) != list:
            markers = np.full(len(idx_lst), markers).tolist()
        else:
            assert len(markers) == len(idx_lst)
        if type(cmap_str) != list:
            cmap_str = np.full(len(idx_lst), cmap_str).tolist()
        else:
            assert len(cmap_str) == len(idx_lst)
        if is_discrete:
            for i in range(len(idx_lst)):
                ax.plot(lon[idx_lst[i]], lat[idx_lst[i]], marker=markers[i], ms=marker_size[i],
                        label=category_names[i], c=colors[i], linestyle='')
            if legend_font_size is None:
                ax.legend()
            else:
                ax.legend(prop=dict(size=legend_font_size))
        else:
            for i in range(len(idx_lst)):
                scat = plt.scatter(lon[idx_lst[i]], lat[idx_lst[i]], c=data[idx_lst[i]], marker=markers[i],
                                   s=marker_size[i], cmap=cmap_str[i], vmin=vmin, vmax=vmax)
            if need_colorbar:
                cbar = fig.colorbar(scat, ax=ax, pad=0.01)
                if colorbar_font_size is not None:
                    cbar.ax.tick_params(labelsize=colorbar_font_size)
    else:
        if is_discrete:
            scatter = ax.scatter(lon, lat, c=data, s=marker_size)
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
            ax.add_artist(legend1)
        else:
            scat = plt.scatter(lon, lat, c=data, s=marker_size, cmap=cmap_str, vmin=vmin, vmax=vmax)
            if need_colorbar:
                if colorbar_size is not None:
                    cbar_ax = fig.add_axes(colorbar_size)
                    cbar = fig.colorbar(scat, cax=cbar_ax, orientation='vertical')
                else:
                    cbar = fig.colorbar(scat, ax=ax, pad=0.01)
                if colorbar_font_size is not None:
                    cbar.ax.tick_params(labelsize=colorbar_font_size)
    return ax


def plot_ts_matplot(t, y, color='r', ax=None, title=None):
    assert type(t) == list
    assert type(y) == list
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    ax.plot(t, y[0], color=color, label='pred')
    ax.plot(t, y[1], label='obs')
    ax.legend()
    if title is not None:
        ax.set_title(title, loc='center')
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_ts_map(dataMap, dataTs, lat, lon, t, sites_id, pertile_range=None):
    # show the map in a pop-up window
    matplotlib.use('TkAgg')
    assert type(dataMap) == list
    assert type(dataTs) == list
    # setup axes
    fig = plt.figure(figsize=(8, 8), dpi=100)
    gs = gridspec.GridSpec(2, 1)
    # plt.subplots_adjust(left=0.13, right=0.89, bottom=0.05)
    # plot maps
    ax1 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = plot_map_carto(dataMap, lat=lat, lon=lon, fig=fig, ax=ax1, pertile_range=pertile_range)
    # line plot
    ax2 = plt.subplot(gs[1])

    # plot ts
    def onclick(event):
        print("click event")
        # refresh the ax2, then new ts data can be showed without previous one
        ax2.cla()
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon) ** 2 + (yClick - lat) ** 2)
        ind = np.argmin(d)
        titleStr = 'site_id %s, lat %.3f, lon %.3f' % (sites_id[ind], lat[ind], lon[ind])
        tsLst = dataTs[ind]
        plot_ts_matplot(t, tsLst, ax=ax2, title=titleStr)
        # following funcs both work
        fig.canvas.draw()
        # plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
