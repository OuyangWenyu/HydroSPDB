"""使用seaborn库绘制各类统计相关的图形"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def flatData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return (xSort)


def plotMap(data,
            *,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None,
            pts=None,
            figsize=(8, 4),
            clbar=True,
            cRangeint=False,
            cmap=plt.cm.jet,
            bounding=None,
            prj='cyl'):
    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True
    if bounding is None:
        bounding = [np.min(lat) - 0.5, np.max(lat) + 0.5,
                    np.min(lon) - 0.5, np.max(lon) + 0.5]

    mm = basemap.Basemap(
        llcrnrlat=bounding[0],
        urcrnrlat=bounding[1],
        llcrnrlon=bounding[2],
        urcrnrlon=bounding[3],
        projection=prj,
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    mm.drawcountries(linewidth=1.0, linestyle='-.')
    x, y = mm(lon, lat)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet(np.arange(0, 1, 0.1)),
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:
        cs = mm.scatter(
            x, y, c=data, s=30, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)

    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    if pts is not None:
        mm.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(
                pts[1][k],
                pts[0][k],
                string.ascii_uppercase[k],
                fontsize=18)
    if clbar is True:
        mm.colorbar(cs, pad='5%')
    if title is not None:
        ax.set_title(title)
        if ax is None:
            return fig, ax, mm
        else:
            return mm, cs
