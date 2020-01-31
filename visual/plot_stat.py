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
