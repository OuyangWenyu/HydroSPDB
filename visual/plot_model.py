"""本项目调用可视化函数进行可视化的一些函数"""
import pandas as pd

from visual.plot_stat import plot_ts, plot_boxs


def plot_box_inds(indicators):
    """绘制观测值和预测值比较的时间序列图"""
    data = pd.DataFrame(indicators)
    plot_boxs(data)


def plot_ts_obs_pred(obs, pred):
    """绘制观测值和预测值比较的时间序列图"""
    values = [obs, pred]
    data = pd.DataFrame(values)
    plot_ts(data)
