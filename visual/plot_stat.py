"""使用seaborn库绘制各类统计相关的图形"""

import numpy as np
import pandas as pd
import seaborn as sns


def plot_boxs():
    """绘制箱型图"""
    sns.set(style="ticks", palette="pastel")

    # Load the example tips dataset
    tips = sns.load_dataset("tips")

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x="day", y="total_bill",
                hue="smoker", palette=["m", "g"],
                data=tips)
    sns.despine(offset=10, trim=True)


def plot_ts():
    """绘制时间序列图"""
    sns.set(style="whitegrid")

    rs = np.random.RandomState(365)
    values = rs.randn(365, 4).cumsum(axis=0)
    dates = pd.date_range("1 1 2016", periods=365, freq="D")
    data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    data = data.rolling(7).mean()

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
