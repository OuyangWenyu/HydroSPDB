"""使用seaborn库绘制各类统计相关的图形"""

import numpy as np
import pandas as pd
import seaborn as sns


def plot_boxs(data):
    """绘制箱型图"""
    sns.set(style="ticks", palette="pastel")

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x="day", y="total_bill",
                hue="smoker", palette=["m", "g"],
                data=data)
    sns.despine(offset=10, trim=True)


def plot_ts(data):
    """绘制时间序列图
    :parameter
        data: pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])"""
    sns.set(style="whitegrid")

    data = data.rolling(7).mean()

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
