"""补充时间处理相关函数"""
import datetime as dt
import numpy as np


def t2dt(t, hr=False):
    t_out = None
    if type(t) is int:
        if 30000000 > t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            t_out = t if hr is False else t.datetime()

    if type(t) is dt.date:
        t_out = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        t_out = t.date() if hr is False else t

    if t_out is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return t_out


def t_range2_array(t_range, *, step=np.timedelta64(1, 'D')):
    sd = t2dt(t_range[0])
    ed = t2dt(t_range[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def t_range_days(t_range, *, step=np.timedelta64(1, 'D')):
    """将给定的一个区间，转换为每日一个值的数组"""
    sd = dt.datetime.strptime(t_range[0], '%Y-%m-%d')
    ed = dt.datetime.strptime(t_range[1], '%Y-%m-%d')
    t_array = np.arange(sd, ed, step)
    return t_array


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2
