import datetime as dt
import numpy as np


def t2dt(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, 'D')):
    sd = t2dt(tRange[0])
    ed = t2dt(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def t_range_days(tRange, *, step=np.timedelta64(1, 'D')):
    """将给定的一个区间，转换为每日一个值的数组"""
    sd = dt.datetime.strptime(tRange[0], '%Y-%m-%d')
    ed = dt.datetime.strptime(tRange[1], '%Y-%m-%d')
    t_array = np.arange(sd, ed, step)
    return t_array


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind1, ind2
