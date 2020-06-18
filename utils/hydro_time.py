"""补充时间处理相关函数"""
import datetime as dt, datetime
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
    # arange函数结果是左闭右开区间
    t_array = np.arange(sd, ed, step)
    return t_array


def t_days_lst2range(t_array):
    if type(t_array[0]) == np.datetime64:
        t0 = t_array[0].astype(datetime.datetime)
        t1 = t_array[-1].astype(datetime.datetime)
    else:
        t0 = t_array[0]
        t1 = t_array[-1]
    sd = t0.strftime('%Y-%m-%d')
    ed = t1.strftime('%Y-%m-%d')
    return [sd, ed]


def t_range_years(t_range):
    """t_range is a left-closed and right-open interval, if t_range[1] is not Jan.1 then end_year should be included"""
    start_year = int(t_range[0].split("-")[0])
    end_year = int(t_range[1].split("-")[0])
    end_month = int(t_range[1].split("-")[1])
    end_day = int(t_range[1].split("-")[2])
    if end_month == 1 and end_day == 1:
        year_range_list = np.arange(start_year, end_year)
    else:
        year_range_list = np.arange(start_year, end_year + 1)
    return year_range_list


def get_year(a_time):
    """返回时间的年份"""
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype('datetime64[Y]').astype(int) + 1970
    else:
        return int(a_time[0:4])


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2


def date_to_julian(a_time):
    if type(a_time) == str:
        fmt = '%Y-%m-%d'
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    julian_date = tt.tm_yday
    return julian_date


def t_range_to_julian(t_range):
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    julian_dates = [date_to_julian(a_time[0:10]) for a_time in t_array_str]
    return julian_dates
