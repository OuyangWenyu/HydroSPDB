import os
import unittest
import numpy as np
import pandas as pd

from utils.dataset_format import trans_daymet_forcing_file_to_camels
from utils.hydro_time import t_range_years, t_range_days, get_year
from datetime import datetime, timedelta


def test_numpy():
    date = np.arange(1996, 2012).reshape(4, 4)
    print("------------------")
    print(date[0, 1])
    print(date[0][1])
    print("对应坐标(0,0)和(1,1)的数字：", date[[0, 1], [0, 1]])
    C_A = date[[0, 2]]  # 先取出想要的行数据
    C_A = C_A[:, [2, 3]]  # 再取出要求的列数据
    print("第0,2行的第3,3列", C_A)
    print(np.arange(5))
    print(np.arange(5).shape[0])


def test_get_year():
    str_time = "1995-01-01"
    print()
    year1 = get_year(str_time)
    print("年份是：", year1)
    text = '2012-09-20'
    y = datetime.strptime(text, '%Y-%m-%d')
    print()
    year2 = get_year(y)
    print("年份是：", year2)
    print("生成年份序列字符串：", np.arange(year1, year2).astype(str))
    a_time = np.datetime64('1995-01-01T00:00:00.000000')
    year3 = get_year(a_time)
    print(type(year3))
    print("年份是：", year3)


def test_os_func():
    files = os.listdir()
    print(type(files))
    print(files)


def test_pandas():
    df = pd.DataFrame(np.arange(16).reshape((4, 4)), columns=['one', 'two', 'three', 'four'],
                      index=['a', 'b', 'c', 'd'])
    print(df["one"].values.astype(str))
    df0 = pd.DataFrame(
        [["A", 1], ["A", 2], ["A", 3], ["B", 1], ["B", 2], ["B", 3], ["C", 1], ["C", 2], ["C", 3],
         ["A", 4], ["A", 5], ["A", 6], ["A", 7], ["B", 4], ["B", 5], ["B", 6], ["B", 7], ["C", 4], ["C", 5], ["C", 6],
         ["C", 7],
         ["A", 8], ["A", 9], ["A", 10], ["B", 8], ["B", 9], ["B", 10], ["C", 8], ["C", 9], ["C", 10]],
        columns=['gage_id', 'time_start'])
    # 接下来把df0转为如下形式：
    df_result = pd.DataFrame(
        [["A", 1], ["A", 2], ["A", 3], ["A", 4], ["A", 5], ["A", 6], ["A", 7], ["A", 8], ["A", 9], ["A", 10],
         ["B", 1], ["B", 2], ["B", 3], ["B", 4], ["B", 5], ["B", 6], ["B", 7], ["B", 8], ["B", 9], ["B", 10],
         ["C", 1], ["C", 2], ["C", 3], ["C", 4], ["C", 5], ["C", 6], ["C", 7], ["C", 8], ["C", 9], ["C", 10]])
    names = ["A", "B", "C"]
    df2 = pd.DataFrame()
    for name in names:
        df_name = df0[df0['gage_id'] == name]
        print("目前的名称：", name, df_name)
        df2 = [df2, df_name]
        df2 = pd.concat(df2)
    np1 = df2.values
    np2 = np.expand_dims(np1, axis=0)
    print(np2)
    np3 = np2.reshape(3, 10, 2)
    print(np3)
    np11 = df_result.values
    np21 = np.expand_dims(np11, axis=0)
    np31 = np21.reshape(3, 10, 2)
    np.testing.assert_equal(np3, np31)


class MyTestCase(unittest.TestCase):
    t_range = ['1995-01-01', '2000-01-01']

    def test_t_range_years(self):
        t_range = self.t_range
        t_list = t_range_years(t_range)
        t_result = np.array([1995, 1996, 1997, 1998, 1999])
        # np的数组assert需要调用numnpy自己的assert函数
        np.testing.assert_equal(t_list, t_result)

    def test_url(self):
        t_range = self.t_range
        start_time_str = datetime.strptime(t_range[0], '%Y-%m-%d')
        end_time_str = datetime.strptime(t_range[1], '%Y-%m-%d') - timedelta(days=1)
        streamflow_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}'
        url = streamflow_url.format('03010101', start_time_str.year, start_time_str.month,
                                    start_time_str.day, end_time_str.year, end_time_str.month, end_time_str.day)
        print(url)

    def test_t_range_days(self):
        t_range = self.t_range
        t_lst = t_range_days(t_range)
        print(t_lst)

    def test_intersect(self):
        t_range = self.t_range
        t_lst = t_range_years(t_range)
        nt = len(t_lst)
        print(t_lst)

        out = np.full([nt], np.nan)

        date = np.array([1996, 1997, 1999, 2000, 2003])
        obs = np.array([1, 2, 3, 4, 5])

        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
        print(out)
        result = np.array([np.nan, 1, 2, np.nan, 3])
        np.testing.assert_equal(out, result)

    # def test_trans_daymet_forcing_file_to_camels(self):
    #     daymet_dir = ''
    #     output_dir = ''
    #     result = pd.read_csv(output_dir)
    #     self.assertEqual(trans_daymet_forcing_file_to_camels(daymet_dir, output_dir), result)


if __name__ == '__main__':
    unittest.main()
