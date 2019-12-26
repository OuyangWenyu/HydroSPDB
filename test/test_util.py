import unittest
import numpy as np
from utils.hydro_time import t_range_years, t_range_days
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


class MyTestCase(unittest.TestCase):
    t_range = ['1995-01-01', '2000-01-01']

    def test_something(self):
        self.assertEqual(True, False)

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


if __name__ == '__main__':
    unittest.main()
