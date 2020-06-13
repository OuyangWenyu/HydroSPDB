import os
import shutil
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import definitions
from data import GagesConfig, GagesSource
from data.gages_input_dataset import GagesModels
from data.susquehanna_input import SusquehannaSource, SusquehannaConfig
from utils import serialize_pickle, unserialize_pickle, hydro_time
from utils.dataset_format import trans_daymet_to_camels, subset_of_dict, trans_susquehanna_daymet_to_camels
from utils.hydro_math import random_choice_no_return, pair_comb
from utils.hydro_time import t_range_years, t_range_days, get_year, t_range_to_julian
from datetime import datetime, timedelta


class MyTestCase(unittest.TestCase):
    def test_numpy(self):
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

    def test_get_year(self):
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

    def test_os_func(self):
        files = os.listdir()
        print(type(files))
        print(files)

    def test_pandas(self):
        df = pd.DataFrame(np.arange(16).reshape((4, 4)), columns=['one', 'two', 'three', 'four'],
                          index=['a', 'b', 'c', 'd'])
        print(df["one"].values.astype(str))
        df0 = pd.DataFrame(
            [["A", 1], ["A", 2], ["A", 3], ["B", 1], ["B", 2], ["B", 3], ["C", 1], ["C", 2], ["C", 3],
             ["A", 4], ["A", 5], ["A", 6], ["A", 7], ["B", 4], ["B", 5], ["B", 6], ["B", 7], ["C", 4], ["C", 5],
             ["C", 6],
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

    def test_subset_of_dict(self):
        prices = {
            'ACME': 45.23,
            'AAPL': 612.78,
            'IBM': 205.55,
            'HPQ': 37.20,
            'FB': 10.75
        }
        tech_names = ['AAPL', 'IBM', 'HPQ', 'MSFT']
        print(subset_of_dict(prices, tech_names))

    t_range = ['1995-01-01', '2000-01-01']

    def test_t_range_years(self):
        t_range = self.t_range
        t_list = t_range_years(t_range)
        t_result = np.array([1995, 1996, 1997, 1998, 1999])
        # np的数组assert需要调用numnpy自己的assert函数
        np.testing.assert_equal(t_list, t_result)
        t_range_another = ['1995-01-01', '2000-11-01']
        t_result_another = np.array([1995, 1996, 1997, 1998, 1999, 2000])
        np.testing.assert_equal(t_range_years(t_range_another), t_result_another)

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

    def test_t_julian(self):
        t_range = t_range_to_julian(["1985-10-01", "1995-10-01"])
        print(t_range)

    def test_random_choice_no_return(self):
        # arr = [2, 3, 4, 5, 6, 7, 8, 9, 1]
        arr = np.arange(10)
        # arr = ["2", "3", "4", "5", "6", "7", "8", "9", "1"]
        num_lst = [2, 3]
        results = random_choice_no_return(arr, num_lst)
        print(results)

    def test_comb_pair(self):
        # combine_attrs = [{"diversion": [False, True]}, {"DOR": [-0.02, 0.02]}]
        combine_attrs = [{"diversion": [False]}]
        print(pair_comb(combine_attrs))

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

    def setUp(self):
        config_dir = definitions.CONFIG_DIR
        # config_file = os.path.join(config_dir, "transdata/config_exp1.ini")
        # subdir = r"transdata/exp1"
        # config_file = os.path.join(config_dir, "transdata/config_exp2.ini")
        # subdir = r"transdata/exp2"
        # config_file = os.path.join(config_dir, "transdata/config_exp3.ini")
        # subdir = r"transdata/exp3"
        # config_file = os.path.join(config_dir, "transdata/config_exp4.ini")
        # subdir = r"transdata/exp4"
        # config_file = os.path.join(config_dir, "transdata/config_exp5.ini")
        # subdir = r"transdata/exp5"
        # config_file = os.path.join(config_dir, "transdata/config_exp6.ini")
        # subdir = r"transdata/exp6"
        # config_file = os.path.join(config_dir, "transdata/config_exp7.ini")
        # subdir = r"transdata/exp7"
        # config_file = os.path.join(config_dir, "transdata/config_exp8.ini")
        # subdir = r"transdata/exp8"
        # config_file = os.path.join(config_dir, "transdata/config_exp9.ini")
        # subdir = r"transdata/exp9"
        # config_file = os.path.join(config_dir, "transdata/config_exp10.ini")
        # subdir = r"transdata/exp10"
        config_file = os.path.join(config_dir, "transdata/config_exp11.ini")
        subdir = r"transdata/exp11"
        self.config_data = GagesConfig.set_subdir(config_file, subdir)

    def test_data_source(self):
        source_data = GagesSource(self.config_data, self.config_data.model_dict["data"]["tRangeTrain"],
                                  screen_basin_area_huc4=False)
        my_file = os.path.join(self.config_data.data_path["Temp"], 'data_source.txt')
        serialize_pickle(source_data, my_file)

    def test_trans_all_forcing_file_to_camels(self):
        data_source_dump = os.path.join(self.config_data.data_path["Temp"], 'data_source.txt')
        source_data = unserialize_pickle(data_source_dump)
        output_dir = os.path.join(self.config_data.data_path["DB"], "basin_mean_forcing", "daymet")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        region_names = [region_temp.split("_")[-1] for region_temp in source_data.all_configs['regions']]
        # forcing data file generated is named as "allref", so rename the "all"
        region_names = ["allref" if r == "all" else r for r in region_names]
        year_start = int(source_data.t_range[0].split("-")[0])
        year_end = int(source_data.t_range[1].split("-")[0])
        years = np.arange(year_start, year_end)
        assert (all(x < y for x, y in zip(source_data.gage_dict['STAID'], source_data.gage_dict['STAID'][1:])))

        config_dir = definitions.CONFIG_DIR
        for i in range(len(region_names)):
            config_file_i = os.path.join(config_dir, "transdata/config_exp" + str(i + 1) + ".ini")
            subdir_i = "transdata/exp" + str(i + 1)
            config_data_i = GagesConfig.set_subdir(config_file_i, subdir_i)
            source_data_i = GagesSource(config_data_i, config_data_i.model_dict["data"]["tRangeTrain"],
                                        screen_basin_area_huc4=False)
            for year in years:
                trans_daymet_to_camels(source_data.all_configs["forcing_dir"], output_dir, source_data_i.gage_dict,
                                       region_names[i], year)

    def test_insert_leap_year_value(self):
        """interpolation for the 12.31 data in leap year"""
        data_dir = os.path.join(self.config_data.data_path["DB"], "basin_mean_forcing", "daymet")
        subdir_str = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10L", "10U", "11", "12", "13", "14", "15",
                      "16", "17", "18"]
        t_range = ["1980-01-01", "2020-01-01"]
        col_lst = ["dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
        for i in range(len(subdir_str)):
            subdir = os.path.join(data_dir, subdir_str[i])
            path_list = os.listdir(subdir)
            path_list.sort()  # 对读取的路径进行排序
            for filename in path_list:
                data_file = os.path.join(subdir, filename)
                is_leap_file_name = data_file[-8:]
                if "leap" in is_leap_file_name:
                    continue
                print("reading", data_file)
                data_temp = pd.read_csv(data_file, sep=r'\s+')
                data_temp.rename(columns={'Mnth': 'Month'}, inplace=True)
                df_date = data_temp[['Year', 'Month', 'Day']]
                date = pd.to_datetime(df_date).values.astype('datetime64[D]')
                # daymet file not for leap year, there is no data in 12.31 in leap year
                assert (all(x < y for x, y in zip(date, date[1:])))
                t_range_list = hydro_time.t_range_days(t_range)
                [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
                assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
                nt = t_range_list.size
                out = np.full([nt, 7], np.nan)
                out[ind2, :] = data_temp[col_lst].values[ind1]
                x = pd.DataFrame(out, columns=col_lst)
                x_intepolate = x.interpolate(method='linear', limit_direction='forward', axis=0)
                csv_date = pd.to_datetime(t_range_list)
                year_month_day_hour = pd.DataFrame(
                    [[dt.year, dt.month, dt.day, dt.hour] for dt in csv_date], columns=['Year', 'Mnth', 'Day', "Hr"])
                # concat
                new_data_df = pd.concat([year_month_day_hour, x_intepolate], axis=1)
                output_file = data_file[:-4] + "_leap.txt"
                new_data_df.to_csv(output_file, header=True, index=False, sep=' ', float_format='%.2f')
                os.remove(data_file)

    def test_screen_some_gauge_and_save(self):
        config_dir = definitions.CONFIG_DIR
        config_file = os.path.join(config_dir, "transdata/config_exp12.ini")
        subdir = r"transdata/exp12"
        config_data = GagesConfig.set_subdir(config_file, subdir)

        ref_source_data = GagesSource.choose_some_basins(self.config_data,
                                                         self.config_data.model_dict["data"]["tRangeTrain"],
                                                         screen_basin_area_huc4=False, ref="Ref")
        ref_sites_id = ref_source_data.all_configs['flow_screen_gage_id']
        ref_sites_id_df = pd.DataFrame({"STAID": ref_sites_id})
        dapeng_dir = os.path.join(self.config_data.data_path["DB"], "dapeng")
        if not os.path.isdir(dapeng_dir):
            os.makedirs(dapeng_dir)
        dapeng_v2_gageid_file = os.path.join(dapeng_dir, "v2.csv")
        ref_sites_id_df.to_csv(dapeng_v2_gageid_file, index=False)

        gages_model = GagesModels(config_data, screen_basin_area_huc4=False, major_dam_num=0)
        sites_id_df = pd.DataFrame({"STAID": gages_model.data_model_train.t_s_dict["sites_id"]})
        dapeng_v1_gageid_file = os.path.join(dapeng_dir, "v1.csv")
        sites_id_df.to_csv(dapeng_v1_gageid_file, index=False)

        print("read and save data screen")

    def test_choose_some_gauge(self):
        ashu_gageid_file = os.path.join(self.config_data.data_path["DB"], "ashu", "AshuGagesId.txt")

        # farshid_gageid_file = os.path.join(self.config_data.data_path["DB"], "farshid", "sites.csv")
        farshid_gageid_file = os.path.join(self.config_data.data_path["DB"], "farshid", "gagelist1713.feather")

        # dapeng_v1_gageid_file = os.path.join(self.config_data.data_path["DB"], "dapeng", "v1.csv")
        dapeng_v2_gageid_file = os.path.join(self.config_data.data_path["DB"], "dapeng", "v2.csv")

        # gauge_df = pd.read_csv(dapeng_v2_gageid_file, dtype={"STAID": str})
        # gauge_list = gauge_df["STAID"].values
        gauge_df = pd.read_feather(farshid_gageid_file)
        gauge_list = gauge_df["site_no"].values

        # np.array(
        #     ['01013500', '01401650', '01585500', '02120780', '02324400', '03139000', '04086600', '05087500',
        #      '05539900', '06468170', '07184000', '08158810', '09404450', '11055800', '12134500', '14166500'])
        data_dir = os.path.join(self.config_data.data_path["DB"], "basin_mean_forcing", "daymet")
        # output_dir = os.path.join(self.config_data.data_path["DB"], "forcing_data_ashu")
        output_dir = os.path.join(self.config_data.data_path["DB"], "forcing_data_farshid")
        # output_dir = os.path.join(self.config_data.data_path["DB"], "forcing_data_dapeng_v1")
        # output_dir = os.path.join(self.config_data.data_path["DB"], "forcing_data_dapeng_v2")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        data_source_dump = os.path.join(self.config_data.data_path["Temp"], 'data_source.txt')
        source_data = unserialize_pickle(data_source_dump)
        gageids = np.array(source_data.gage_dict['STAID'])
        xy, x_ind, y_ind = np.intersect1d(gauge_list, gageids, return_indices=True)
        index = np.array([np.where(gageids == i) for i in xy]).flatten()
        print(index)
        for j in index:
            huc_id = source_data.gage_dict['HUC02'][j]
            data_huc_dir = os.path.join(data_dir, huc_id)
            src = os.path.join(data_huc_dir, source_data.gage_dict['STAID'][j] + '_lump_daymet_forcing_leap.txt')
            output_huc_dir = os.path.join(output_dir, huc_id)
            if not os.path.isdir(output_huc_dir):
                os.mkdir(output_huc_dir)
            dst = os.path.join(output_huc_dir, source_data.gage_dict['STAID'][j] + '_lump_daymet_forcing_leap.txt')
            print("write into", dst)
            shutil.copy(src, dst)

    def test_trans_susquehanna_forcing_file_to_camels(self):
        config_dir = definitions.CONFIG_DIR
        config_file = os.path.join(config_dir, "transdata/config_exp12.ini")
        subdir = r"transdata/exp12"
        config_data = SusquehannaConfig.set_subdir(config_file, subdir)
        source_data = SusquehannaSource(config_data, config_data.model_dict["data"]["tRangeTrain"])
        output_dir = os.path.join(config_data.data_path["DB"], "basin_mean_forcing_huc10", "daymet")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        year_start = int(source_data.t_range[0].split("-")[0])
        year_end = int(source_data.t_range[1].split("-")[0])
        years = np.arange(year_start, year_end)
        assert (all(x < y for x, y in zip(source_data.gage_dict['HUC10'], source_data.gage_dict['HUC10'][1:])))

        for year in years:
            trans_susquehanna_daymet_to_camels(source_data.all_configs["forcing_dir"], output_dir,
                                               source_data.gage_dict,
                                               "HUC10_Susquehanna", year)

    def test_insert_susquehanan_leap_year_value(self):
        """interpolation for the 12.31 data in leap year"""
        config_dir = definitions.CONFIG_DIR
        config_file = os.path.join(config_dir, "transdata/config_exp12.ini")
        subdir = r"transdata/exp12"
        config_data = SusquehannaConfig.set_subdir(config_file, subdir)
        data_dir = os.path.join(config_data.data_path["DB"], "basin_mean_forcing_huc10", "daymet")
        t_range = ["1980-01-01", "2015-01-01"]
        col_lst = ["dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
        path_list = os.listdir(data_dir)
        path_list.sort()  # 对读取的路径进行排序
        for filename in path_list:
            data_file = os.path.join(data_dir, filename)
            is_leap_file_name = data_file[-8:]
            if "leap" in is_leap_file_name:
                continue
            print("reading", data_file)
            data_temp = pd.read_csv(data_file, sep=r'\s+')
            data_temp.rename(columns={'Mnth': 'Month'}, inplace=True)
            df_date = data_temp[['Year', 'Month', 'Day']]
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            # daymet file not for leap year, there is no data in 12.31 in leap year
            assert (all(x < y for x, y in zip(date, date[1:])))
            t_range_list = hydro_time.t_range_days(t_range)
            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
            nt = t_range_list.size
            out = np.full([nt, 7], np.nan)
            out[ind2, :] = data_temp[col_lst].values[ind1]
            x = pd.DataFrame(out, columns=col_lst)
            x_intepolate = x.interpolate(method='linear', limit_direction='forward', axis=0)
            csv_date = pd.to_datetime(t_range_list)
            year_month_day_hour = pd.DataFrame(
                [[dt.year, dt.month, dt.day, dt.hour] for dt in csv_date], columns=['Year', 'Mnth', 'Day', "Hr"])
            # concat
            new_data_df = pd.concat([year_month_day_hour, x_intepolate], axis=1)
            output_file = data_file[:-4] + "_leap.txt"
            new_data_df.to_csv(output_file, header=True, index=False, sep=' ', float_format='%.2f')
            os.remove(data_file)

    def test_check_streamflow_data(self):
        source_data = GagesSource(self.config_data, self.config_data.model_dict["data"]["tRangeTrain"],
                                  screen_basin_area_huc4=False)
        t_range_list = hydro_time.t_range_days(["1990-01-01", "2010-01-01"])
        # data_temp = source_data.read_usge_gage("01", '01052500', t_range_list)
        data_temp = source_data.read_usge_gage("08", '08013000', t_range_list)
        print(data_temp)
        print(np.argwhere(np.isnan(data_temp)))

    def test_gpu(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda is TITAN
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # cuda is geforce 0
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # cuda is geforce 2
        x = torch.tensor([1., 2.]).cuda()
        # x.device is device(type='cuda', index=0)
        y = torch.tensor([1., 2.]).cuda()
        print(x)
        print(y)

    def test_dataparallel(self):
        # Parameters and DataLoaders
        input_size = 5
        output_size = 2

        batch_size = 30
        data_size = 100
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class RandomDataset(Dataset):

            def __init__(self, size, length):
                self.len = length
                self.data = torch.randn(length, size)

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return self.len

        rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                                 batch_size=batch_size, shuffle=True)

        class Model(nn.Module):
            # Our model

            def __init__(self, input_size, output_size):
                super(Model, self).__init__()
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, input):
                output = self.fc(input)
                print("\tIn Model: input size", input.size(),
                      "output size", output.size())

                return output

        model = Model(input_size, output_size)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # model = nn.DataParallel(model)
            model = nn.DataParallel(model, device_ids=[0, 1])

        model.to(device)
        for data in rand_loader:
            input = data.to(device)
            output = model(input)
            print("Outside: input size", input.size(),
                  "output_size", output.size())


if __name__ == '__main__':
    unittest.main()
