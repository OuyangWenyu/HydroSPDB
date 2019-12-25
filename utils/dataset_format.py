"""
Define 数据处理所需的一些基础类型工具类型的函数
"""
import fnmatch
import zipfile

import numpy as np
import pandas as pd
import glob
import os


def trans_nldas_forcing_file_to_camels():
    """transform forcing data of nldas formats to the one in camels dataset"""
    # you can add features or delete features, or change the order, which depends on your txt content
    your_dataset = ['temperature', "total_precipitation", "pressure", "wind_u", "wind_v", "longwave_radiation",
                    "convective_fraction", "potential_energy"]

    csv_ = glob.glob('_csv/*.csv')
    num_basin = len(csv_)  # the number of basin
    num_your_dataset = int(len(your_dataset))

    # don't change!
    name_dataset = ['temperature', "total_precipitation", "pressure", "wind_u", "wind_v", "longwave_radiation",
                    "convective_fraction", "potential_energy", "potential_evaporation", "shortwave_radiation"]
    for i_basin in range(num_basin):
        # name csv
        name_csv = 'basin' + str(i_basin) + '_s.csv'
        # read csv by data
        csv_data = pd.read_csv('_csv/' + name_csv)
        # create an empty DataFrame
        data_df = pd.DataFrame()

        # get Year,Month,Day,Hour info, DataFrame format
        csv_date = csv_data['system:index'].str[1:12]
        csv_year = pd.to_numeric(csv_date.str[0:4])
        data_df['Year'] = csv_year
        csv_month = pd.to_numeric(csv_date.str[4:6])
        data_df['Month'] = csv_month
        csv_day = pd.to_numeric(csv_date.str[6:8])
        data_df['Day'] = csv_day
        csv_hour = pd.to_numeric(csv_date.str[9:11])
        data_df['Hour'] = csv_hour

        # get features info
        data = csv_data['Meann'].str.split(',', expand=True)
        # split them
        temperature = pd.to_numeric(data[0].str.split('=', expand=True)[1])
        total_precipitation = pd.to_numeric(data[1].str.split('=', expand=True)[1])
        pressure = pd.to_numeric(data[2].str.split('=', expand=True)[1])
        wind_u = pd.to_numeric(data[3].str.split('=', expand=True)[1])
        wind_v = pd.to_numeric(data[4].str.split('=', expand=True)[1])
        longwave_radiation = pd.to_numeric(data[5].str.split('=', expand=True)[1])
        convective_fraction = pd.to_numeric(data[6].str.split('=', expand=True)[1])
        potential_energy = pd.to_numeric(data[7].str.split('=', expand=True)[1])
        potential_evaporation = pd.to_numeric(data[8].str.split('=', expand=True)[1])
        shortwave_radiation = data[9].str.split('=', expand=True)[1]
        shortwave_radiation = pd.to_numeric(shortwave_radiation.str.split('}', expand=True)[0])

        # add your feature in the DataFrame
        for i_database in range(num_your_dataset):
            data_df[your_dataset[i_database]] = eval(your_dataset[i_database])

        # output the result(hourly data)
        data_df.to_csv('result_hour/basin_' + str(i_basin) + '_result.txt', header=True, index=False, sep=' ',
                       float_format='%.2f')

        # # convert hourly data to daily data
        # index for convert (from hour to day by mean or sum method)
        index_mean = ['temperature', 'specific_humidity', 'pressure', 'wind_u', 'wind_v',
                      'longwave_radiation', 'convective_fraction', 'potential_energy',
                      'shortwave_radiation']  # mean
        index_sum = ['potential_evaporation', 'total_precipitation']  # sum
        # the number of days
        num_day = int(len(data_df) / 24)
        # create an empty DataFrame to contain daily data
        data_day_df = pd.DataFrame()
        Year_day = []
        Month_day = []
        Day_day = []
        Hour_day = []
        # storage date info firstly
        for i in range(num_day):
            # get the daily date
            Year_day.append(csv_year[0 + 24 * i])
            Month_day.append(csv_month[0 + 24 * i])
            Day_day.append(csv_day[0 + 24 * i])
            Hour_day.append(csv_hour[0 + 24 * i])

        data_day_df['Year'] = Year_day
        data_day_df['Month'] = Month_day
        data_day_df['Day'] = Day_day
        data_day_df['Hour'] = Hour_day

        # convert hourly to daily by sum or mean, which depends on index_mean and index_sum
        for i_database in range(num_your_dataset):
            data_day = []
            if your_dataset[i_database] in index_mean:
                for i in range(num_day):
                    i_data_df = data_df[your_dataset[i_database]]
                    data_day.append(i_data_df[0 + 24 * i:24 + 24 * i].mean())

            elif your_dataset[i_database] in index_sum:
                for i in range(num_day):
                    i_data_df = data_df[your_dataset[i_database]]
                    data_day.append(i_data_df[0 + 24 * i:24 + 24 * i].sum())
                    # storage daily data
            data_day_df[your_dataset[i_database]] = data_day
        # output daily data
        data_day_df.to_csv('result_day/basin_' + str(i_basin) + '_result.txt', header=True, index=False, sep=' ',
                           float_format='%.2f')


def trans_daymet_forcing_file_to_camels(daymet_dir, output_dir):
    """transform forcing data of daymet formats to the one in camels dataset"""
    # don't change!
    name_dataset = ['gage_id', "time_start", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    camels_index = ['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)',
                    'tmin(C)', 'vp(Pa)']
    # The first three lines are:
    #   latitude of gauge
    #   elevation of gauge (m)
    #   area of basin (m^2)
    # find all dirs and files  TODO: all files
    for entry in os.listdir(daymet_dir):
        dir = os.path.join(daymet_dir, entry)
        if os.path.isdir(dir):
            print(dir)
            for f_name in os.listdir(dir):
                if fnmatch.fnmatch(f_name, '*.csv'):
                    print(f_name)
                    csv_ = pd.read_csv(f_name)
                    all_gages_ids = np.unique(csv_[name_dataset[0]].values)
                    all_periods = np.unique(csv_[name_dataset[1]].values)
                    all_data = csv_.values.reshape(all_gages_ids.size, all_periods.size, len(name_dataset))
                    for i_basin in range(all_gages_ids.size):
                        # name csv
                        basin_data = all_data[i_basin]
                        # get Year,Month,Day,Hour info
                        csv_date = basin_data[:, 1]
                        year_month_day_hour = [[dt.year, dt.month, dt.day, dt.hour] for dt in csv_date]
                        # concat arrs to a new one
                        new_arr = np.concatenate((year_month_day_hour, basin_data[:, 2:]), axis=1)
                        # create a DataFrame
                        data_df = pd.DataFrame(new_arr, columns=camels_index)
                        # output the result(hourly data)
                        output_file = os.path.join(output_dir, str(i_basin) + '_forcing.txt')
                        data_df.to_csv(output_file, header=True, index=False, sep=' ', float_format='%.2f')


def unzip_file(dataset_zip, path_unzip):
    """TODO 递归操作（文件夹内的压缩文件也要解压）：把zip文件 dataset_zip 解压到 path_unzip 文件夹下"""
    with zipfile.ZipFile(dataset_zip, 'r') as zip_temp:
        zip_temp.extractall(path_unzip)


def index2d(ind, ny, nx):
    iy = np.floor(ind / nx)
    ix = np.floor(ind % nx)
    return int(iy), int(ix)


def fillNan(mat, mask):
    temp = mat.copy()
    temp[~mask] = np.nan
    return temp
