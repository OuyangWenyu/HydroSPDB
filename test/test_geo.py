import os
import unittest
import wget
import definitions
from data.download_data import download_one_zip
from explore.stat import statError1d
from utils import serialize_numpy, unserialize_numpy
from utils.hydro_geo import shps_trans_coord, basin_avg_netcdf, trans_points
import numpy as np
from pyproj import Proj
import pandas as pd


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = "camels"
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)

    zip_file = os.path.join(dir_db, "basin_set_full_res.zip")
    unzip_dir = os.path.join(dir_db, "basin_set_full_res")
    shpfile_folder = os.path.join(dir_db, "basin_set_full_res_wgs84")
    netcdf_dir = os.path.join(dir_db, "daymet_netcdf")
    year = 2000
    par = 'tmax'
    netcdf_file = os.path.join(netcdf_dir, "daymet_v3_{par}_{year}_na.nc4".format(year=year, par=par))

    def test_download_camels_shp(self):
        camels_shp_url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_set_full_res.zip"
        data_dir = self.dir_db
        download_one_zip(camels_shp_url, data_dir)

    def test_trans_points(self):
        outProj = Proj(init='epsg:4326')
        outProj_another = Proj(init='epsg:4269')
        pxs = np.full([100000], -77.862487)
        pys = np.full([100000], 40.79461)
        pxys_out = trans_points(outProj_another, outProj, pxs, pys)
        print(pxys_out)

    def test_shp_trans_coord(self):
        """首先读取shapefile文件，然后对每个polygon的每个坐标点进行坐标变换，接着再重新构建一个个的shapefile
        程序需要优化，几个方面：
        1.把能放到循环外的都放到循环外处理
        2.for循环用更贴近C的map等代替
        3.查查有没有直接转换一组点坐标的方法
        4.并行计算算法
        5.把shapefile在arcgis上拆解后，多找几个电脑算
        6.重新用arcgis弄
        """
        input_folder = r"../../example/data/gages/boundaries-shapefiles-by-aggeco"
        input_folder = self.unzip_dir
        output_folder = r"../../example/data/gages/gagesII_basin_shapefile_wgs84"
        output_folder = self.shpfile_folder
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        shps_trans_coord(input_folder, output_folder)

    def test_download_netcdf(self):
        year = self.year
        par = self.par
        url_pattern = "https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/1328/{year}/daymet_v3_{par}_{year}_na.nc4"
        url = url_pattern.format(year=year, par=par)
        # 下载到指定文件夹内
        if os.path.isfile(self.netcdf_file):
            print("data downloaded...")
        else:
            wget.download(url, self.netcdf_dir)

    def test_basin_avg_netcdf(self):
        """读取netcdf文件，计算给定的shapefile代表的范围内该netcdf文件的给定变量的流域平均值
        算法需要优化：
        1.判断区域那块，可以根据bound迅速地排除大部分不需要判断的点，只判断在bound内的点
        2.其他的优化和shp_trans_coord下的差不多
        """
        # 先读取一个netcdf文件，然后把shapefile选择一张，先测试下上面的程序。
        file_path = self.netcdf_file
        shp_file = os.path.join(self.shpfile_folder, "01013500.shp")
        mask_file = os.path.join(self.shpfile_folder, "mask_01013500")
        avgs = basin_avg_netcdf(file_path, shp_file, mask_file)
        daymet_myself_file = os.path.join(self.netcdf_dir, "daymet_01013500_mean_2000_myself")
        serialize_numpy(np.array(avgs), daymet_myself_file)

    def test_daymet_avg_from_diff(self):

        daymet_myself_file = os.path.join(self.netcdf_dir, "daymet_01013500_mean_2000_myself.npy")
        myself_data_tmax = unserialize_numpy(daymet_myself_file)

        camels_data = pd.read_csv(os.path.join(self.dir_db,
                                               "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/01/01013500_lump_cida_forcing_leap.txt"),
                                  sep=r'\s+', header=None, skiprows=4)
        camels_data_tmax = camels_data[8].values[7305:7670]

        gee_data = pd.read_csv(os.path.join(self.netcdf_dir, "daymet_01013500_mean_2000.csv"))
        gee_data_tmax = gee_data["tmax"].values
        print()
        print("Bias, RMSE, NSE", statError1d(myself_data_tmax, camels_data_tmax))
        print("Bias, RMSE, NSE", statError1d(myself_data_tmax, gee_data_tmax))
        print("Bias, RMSE, NSE", statError1d(camels_data_tmax, gee_data_tmax))
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")

        values = np.array([myself_data_tmax, camels_data_tmax, gee_data_tmax]).T
        print(values)
        dates = pd.date_range("1 1 2000", periods=365, freq="D")
        data = pd.DataFrame(values, dates, columns=["myself", "camels", "gee"])
        print(data)
        sns.lineplot(data=data.iloc[:, 0:2], palette="tab10", linewidth=2.5)
        plt.show()
        sns.lineplot(data=data.iloc[:, 1:3], palette="tab10", linewidth=2.5)
        plt.show()


if __name__ == '__main__':
    unittest.main()
