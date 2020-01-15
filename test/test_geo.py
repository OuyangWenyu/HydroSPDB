import os
import unittest

import definitions
from data.download_data import download_one_zip
from utils.hydro_geo import shps_trans_coord, basin_avg_netcdf, trans_points
import numpy as np
from pyproj import Proj


class MyTestCase(unittest.TestCase):
    config_file = definitions.CONFIG_FILE
    project_dir = definitions.ROOT_DIR
    dataset = "camels"
    dir_db = os.path.join(project_dir, 'example/data', dataset)
    dir_out = os.path.join(project_dir, 'example/output', dataset)
    dir_temp = os.path.join(project_dir, 'example/temp', dataset)

    zip_file = os.path.join(dir_db, "basin_set_full_res.zip")
    unzip_dir = os.path.join(dir_db, "basin_set_full_res")

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
        output_folder = os.path.join(self.dir_db, "basin_set_full_res_wgs84")
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        shps_trans_coord(input_folder, output_folder)

    def test_basin_avg_netcdf(self):
        """读取netcdf文件，计算给定的shapefile代表的范围内该netcdf文件的给定变量的流域平均值
        算法需要优化：
        1.判断区域那块，可以根据bound迅速地排除大部分不需要判断的点，只判断在bound内的点
        2.其他的优化和shp_trans_coord下的差不多
        """
        # 先读取一个netcdf文件，然后把shapefile选择一张，先测试下上面的程序。
        # Define path to folder，以r开头表示相对路径

        input_folder = r"examples_data"

        # Join folder path and filename
        netcdf_file = "daymet_v3_prcp_2000_na.nc4"
        file_path = os.path.join(input_folder, netcdf_file)
        shp_file = os.path.join(input_folder, "03144816.shp")
        basin_avg_netcdf(file_path, shp_file)


if __name__ == '__main__':
    unittest.main()
