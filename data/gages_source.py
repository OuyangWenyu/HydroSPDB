from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
from data import DataSource
from data.download_data import download_small_file, download_google_drive
from utils import *


class GagesSource(DataSource):
    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)

    def read_site_info(self, ids_specific=None, screen_basin_area_huc4=True):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area

        Parameter:
            ids_specific： given sites' ids
            screen_basin_area_huc4: 是否取出流域面积大于等于所处HUC流域的面积的流域
        Return：
            各个站点的attibutes in basinid.txt
        """
        # 数据从第二行开始，因此跳过第一行。
        gage_id_file = self.all_configs.get("gage_id_file")
        points_file = self.all_configs.get("gage_point_file")
        huc4_shp_file = self.all_configs.get("huc4_shp_file")
        gage_region_dir = self.all_configs.get("gage_region_dir")
        region_shapefiles = self.all_configs.get("regions")
        data = pd.read_csv(gage_id_file, sep=',', dtype={0: str})
        gage_fld_lst = data.columns.values
        out = dict()
        if len(region_shapefiles):
            # read sites from shapefile of region, get id from it.
            # Read file using gpd.read_file() TODO:多个regins情况还未完成
            shapefile = os.path.join(gage_region_dir, region_shapefiles[0] + '.shp')
            shape_data = gpd.read_file(shapefile)
            print(shape_data.columns)
            gages_id = shape_data['GAGE_ID'].values
            if screen_basin_area_huc4:
                # using shapefile of all basins to check if their basin area satisfy the criteria
                # remove stations with catchment areas greater than the HUC4 basins in which they are located
                # firstly, get the HUC4 basin's area of the site
                print("screen big area basins")
                join_points = spatial_join(points_file, huc4_shp_file)
                # get "AREASQKM" attribute data to filter
                join_points = join_points[join_points["DRAIN_SQKM"] < join_points["AREASQKM"]]
                gages_huc4_id = join_points['STAID'].values
                gages_id, ind1, ind2 = np.intersect1d(gages_id, gages_huc4_id, return_indices=True)
            df_id_region = data.iloc[:, 0].values
            c, ind1, ind2 = np.intersect1d(df_id_region, gages_id, return_indices=True)
            data = data.iloc[ind1, :]
        if ids_specific:
            df_id_test = data.iloc[:, 0].values
            c, ind1, ind2 = np.intersect1d(df_id_test, ids_specific, return_indices=True)
            data = data.iloc[ind1, :]
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data[s].values.tolist()
            else:
                out[s] = data[s].values
        return out, gage_fld_lst

    def prepare_forcing_data(self):
        """如果没有给url或者查到没有数据，就只能报错了，提示需要手动下载. 可以使用google drive下载数据"""
        url_forcing = self.all_configs.get("forcing_url")
        if url_forcing is None:
            print("Downloading dataset from google drive directly...")
            # 个人定义的：google drive上的forcing数据文件夹名和forcing类型一样的
            dir_name = self.all_configs.get("forcing_type")
            download_dir_name = self.all_configs.get("forcing_dir")
            if not os.path.isdir(download_dir_name):
                os.mkdir(download_dir_name)
            # 如果已经有了数据，那么就不必再下载了
            regions = self.all_configs["regions"]
            regions_shps = [r.split('_')[-1] for r in regions]
            year_range_list = hydro_time.t_range_years(self.t_range)
            # 如果有某个文件没有，那么就下载数据
            shp_files_now = []
            for f_name in os.listdir(download_dir_name):
                if f_name.endswith('.csv'):
                    shp_files_now.append(f_name)
            is_download = False
            for r_shp in regions_shps:
                r_files = [dir_name + "_" + r_shp + "_mean_" + str(t_range_temp) + ".csv" for t_range_temp in
                           year_range_list]
                r_file_is_download = []
                for r_file_temp in r_files:
                    if r_file_temp not in shp_files_now:
                        is_download_temp = True
                        r_file_is_download.append(is_download_temp)
                if True in r_file_is_download:
                    is_download = True
                    break

            if is_download:
                # 然后下载数据到这个文件夹下，这里从google drive下载数据
                download_google_drive(dir_name, download_dir_name)
            else:
                print("forcing数据已经下载好了")
        print("forcing数据准备完毕")

    def prepare_flow_data(self, gage_dict, gage_fld_lst):
        """检查数据是否齐全，不够的话进行下载，下载数据的时间范围要设置的大一些，这里暂时例子都是以1980-01-01到2015-12-31
        parameters:
            gage_dict: read_gage_info返回值--out
            gage_dict: read_gage_info返回值--gage_fld_lst
        """
        dir_gage_flow = self.all_configs.get("flow_dir")
        streamflow_url = self.all_configs.get("flow_url")
        t_download_range = self.all_configs.get("t_range_all")
        if not os.path.isdir(dir_gage_flow):
            os.mkdir(dir_gage_flow)
        dir_list = os.listdir(dir_gage_flow)
        # 区域一共有18个，为了便于后续处理，把属于不同region的站点的文件放到不同的文件夹下面
        # 判断usgs_id_lst中没有对应径流文件的要从网上下载
        usgs_id_lst = gage_dict[gage_fld_lst[0]]
        for ind in range(len(usgs_id_lst)):
            # different hucs different directories
            huc_02 = gage_dict[gage_fld_lst[3]][ind]
            dir_huc_02 = str(huc_02)
            if dir_huc_02 not in dir_list:
                dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
                os.mkdir(dir_huc_02)
                dir_list = os.listdir(dir_gage_flow)
            dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
            file_list = os.listdir(dir_huc_02)
            file_usgs_id = str(usgs_id_lst[ind]) + ".txt"
            if file_usgs_id not in file_list:
                # 通过直接读取网页的方式获取数据，然后存入txt文件
                start_time_str = datetime.strptime(t_download_range[0], '%Y-%m-%d')
                end_time_str = datetime.strptime(t_download_range[1], '%Y-%m-%d') - timedelta(days=1)
                url = streamflow_url.format(usgs_id_lst[ind], start_time_str.year, start_time_str.month,
                                            start_time_str.day, end_time_str.year, end_time_str.month, end_time_str.day)

                # 存放的位置是对应HUC02区域的文件夹下
                temp_file = os.path.join(dir_huc_02, str(usgs_id_lst[ind]) + '.txt')
                download_small_file(url, temp_file)
                print("成功写入 " + temp_file + " 径流数据！")
        print("径流量数据准备好了...")
