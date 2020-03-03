"""获取源数据，源数据不考虑格式，只是最原始所需下载的数据，先以gages数据集测试编写，后面其他数据集采用继承方式修改"""

# 数据类型包括：径流数据（从usgs下载），forcing数据（从daymet或者nldas下载），属性数据（从usgs属性表读取）
# 定义选择哪些源数据
import os

from data.download_data import download_one_zip


class DataSource(object):
    """获取源数据的思路是：
    首先准备好属性文件，主要是从网上下载获取；
    然后读取配置文件及相关属性文件了解到模型计算的对象；
    接下来获取forcing数据和streamflow数据
    """

    def __init__(self, config_data, t_range, screen_basin_area_huc4=True):
        """read configuration of data source. 读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        self.data_config = config_data
        self.all_configs = config_data.read_data_config()
        # t_range: 训练数据还是测试数据，需要外部指定
        self.t_range = t_range
        self.prepare_attr_data()
        self.prepare_forcing_data()
        gage_dict, gage_fld_lst = self.read_site_info(screen_basin_area_huc4=screen_basin_area_huc4)
        self.prepare_flow_data(gage_dict, gage_fld_lst)
        # 一些后面常用的变量也在这里赋值到SourceData对象中
        self.gage_dict = gage_dict
        self.gage_fld_lst = gage_fld_lst

    def prepare_attr_data(self):
        """根据时间读取数据，没有的数据下载"""
        configs = self.all_configs
        data_dir = configs.get('root_dir')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        attr_urls = configs.get('attr_url')
        [download_one_zip(attr_url, data_dir) for attr_url in attr_urls]
        print("attribute data Ready! ...")

    def read_site_info(self, ids_specific=None, screen_basin_area_huc4=True):
        """read basic information of sites"""
        print("get infomation of sites...")

    def prepare_forcing_data(self):
        """DOWNLOAD forcing data from website"""
        print("forcing data Ready! ...")

    def prepare_flow_data(self, gage_dict, gage_fld_lst):
        """download streamflow data"""
        print("streamflow data Ready! ...")

    def read_usgs(self):
        """read streamflow data"""
        pass

    def usgs_screen_streamflow(self, streamflow, usgs_ids=None, time_range=None):
        """choose some gauges"""
        pass

    def read_attr(self, usgs_id_lst, var_lst):
        """read attributes data"""
        pass

    def read_forcing(self, usgs_id_lst, t_range_lst):
        """read forcing data"""
        pass
