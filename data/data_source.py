"""download and read source data. parent class"""

# Data type：streamflow data（download from USGS），forcing data（from Daymet），attr data（from USGS GAGES-II）
import os

from data.download_data import download_one_zip


class DataSource(object):
    """Read config file, then prepare data in the correct dir"""
    def __init__(self, config_data, t_range, screen_basin_area_huc4=True):
        """read configuration of data source"""
        self.data_config = config_data
        self.all_configs = config_data.read_data_config()
        # t_range: training or test
        self.t_range = t_range
        gage_dict, gage_fld_lst = self.read_site_info(screen_basin_area_huc4=screen_basin_area_huc4)
        self.gage_dict = gage_dict
        self.gage_fld_lst = gage_fld_lst

    def read_site_info(self, screen_basin_area_huc4=True):
        """read basic information of sites"""
        print("get infomation of sites...")

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
