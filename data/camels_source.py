import pandas as pd
from data import DataSource
from data.download_data import download_one_zip
from utils import *


class CamelsSource(DataSource):
    def __init__(self, config_data, t_range):
        super().__init__(config_data, t_range)

    def read_site_info(self, **kwargs):
        """根据配置读取所需的gages-ii站点信息及流域基本location等信息。
        从中选出field_lst中属性名称对应的值，存入dic中。
                    # using shapefile of all basins to check if their basin area satisfy the criteria
                    # read shpfile from data directory and calculate the area
        param **kwargs: none
        Return：
            各个站点的attibutes in basinid.txt

        """
        gage_file = self.all_configs["gauge_id_file"]

        data = pd.read_csv(gage_file, sep='\t', header=None, skiprows=1)
        # header gives some troubles. Skip and hardcode
        field_lst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
        out = dict()
        for s in field_lst:
            if s is 'name':
                out[s] = data[field_lst.index(s)].values.tolist()
            else:
                out[s] = data[field_lst.index(s)].values
        return out, field_lst

    def prepare_forcing_data(self):
        configs = self.all_configs
        data_dir = configs.get('root_dir')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        forcing_url = configs.get('forcing_url')
        download_one_zip(forcing_url, data_dir)
