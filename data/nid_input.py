"""a data downloader and formatter for NID dataset"""
import os
import pandas as pd
import definitions
from data.download_data import download_excel
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point


class NidConfig(object):
    nidFile = 'NID2018_U.xlsx'
    # nidFile = 'PA_U.xlsx'
    # nidFile = 'OH_U.xlsx'
    nidUrl = 'https://nid.sec.usace.army.mil/ords/NID_R.DOWNLOADFILE?InFileName={nidFile}'.format(nidFile=nidFile)
    nidDir = os.path.join(definitions.ROOT_DIR, "example", 'data', 'nid')
    # EPSG:4269 --  https://epsg.io/4269
    nidEpsg = 4269

    def __init__(self):
        if not os.path.isdir(NidConfig.nidDir):
            os.mkdir(NidConfig.nidDir)
        self.nid_url = NidConfig.nidUrl
        self.nid_dir = NidConfig.nidDir
        self.nid_file = os.path.join(NidConfig.nidDir, NidConfig.nidFile)
        self.nid_epsg = NidConfig.nidEpsg


class NidSource(object):

    def __init__(self, config_data):
        """read configuration of data source. 读取配置，准备数据，关于数据读取部分，可以放在外部需要的时候再执行"""
        self.data_config = config_data
        self.prepare_data()

    def prepare_data(self):
        download_excel(self.data_config.nid_url, self.data_config.nid_file)

    def read_nid(self):
        df = pd.read_excel(self.data_config.nid_file)
        """transform data to geopandas"""
        data = gpd.GeoDataFrame(df, crs=CRS.from_epsg(self.data_config.nid_epsg).to_wkt())
        data['geometry'] = None
        for idx in range(df.shape[0]):
            # create a point based on x and y column values on this row:
            point = Point(df['LONGITUDE'][idx], df['LATITUDE'][idx])
            # Add the point object to the geometry column on this row:
            data.at[idx, 'geometry'] = point

        return data


class NidModel(object):
    """data formatter， utilizing function of DataSource object to read data and transform"""

    def __init__(self):
        """:parameter data_source: DataSource object"""
        nid_config = NidConfig()
        self.nid_source = NidSource(nid_config)
        self.nid_data = self.nid_source.read_nid()
