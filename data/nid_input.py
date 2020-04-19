"""a data downloader and formatter for NID dataset"""
import os
import pandas as pd
import definitions
from data.download_data import download_excel
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point

from utils import serialize_pickle
from utils.hydro_util import serialize_geopandas, unserialize_pickle, unserialize_geopandas


def save_nidinput(nid_model, data_path, num_str=None, **kwargs):
    if num_str is not None:
        data_path = os.path.join(data_path, num_str)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    nid_source_file = os.path.join(data_path, kwargs['nid_source_file_name'])
    nid_data_file = os.path.join(data_path, kwargs['nid_data_file_name'])
    serialize_pickle(nid_model.nid_source, nid_source_file)
    serialize_geopandas(nid_model.nid_data, nid_data_file)


class NidConfig(object):
    nidUrl = 'https://nid.sec.usace.army.mil/ords/NID_R.DOWNLOADFILE?InFileName={nidFile}'
    nidDir = os.path.join(definitions.ROOT_DIR, "example", 'data', 'nid')
    # EPSG:4269 --  https://epsg.io/4269
    nidEpsg = 4269

    def __init__(self, nid_file_name):
        if not os.path.isdir(NidConfig.nidDir):
            os.mkdir(NidConfig.nidDir)
        self.nid_url = NidConfig.nidUrl.format(nidFile=nid_file_name)
        self.nid_dir = NidConfig.nidDir
        self.nid_file = os.path.join(NidConfig.nidDir, nid_file_name)
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

    def __init__(self, nid_file='NID2018_U.xlsx', *args):
        """:parameter data_source: DataSource object"""
        if len(args) == 0:
            nid_config = NidConfig(nid_file)
            self.nid_source = NidSource(nid_config)
            self.nid_data = self.nid_source.read_nid()
        else:
            self.nid_source = args[0]
            self.nid_data = args[1]

    @classmethod
    def load_nidmodel(cls, data_path, num_str=None, nid_file='NID2018_U.xlsx', **kwargs):
        if num_str is not None:
            data_path = os.path.join(data_path, num_str)
        nid_source_file = os.path.join(data_path, kwargs['nid_source_file_name'])
        nid_data_file = os.path.join(data_path, kwargs['nid_data_file_name'])
        nid_source = unserialize_pickle(nid_source_file)
        nid_data = unserialize_geopandas(nid_data_file)
        nid_model = cls(nid_file, nid_source, nid_data)
        return nid_model
