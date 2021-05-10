"""a data downloader and formatter for NID dataset"""
import collections
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point

from data.data_base import DatasetBase
from utils.hydro_utils import serialize_pickle, serialize_geopandas, unserialize_pickle, unserialize_geopandas, \
    download_excel


class Nid(DatasetBase):
    """Dataset of NID"""

    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.dataset_description = self.set_dataset_describe()
        if download:
            self.download_dataset()
        self.init_cache()

    def init_cache(self):
        """stor the sheet data to binary format"""
        if os.path.isfile(self.dataset_description["NID_GEO_FILE"]):
            print("cache has existed")
        else:
            df = pd.read_excel(self.dataset_description["NID_FILE"])
            """transform data to geopandas"""
            data = gpd.GeoDataFrame(df, crs=CRS.from_epsg(self.dataset_description["NID_EPSG"]).to_wkt())
            data['geometry'] = None
            for idx in range(df.shape[0]):
                # create a point based on x and y column values on this row:
                point = Point(df['LONGITUDE'][idx], df['LATITUDE'][idx])
                # Add the point object to the geometry column on this row:
                data.at[idx, 'geometry'] = point
            serialize_geopandas(data, self.dataset_description["NID_GEO_FILE"])

    def get_name(self):
        return "NID"

    def set_dataset_describe(self):
        nid_db = self.dataset_dir
        nid_file = os.path.join(nid_db, "NID2018_U.xlsx")
        nid_geo_file = os.path.join(nid_db, "dam_points.geojson")
        download_url = 'https://nid.sec.usace.army.mil/ords/NID_R.DOWNLOADFILE?InFileName={nidFile}'.format(
            nidFile="NID2018_U.xlsx")
        # EPSG:4269 --  https://epsg.io/4269
        nid_epsg = 4269
        return collections.OrderedDict(NID_DIR=nid_db, NID_FILE=nid_file, NID_GEO_FILE=nid_geo_file, NID_EPSG=nid_epsg,
                                       NID_URL=download_url)

    def download_dataset(self):
        # TODO: the USACE website connot be accessed now
        nid_config = self.dataset_description
        if not os.path.isdir(nid_config["NID_DIR"]):
            os.makedirs(nid_config["NID_DIR"])
        if not os.path.isfile(nid_config["NID_FILE"]):
            download_excel(nid_config["NID_URL"], nid_config["NID_FILE"])
        print("The NID dataset has been downloaded!")

    def read_object_ids(self, object_params=None) -> np.array:
        pass

    def read_target_cols(self, object_ids=None, t_range_list=None, target_cols=None, **kwargs) -> np.array:
        # no target cols now
        pass

    def read_relevant_cols(self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs) -> np.array:
        # no relevant cols now
        pass

    def read_constant_cols(self, object_ids=None, constant_cols=None, **kwargs) -> np.array:
        if os.path.isfile(self.dataset_description["NID_GEO_FILE"]):
            df = unserialize_geopandas(self.dataset_description["NID_GEO_FILE"])
        else:
            df = pd.read_excel(self.dataset_description["NID_FILE"])
        return df

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        pass

    def get_constant_cols(self):
        if os.path.isfile(self.dataset_description["NID_GEO_FILE"]):
            df = unserialize_geopandas(self.dataset_description["NID_GEO_FILE"])
        else:
            df = pd.read_excel(self.dataset_description["NID_FILE"])
        return df.columns

    def get_relevant_cols(self):
        pass

    def get_target_cols(self):
        pass

    def get_other_cols(self):
        pass
