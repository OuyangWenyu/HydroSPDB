"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-20 17:42:37
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /HydroSPDB/hydrospdb/data/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrospdb.data.source.data_camels import Camels

# more data types which cannot be easily treated same with attribute or forcing data
from hydrospdb.data.loader.data_loaders import (
    BasinFlowDataModel,
    XczDataModel,
)
from hydrospdb.data.loader.data_sets import (
    BasinFlowDataset,
    BasinSingleFlowDataset,
)
from hydrospdb.data.source.data_gages import Gages

other_data_source_list = ["RES_STOR_HIST", "GAGES_TS", "FDC", "DI", "WATER_SURFACE"]

data_sources_dict = {
    "CAMELS": Camels,
    "GAGES": Gages,
}

dataloaders_dict = {
    "StreamflowDataset": BasinFlowDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "StreamflowDataModel": BasinFlowDataModel,
    "KernelFlowDataModel": XczDataModel,
}
