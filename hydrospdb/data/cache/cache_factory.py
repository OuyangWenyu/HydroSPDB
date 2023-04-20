"""
Author: Wenyu Ouyang
Date: 2021-12-17 18:02:27
LastEditTime: 2023-04-20 17:52:44
LastEditors: Wenyu Ouyang
Description: serialize data so that we can access them quickly
FilePath: /HydroSPDB/hydrospdb/data/cache/cache_factory.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydrospdb.data.cache.cache_base import DataSourceCache
from hydrospdb.data.loader.data_scalers import wrap_t_s_dict


def cache_data_source(data_params, data_source):
    """
    Save data from source so that we can read data quickly

    Parameters
    ----------
    data_params
        data parameters
    data_source
        instance of data source class

    Returns
    -------
    None
    """
    modes = ["train", "valid", "test"]
    for mode in modes:
        if mode == "valid":
            if data_params["t_range_valid"] is None:
                continue
        cache = DataSourceCache(data_params["test_path"], mode, data_source)
        t_s_dict = wrap_t_s_dict(data_source, data_params, mode)
        basins_id = t_s_dict["sites_id"]
        t_range_list = t_s_dict["t_final_range"]
        cache.read_save_data_source(
            basins_id,
            t_range_list,
            target_cols=data_params["target_cols"],
            attr_cols=data_params["constant_cols"],
            forcing_cols=data_params["relevant_cols"],
            forcing_type=data_params["relevant_types"][0],
            other_cols=data_params["other_cols"],
        )
