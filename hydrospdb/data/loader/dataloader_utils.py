from functools import wraps

import torch

from hydrospdb.data.cache.cache_base import DataSourceCache
from hydrospdb.data.loader.data_scalers import wrap_t_s_dict
from hydrospdb.data.source.data_base import DataSourceBase
from hydrospdb.utils.hydro_utils import check_np_array_nan


@check_np_array_nan
def read_yxc(data_source: DataSourceBase, data_params: dict, loader_type: str) -> tuple:
    """
    Read output, dynamic inputs and static inputs

    Parameters
    ----------
    data_source
        object for reading source data
    data_params
        parameters for reading source data
    loader_type
        train, vaild or test

    Returns
    -------
    tuple [np.array]
        data_flow, data_forcing, data_attr
    """
    t_s_dict = wrap_t_s_dict(data_source, data_params, loader_type)
    basins_id = t_s_dict["sites_id"]
    t_range_list = t_s_dict["t_final_range"]
    target_cols = data_params["target_cols"]
    relevant_cols = data_params["relevant_cols"]
    constant_cols = data_params["constant_cols"]
    cache_read = data_params["cache_read"]
    if "other_cols" in data_params.keys():
        other_cols = data_params["other_cols"]
    else:
        other_cols = None
    if cache_read:
        # Don't wanna the cache impact the implemention of data_sources' read_xxx functions
        # Hence, here we follow "Convention over configuration", and set the cache files' name in DataSourceCache
        data_source_cache = DataSourceCache(
            data_params["cache_path"], loader_type, data_source
        )
        caches = data_source_cache.load_data_source()
        data_dict = caches[3]
        # judge if the configs are correct
        if not (
            basins_id == data_dict[data_source_cache.key_sites]
            and t_range_list == data_dict[data_source_cache.key_t_range]
            and target_cols == data_dict[data_source_cache.key_target_cols]
            and relevant_cols == data_dict[data_source_cache.key_relevant_cols]
            and constant_cols == data_dict[data_source_cache.key_constant_cols]
        ):
            raise RuntimeError(
                "You chose a wrong cache, please set cache_write=1 to get correct cache or just set cache_read=0"
            )
        if other_cols is not None:
            for key, value in other_cols.items():
                assert value == data_dict[data_source_cache.key_other_cols][key]
            return caches[0], caches[1], caches[2], caches[4]
        return caches[0], caches[1], caches[2]
    if "relevant_types" in data_params.keys():
        forcing_type = data_params["relevant_types"][0]
    else:
        forcing_type = None
    # read streamflow
    data_flow = data_source.read_target_cols(basins_id, t_range_list, target_cols)
    # read forcing
    data_forcing = data_source.read_relevant_cols(
        basins_id, t_range_list, relevant_cols, forcing_type=forcing_type
    )
    # read attributes
    data_attr = data_source.read_constant_cols(basins_id, constant_cols)
    if other_cols is not None:
        # read other data
        data_other = data_source.read_other_cols(basins_id, other_cols=other_cols)
        return data_flow, data_forcing, data_attr, data_other
    return data_flow, data_forcing, data_attr


def check_data_loader(func):
    """
    Check if the data loader will load an input and output with NaN.
    If NaN exist in inputs, raise an Error;
    If all elements in output are NaN, also raise an Error

    Parameters
    ----------
    func
        function to run

    Returns
    -------
    function(*args, **kwargs)
        the wrapper
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        for a_tensor in result[:-1]:
            if torch.isnan(a_tensor).any():
                raise ValueError(
                    "We don't support an input with NaN value for deep learning models;\n"
                    "Please check your input data"
                )
        if torch.isnan(result[-1]).all():
            raise ValueError(
                "We don't support an output with all NaN value for deep learning models;\n"
                "Please check your output data"
            )
        return result

    return wrapper
