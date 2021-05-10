from data.cache.cache_base import DatasetCache
from data.data_scalers import wrap_t_s_dict


def cache_dataset(dataset_params, dataset):
    modes = ["train", "valid", "test"]
    for mode in modes:
        if mode == "valid":
            if dataset_params["t_range_valid"] is None:
                continue
        cache = DatasetCache(dataset_params["test_path"], mode, dataset)
        t_s_dict = wrap_t_s_dict(dataset, dataset_params, mode)
        basins_id = t_s_dict["sites_id"]
        t_range_list = t_s_dict["t_final_range"]
        cache.read_save_dataset(basins_id, t_range_list, flow_cols=dataset_params["target_cols"],
                                attr_cols=dataset_params["constant_cols"],
                                forcing_cols=dataset_params["relevant_cols"],
                                forcing_type=dataset_params["relevant_types"][0],
                                other_cols=dataset_params["other_cols"])
