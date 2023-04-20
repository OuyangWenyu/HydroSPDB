"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-04-20 17:55:52
LastEditors: Wenyu Ouyang
Description: Select sites according to some conditions
FilePath: /HydroSPDB/hydrospdb/data/source_pro/select_gages_ids.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np

from hydrospdb.data.source.data_gages import Gages
from hydrospdb.data.source.data_constant import Q_CAMELS_US_NAME


def usgs_screen_streamflow(
    gages: Gages,
    usgs_ids: list,
    time_range: list,
    flow_type=Q_CAMELS_US_NAME,
    **kwargs
) -> list:
    """
    according to the criteria and its ancillary condition--thresh of streamflow data,
    choose appropriate ones from the given usgs sites

    Parameters
    ----------
    gages
        Camels, CamelsSeries, Gages or GagesPro object
    usgs_ids: list
        given sites' ids
    time_range: list
        chosen time range
    flow_type
        flow's name in data file; default is usgsFlow for CAMELS-US
    kwargs
        all criteria

    Returns
    -------
    list
        sites_chosen: [] -- ids of chosen gages

    Examples
    --------
        >>> usgs_screen_streamflow(gages, ["02349000","08168797"], ["1995-01-01","2015-01-01"], **{'missing_data_ratio': 0, 'zero_value_ratio': 1})
    """
    usgs_values = gages.read_target_cols(usgs_ids, time_range, [flow_type])[:, :, 0]
    sites_index = np.arange(usgs_values.shape[0])
    sites_chosen = np.ones(usgs_values.shape[0])
    for i in range(sites_index.size):
        # loop for every site
        runoff = usgs_values[i, :]
        for criteria in kwargs:
            # if any criteria is not matched, we can filter this site
            if sites_chosen[sites_index[i]] == 0:
                break
            if criteria == "missing_data_ratio":
                nan_length = runoff[np.isnan(runoff)].size
                # then calculate the length of consecutive nan
                thresh = kwargs[criteria]
                if nan_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1

            elif criteria == "zero_value_ratio":
                zero_length = runoff.size - np.count_nonzero(runoff)
                thresh = kwargs[criteria]
                if zero_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1
            else:
                print(
                    "Oops! That is not valid value. Try missing_data_ratio or zero_value_ratio ..."
                )
    gages_chosen_id = [
        usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0
    ]
    # assert all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:]))
    return gages_chosen_id


def choose_basins_with_area(
    gages: Gages,
    usgs_ids: list,
    smallest_area: float,
    largest_area: float,
) -> list:
    """
    choose basins with not too large or too small area

    Parameters
    ----------
    gages
        Camels, CamelsSeries, Gages or GagesPro object
    usgs_ids: list
        given sites' ids
    smallest_area
        lower limit; unit is km2
    largest_area
        upper limit; unit is km2

    Returns
    -------
    list
        sites_chosen: [] -- ids of chosen gages

    """
    basins_areas = gages.read_basin_area(usgs_ids).flatten()
    sites_index = np.arange(len(usgs_ids))
    sites_chosen = np.ones(len(usgs_ids))
    for i in range(sites_index.size):
        # loop for every site
        if basins_areas[i] < smallest_area or basins_areas[i] > largest_area:
            sites_chosen[sites_index[i]] = 0
        else:
            sites_chosen[sites_index[i]] = 1
    gages_chosen_id = [
        usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0
    ]
    # assert all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:]))
    return gages_chosen_id
