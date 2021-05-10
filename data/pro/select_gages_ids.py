from typing import Union
import numpy as np

from data.data_gages import Gages
from data.pro.data_gages_pro import GagesPro, get_dor_values


def dor_reservoirs_chosen(gages: Union[Gages, GagesPro], usgs_id, dor_chosen):
    """choose basins of small DOR(calculated by NOR_STORAGE/RUNAVE7100)"""
    if type(gages) == GagesPro:
        gages = gages.gages
    dors = get_dor_values(gages, usgs_id)
    if type(dor_chosen) == list or type(dor_chosen) == tuple:
        # right half-open range
        chosen_id = [usgs_id[i] for i in range(dors.size) if dor_chosen[0] <= dors[i] < dor_chosen[1]]
    else:
        if dor_chosen < 0:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] < -dor_chosen]
        else:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] >= dor_chosen]

    assert (all(x < y for x, y in zip(chosen_id, chosen_id[1:])))
    return chosen_id


def usgs_screen_streamflow(gages: Union[Gages, GagesPro], usgs_ids: list, time_range: list, **kwargs):
    """according to the criteria and its ancillary condition--thresh of streamflow data,
        choose appropriate ones from the given usgs sites
        Parameters
        ----------
        gages: Gages or GagesPro object
        usgs_ids: list -- given sites' ids
        time_range: list -- chosen time range
        kwargs: all criteria
        Returns
        -------
        usgs_out : ndarray -- streamflow  1d-var is gage, 2d-var is day
        sites_chosen: [] -- ids of chosen gages
        Examples
        --------
        usgs_screen(gages, ["02349000","08168797"], [‘1995-01-01’,‘2015-01-01’], **{'missing_data_ratio': 0, 'zero_value_ratio': 1})
    """
    usgs_values = gages.read_target_cols(usgs_ids, time_range, "usgsFlow")
    sites_index = np.arange(usgs_values.shape[0])
    sites_chosen = np.ones(usgs_values.shape[0])
    for i in range(sites_index.size):
        # loop for every site
        runoff = usgs_values[i, :]
        for criteria in kwargs:
            # if any criteria is not matched, we can filter this site
            if sites_chosen[sites_index[i]] == 0:
                break
            if criteria == 'missing_data_ratio':
                nan_length = runoff[np.isnan(runoff)].size
                # then calculate the length of consecutive nan
                thresh = kwargs[criteria]
                if nan_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1

            elif criteria == 'zero_value_ratio':
                zero_length = runoff.size - np.count_nonzero(runoff)
                thresh = kwargs[criteria]
                if zero_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1
            else:
                print("Oops!  That is not valid value.  Try again...")
    gages_chosen_id = [usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]
    assert (all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:])))
    return gages_chosen_id
