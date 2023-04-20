"""
Author: Wenyu Ouyang
Date: 2022-12-08 09:24:54
LastEditTime: 2022-12-16 22:31:09
LastEditors: Wenyu Ouyang
Description: some constant for hydro model
FilePath: /HydroSPB/hydroSPB/utils/hydro_constant.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np


class HydroVar:
    """
    Hydrological variables
    """

    def __init__(
        self, name, unit=None, data: np.ndarray = None, alias=None, ChineseName=None
    ):
        """Initialize a HydroVar object

        Parameters
        ----------
        name : str
            name of variable
        unit : str
            unit of variable
        data : np.ndarray
            data of variable
        alias : str
            sometimes we use a different name for the same variable
        ChineseName : str
            Chinese name of variable
        """
        self.name = name
        self.unit = unit
        # data of variable
        self.data = data
        if self.data is not None and not isinstance(self.data, np.ndarray):
            raise ValueError("data should be numpy array")
        # sometimes we use a different name for the same variable
        self.alias = alias
        self.ChineseName = ChineseName

    def convert_var_unit(self, unit_final, **kwargs):
        """
        convert unit of variable

        Parameters
        ----------
        unit_final
            unit of variable after conversion
        **kwargs
            other parameters required for conversion

        Returns
        -------
        data
            data after conversion
        """
        if self.unit != unit_final:
            self.data = convert_unit(self.data, self.unit, unit_final, **kwargs)
            self.unit = unit_final


streamflow = HydroVar("streamflow", "m3/s", ChineseName="径流")
evapotranspiration = HydroVar("evapotranspiration", "mm/day", ChineseName="蒸散发")


# unify the unit of each variable
unit = {"streamflow": "m3/s"}


def convert_unit(data, unit_now, unit_final, **kwargs):
    """
    convert unit of variable

    Parameters
    ----------
    data
        data to be converted
    unit_now
        unit of variable now
    unit_final
        unit of variable after conversion
    **kwargs
        other parameters required for conversion

    Returns
    -------
    data
        data after conversion
    """
    if unit_now == "mm/day" and unit_final == "m3/s":
        basin_area = kwargs["basin_area"]
        if basin_area is None:
            raise ValueError("basin area is required for unit conversion")
        elif isinstance(basin_area, (list, np.ndarray)):
            if len(basin_area) != data.shape[0]:
                raise ValueError("basin area should have the same length as data")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            basin_area = np.repeat(basin_area, data.shape[1]).reshape(data.shape)
        result = mm_per_day_to_m3_per_sec(basin_area=basin_area, q=data)
    elif unit_now == "ft3/s" and unit_final == "m3/s":
        result = data / 35.314666721489
    elif unit_now == "0.1mm/day" and unit_final == "mm/day":
        result = data * 0.1
    else:
        raise ValueError("unit conversion not supported")
    return result


def mm_per_day_to_m3_per_sec(basin_area, q):
    """
    trans mm/day to m3/s for xaj models

    Parameters
    ----------
    basin_area
        we need to know the area of a basin so that we can perform this transformation
    q
        original streamflow data

    Returns
    -------

    """
    # 1 ft3 = 0.02831685 m3
    # ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    return q * basin_area * km2tom2 / (mtomm * daytos)
