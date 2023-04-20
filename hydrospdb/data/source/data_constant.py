"""
Author: Wenyu Ouyang
Date: 2022-12-02 17:52:48
LastEditTime: 2023-04-20 17:27:36
LastEditors: Wenyu Ouyang
Description: Some constant for data processing
FilePath: /HydroSPDB/hydrospdb/data/source/data_constant.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""


from hydrospdb.utils.hydro_constant import HydroVar


DAYMET_NAME = "daymet"
NLDAS_NAME = "nldas"

Q_CAMELS_US_NAME = "usgsFlow"
PRCP_DAYMET_NAME = "prcp"
PRCP_NLDAS_NAME = "total_precipitation"
PET_DAYMET_NAME = "PET"
PET_NLDAS_NAME = "potential_evaporation"

streamflow_camels_us = HydroVar(
    name=Q_CAMELS_US_NAME,
    unit="ft3/s",
    ChineseName="径流",
)
