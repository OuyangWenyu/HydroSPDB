"""
Author: Wenyu Ouyang
Date: 2023-04-20 11:51:06
LastEditTime: 2023-04-20 17:46:20
LastEditors: Wenyu Ouyang
Description: 
FilePath: /HydroSPDB/scripts/streamflow/script_constant.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


from hydrospdb.data.source.data_constant import (
    PET_NLDAS_NAME,
    PRCP_DAYMET_NAME,
    PRCP_NLDAS_NAME,
)


VAR_C_CHOSEN_FROM_CAMELS_US = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "geol_1st_class",
    "geol_2nd_class",
    "geol_porostiy",
    "geol_permeability",
]

VAR_C_CHOSEN_FROM_GAGES_II = [
    "DRAIN_SQKM",
    "ELEV_MEAN_M_BASIN",
    "SLOPE_PCT",
    "DEVNLCD06",
    "FORESTNLCD06",
    "PLANTNLCD06",
    "WATERNLCD06",
    "SNOWICENLCD06",
    "BARRENNLCD06",
    "SHRUBNLCD06",
    "GRASSNLCD06",
    "WOODYWETNLCD06",
    "EMERGWETNLCD06",
    "AWCAVE",
    "PERMAVE",
    "RFACT",
    "ROCKDEPAVE",
    "GEOL_REEDBUSH_DOM",
    "GEOL_REEDBUSH_DOM_PCT",
    "STREAMS_KM_SQ_KM",
    "NDAMS_2009",
    "STOR_NOR_2009",
    "RAW_DIS_NEAREST_MAJ_DAM",
    "CANALS_PCT",
    "RAW_DIS_NEAREST_CANAL",
    "FRESHW_WITHDRAWAL",
    "POWER_SUM_MW",
    "PDEN_2000_BLOCK",
    "ROADS_KM_SQ_KM",
    "IMPNLCD06",
]

VAR_T_CHOSEN_FROM_DAYMET = [
    "dayl",
    PRCP_DAYMET_NAME,
    "srad",
    "swe",
    "tmax",
    "tmin",
    "vp",
]

VAR_T_CHOSEN_FROM_NLDAS = [
    PRCP_NLDAS_NAME,
    PET_NLDAS_NAME,
    "temperature",
    "specific_humidity",
    "shortwave_radiation",
    "potential_energy",
]
