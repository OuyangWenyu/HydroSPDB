"""
Author: Wenyu Ouyang
Date: 2023-04-20 11:51:06
LastEditTime: 2023-04-20 16:12:02
LastEditors: Wenyu Ouyang
Description: unzip downloaded data
FilePath: /HydroSPDB/scripts/prepare_data.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydrospdb.utils.hydro_utils import unzip_nested_zip

print("Please download data manually!")
if not os.path.isdir(definitions.DATASET_DIR):
    raise RuntimeError(f"{definitions.DATASET_DIR} not found! Please download the data")
zip_files = [
    "59692a64e4b0d1f9f05fbd39",
    "basin_mean_forcing.zip",
    "basinchar_and_report_sept_2011.zip",
    "boundaries_shapefiles_by_aggeco.zip",
    "gages_streamflow.zip",
    "gagesII_9322_point_shapefile.zip",
]
download_zip_files = [
    os.path.join(definitions.DATASET_DIR, zip_file) for zip_file in zip_files
]
for download_zip_file in download_zip_files:
    if not os.path.isfile(download_zip_file):
        raise RuntimeError(f"{download_zip_file} not found! Please download the data")
unzip_dirs = [
    os.path.join(definitions.DATASET_DIR, zip_file[:-4]) for zip_file in zip_files
]
for i in range(len(unzip_dirs)):
    if not os.path.isdir(unzip_dirs[i]):
        print(f"unzip directory:{unzip_dirs[i]}")
        unzip_nested_zip(download_zip_files[i], unzip_dirs[i])
    else:
        print(f"unzip directory -- {unzip_dirs[i]} has existed")
