"""
Author: Wenyu Ouyang
Date: 2023-04-11 10:00:42
LastEditTime: 2023-04-20 16:09:43
LastEditors: Wenyu Ouyang
Description: Config for the main folders
FilePath: /HydroSPDB/definitions.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
path = Path(ROOT_DIR)
try:
    import definitions_private
    DATASET_DIR = definitions_private.DATASET_DIR
    RESULT_DIR = definitions_private.RESULT_DIR
except ImportError:
    # default data source directory
    DATASET_DIR = os.path.join(
        path.parent.parent.absolute(), "data", "hydro-dl-reservoir-data"
    )  # This is your Data source directory
    # default result directory
    RESULT_DIR = os.path.join(ROOT_DIR, "results")  # This is your result directory
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
print("Please Check your directory:")
print("ROOT_DIR of the repo: ", ROOT_DIR)
print("DATA_SOURCE_DIR of the repo: ", DATASET_DIR)
print("RESULT_DIR of the repo: ", RESULT_DIR)
