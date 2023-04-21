"""
Author: Wenyu Ouyang
Date: 2023-04-20 23:05:39
LastEditTime: 2023-04-21 10:53:33
LastEditors: Wenyu Ouyang
Description: Plot results
FilePath: /HydroSPDB/scripts/plot_results.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydrospdb.models.trainer import load_result
from scripts.streamflow_utils import get_json_file
from hydrospdb.visual.plot_model import plot_gages_map_and_box
from hydrospdb.utils.hydro_stat import stat_error
from hydrospdb.data.source.data_gages import Gages


exp = "exp0010"
test_epoch = 300
gages_exp_dir = os.path.join(definitions.RESULT_DIR, "gages", exp)
config_data = get_json_file(gages_exp_dir)
sites_id = config_data["data_params"]["object_ids"]
pred, obs = load_result(gages_exp_dir, test_epoch)
idx_lst = np.arange(len(sites_id)).tolist()
nse_range = [0, 1]
inds_df = pd.DataFrame(stat_error(obs, pred))
keys_nse = "NSE"
idx_lstl_nse = inds_df[
    (inds_df[keys_nse] >= nse_range[0]) & (inds_df[keys_nse] <= nse_range[1])
].index.tolist()
gages = Gages(definitions.DATASET_DIR, False)
gages_df = pd.DataFrame(gages.gages_sites)
# choose sites with given STAID in gages_df
chosen_sites_df = gages_df[gages_df["STAID"].isin(sites_id)]
fig = plot_gages_map_and_box(
    chosen_sites_df,
    inds_df,
    keys_nse,
    idx_lst=idx_lstl_nse,
)
FIGURE_DPI = 600
plt.savefig(
    os.path.join(config_data["data_params"]["test_path"], "sites_NSE_map_box.png"),
    dpi=FIGURE_DPI,
    bbox_inches="tight",
)
plt.figure()
