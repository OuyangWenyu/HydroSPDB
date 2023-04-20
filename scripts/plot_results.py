"""
Author: Wenyu Ouyang
Date: 2023-04-20 23:05:39
LastEditTime: 2023-04-20 23:08:04
LastEditors: Wenyu Ouyang
Description: Plot results
FilePath: /HydroSPDB/scripts/plot_results.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import definitions

# TODO: unfinished
# cfg_dir_gages = os.path.join(definitions.RESULT_DIR, "gages")
# cfg_files = [os.path.join(cfg_dir_gages, "exp001", "05_April_202105_47AM.json"),
#              os.path.join(cfg_dir_gages, "exp14", "21_April_202102_59PM.json"), ]
# cfgs = []
# for cfg_file in cfg_files:
#     cfgs.append(unserialize_json(cfg_file))
# test_epoch = cfgs[0]["training_params"]["epochs"]
# keys_nse = "NSE"
# dpi = 600

# gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
# basins_id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values

# gage_2909id_file = os.path.join(definitions.ROOT_DIR, "example", "2909basins_NSE.csv")
# basins_2909id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values
# idx_lst_2909_in_3557 = [i for i in range(len(basins_id)) if basins_id[i] in basins_2909id]

# inds_df1 = stat_result(cfgs[0]["dataset_params"]["test_path"], test_epoch)
# inds_df2 = stat_result(cfgs[1]["dataset_params"]["test_path"], test_epoch)
# # another_test_epoch = 150
# # inds_df2 = stat_result(os.path.join(cfg_dir_gages, "exp15"), another_test_epoch)

# xs1 = []
# ys1 = []
# cases_exps_legends_together_1 = ["2909in_vanilla3557", "2909more_attr"]
# x1, y1 = ecdf(np.nan_to_num(inds_df1[keys_nse].iloc[idx_lst_2909_in_3557]))
# xs1.append(x1)
# ys1.append(y1)

# x2, y2 = ecdf(np.nan_to_num(inds_df2[keys_nse]))
# xs1.append(x2)
# ys1.append(y2)

# plot_ecdfs_matplot(xs1, ys1, cases_exps_legends_together_1,
#                    colors=["b", "r"],
#                    dash_lines=[False, False], x_str="NSE", y_str="CDF")
# plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], 'more_attr2909_' + str(test_epoch) + 'epoch.png'),
#             dpi=dpi, bbox_inches="tight")

# plt.show()
