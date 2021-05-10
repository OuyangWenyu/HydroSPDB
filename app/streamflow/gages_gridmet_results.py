import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import definitions
from explore.stat import ecdf
from hydroDL.trainer import stat_result
from utils.hydro_utils import unserialize_json
from visual.plot_stat import plot_ecdfs_matplot

cfg_dir = os.path.join(definitions.ROOT_DIR, "example", "gages_gridmet")
cfg_files = [os.path.join(cfg_dir, "exp2", "12_April_202107_06PM.json"),
             os.path.join(cfg_dir, "exp3", "12_April_202108_27PM.json"),
             os.path.join(cfg_dir, "exp4", "12_April_202108_44PM.json"),
             os.path.join(cfg_dir, "exp5", "13_April_202112_12AM.json"),
             os.path.join(cfg_dir, "exp6", "13_April_202111_54AM.json"),
             os.path.join(cfg_dir, "exp7", "13_April_202106_01PM.json"),
             # os.path.join(cfg_dir, "exp8", "15_April_202111_03AM.json")]
             # os.path.join(cfg_dir, "exp8", "15_April_202109_02PM.json")]
             # os.path.join(cfg_dir, "exp8", "15_April_202110_06PM.json")]
             # os.path.join(cfg_dir, "exp8", "15_April_202110_15PM.json")]
             # os.path.join(cfg_dir, "exp8", "15_April_202110_25PM.json")]
             # os.path.join(cfg_dir, "exp8", "15_April_202110_39PM.json")]
             # os.path.join(cfg_dir, "exp8", "16_April_202109_11AM.json")]
             os.path.join(cfg_dir, "exp8", "18_April_202110_32AM.json")]
cfgs = []
for cfg_file in cfg_files:
    cfgs.append(unserialize_json(cfg_file))
test_epoch = cfgs[0]["training_params"]["epochs"]
keys_nse = "NSE"
dpi = 600
irrigation328_gage_id = os.path.join(definitions.ROOT_DIR, "example", "328irrigation_gage_id.csv")
irri_basins_id = pd.read_csv(irrigation328_gage_id, dtype={0: str}).iloc[:, 0].values
gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
basins_id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values

inds_df1 = stat_result(cfgs[0]["dataset_params"]["test_path"], test_epoch)
inds_df2 = stat_result(cfgs[1]["dataset_params"]["test_path"], test_epoch)
inds_df3 = stat_result(cfgs[2]["dataset_params"]["test_path"], test_epoch)
xs1 = []
ys1 = []
cases_exps_legends_together_1 = ["328_no-et", "328_et", "328_cet"]

x1, y1 = ecdf(np.nan_to_num(inds_df1[keys_nse]))
xs1.append(x1)
ys1.append(y1)

x2, y2 = ecdf(np.nan_to_num(inds_df2[keys_nse]))
xs1.append(x2)
ys1.append(y2)

x3, y3 = ecdf(np.nan_to_num(inds_df3[keys_nse]))
xs1.append(x3)
ys1.append(y3)

plot_ecdfs_matplot(xs1, ys1, cases_exps_legends_together_1,
                   colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                   dash_lines=[False, False, False], x_str="NSE", y_str="CDF")

plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '328_cet_et_no-et.png'), dpi=dpi, bbox_inches="tight")

inds_df4 = stat_result(cfgs[3]["dataset_params"]["test_path"], test_epoch)
inds_df5 = stat_result(cfgs[4]["dataset_params"]["test_path"], test_epoch)
inds_df6 = stat_result(cfgs[5]["dataset_params"]["test_path"], test_epoch)
xs2 = []
ys2 = []
cases_exps_legends_together_2 = ["3557_no-et", "3557_et", "3557_cet"]

x4, y4 = ecdf(np.nan_to_num(inds_df4[keys_nse]))
xs2.append(x4)
ys2.append(y4)

x5, y5 = ecdf(np.nan_to_num(inds_df5[keys_nse]))
xs2.append(x5)
ys2.append(y5)

x6, y6 = ecdf(np.nan_to_num(inds_df6[keys_nse]))
xs2.append(x6)
ys2.append(y6)

plot_ecdfs_matplot(xs2, ys2, cases_exps_legends_together_2,
                   colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                   dash_lines=[False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '3557_cet_et_no-et.png'), dpi=dpi,
            bbox_inches="tight")

xs3 = []
ys3 = []
cases_exps_legends_together_3 = ["328in3557_no-et", "328in3557_et", "328in3557_cet", "328in3557_et_tl-cet"]
idx_lst_328in3557 = [i for i in range(len(basins_id)) if basins_id[i] in irri_basins_id]
x7, y7 = ecdf(np.nan_to_num(inds_df4[keys_nse].iloc[idx_lst_328in3557]))
xs3.append(x7)
ys3.append(y7)

x8, y8 = ecdf(np.nan_to_num(inds_df5[keys_nse].iloc[idx_lst_328in3557]))
xs3.append(x8)
ys3.append(y8)

x9, y9 = ecdf(np.nan_to_num(inds_df6[keys_nse].iloc[idx_lst_328in3557]))
xs3.append(x9)
ys3.append(y9)

inds_df10 = stat_result(cfgs[6]["dataset_params"]["test_path"], cfgs[6]["training_params"]["epochs"])
x10, y10 = ecdf(np.nan_to_num(inds_df10[keys_nse]))
xs3.append(x10)
ys3.append(y10)
plot_ecdfs_matplot(xs3, ys3, cases_exps_legends_together_3,
                   colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                   dash_lines=[False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '328in3557_cet_et_no-et_tl.png'), dpi=dpi,
            bbox_inches="tight")
plt.show()
