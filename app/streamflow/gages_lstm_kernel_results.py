import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import definitions
from data.data_gages import Gages
from data.pro.select_gages_ids import dor_reservoirs_chosen
from explore.stat import ecdf
from hydroDL.trainer import stat_result
from utils.hydro_utils import unserialize_json
from visual.plot_stat import plot_ecdfs_matplot

cfg_dir = os.path.join(definitions.ROOT_DIR, "example", "dam")
cfg_dir_gages = os.path.join(definitions.ROOT_DIR, "example", "gages")
cfg_files = [os.path.join(cfg_dir_gages, "exp2", "05_April_202105_47AM.json"),
             os.path.join(cfg_dir, "exp5", "20_April_202110_59AM.json"),
             os.path.join(cfg_dir, "exp3", "17_April_202101_31PM.json"),
             os.path.join(cfg_dir, "exp4", "19_April_202112_09AM.json"),
             os.path.join(cfg_dir_gages, "exp8", "18_April_202102_54AM.json"),
             os.path.join(cfg_dir_gages, "exp9", "18_April_202109_11PM.json"), ]
cfgs = []
for cfg_file in cfg_files:
    cfgs.append(unserialize_json(cfg_file))
test_epoch = cfgs[0]["training_params"]["epochs"]
keys_nse = "NSE"
dpi = 600

gage_3557id_file = os.path.join(definitions.ROOT_DIR, "example", "3557basins_ID_NSE_DOR.csv")
basins_id = pd.read_csv(gage_3557id_file, dtype={0: str}).iloc[:, 0].values

gages_path = os.path.join(definitions.DATASET_DIR, "gages")
gages = Gages(gages_path, False)
large_dor_ids = dor_reservoirs_chosen(gages, basins_id, 0.1)

inds_df1 = stat_result(cfgs[0]["dataset_params"]["test_path"], test_epoch)
inds_df2 = stat_result(cfgs[1]["dataset_params"]["test_path"], test_epoch)
inds_df3 = stat_result(cfgs[2]["dataset_params"]["test_path"], test_epoch)
inds_df4 = stat_result(cfgs[3]["dataset_params"]["test_path"], test_epoch)
inds_df5 = stat_result(cfgs[4]["dataset_params"]["test_path"], test_epoch)
inds_df6 = stat_result(cfgs[5]["dataset_params"]["test_path"], test_epoch)

xs1 = []
ys1 = []
cases_exps_legends_together_1 = ["dor>=0.1in3557", "dor>=0.1", "dor>=0.1natural_flow", "dor>=0.1delta_storage",
                                 "dor>=0.1natural_flowin3557", "dor>=0.1delta_storagein3557"]
idx_lst_large_dor_in_3557 = [i for i in range(len(basins_id)) if basins_id[i] in large_dor_ids]
x1, y1 = ecdf(np.nan_to_num(inds_df1[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs1.append(x1)
ys1.append(y1)

x2, y2 = ecdf(np.nan_to_num(inds_df2[keys_nse]))
xs1.append(x2)
ys1.append(y2)

x3, y3 = ecdf(np.nan_to_num(inds_df3[keys_nse]))
xs1.append(x3)
ys1.append(y3)

x4, y4 = ecdf(np.nan_to_num(inds_df4[keys_nse]))
xs1.append(x4)
ys1.append(y4)

x5, y5 = ecdf(np.nan_to_num(inds_df5[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs1.append(x5)
ys1.append(y5)

x6, y6 = ecdf(np.nan_to_num(inds_df6[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs1.append(x6)
ys1.append(y6)

plot_ecdfs_matplot(xs1, ys1, cases_exps_legends_together_1,
                   colors=["b", "g", "r", "c", "m", "y"],
                   dash_lines=[False, False, False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[1]["dataset_params"]["test_path"], 'large_dor_lstm_kernel.png'), dpi=dpi,
            bbox_inches="tight")

xs3 = []
ys3 = []
cases_exps_legends_together_3 = ["dor>=0.1in3557", "dor>=0.1", "dor>=0.1natural_flow",
                                 "dor>=0.1natural_flowin3557"]
idx_lst_large_dor_in_3557 = [i for i in range(len(basins_id)) if basins_id[i] in large_dor_ids]
xs3.append(x1)
ys3.append(y1)

xs3.append(x2)
ys3.append(y2)

xs3.append(x3)
ys3.append(y3)

xs3.append(x5)
ys3.append(y5)

plot_ecdfs_matplot(xs3, ys3, cases_exps_legends_together_3,
                   colors=["b", "g", "r", "c"],
                   dash_lines=[False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], 'large_dor_lstm_kernel_natural_flow.png'), dpi=dpi,
            bbox_inches="tight")

xs2 = []
ys2 = []
cases_exps_legends_together_2 = ["vanilla3557", "natural_flow3557", "delta_storage3557"]
x7, y7 = ecdf(np.nan_to_num(inds_df1[keys_nse]))
xs2.append(x7)
ys2.append(y7)

x8, y8 = ecdf(np.nan_to_num(inds_df5[keys_nse]))
xs2.append(x8)
ys2.append(y8)

x9, y9 = ecdf(np.nan_to_num(inds_df6[keys_nse]))
xs2.append(x9)
ys2.append(y9)
plot_ecdfs_matplot(xs2, ys2, cases_exps_legends_together_2,
                   colors=["b", "g", "r"],
                   dash_lines=[False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '3557_lstm_kernel.png'), dpi=dpi,
            bbox_inches="tight")

xs4 = []
ys4 = []
cases_exps_legends_together_4 = ["vanilla3557", "natural_flow3557"]
xs4.append(x7)
ys4.append(y7)

xs4.append(x8)
ys4.append(y8)

plot_ecdfs_matplot(xs4, ys4, cases_exps_legends_together_4,
                   colors=["b", "r"],
                   dash_lines=[False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '3557_lstm_kernel_natural_flow.png'), dpi=dpi,
            bbox_inches="tight")

plt.show()
