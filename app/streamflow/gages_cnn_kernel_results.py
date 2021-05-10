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

cfg_dir_gages = os.path.join(definitions.ROOT_DIR, "example", "gages")
cfg_files = [os.path.join(cfg_dir_gages, "exp2", "05_April_202105_47AM.json"),
             os.path.join(cfg_dir_gages, "exp10", "19_April_202104_09PM.json"),
             os.path.join(cfg_dir_gages, "exp11", "20_April_202112_54AM.json"),
             os.path.join(cfg_dir_gages, "exp12", "20_April_202105_35PM.json"),
             os.path.join(cfg_dir_gages, "exp13", "21_April_202101_55AM.json"), ]
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
zero_small_dor_ids = dor_reservoirs_chosen(gages, basins_id, -0.1)

inds_df1 = stat_result(cfgs[0]["dataset_params"]["test_path"], test_epoch)
inds_df2 = stat_result(cfgs[1]["dataset_params"]["test_path"], test_epoch)
inds_df3 = stat_result(cfgs[2]["dataset_params"]["test_path"], test_epoch)
inds_df4 = stat_result(cfgs[3]["dataset_params"]["test_path"], test_epoch)
inds_df5 = stat_result(cfgs[4]["dataset_params"]["test_path"], test_epoch)

xs1 = []
ys1 = []
cases_exps_legends_together_1 = ["vanilla3557", "cat_first_stor_cnn_kernel3557", "relu_first_stor_cnn_kernel3557",
                                 "cat_first_dor_cnn_kernel3557", "relu_first_dor_cnn_kernel3557"]
x1, y1 = ecdf(np.nan_to_num(inds_df1[keys_nse]))
xs1.append(x1)
ys1.append(y1)

x2, y2 = ecdf(np.nan_to_num(inds_df2[keys_nse]))
xs1.append(x2)
ys1.append(y2)

x3, y3 = ecdf(np.nan_to_num(inds_df3[keys_nse]))
xs1.append(x3)
ys1.append(y3)

x14, y14 = ecdf(np.nan_to_num(inds_df4[keys_nse]))
xs1.append(x14)
ys1.append(y14)

x15, y15 = ecdf(np.nan_to_num(inds_df5[keys_nse]))
xs1.append(x15)
ys1.append(y15)

plot_ecdfs_matplot(xs1, ys1, cases_exps_legends_together_1,
                   colors=["b", "g", "r", "c", "m"],
                   dash_lines=[False, False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], '3557_cnn_kernel.png'), dpi=dpi,
            bbox_inches="tight")

xs2 = []
ys2 = []
cases_exps_legends_together_2 = ["dor>=0.1vanilla3557", "dor>=0.1cat_first_stor_cnn_kernel3557",
                                 "dor>=0.1relu_first_stor_cnn_kernel3557", "dor>=0.1cat_first_dor_cnn_kernel3557",
                                 "dor>=0.1relu_first_dor_cnn_kernel3557", ]
idx_lst_large_dor_in_3557 = [i for i in range(len(basins_id)) if basins_id[i] in large_dor_ids]
x4, y4 = ecdf(np.nan_to_num(inds_df1[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs2.append(x4)
ys2.append(y4)

x5, y5 = ecdf(np.nan_to_num(inds_df2[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs2.append(x5)
ys2.append(y5)

x6, y6 = ecdf(np.nan_to_num(inds_df3[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs2.append(x6)
ys2.append(y6)

x16, y16 = ecdf(np.nan_to_num(inds_df4[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs2.append(x16)
ys2.append(y16)

x17, y17 = ecdf(np.nan_to_num(inds_df5[keys_nse].iloc[idx_lst_large_dor_in_3557]))
xs2.append(x17)
ys2.append(y17)

plot_ecdfs_matplot(xs2, ys2, cases_exps_legends_together_2,
                   colors=["b", "g", "r", "c", "m"],
                   dash_lines=[False, False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], 'largedor3557_cnn_kernel.png'), dpi=dpi,
            bbox_inches="tight")

xs3 = []
ys3 = []
cases_exps_legends_together_3 = ["dor<0.1vanilla3557", "dor<0.1cat_first_stor_cnn_kernel3557",
                                 "dor<0.1relu_first_stor_cnn_kernel3557", "dor<0.1cat_first_dor_cnn_kernel3557",
                                 "dor<0.1relu_first_dor_cnn_kernel3557"]
idx_lst_zero_small_dor_in_3557 = [i for i in range(len(basins_id)) if basins_id[i] in zero_small_dor_ids]
x7, y7 = ecdf(np.nan_to_num(inds_df1[keys_nse].iloc[idx_lst_zero_small_dor_in_3557]))
xs3.append(x7)
ys3.append(y7)

x8, y8 = ecdf(np.nan_to_num(inds_df2[keys_nse].iloc[idx_lst_zero_small_dor_in_3557]))
xs3.append(x8)
ys3.append(y8)

x9, y9 = ecdf(np.nan_to_num(inds_df3[keys_nse].iloc[idx_lst_zero_small_dor_in_3557]))
xs3.append(x9)
ys3.append(y9)

x18, y18 = ecdf(np.nan_to_num(inds_df4[keys_nse].iloc[idx_lst_zero_small_dor_in_3557]))
xs3.append(x18)
ys3.append(y18)

x19, y19 = ecdf(np.nan_to_num(inds_df5[keys_nse].iloc[idx_lst_zero_small_dor_in_3557]))
xs3.append(x19)
ys3.append(y19)

plot_ecdfs_matplot(xs3, ys3, cases_exps_legends_together_3,
                   colors=["b", "g", "r", "c", "m"],
                   dash_lines=[False, False, False, False, False], x_str="NSE", y_str="CDF")
plt.savefig(os.path.join(cfgs[0]["dataset_params"]["test_path"], 'smalldor3557_cnn_kernel.png'), dpi=dpi,
            bbox_inches="tight")
plt.show()
