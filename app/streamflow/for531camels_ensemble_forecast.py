import sys

from data.data_input import GagesModel, save_datamodel
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp, GagesModels
from data.config import cfg
from explore.stat import ecdf
from visual.plot_stat import plot_ecdfs

sys.path.append("../..")
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# conus_exps = ["basic_exp37", "basic_exp39", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43"]
# exp_lst = ["basic_exp31", "basic_exp32", "basic_exp33", "basic_exp34", "basic_exp49", "basic_exp36"]
# gpu_lst = [1, 1, 0, 0, 1, 1]
conus_exps = ["basic_exp37"]
exp_lst = ["basic_exp31"]
gpu_lst = [0]
doLst = list()
# doLst.append('cache')
test_epoch = 20

all_config_Data = load_dataconfig_case_exp(cfg, conus_exps[0])
config_data = load_dataconfig_case_exp(cfg, exp_lst[0])

if 'cache' in doLst:
    quick_data_dir = os.path.join(all_config_Data.data_path["DB"], "quickdata")
    data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
    data_model_train = GagesModel.load_datamodel(data_dir,
                                                 data_source_file_name='data_source.txt',
                                                 stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                 forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                 f_dict_file_name='dictFactorize.json',
                                                 var_dict_file_name='dictAttribute.json',
                                                 t_s_dict_file_name='dictTimeSpace.json')
    data_model_test = GagesModel.load_datamodel(data_dir,
                                                data_source_file_name='test_data_source.txt',
                                                stat_file_name='test_Statistics.json',
                                                flow_file_name='test_flow.npy',
                                                forcing_file_name='test_forcing.npy',
                                                attr_file_name='test_attr.npy',
                                                f_dict_file_name='test_dictFactorize.json',
                                                var_dict_file_name='test_dictAttribute.json',
                                                t_s_dict_file_name='test_dictTimeSpace.json')

    gages_model_train = GagesModel.update_data_model(all_config_Data, data_model_train, data_attr_update=True,
                                                     screen_basin_area_huc4=False)
    gages_model_test = GagesModel.update_data_model(all_config_Data, data_model_test, data_attr_update=True,
                                                    train_stat_dict=gages_model_train.stat_dict,
                                                    screen_basin_area_huc4=False)
    save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                   stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                   forcing_file_name='test_forcing', attr_file_name='test_attr',
                   f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                   t_s_dict_file_name='test_dictTimeSpace.json')
    print("read and save gages conus data model")

    camels531_gageid_file = os.path.join(config_data.data_path["DB"], "camels531", "camels531.txt")
    gauge_df = pd.read_csv(camels531_gageid_file, dtype={"GaugeID": str})
    gauge_list = gauge_df["GaugeID"].values
    all_sites_camels_531 = np.sort([str(gauge).zfill(8) for gauge in gauge_list])
    gages_model = GagesModels(config_data, screen_basin_area_huc4=False,
                              sites_id=all_sites_camels_531.tolist())
    save_datamodel(gages_model.data_model_test, data_source_file_name='test_data_source.txt',
                   stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                   forcing_file_name='test_forcing', attr_file_name='test_attr',
                   f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                   t_s_dict_file_name='test_dictTimeSpace.json')
    print("read and save camels 531 data model")
# plot
data_model = GagesModel.load_datamodel(config_data.data_path["Temp"],
                                       data_source_file_name='test_data_source.txt',
                                       stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                       forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                       f_dict_file_name='test_dictFactorize.json',
                                       var_dict_file_name='test_dictAttribute.json',
                                       t_s_dict_file_name='test_dictTimeSpace.json')
inds_df_camels, pred_mean, obs_mean = load_ensemble_result(cfg, exp_lst, test_epoch, return_value=True)
# matplotlib.use('TkAgg')
# plot box
# keys = ["Bias", "RMSE", "NSE"]
# box_fig = plot_diff_boxes(inds_df_camels[keys])

data_model_conus = GagesModel.load_datamodel(all_config_Data.data_path["Temp"],
                                             data_source_file_name='test_data_source.txt',
                                             stat_file_name='test_Statistics.json',
                                             flow_file_name='test_flow.npy',
                                             forcing_file_name='test_forcing.npy',
                                             attr_file_name='test_attr.npy',
                                             f_dict_file_name='test_dictFactorize.json',
                                             var_dict_file_name='test_dictAttribute.json',
                                             t_s_dict_file_name='test_dictTimeSpace.json')
all_sites = data_model_conus.t_s_dict["sites_id"]
idx_lst_camels = [i for i in range(len(all_sites)) if all_sites[i] in data_model.t_s_dict["sites_id"]]

inds_df = load_ensemble_result(cfg, conus_exps, test_epoch)
keys_nse = "NSE"
xs = []
ys = []
# cases_exps_legends = ["523sites_from_LSTM-CONUS", "523sites_trained_in_LSTM-CAMELS"]
cases_exps_legends = ["Train: LSTM-CONUS; Test: 523 basins in CAMES", "Train: 523 basins in CAMELS; Test: 523 basins in CAMES"]
x1, y1 = ecdf(inds_df[keys_nse].iloc[idx_lst_camels])
xs.append(x1)
ys.append(y1)

x2, y2 = ecdf(inds_df_camels[keys_nse])
xs.append(x2)
ys.append(y2)

plot_ecdfs(xs, ys, cases_exps_legends, x_str="NSE", y_str="CDF")
sns.despine()
plt.savefig(os.path.join(config_data.data_path["Out"], 'camels_synergy.png'), dpi=500, bbox_inches="tight")
plt.show()
