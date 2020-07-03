import sys

from data import GagesSource
from data.data_input import GagesModel
from data.gages_input_dataset import load_ensemble_result, load_dataconfig_case_exp
from explore.stat import ecdf
from visual.plot_stat import plot_ecdfs

sys.path.append("../..")
import os
import definitions

conus_exps = ["basic_exp37", "basic_exp39", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43"]
smalldor_exp_lst = ["dam_exp1", "dam_exp2", "dam_exp3", "dam_exp7", "dam_exp8", "dam_exp9"]
largedor_exp_lst = ["dam_exp4", "dam_exp5", "dam_exp6", "dam_exp13", "dam_exp16", "dam_exp19"]
gpu_lst = [1, 1, 0, 0, 2, 2]
doLst = list()
# doLst.append('cache')
config_dir = definitions.CONFIG_DIR
test_epoch = 300

smalldor_config_data = load_dataconfig_case_exp(smalldor_exp_lst[0])
largedor_config_data = load_dataconfig_case_exp(largedor_exp_lst[0])
all_config_Data = load_dataconfig_case_exp(conus_exps[0])

quick_data_dir = os.path.join(all_config_Data.data_path["DB"], "quickdata")
data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
data_model = GagesModel.load_datamodel(data_dir,
                                       data_source_file_name='test_data_source.txt',
                                       stat_file_name='test_Statistics.json',
                                       flow_file_name='test_flow.npy',
                                       forcing_file_name='test_forcing.npy',
                                       attr_file_name='test_attr.npy',
                                       f_dict_file_name='test_dictFactorize.json',
                                       var_dict_file_name='test_dictAttribute.json',
                                       t_s_dict_file_name='test_dictTimeSpace.json')
all_sites = data_model.t_s_dict["sites_id"]

dor_1 = - 0.02
dor_2 = 0.02
source_data_dor1 = GagesSource.choose_some_basins(all_config_Data,
                                                  all_config_Data.model_dict["data"]["tRangeTrain"],
                                                  screen_basin_area_huc4=False,
                                                  DOR=dor_1)
source_data_dor2 = GagesSource.choose_some_basins(all_config_Data,
                                                  all_config_Data.model_dict["data"]["tRangeTrain"],
                                                  screen_basin_area_huc4=False,
                                                  DOR=dor_2)
sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
sites_id_dor2 = source_data_dor2.all_configs['flow_screen_gage_id']
idx_lst_smalldor = [i for i in range(len(all_sites)) if all_sites[i] in sites_id_dor1]
idx_lst_largedor = [i for i in range(len(all_sites)) if all_sites[i] in sites_id_dor2]

cases_exps = conus_exps
inds_df = load_ensemble_result(cases_exps, test_epoch)
keys_nse = "NSE"
xs = []
ys = []
cases_exps_legends_together = ["small_dor", "large_dor"]

x1, y1 = ecdf(inds_df[keys_nse].iloc[idx_lst_smalldor])
xs.append(x1)
ys.append(y1)

x2, y2 = ecdf(inds_df[keys_nse].iloc[idx_lst_largedor])
xs.append(x2)
ys.append(y2)

# compare_item = 0
compare_item = 1
if compare_item == 0:
    plot_ecdfs(xs, ys, cases_exps_legends_together)
elif compare_item == 1:
    cases_exps = smalldor_exp_lst
    inds_df_smalldor = load_ensemble_result(cases_exps, test_epoch)
    x3, y3 = ecdf(inds_df_smalldor[keys_nse])
    xs.append(x3)
    ys.append(y3)
    cases_exps = largedor_exp_lst
    inds_df_largedor = load_ensemble_result(cases_exps, test_epoch)
    x4, y4 = ecdf(inds_df_largedor[keys_nse])
    xs.append(x4)
    ys.append(y4)
    cases_exps_legends_separate = ["small_dor", "large_dor"]
    plot_ecdfs(xs, ys, cases_exps_legends_together + cases_exps_legends_separate,
               style=["together", "together", "separate", "separate"], case_str="dor_value",
               event_str="is_pooling_together", x_str="NSE", y_str="CDF")
