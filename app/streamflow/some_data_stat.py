"""some analysis for the dataset we used"""
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import GagesSource
from data.data_input import GagesModel
from data.gages_input_dataset import load_dataconfig_case_exp, load_ensemble_result
from data.config import cfg, update_cfg, cmd
from utils.hydro_util import hydro_logger
from visual.plot_model import plot_sites_and_attr, plot_scatter_multi_attrs

# calculate the dor values of this paper(https://doi.org/10.1029/2007WR005971) according to its table 1
data_validate = pd.read_csv("paper10.1029_2007WR005971-table1.csv")

idx4paper = 0
paper_dors = []
while idx4paper < data_validate.shape[0]:
    dam_num_tmp = data_validate['Number of Main dams'][idx4paper]
    if math.isnan(dam_num_tmp):
        hydro_logger.error("miss it")
    else:
        dam_num_tmp = int(dam_num_tmp)
        if type(data_validate['Watershed Area, km2'][idx4paper]) == str:
            watershed_area = float(data_validate['Watershed Area, km2'][idx4paper].replace(',', ''))
        else:
            watershed_area = data_validate['Watershed Area, km2'][idx4paper]
        if type(data_validate['Mean Runoff, mm/y'][idx4paper]) == str:
            mean_runoff = float(data_validate['Mean Runoff, mm/y'][idx4paper].replace(',', ''))
        else:
            mean_runoff = data_validate['Mean Runoff, mm/y'][idx4paper]

        capacity_reservoir = 0
        for idx4paper_tmp in range(dam_num_tmp):
            if type(data_validate['Capacity of Reservoirs, hm3'][idx4paper]) == str:
                capa_res_tmp = float(data_validate['Capacity of Reservoirs, hm3'][idx4paper].replace(',', ''))
            else:
                capa_res_tmp = data_validate['Capacity of Reservoirs, hm3'][idx4paper]
            capacity_reservoir = capacity_reservoir + capa_res_tmp
            idx4paper = idx4paper + 1

    paper_dor_tmp = (capacity_reservoir * 1000000 / (watershed_area * 1000000)) * 1000 / mean_runoff
    paper_dors.append(paper_dor_tmp)
paper_dors = np.array(paper_dors)
hydro_logger.info("The dor values of those basins: %s", paper_dors)
hydro_logger.info("Are dor values of those basins bigger than 0.02: %s", paper_dors > 0.02)
hydro_logger.info("Are dor values of those basins bigger than 0.1: %s", paper_dors > 0.1)
hydro_logger.info("Are dor values of those basins bigger than 1: %s", paper_dors > 1)
test_epoch = 300

all_exps_lst = ["basic_exp39", "basic_exp37", "basic_exp40", "basic_exp41", "basic_exp42", "basic_exp43",
                "basic_exp32", "basic_exp31", "basic_exp33", "basic_exp34", "basic_exp35", "basic_exp36"]
all_exps_random_seeds = ["123", "1234", "12345", "111", "1111", "11111", "123", "1234", "12345", "111", "1111", "11111"]
idx_tmp_now = 0
for exp_tmp in all_exps_lst:
    exp_tmp_lst = [exp_tmp]
    inds_df_tmp = load_ensemble_result(cfg, exp_tmp_lst, test_epoch)
    hydro_logger.info("The median NSE value of %s is %.2f (random seed: %s)", exp_tmp, inds_df_tmp.median()["NSE"],
                      all_exps_random_seeds[idx_tmp_now])
    idx_tmp_now = idx_tmp_now + 1

exp_lst = ["basic_exp37"]
config_data = load_dataconfig_case_exp(cfg, exp_lst[0])
data_model = GagesModel.load_datamodel(config_data.data_path["Temp"],
                                       data_source_file_name='test_data_source.txt',
                                       stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                       forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                       f_dict_file_name='test_dictFactorize.json',
                                       var_dict_file_name='test_dictAttribute.json',
                                       t_s_dict_file_name='test_dictTimeSpace.json')
camels_gageid_file = os.path.join(config_data.data_path["DB"], "camels_attributes_v2.0", "camels_attributes_v2.0",
                                  "camels_name.txt")
gauge_df = pd.read_csv(camels_gageid_file, dtype={"gauge_id": str}, sep=';')
gauge_list = gauge_df["gauge_id"].values

"show the relationship between NSE and some attrs"
attr_lst_shown = ["NDAMS_2009", "STOR_NOR_2009", "RAW_DIS_NEAREST_MAJ_DAM", "RAW_AVG_DIS_ALLDAMS", "FRESHW_WITHDRAWAL",
                  "PCT_IRRIG_AG", "POWER_SUM_MW", "PDEN_2000_BLOCK", "ROADS_KM_SQ_KM", "IMPNLCD06"]
attr_lst_shown_names = ["NDAMS", "STOR_NOR", "RAW_DIS_NEAREST_MAJ_DAM", "RAW_AVG_DIS_ALLDAMS", "FRESHW_WITHDRAWAL",
                        "PCT_IRRIG_AG", "POWER_SUM_MW", "PDEN_BLOCK", "ROADS_KM_SQ_KM", "IMPNLCD"]
sites_nonref = data_model.t_s_dict["sites_id"]
attrs_value = data_model.data_source.read_attr_origin(sites_nonref, attr_lst_shown)
inds_df, pred_mean, obs_mean = load_ensemble_result(cfg, exp_lst, test_epoch, return_value=True)
# nse_range = [-100000, 1]
nse_range = [0, 1]
# nse_range = [-1, 1]
show_ind_key = 'NSE'
idx_lst_nse_range = inds_df[
    (inds_df[show_ind_key] >= nse_range[0]) & (inds_df[show_ind_key] < nse_range[1])].index.tolist()
nse_values = inds_df["NSE"].values[idx_lst_nse_range]
for i in range(len(attr_lst_shown)):
    df = pd.DataFrame({attr_lst_shown_names[i]: attrs_value[i, idx_lst_nse_range], show_ind_key: nse_values})
    # g = sns.jointplot(x=attr_lst_shown_names[i], y=show_ind_key, data=df, kind="reg")
    sns.set(color_codes=True)
    g = sns.regplot(x=attr_lst_shown_names[i], y=show_ind_key, data=df, scatter_kws={'s': 10})
    show_max = attrs_value[i, idx_lst_nse_range].max()
    show_min = attrs_value[i, idx_lst_nse_range].min()
    if show_min < 0:
        show_min = 0
    # g.ax_marg_x.set_xlim(show_min, show_max)
    # g.ax_marg_y.set_ylim(0, 1)
    plt.ylim(0, 1)
    plt.xlim(show_min, show_max)
    plt.savefig(
        os.path.join(config_data.data_path["Out"], 'NSE-min-' + str(nse_range[0]) + '~' + attr_lst_shown[i] + '.png'),
        dpi=500, bbox_inches="tight")
    plt.show()

show_ind_key = 'NSE'
y_var_lst = [show_ind_key]
plot_scatter_multi_attrs(data_model, inds_df, idx_lst_nse_range, attr_lst_shown, y_var_lst)
plt.show()

"how many zero-dor, small-dor and large-dor basins in the CAMELS dataset"
t_train = config_data.model_dict["data"]["tRangeTrain"]
t_test = config_data.model_dict["data"]["tRangeTest"]
t_train_test = [t_train[0], t_test[1]]

source_data0 = GagesSource.choose_some_basins(config_data, t_train_test, screen_basin_area_huc4=False, dam_num=0)
sites_id_nodam = source_data0.all_configs['flow_screen_gage_id']

dor1 = -0.02
source_data1 = GagesSource.choose_some_basins(config_data, t_train_test, screen_basin_area_huc4=False, DOR=dor1)
sites_id_zerosmalldor = source_data1.all_configs['flow_screen_gage_id']
sites_id_zerodor = np.intersect1d(sites_id_zerosmalldor, sites_id_nodam)
sites_id_smalldor = [site_tmp for site_tmp in sites_id_zerosmalldor if site_tmp not in sites_id_nodam]

dor2 = 0.02
source_data2 = GagesSource.choose_some_basins(config_data, t_train_test, screen_basin_area_huc4=False, DOR=dor2)
sites_id_largedor = source_data2.all_configs['flow_screen_gage_id']

largedor_in_camels = np.intersect1d(sites_id_largedor, gauge_list)
smalldor_in_camels = np.intersect1d(sites_id_smalldor, gauge_list)
zerodor_in_camels = np.intersect1d(sites_id_zerodor, gauge_list)
hydro_logger.info("The number of large-dor basins in CAMELS: %d", largedor_in_camels.size)
hydro_logger.info("The number of small-dor basins in CAMELS: %d", smalldor_in_camels.size)
hydro_logger.info("The number of zero-dor basins in CAMELS: %d", zerodor_in_camels.size)

"how many zero-dor, small-dor and large-dor basins in the 523-CAMELS dataset"
exp_523basins_lst = ["basic_exp31"]
config_data_523basins = load_dataconfig_case_exp(cfg, exp_523basins_lst[0])
data_model_523basins = GagesModel.load_datamodel(config_data_523basins.data_path["Temp"],
                                                 data_source_file_name='test_data_source.txt',
                                                 stat_file_name='test_Statistics.json', flow_file_name='test_flow.npy',
                                                 forcing_file_name='test_forcing.npy', attr_file_name='test_attr.npy',
                                                 f_dict_file_name='test_dictFactorize.json',
                                                 var_dict_file_name='test_dictAttribute.json',
                                                 t_s_dict_file_name='test_dictTimeSpace.json')
basins523 = data_model_523basins.t_s_dict["sites_id"]
largedor_in_523acamels = np.intersect1d(sites_id_largedor, basins523)
smalldor_in_523camels = np.intersect1d(sites_id_smalldor, basins523)
zerodor_in_523camels = np.intersect1d(sites_id_zerodor, basins523)
hydro_logger.info("The number of large-dor basins in 523-CAMELS: %d", largedor_in_523acamels.size)
hydro_logger.info("The number of small-dor basins in 523-CAMELS: %d", smalldor_in_523camels.size)
hydro_logger.info("The number of zero-dor basins in 523-CAMELS: %d", zerodor_in_523camels.size)

"plot points of all 3557 sites and camels sites with different colors; plot polygons of all 3557 basins and camels " \
"basins with different colors "
chosen_sites = np.intersect1d(gauge_list, data_model.t_s_dict["sites_id"])
remain_sites = [site_tmp for site_tmp in data_model.t_s_dict["sites_id"] if site_tmp not in chosen_sites]

all_lat = data_model.data_source.gage_dict["LAT_GAGE"]
all_lon = data_model.data_source.gage_dict["LNG_GAGE"]
all_sites_id = data_model.data_source.gage_dict["STAID"]

is_camels = np.array([1 if data_model.t_s_dict["sites_id"][i] in chosen_sites else 0 for i in
                      range(len(data_model.t_s_dict["sites_id"]))])
plot_sites_and_attr(all_sites_id, all_lon, all_lat, chosen_sites, remain_sites, is_camels, is_discrete=True,
                    markers=["x", "o"], marker_sizes=[4, 2], colors=["b", "r"])
plt.savefig(os.path.join(config_data.data_path["Out"], 'map_camels_or_not.png'), dpi=500, bbox_inches="tight")

attrs_lst = ["SLOPE_PCT", "FORESTNLCD06", "PERMAVE", "GEOL_REEDBUSH_DOM_PCT", "STOR_NOR_2009",
             "FRESHW_WITHDRAWAL"]
data_attrs = data_model.data_source.read_attr_origin(all_sites_id.tolist(), attrs_lst)
for i in range(len(attrs_lst)):
    data_attr = data_attrs[i]
    if attrs_lst[i] == "STOR_NOR_2009" or attrs_lst[i] == "FRESHW_WITHDRAWAL":
        plot_sites_and_attr(all_sites_id, all_lon, all_lat, chosen_sites, remain_sites, data_attr,
                            pertile_range=[0, 95], markers=["x", "o"], marker_sizes=[20, 10], cmap_str="jet")
    else:
        plot_sites_and_attr(all_sites_id, all_lon, all_lat, chosen_sites, remain_sites, data_attr,
                            markers=["x", "o"], marker_sizes=[20, 10], cmap_str="jet")
    plt.savefig(os.path.join(config_data.data_path["Out"], attrs_lst[i] + '.png'), dpi=500, bbox_inches="tight")
