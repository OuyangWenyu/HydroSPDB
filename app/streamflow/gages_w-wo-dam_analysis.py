"""1. zero-dor + large-dor basins 2. small-dor + large-dor basins"""
import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig, GagesSource
from data.data_input import _basin_norm, save_result, save_datamodel
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
import numpy as np


def dam_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    dam_plan = cfg.DAM_PLAN
    cache = cfg.CACHE.STATE
    print("train and test in basins with different combination: \n")
    config_data = GagesConfig(cfg)
    if dam_plan == 2:
        dam_num = 0
        dor = 0.02
        source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                          config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          DOR=dor)
        # basins with dams
        source_data_withoutdams = GagesSource.choose_some_basins(config_data,
                                                                 config_data.model_dict["data"]["tRangeTrain"],
                                                                 screen_basin_area_huc4=False,
                                                                 dam_num=dam_num)

        sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
        sites_id_withoutdams = source_data_withoutdams.all_configs['flow_screen_gage_id']
        sites_id_chosen = np.sort(np.union1d(np.array(sites_id_dor1), np.array(sites_id_withoutdams))).tolist()
    elif dam_plan == 3:
        dam_num = [1, 100000]
        # basins with dams
        source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                              config_data.model_dict["data"]["tRangeTrain"],
                                                              screen_basin_area_huc4=False,
                                                              dam_num=dam_num)
        sites_id_chosen = source_data_withdams.all_configs['flow_screen_gage_id']
    else:
        print("wrong choice")
        sites_id_chosen = None
    gages_model = GagesModels(config_data, screen_basin_area_huc4=False, sites_id=sites_id_chosen)
    gages_model_train = gages_model.data_model_train
    gages_model_test = gages_model.data_model_test
    if cache:
        save_datamodel(gages_model_train, data_source_file_name='data_source.txt',
                       stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                       attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                       var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_datamodel(gages_model_test, data_source_file_name='test_data_source.txt',
                       stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                       forcing_file_name='test_forcing', attr_file_name='test_attr',
                       f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                       t_s_dict_file_name='test_dictTimeSpace.json')
    with torch.cuda.device(gpu_num):
        if train_mode:
            master_train(gages_model_train, random_seed=random_seed)
        pred, obs = master_test(gages_model_test, epoch=test_epoch)
        basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                            is_return_dict=False)
        mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                           is_return_dict=False)
        mean_prep = mean_prep / 365 * 10
        pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
        obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
        save_result(gages_model_test.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)


# python gages_w-wo-dam_analysis.py --sub nodam/exp7 --cache_state 1 --dam_plan 2 --train_epoch 20 --save_epoch 10 --te 20
# python gages_w-wo-dam_analysis.py --sub dam/exp27 --cache_state 1 --dam_plan 3 --train_epoch 20 --save_epoch 10 --te 20 --rs 123
# python gages_w-wo-dam_analysis.py --cfg dam/config_exp26.ini --ctx 2 --dam_plan 3 --rs 1234 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg nodam/config_exp8.ini --ctx 0 --dam_plan 2 --rs 123 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg nodam/config_exp9.ini --ctx 1 --dam_plan 2 --rs 12345 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg dam/config_exp28.ini --ctx 0 --dam_plan 3 --rs 12345 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg nodam/config_exp10.ini --ctx 2 --dam_plan 2 --rs 111 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg nodam/config_exp11.ini --ctx 1 --dam_plan 2 --rs 1111 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg dam/config_exp29.ini --ctx 2 --dam_plan 3 --rs 111 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg dam/config_exp30.ini --ctx 1 --dam_plan 3 --rs 1111 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg nodam/config_exp12.ini --ctx 2 --dam_plan 2 --rs 11111 --te 300 --train_mode True
# python gages_w-wo-dam_analysis.py --cfg dam/config_exp31.ini --ctx 0 --dam_plan 3 --rs 11111 --te 300 --train_mode True
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    dam_lstm(args)
    print("End\n")
