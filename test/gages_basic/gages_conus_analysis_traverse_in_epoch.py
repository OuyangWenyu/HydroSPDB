"""model in the CONUS"""
import os

import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig
from data.data_input import _basin_norm, save_result, save_quick_data, save_datamodel, StreamflowInputDataset
from data.gages_input_dataset import GagesModels
from hydroDL.master.master import master_train_easier_lstm, master_test_easier_lstm


def conus_other_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    print("train and test in CONUS: \n")
    print(cfg)
    config_data = GagesConfig(cfg)

    gages_model = GagesModels(config_data, screen_basin_area_huc4=False)
    gages_model_train = gages_model.data_model_train
    gages_model_test = gages_model.data_model_test
    if cfg.CACHE.GEN_QUICK_DATA:
        if not os.path.isdir(cfg.CACHE.DATA_DIR):
            os.makedirs(cfg.CACHE.DATA_DIR)
        save_quick_data(gages_model_train, cfg.CACHE.DATA_DIR, data_source_file_name='data_source.txt',
                        stat_file_name='Statistics.json', flow_file_name='flow', forcing_file_name='forcing',
                        attr_file_name='attr', f_dict_file_name='dictFactorize.json',
                        var_dict_file_name='dictAttribute.json', t_s_dict_file_name='dictTimeSpace.json')
        save_quick_data(gages_model_test, cfg.CACHE.DATA_DIR, data_source_file_name='test_data_source.txt',
                        stat_file_name='test_Statistics.json', flow_file_name='test_flow',
                        forcing_file_name='test_forcing', attr_file_name='test_attr',
                        f_dict_file_name='test_dictFactorize.json', var_dict_file_name='test_dictAttribute.json',
                        t_s_dict_file_name='test_dictTimeSpace.json')
    if cfg.CACHE.STATE:
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
            data_set = StreamflowInputDataset(gages_model_train)
            master_train_easier_lstm(data_set)
        test_data_set = StreamflowInputDataset(gages_model_test, train_mode=False)
        pred, obs = master_test_easier_lstm(test_data_set, load_epoch=test_epoch)
        basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                            is_return_dict=False)
        mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                           is_return_dict=False)
        mean_prep = mean_prep / 365 * 10
        pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
        obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
        save_result(gages_model_test.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)


# python gages_conus_analysis_traverse_in_epoch.py --sub basic/exp38 --cache_state 1 --mini_batch 256 365 --train_epoch 50 --save_epoch 2 --te 30
if __name__ == '__main__':
    print("Begin\n")
    # args = cmd(sub="basic/exp38", cache_state=1, mini_batch=[256, 365], train_epoch=50, save_epoch=2,
    #            te=30)
    args = cmd()
    conus_other_lstm(args)
    print("End\n")
