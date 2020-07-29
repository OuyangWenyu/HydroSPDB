"""
target：utilizing camels/gages dataset to train LSTM model and test
pipeline:
    data pre-processing——statistical analysis——model training and testing——visualization of outputs——tune parameters

目的：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
基本流程： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
import json
import argparse
import os

import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg
from data import GagesConfig
from data.data_input import _basin_norm, save_result, save_quick_data, save_datamodel
from data.gages_input_dataset import GagesModels, generate_gages_models
from hydroDL import master_train, master_test


def conus_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    print("train and test in CONUS: \n")
    print(cfg)
    config_data = GagesConfig(cfg)
    if cfg.CACHE.QUICK_DATA:
        data_dir = cfg.CACHE.DATA_DIR
        gages_model_train, gages_model_test = generate_gages_models(config_data, data_dir, screen_basin_area_huc4=False)
    else:
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


def cmd():
    """input args from cmd"""
    parser = argparse.ArgumentParser(description='Train the CONUS model')
    parser.add_argument('--sub', dest='sub', help='subset and sub experiment', default="basic/exp37", type=str)
    parser.add_argument('--ctx', dest='ctx',
                        help='Running Context -- gpu num. E.g `--ctx 0` means run code in the context of gpu 0',
                        type=int, default=0)
    parser.add_argument('--rs', dest='rs', help='random seed', default=1234, type=int)
    parser.add_argument('--te', dest='te', help='test epoch', default=300, type=int)
    # There is something wrong with "bool", so I used 1 as True, 0 as False
    parser.add_argument('--train_mode', dest='train_mode', help='train or test', default=1, type=int)
    parser.add_argument('--train_epoch', dest='train_epoch', help='epoches of training period', default=340, type=int)
    parser.add_argument('--save_epoch', dest='save_epoch', help='save for every save_epoch epoches', default=20,
                        type=int)
    parser.add_argument('--regions', dest='regions',
                        help='There are 10 regions in GAGES-II. One is reference region, others are non-ref regions',
                        default=['bas_ref_all', 'bas_nonref_CntlPlains', 'bas_nonref_EastHghlnds',
                                 'bas_nonref_MxWdShld', 'bas_nonref_NorthEast', 'bas_nonref_SECstPlain',
                                 'bas_nonref_SEPlains', 'bas_nonref_WestMnts', 'bas_nonref_WestPlains',
                                 'bas_nonref_WestXeric'], type=list)
    # parser.add_argument('--regions', dest='regions',
    #                     help='There are 10 regions in GAGES-II. One is reference region, others are non-ref regions',
    #                     default=['bas_nonref_MxWdShld'], type=list)
    parser.add_argument('--gage_id', dest='gage_id', help='just select some sites',
                        default=None, type=list)
    parser.add_argument('--flow_screen', dest='flow_screen',
                        help='screen some sites according to their streamflow record',
                        default={'missing_data_ratio': 0, 'zero_value_ratio': 1}, type=json.loads)
    parser.add_argument('--var_c', dest='var_c', help='types of attributes', default=None, type=list)
    parser.add_argument('--var_t', dest='var_t', help='types of forcing', default=None, type=list)
    parser.add_argument('--gen_quick_data', dest='gen_quick_data', help='do I generate quick data?', default=0,
                        type=int)
    parser.add_argument('--quick_data', dest='quick_data', help='Has quick data existed?', default=1, type=int)
    parser.add_argument('--cache_state', dest='cache_state', help='Does save the data model for the sub experiment?',
                        default=0, type=int)
    args = parser.parse_args()
    return args


# python gages_conus_analysis.py --sub basic/exp37 --gen_quick_data 1 --quick_data 0 --cache_state 1
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    conus_lstm(args)
    print("End\n")
