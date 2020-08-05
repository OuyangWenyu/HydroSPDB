"""
target：utilizing camels/gages dataset to train LSTM model and test
pipeline:
    data pre-processing——statistical analysis——model training and testing——visualization of outputs——tune parameters

目的：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
基本流程： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
import os
import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig
from data.data_input import _basin_norm, save_result, save_datamodel
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
import numpy as np
import pandas as pd


def camels_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    cache = cfg.CACHE.STATE
    print("train and test in CAMELS: \n")
    config_data = GagesConfig(cfg)

    camels531_gageid_file = os.path.join(config_data.data_path["DB"], "camels531", "camels531.txt")
    gauge_df = pd.read_csv(camels531_gageid_file, dtype={"GaugeID": str})
    gauge_list = gauge_df["GaugeID"].values
    all_sites_camels_531 = np.sort([str(gauge).zfill(8) for gauge in gauge_list])
    gages_model = GagesModels(config_data, screen_basin_area_huc4=False, sites_id=all_sites_camels_531.tolist())
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


# python for531camels_conus_analysis.py --sub basic/exp31  --cache_state 1 --ctx 0 --te 20 --train_epoch 20 --save_epoch 10
# python for531camels_conus_analysis.py --cfg basic/config_exp32.ini --ctx 1 --rs 123 --te 300 --train_mode True
# python for531camels_conus_analysis.py --cfg basic/config_exp33.ini --ctx 0 --rs 12345 --te 300 --train_mode True
# python for531camels_conus_analysis.py --cfg basic/config_exp34.ini --ctx 0 --rs 111 --te 300 --train_mode True
# python for531camels_conus_analysis.py --cfg basic/config_exp35.ini --ctx 1 --rs 1111 --te 300 --train_mode True
# retrain: python for531camels_conus_analysis.py --cfg basic/config_exp49.ini --ctx 1 --rs 1111 --te 300 --train_mode True
# python for531camels_conus_analysis.py --cfg basic/config_exp36.ini --ctx 1 --rs 11111 --te 300 --train_mode True
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    camels_lstm(args)
    print("End\n")
