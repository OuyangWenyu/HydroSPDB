"""
target：utilizing camels/gages dataset to train LSTM model and test
pipeline:
    data pre-processing——statistical analysis——model training and testing——visualization of outputs——tune parameters

目的：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
基本流程： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
import os
import argparse
import torch
from easydict import EasyDict as edict
import sys

sys.path.append("../..")
import definitions
from data import GagesConfig
from data.data_input import GagesModel, _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
from visual.plot_model import plot_we_need

config_dir = definitions.CONFIG_DIR
cfg = edict()


def nodam_lstm(args):
    exp_config_file = cfg.EXP
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    dam_num = cfg.DAM_NUM
    print("train and test in CONUS: \n")
    config_file = os.path.join(config_dir, exp_config_file)
    temp_file_subname = exp_config_file.split("/")
    subexp = temp_file_subname[1].split("_")[1][:-4]
    subdir = temp_file_subname[0] + "/" + subexp
    config_data = GagesConfig.set_subdir(config_file, subdir)

    gages_model = GagesModels(config_data, screen_basin_area_huc4=False, dam_num=dam_num)
    gages_model_train = gages_model.data_model_train
    gages_model_test = gages_model.data_model_test
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
    parser = argparse.ArgumentParser(description='Train the basins with dor range in CONUS')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', default="nodam/config_exp1.ini",
                        type=str)
    parser.add_argument('--ctx', dest='ctx',
                        help='Running Context -- gpu num. E.g `--ctx 0` means run code in the context of gpu 0',
                        type=int, default=1)
    parser.add_argument('--rs', dest='rs', help='random seed', default=1234, type=int)
    parser.add_argument('--te', dest='te', help='test epoch', default=20, type=int)
    parser.add_argument('--train_mode', dest='train_mode', help='train or test',
                        default=True, type=bool)
    parser.add_argument('--dam_num', dest='dam_num', help='number of dams in a basin', default=0, type=int)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg.EXP = args.cfg_file
    if args.ctx is not None:
        cfg.CTX = args.ctx
    if args.rs is not None:
        cfg.RANDOM_SEED = args.rs
    if args.te is not None:
        cfg.TEST_EPOCH = args.te
    if args.train_mode is not None:
        cfg.TRAIN_MODE = args.train_mode
    if args.dam_num is not None:
        cfg.DAM_NUM = args.dam_num
    return args


# python gages_nodam_analysis.py --cfg nodam/config_exp1.ini --ctx 1 --dam_num 0 --rs 1234 --te 300 --train_mode True
# python gages_nodam_analysis.py --cfg nodam/config_exp2.ini --ctx 1 --dam_num 0 --rs 123 --te 300 --train_mode True
# python gages_nodam_analysis.py --cfg nodam/config_exp3.ini --ctx 2 --dam_num 0 --rs 12345 --te 300 --train_mode True
# python gages_nodam_analysis.py --cfg nodam/config_exp4.ini --ctx 2 --dam_num 0 --rs 111 --te 300 --train_mode True
# python gages_nodam_analysis.py --cfg nodam/config_exp5.ini --ctx 1 --dam_num 0 --rs 1111 --te 300 --train_mode True
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    nodam_lstm(args)
    print("End\n")
