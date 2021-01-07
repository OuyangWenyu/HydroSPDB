"""
target：utilizing camels/gages dataset to train LSTM model and test
pipeline:
    data pre-processing——statistical analysis——model training and testing——visualization of outputs——tune parameters

目的：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
基本流程： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig
from data.data_input import _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test


def nodam_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    dam_num = cfg.GAGES.attrScreenParams.dam_num
    print("train and test in basins without dams: \n")
    config_data = GagesConfig(cfg)

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


# python gages_nodam_analysis.py --sub nodam/exp1 --ctx 0 --attr_screen {\"dam_num\":0} --train_mode 1
# python gages_nodam_analysis.py --sub nodam/exp2 --ctx 0 --attr_screen {\"dam_num\":0} --rs 123 --train_mode 1
# python gages_nodam_analysis.py --sub nodam/exp3 --ctx 0 --attr_screen {\"dam_num\":0} --rs 12345 --train_mode 1
# python gages_nodam_analysis.py --sub nodam/exp4 --ctx 0 --attr_screen {\"dam_num\":0} --rs 111 --train_mode 1
# python gages_nodam_analysis.py --sub nodam/exp5 --ctx 0 --attr_screen {\"dam_num\":0} --rs 1111 --train_mode 1
# python gages_nodam_analysis.py --sub nodam/exp6 --ctx 0 --attr_screen {\"dam_num\":0} --rs 11111 --train_mode 1
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    nodam_lstm(args)
    print("End\n")
