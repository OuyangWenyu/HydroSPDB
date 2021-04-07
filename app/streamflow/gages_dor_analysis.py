"""1. zero-dor and small-dor basins 2. large-dor basins"""
import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig
from data.data_input import _basin_norm, save_result, save_datamodel
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test


def dor_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    dor = cfg.GAGES.attrScreenParams.DOR
    cache = cfg.CACHE.STATE
    print("train and test in some dor basins: \n")
    config_data = GagesConfig(cfg)

    gages_model = GagesModels(config_data, screen_basin_area_huc4=False, DOR=dor)
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


# python gages_dor_analysis.py --sub dam/exp1 --ctx 0 --attr_screen {\"DOR\":-0.02} --cache_state 1
# python gages_dor_analysis.py --sub dam/exp2 --attr_screen {\"DOR\":-0.02} --rs 123
# python gages_dor_analysis.py --sub dam/exp3 --ctx 0 --attr_screen {\"DOR\":-0.02} --rs 12345
# python gages_dor_analysis.py --sub dam/exp7 --ctx 0 --attr_screen {\"DOR\":-0.02} --rs 111
# python gages_dor_analysis.py --sub dam/exp8 --ctx 0 --attr_screen {\"DOR\":-0.02} --rs 1111
# python gages_dor_analysis.py --sub dam/exp9 --ctx 0 --attr_screen {\"DOR\":-0.02} --rs 11111
# python gages_dor_analysis.py --sub dam/exp4 --ctx 0 --attr_screen {\"DOR\":0.02}
# python gages_dor_analysis.py --sub dam/exp5 --ctx 0 --attr_screen {\"DOR\":0.02} --rs 123
# python gages_dor_analysis.py --sub dam/exp6 --ctx 0 --attr_screen {\"DOR\":0.02} --rs 12345
# python gages_dor_analysis.py --sub dam/exp13 --ctx 0 --attr_screen {\"DOR\":0.02} --rs 111
# python gages_dor_analysis.py --sub dam/exp16 --ctx 0 --attr_screen {\"DOR\":0.02} --rs 1111
# python gages_dor_analysis.py --sub dam/exp19 --ctx 0 --attr_screen {\"DOR\":0.02} --rs 11111

# python gages_dor_analysis.py --sub dam/exp12 --ctx 0 --attr_screen {\"DOR\":0.003}
# python gages_dor_analysis.py --sub dam/exp18 --ctx 0 --attr_screen {\"DOR\":0.08}
# python gages_dor_analysis.py --sub dam/exp33 --ctx 0 --attr_screen {\"DOR\":1}
# python gages_dor_analysis.py --sub dam/exp34 --ctx 0 --attr_screen {\"DOR\":-0.003}
# python gages_dor_analysis.py --sub dam/exp35 --ctx 0 --attr_screen {\"DOR\":-0.08}
# python gages_dor_analysis.py --sub dam/exp36 --ctx 0 --attr_screen {\"DOR\":-1}

# python gages_dor_analysis.py --sub dam/exp37 --ctx 0 --attr_screen {\"DOR\":-0.1} --cache_state 1
# python gages_dor_analysis.py --sub dam/exp38 --ctx 0 --attr_screen {\"DOR\":0.1} --cache_state 1
# python gages_dor_analysis.py --sub dam/exp40 --ctx 0 --attr_screen {\"DOR\":-0.1} --rs 123
# python gages_dor_analysis.py --sub dam/exp41 --ctx 0 --attr_screen {\"DOR\":0.1} --rs 123
# python gages_dor_analysis.py --sub dam/exp43 --ctx 0 --attr_screen {\"DOR\":-0.1} --rs 12345
# python gages_dor_analysis.py --sub dam/exp44 --ctx 0 --attr_screen {\"DOR\":0.1} --rs 12345
# python gages_dor_analysis.py --sub dam/exp46 --ctx 0 --attr_screen {\"DOR\":-0.1} --rs 111
# python gages_dor_analysis.py --sub dam/exp47 --ctx 0 --attr_screen {\"DOR\":0.1} --rs 111
# python gages_dor_analysis.py --sub dam/exp49 --ctx 0 --attr_screen {\"DOR\":-0.1} --rs 1111
# python gages_dor_analysis.py --sub dam/exp50 --ctx 0 --attr_screen {\"DOR\":0.1} --rs 1111
# python gages_dor_analysis.py --sub dam/exp52 --ctx 0 --attr_screen {\"DOR\":-0.1} --rs 11111
# python gages_dor_analysis.py --sub dam/exp53 --ctx 0 --attr_screen {\"DOR\":0.1} --rs 11111
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    dor_lstm(args)
    print("End\n")
