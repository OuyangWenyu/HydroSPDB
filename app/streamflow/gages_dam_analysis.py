"""small-dor basins"""
import torch
import sys

sys.path.append("../..")
from data.config import cfg, update_cfg, cmd
from data import GagesConfig, GagesSource
from data.data_input import _basin_norm, save_result
from data.gages_input_dataset import GagesModels
from hydroDL import master_train, master_test
import numpy as np


def dam_lstm(args):
    update_cfg(cfg, args)
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    dor = cfg.GAGES.attrScreenParams.DOR
    print("train and test in basins with dams: \n")
    config_data = GagesConfig(cfg)

    source_data_dor1 = GagesSource.choose_some_basins(config_data,
                                                      config_data.model_dict["data"]["tRangeTrain"],
                                                      screen_basin_area_huc4=False,
                                                      DOR=dor)
    # basins with dams
    source_data_withdams = GagesSource.choose_some_basins(config_data,
                                                          config_data.model_dict["data"]["tRangeTrain"],
                                                          screen_basin_area_huc4=False,
                                                          dam_num=[1, 100000])

    sites_id_dor1 = source_data_dor1.all_configs['flow_screen_gage_id']
    sites_id_withdams = source_data_withdams.all_configs['flow_screen_gage_id']
    sites_id_chosen = np.intersect1d(np.array(sites_id_dor1), np.array(sites_id_withdams)).tolist()

    gages_model = GagesModels(config_data, screen_basin_area_huc4=False, sites_id=sites_id_chosen)
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


# python gages_dam_analysis.py --sub dam/exp20 --ctx 0 --attr_screen {\"DOR\":-0.02} --train_mode 1
# python gages_dam_analysis.py --sub dam/exp21 --ctx 0 --attr_screen {\"DOR\":-0.02}  --rs 123 --train_mode 1
# python gages_dam_analysis.py --sub dam/exp22 --ctx 0 --attr_screen {\"DOR\":-0.02}  --rs 12345 --train_mode 1
# python gages_dam_analysis.py --sub dam/exp23 --ctx 0 --attr_screen {\"DOR\":-0.02}  --rs 111 --train_mode 1
# python gages_dam_analysis.py --sub dam/exp24 --ctx 0 --attr_screen {\"DOR\":-0.02}  --rs 1111 --train_mode 1
# python gages_dam_analysis.py --sub dam/exp25 --ctx 0 --attr_screen {\"DOR\":-0.02}  --rs 11111 --train_mode 1
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    dam_lstm(args)
    print("End\n")
