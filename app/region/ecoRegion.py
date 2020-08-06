"""train and test in basins from different regions"""
import argparse
import os
import sys
from easydict import EasyDict as edict
import numpy as np
import torch

sys.path.append("../..")
import definitions
from data import GagesConfig, GagesSource
from data.data_input import GagesModel, save_datamodel, _basin_norm, save_result
from hydroDL import master_train, master_test

config_dir = definitions.CONFIG_DIR
cfg = edict()


def synergy_ecoregion(args):
    exp_config_file = cfg.EXP
    random_seed = cfg.RANDOM_SEED
    test_epoch = cfg.TEST_EPOCH
    gpu_num = cfg.CTX
    train_mode = cfg.TRAIN_MODE
    cache = cfg.CACHE
    print("train and test in CONUS: \n")
    config_file = os.path.join(config_dir, exp_config_file)
    temp_file_subname = exp_config_file.split("/")
    subexp = temp_file_subname[1].split("_")[1][:-4]
    subdir = temp_file_subname[0] + "/" + subexp
    config_data = GagesConfig.set_subdir(config_file, subdir)
    eco_names = [("ECO2_CODE", 5.2), ("ECO2_CODE", 5.3), ("ECO2_CODE", 6.2), ("ECO2_CODE", 7.1),
                 ("ECO2_CODE", 8.1), ("ECO2_CODE", 8.2), ("ECO2_CODE", 8.3), ("ECO2_CODE", 8.4),
                 ("ECO2_CODE", 8.5), ("ECO2_CODE", 9.2), ("ECO2_CODE", 9.3), ("ECO2_CODE", 9.4),
                 ("ECO2_CODE", 9.5), ("ECO2_CODE", 9.6), ("ECO2_CODE", 10.1), ("ECO2_CODE", 10.2),
                 ("ECO2_CODE", 10.4), ("ECO2_CODE", 11.1), ("ECO2_CODE", 12.1), ("ECO2_CODE", 13.1)]

    quick_data_dir = os.path.join(config_data.data_path["DB"], "quickdata")
    data_dir = os.path.join(quick_data_dir, "conus-all_90-10_nan-0.0_00-1.0")
    data_model_train = GagesModel.load_datamodel(data_dir,
                                                 data_source_file_name='data_source.txt',
                                                 stat_file_name='Statistics.json', flow_file_name='flow.npy',
                                                 forcing_file_name='forcing.npy', attr_file_name='attr.npy',
                                                 f_dict_file_name='dictFactorize.json',
                                                 var_dict_file_name='dictAttribute.json',
                                                 t_s_dict_file_name='dictTimeSpace.json')
    data_model_test = GagesModel.load_datamodel(data_dir,
                                                data_source_file_name='test_data_source.txt',
                                                stat_file_name='test_Statistics.json',
                                                flow_file_name='test_flow.npy',
                                                forcing_file_name='test_forcing.npy',
                                                attr_file_name='test_attr.npy',
                                                f_dict_file_name='test_dictFactorize.json',
                                                var_dict_file_name='test_dictAttribute.json',
                                                t_s_dict_file_name='test_dictTimeSpace.json')

    for eco_name in eco_names:
        source_data = GagesSource.choose_some_basins(config_data,
                                                     config_data.model_dict["data"]["tRangeTrain"],
                                                     screen_basin_area_huc4=False, ecoregion=eco_name)
        sites_id = source_data.all_configs['flow_screen_gage_id']
        sites_id_inter = np.intersect1d(data_model_train.t_s_dict["sites_id"], sites_id)
        if sites_id_inter.size < 1:
            continue
        subdir_i = os.path.join(subdir, str(eco_name[1]))
        config_data_i = GagesConfig.set_subdir(config_file, subdir_i)
        gages_model_train = GagesModel.update_data_model(config_data_i, data_model_train,
                                                         sites_id_update=sites_id,
                                                         data_attr_update=True, screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(config_data_i, data_model_test, sites_id_update=sites_id,
                                                        data_attr_update=True,
                                                        train_stat_dict=gages_model_train.stat_dict,
                                                        screen_basin_area_huc4=False)
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
            print("save ecoregion " + str(eco_name[1]) + " data model")

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
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file',
                        default="ecoregion/config_exp15.ini", type=str)
    parser.add_argument('--ctx', dest='ctx',
                        help='Running Context -- gpu num. E.g `--ctx 0` means run code in the context of gpu 0',
                        type=int, default=1)
    parser.add_argument('--rs', dest='rs', help='random seed', default=1234, type=int)
    parser.add_argument('--te', dest='te', help='test epoch', default=20, type=int)
    parser.add_argument('--cache', dest='cache', help='Does save the cache', default=0, type=int)
    parser.add_argument('--train_mode', dest='train_mode', help='train or test',
                        default=True, type=bool)
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
    if args.cache is not None:
        if args.cache > 0:
            cfg.CACHE = True
        else:
            cfg.CACHE = False
    return args


# python ecoRegion.py --cfg ecoregion/config_exp15.ini --ctx 1 --rs 1234 --te 300 --train_mode True --cache 1
# python ecoRegion.py --cfg ecoregion/config_exp16.ini --ctx 1 --rs 123 --te 300 --train_mode True
# python ecoRegion.py --cfg ecoregion/config_exp17.ini --ctx 0 --rs 12345 --te 300 --train_mode True
# python ecoRegion.py --cfg ecoregion/config_exp18.ini --ctx 1 --rs 111 --te 300 --train_mode True
# python ecoRegion.py --cfg ecoregion/config_exp19.ini --ctx 1 --rs 1111 --te 300 --train_mode True
# python ecoRegion.py --cfg ecoregion/config_exp20.ini --ctx 0 --rs 11111 --te 300 --train_mode True
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    synergy_ecoregion(args)
    print("End\n")
