"""train and test in basins from different regions"""
import os
import sys

import numpy as np
import torch

sys.path.append("../..")
from data import GagesConfig, GagesSource
from data.config import cfg, cmd, update_cfg
from data.data_input import GagesModel, save_datamodel, _basin_norm, save_result
from hydroDL import master_train, master_test


def synergy_ecoregion(args):
    update_cfg(cfg, args)
    cache = cfg.CACHE.STATE
    train_mode = cfg.TRAIN_MODE
    test_epoch = cfg.TEST_EPOCH
    config_data = GagesConfig(cfg)
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
        config_data = GagesConfig.set_subdir(cfg, str(eco_name[1]))
        gages_model_train = GagesModel.update_data_model(config_data, data_model_train,
                                                         sites_id_update=sites_id,
                                                         data_attr_update=True, screen_basin_area_huc4=False)
        gages_model_test = GagesModel.update_data_model(config_data, data_model_test, sites_id_update=sites_id,
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

        with torch.cuda.device(0):
            if train_mode:
                master_train(gages_model_train)
            pred, obs = master_test(gages_model_test, epoch=test_epoch)
            basin_area = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['DRAIN_SQKM'],
                                                                is_return_dict=False)
            mean_prep = gages_model_test.data_source.read_attr(gages_model_test.t_s_dict["sites_id"], ['PPTAVG_BASIN'],
                                                               is_return_dict=False)
            mean_prep = mean_prep / 365 * 10
            pred = _basin_norm(pred, basin_area, mean_prep, to_norm=False)
            obs = _basin_norm(obs, basin_area, mean_prep, to_norm=False)
            save_result(gages_model_test.data_source.data_config.data_path['Temp'], test_epoch, pred, obs)


# python ecoregions.py --sub ecoregion/exp1 --train_epoch 2 --save_epoch 1 --te 2
if __name__ == '__main__':
    print("Begin\n")
    args = cmd()
    synergy_ecoregion(args)
    print("End\n")
