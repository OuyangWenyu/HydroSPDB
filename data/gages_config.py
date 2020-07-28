import collections
import os
import shutil

import definitions
from data.data_config import DataConfig, wrap_master
from configparser import ConfigParser


class GagesConfig(DataConfig):
    def __init__(self, config_file):
        super().__init__(config_file)
        opt_data, opt_train, opt_model, opt_loss = self.init_model_param()
        self.model_dict = wrap_master(self.data_path, opt_data, opt_model, opt_loss, opt_train)

    @classmethod
    def set_subdir(cls, config_file, subdir):
        """ set_subdir for "temp" and "output" """
        new_data_config = cls(config_file)
        print("set sub directory")
        new_data_config.data_path["Out"] = os.path.join(new_data_config.data_path["Out"], subdir)
        new_data_config.data_path["Temp"] = os.path.join(new_data_config.data_path["Temp"], subdir)
        if not os.path.isdir(new_data_config.data_path["Out"]):
            os.makedirs(new_data_config.data_path["Out"])
        if not os.path.isdir(new_data_config.data_path["Temp"]):
            os.makedirs(new_data_config.data_path["Temp"])
        new_data_config.model_dict["dir"]["Out"] = new_data_config.data_path["Out"]
        new_data_config.model_dict["dir"]["Temp"] = new_data_config.data_path["Temp"]
        return new_data_config

    def read_data_config(self):
        """put all configs into a new OrderedDict. use more easily-understand keys"""
        cfg = self.config_file
        dir_db_dict = self.data_path
        dir_db = dir_db_dict.get("DB")
        dir_out = dir_db_dict.get("Out")
        dir_temp = dir_db_dict.get("Temp")

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       regions=cfg.GAGES.regions, flow_dir=cfg.GAGES.streamflowDir,
                                       flow_url=cfg.GAGES.streamflowUrl, flow_screen_gage_id=cfg.GAGES.gageIdScreen,
                                       flow_screen_param=cfg.GAGES.streamflowScreenParams,
                                       forcing_chosen=cfg.GAGES.varT, forcing_dir=cfg.GAGES.forcingDir,
                                       forcing_type=cfg.GAGES.forcingType, attr_chosen=cfg.GAGES.varC,
                                       attr_dir=cfg.GAGES.attrDir, attr_url=cfg.GAGES.attrUrl,
                                       gage_files_dir=cfg.GAGES.gage_files_dir, gage_id_file=cfg.GAGES.gage_id_file,
                                       gage_region_dir=cfg.GAGES.gage_region_dir,
                                       gage_point_file=cfg.GAGES.gagesii_points_file,
                                       huc4_shp_file=cfg.GAGES.huc4_shp_file, t_range_all=cfg.GAGES.tRangeAll,
                                       population_file=cfg.GAGES.population_file, wateruse_file=cfg.GAGES.wateruse_file)
