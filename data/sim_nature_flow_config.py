"""a data config class for simulated natural flow generated by model trained by dataset of ref basins"""
import collections
import os
from configparser import ConfigParser

from data import DataConfig, wrap_master


class SimNatureFlowConfig(DataConfig):
    def __init__(self, config_file):
        super().__init__(config_file)
        opt_data, opt_train, opt_model, opt_loss = self.init_model_param()
        self.model_dict = wrap_master(self.data_path, opt_data, opt_model, opt_loss, opt_train)

    def init_data_param(self):
        """read camels or gages dataset configuration"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)

        t_range_all = cfg.get(section, options[0])
        ref_regions = cfg.get(section, options[1])
        nonref_regions = cfg.get(section, options[2])
        streamflow_dir = cfg.get(section, options[3])
        streamflow_url = cfg.get(section, options[4])
        gage_id_of_ref_screen = eval(cfg.get(section, options[5]))
        gage_id_of_non_ref_screen = eval(cfg.get(section, options[6]))

        opt_data = collections.OrderedDict(streamflowUrl=streamflow_url, tRangeAll=t_range_all, refRegions=ref_regions,
                                           nonrefRegions=nonref_regions, streamflowDir=streamflow_dir,
                                           gageIdOfRefScreen=gage_id_of_ref_screen,
                                           gageIdOfNonRefScreen=gage_id_of_non_ref_screen)

        return opt_data

    def read_data_config(self):
        dir_db = self.data_path.get("DB")
        dir_out = self.data_path.get("Out")
        dir_temp = self.data_path.get("Temp")

        data_params = self.init_data_param()

        # 径流数据配置
        flow_dir = os.path.join(dir_db, data_params.get("streamflowDir"))
        flow_screen_gage_id_ref_screen = data_params.get("gageIdOfRefScreen")
        flow_screen_gage_id_non_ref_screen = data_params.get("gageIdOfNonRefScreen")

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       flow_dir=flow_dir, flow_screen_gage_id_ref_screen=flow_screen_gage_id_ref_screen,
                                       flow_screen_gage_id_non_ref_screen=flow_screen_gage_id_non_ref_screen)
