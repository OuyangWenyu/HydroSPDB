import collections
import os

from data.data_config import DataConfig, wrap_master
from configparser import ConfigParser


class CamelsConfig(DataConfig):
    def __init__(self, config_file):
        super().__init__(config_file)
        opt_data, opt_train, opt_model, opt_loss = self.init_model_param()
        self.model_dict = wrap_master(self.data_path, opt_data, opt_model, opt_loss, opt_train)

    @classmethod
    def set_subdir(cls, config_file, subdir):
        """ set_subdir for "temp" and "output" """
        new_data_config = cls(config_file)
        new_data_config.data_path["Out"] = os.path.join(new_data_config.data_path["Out"], subdir)
        new_data_config.data_path["Temp"] = os.path.join(new_data_config.data_path["Temp"], subdir)
        if not os.path.isdir(new_data_config.data_path["Out"]):
            os.makedirs(new_data_config.data_path["Out"])
        if not os.path.isdir(new_data_config.data_path["Temp"]):
            os.makedirs(new_data_config.data_path["Temp"])
        return new_data_config

    def init_data_param(self):
        """read camels or gages dataset configuration
        根据配置文件读取有关输入数据的各项参数"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)

        # forcing
        forcing_dir = cfg.get(section, options[0])
        forcing_type = cfg.get(section, options[1])
        forcing_url = cfg.get(section, options[2])
        forcing_lst = eval(cfg.get(section, options[3]))

        # streamflow
        streamflow_dir = cfg.get(section, options[4])
        gage_id_screen = eval(cfg.get(section, options[5]))

        # attribute
        attr_dir = cfg.get(section, options[6])
        attr_url = eval(cfg.get(section, options[7]))
        attr_str_sel = eval(cfg.get(section, options[8]))

        opt_data = collections.OrderedDict(varT=forcing_lst, forcingDir=forcing_dir, forcingType=forcing_type,
                                           forcingUrl=forcing_url,
                                           varC=attr_str_sel, attrDir=attr_dir, attrUrl=attr_url,
                                           streamflowDir=streamflow_dir, gageIdScreen=gage_id_screen)

        return opt_data

    def read_data_config(self):
        dir_db = self.data_path.get("DB")
        dir_out = self.data_path.get("Out")
        dir_temp = self.data_path.get("Temp")

        data_params = self.init_data_param()
        # 站点的shp file
        camels_shp_file = os.path.join(dir_db, "basin_set_full_res", "HCDN_nhru_final_671.shp")
        # 径流数据配置
        flow_dir = os.path.join(dir_db, data_params.get("streamflowDir"))
        flow_screen_gage_id = data_params.get("gageIdScreen")
        # 所选forcing
        forcing_chosen = data_params.get("varT")
        forcing_dir = os.path.join(dir_db, data_params.get("forcingDir"))
        forcing_type = data_params.get("forcingType")
        # 有了forcing type之后，确定到真正的forcing数据文件夹
        forcing_dir = os.path.join(forcing_dir, forcing_type)
        forcing_url = data_params.get("forcingUrl")
        # 所选属性
        attr_url = data_params.get("attrUrl")
        attr_chosen = data_params.get("varC")
        attr_dir = os.path.join(dir_db, data_params.get("attrDir"))
        gauge_id_dir = "basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata"
        gauge_id_file = os.path.join(dir_db, gauge_id_dir, 'gauge_information.txt')

        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       flow_dir=flow_dir, flow_screen_gage_id=flow_screen_gage_id,
                                       forcing_chosen=forcing_chosen, forcing_dir=forcing_dir,
                                       forcing_type=forcing_type, forcing_url=forcing_url,
                                       attr_url=attr_url, attr_chosen=attr_chosen, attr_dir=attr_dir,
                                       gauge_id_file=gauge_id_file, gauge_shp_file=camels_shp_file)
