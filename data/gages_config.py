import collections
import os

from data.data_config import DataConfig, wrap_master
from configparser import ConfigParser
from data.download_data import download_kaggle_file


class GagesConfig(DataConfig):
    def __init__(self, config_file):
        super().__init__(config_file)
        opt_data, opt_train, opt_model, opt_loss = self.init_model_param()
        self.model_dict = wrap_master(self.data_path, opt_data, opt_model, opt_loss, opt_train)

    def init_data_param(self):
        """read camels or gages dataset configuration
        根据配置文件读取有关输入数据的各项参数"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        sections = cfg.sections()
        section = cfg.get(sections[0], 'data')
        options = cfg.options(section)

        # time and space range of gages data. 时间空间范围配置项
        t_range_all = eval(cfg.get(section, options[0]))
        regions = eval(cfg.get(section, options[1]))

        # forcing
        forcing_dir = cfg.get(section, options[2])
        forcing_type = cfg.get(section, options[3])
        forcing_url = cfg.get(section, options[4])
        if forcing_url == 'None':
            forcing_url = eval(forcing_url)
        forcing_lst = eval(cfg.get(section, options[5]))

        # streamflow
        streamflow_dir = cfg.get(section, options[6])
        streamflow_url = cfg.get(section, options[7])
        gage_id_screen = eval(cfg.get(section, options[8]))
        streamflow_screen_param = eval(cfg.get(section, options[9]))

        # attribute
        attr_dir = cfg.get(section, options[10])
        attr_url = cfg.get(section, options[11])
        attrBasin = eval(cfg.get(section, options[13]))
        attrLandcover = eval(cfg.get(section, options[14]))
        attrSoil = eval(cfg.get(section, options[15]))
        attrGeol = eval(cfg.get(section, options[16]))
        attrHydro = eval(cfg.get(section, options[17]))
        attrHydroModDams = eval(cfg.get(section, options[18]))
        attr_str_sel = eval(cfg.get(section, options[12]))

        opt_data = collections.OrderedDict(varT=forcing_lst, forcingDir=forcing_dir, forcingType=forcing_type,
                                           forcingUrl=forcing_url,
                                           varC=attr_str_sel, attrDir=attr_dir, attrUrl=attr_url,
                                           streamflowDir=streamflow_dir, streamflowUrl=streamflow_url,
                                           gageIdScreen=gage_id_screen, streamflowScreenParam=streamflow_screen_param,
                                           regions=regions, tRangeAll=t_range_all)

        return opt_data

    def read_data_config(self):
        """读取gages数据项的配置，整理gages数据的独特配置，然后一起返回到一个dict中"""
        dir_db_dict = self.data_path

        dir_db = dir_db_dict.get("DB")
        dir_out = dir_db_dict.get("Out")
        dir_temp = dir_db_dict.get("Temp")
        # 几个根目录文件夹，没有的话就建立
        if not os.path.isdir(dir_db):
            os.mkdir(dir_db)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if not os.path.isdir(dir_temp):
            os.mkdir(dir_temp)
        data_params = self.init_data_param()

        t_range_all = data_params.get("tRangeAll")
        # regions
        ref_nonref_regions = data_params.get("regions")
        # region文件夹
        gage_region_dir = os.path.join(dir_db, 'boundaries-shapefiles-by-aggeco')
        # 站点的point文件文件夹
        gagesii_points_file = os.path.join(dir_db, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp")
        # 调用download_kaggle_file从kaggle上下载,
        huc4_shp_dir = os.path.join(dir_db, "huc4")
        huc4_shp_file = os.path.join(huc4_shp_dir, "HUC4.shp")
        # 这步暂时需要手动放置到指定文件夹下
        kaggle_src = os.path.join(dir_db, 'kaggle.json')
        name_of_dataset = "owenyy/wbdhu4-a-us-september2019-shpfile"
        download_kaggle_file(kaggle_src, name_of_dataset, huc4_shp_dir, huc4_shp_file)

        # 径流数据配置
        flow_dir = os.path.join(dir_db, data_params.get("streamflowDir"))
        flow_url = data_params.get("streamflowUrl")
        flow_screen_gage_id = data_params.get("gageIdScreen")
        flow_screen_param = data_params.get("streamflowScreenParam")
        # 所选forcing
        forcing_chosen = data_params.get("varT")
        forcing_dir = os.path.join(dir_db, data_params.get("forcingDir"))
        forcing_type = data_params.get("forcingType")
        # 有了forcing type之后，确定到真正的forcing数据文件夹
        forcing_dir = os.path.join(forcing_dir, forcing_type)
        forcing_url = data_params.get("forcingUrl")
        # 所选属性
        attr_chosen = data_params.get("varC")
        attr_dir = os.path.join(dir_db, data_params.get("attrDir"))
        # USGS所有站点的文件，gages文件夹下载下来之后文件夹都是固定的
        gage_files_dir = os.path.join(attr_dir, 'spreadsheets-in-csv-format')
        gage_id_file = os.path.join(gage_files_dir, 'conterm_basinid.txt')
        attr_url = data_params.get("attrUrl")
        
        return collections.OrderedDict(root_dir=dir_db, out_dir=dir_out, temp_dir=dir_temp,
                                       regions=ref_nonref_regions,
                                       flow_dir=flow_dir, flow_url=flow_url, flow_screen_gage_id=flow_screen_gage_id,
                                       flow_screen_param=flow_screen_param,
                                       forcing_chosen=forcing_chosen, forcing_dir=forcing_dir,
                                       forcing_type=forcing_type,
                                       forcing_url=forcing_url,
                                       attr_chosen=attr_chosen, attr_dir=attr_dir, attr_url=attr_url,
                                       gage_files_dir=gage_files_dir, gage_id_file=gage_id_file,
                                       gage_region_dir=gage_region_dir, gage_point_file=gagesii_points_file,
                                       huc4_shp_file=huc4_shp_file, t_range_all=t_range_all)