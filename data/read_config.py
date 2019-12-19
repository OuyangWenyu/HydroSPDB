import collections
import os
from configparser import ConfigParser

from data.download_data import download_kaggle_file


def init_path(config_file):
    """根据配置文件读取数据源路径"""
    cfg = ConfigParser()
    cfg.read(config_file)
    sections = cfg.sections()
    data_input = cfg.get(sections[0], 'download')
    data_output = cfg.get(sections[0], 'output')

    path_data = collections.OrderedDict(
        DB=os.path.join(data_input, cfg.get(sections[0], 'data')),
        Out=os.path.join(data_output, cfg.get(sections[0], 'data')))

    return path_data


def init_data_param(config_file):
    """根据配置文件读取有关输入数据的各项参数"""
    cfg = ConfigParser()
    cfg.read(config_file)
    sections = cfg.sections()
    section = cfg.get(sections[0], 'data')
    options = cfg.options(section)
    forcing_lst = cfg.get(section, options[0])
    attr_str_sel = cfg.get(section, options[1])
    streamflow_data = cfg.get(section, options[2])
    t_range_train = cfg.get(section, options[3])
    regions = cfg.get(section, options[4])
    do_norm = cfg.get(section, options[5])
    rm_nan = cfg.get(section, options[6])
    da_obs = cfg.get(section, options[7])
    return collections.OrderedDict(varT=forcing_lst, varC=attr_str_sel, streamflowData=streamflow_data,
                                   tRange=t_range_train, regions=regions, doNorm=do_norm, rmNan=rm_nan, daObs=da_obs)


def init_model_param(config_file, optDataParam):
    """根据配置文件读取有关模型的各项参数，返回optModel, optLoss, optTrain三组参数"""
    cfg = ConfigParser()
    cfg.read(config_file)
    section = 'model'
    options = cfg.options(section)
    mini_batch = cfg.get(section, options[0])
    n_epoch = cfg.get(section, options[1])
    save_epoch = cfg.get(section, options[2])
    collection1 = collections.OrderedDict(miniBatch=mini_batch, nEpoch=n_epoch, saveEpoch=save_epoch)

    ny = cfg.get(section, options[3])
    hidden_size = cfg.get(section, options[4])
    do_relu = cfg.get(section, options[5])
    collection2 = collections.OrderedDict(name='hydroDL.model.rnn.CudnnLstmModel',
                                          nx=len(optDataParam['varT']) + len(optDataParam['varC']), ny=ny,
                                          hiddenSize=hidden_size, doReLU=do_relu)

    prior = cfg.get(section, options[6])
    collection3 = collections.OrderedDict(name='hydroDL.model.crit.RmseLoss', prior=prior)
    return collection1, collection2, collection3


def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt


def read_gages_config(config_file):
    """读取gages数据项的配置，整理gages数据的独特配置，然后一起返回到一个dict中"""
    dir_db_dict = init_path(config_file)
    dir_db = dir_db_dict.get("DB")
    # USGS所有站点的文件，gages文件夹下载下来之后文件夹都是固定的
    dir_gage_attr = os.path.join(dir_db, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
    gage_id_file = os.path.join(dir_gage_attr, 'conterm_basinid.txt')

    data_params = init_data_param(config_file)
    # 径流数据配置
    flow_dir = data_params.get("streamflowDir")
    flow_url = data_params.get("streamflowUrl")
    # 所选forcing
    forcing_chosen = data_params.get("varT")
    forcing_dir = data_params.get("forcingDir")
    forcing_type = data_params.get("forcingType")
    forcing_url = data_params.get("forcingUrl")
    # 所选属性
    attr_chosen = data_params.get("varC")
    attr_dir = data_params.get("attrDir")
    attr_url = data_params.get("attrUrl")
    # time range
    t_range_train = data_params.get("tRangeTrain")
    t_range_test = data_params.get("tRangeTest")
    # regions
    ref_nonref_regions = data_params.get("regions")
    # region文件夹
    gage_region_dir = os.path.join(dir_db, 'boundaries-shapefiles-by-aggeco')
    # 站点的point文件文件夹
    gagesii_points_file = os.path.join(dir_db, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp")
    # 调用download_kaggle_file从kaggle上下载,
    huc4_shp_file = os.path.join(dir_db, "huc4", "HUC4.shp")
    # 这步暂时需要手动放置到指定文件夹下
    kaggle_src = os.path.join(dir_db, 'kaggle.json')
    name_of_dataset = "owenyy/wbdhu4-a-us-september2019-shpfile"
    download_kaggle_file(kaggle_src, name_of_dataset, huc4_shp_file)
    return collections.OrderedDict(root_dir=dir_db,
                                   t_range_train=t_range_train, t_range_test=t_range_test, regions=ref_nonref_regions,
                                   flow_dir=flow_dir, flow_url=flow_url,
                                   forcing_chosen=forcing_chosen, forcing_dir=forcing_dir, forcing_type=forcing_type,
                                   forcing_url=forcing_url,
                                   attr_chosen=attr_chosen, attr_dir=attr_dir, attr_url=attr_url,
                                   gage_id_file=gage_id_file, gage_region_dir=gage_region_dir,
                                   gage_point_file=gagesii_points_file, huc4_shp_file=huc4_shp_file
                                   )
