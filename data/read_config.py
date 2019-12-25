import collections
import json
import os
from collections import OrderedDict
from configparser import ConfigParser

from data.download_data import download_kaggle_file


def init_path(config_file):
    """根据配置文件读取数据源路径"""
    cfg = ConfigParser()
    cfg.read(config_file)
    sections = cfg.sections()
    data_input = cfg.get(sections[0], 'download')
    data_output = cfg.get(sections[0], 'output')
    root = os.path.expanduser('~')
    data_input = os.path.join(root, data_input[2:])
    data_output = os.path.join(root, data_output[2:])
    path_data = collections.OrderedDict(
        DB=os.path.join(data_input, cfg.get(sections[0], 'data')),
        Out=os.path.join(data_output, cfg.get(sections[0], 'data')))
    print(path_data)
    return path_data


def init_data_param(config_file):
    """根据配置文件读取有关输入数据的各项参数"""
    cfg = ConfigParser()
    cfg.read(config_file)
    sections = cfg.sections()
    section = cfg.get(sections[0], 'data')
    options = cfg.options(section)

    # 关于数据处理的一些配置项
    do_norm = eval(cfg.get(section, options[0]))
    rm_nan = eval(cfg.get(section, options[1]))
    da_obs = eval(cfg.get(section, options[2]))

    # 时间空间范围配置项
    t_range_train = eval(cfg.get(section, options[3]))
    t_range_test = eval(cfg.get(section, options[4]))
    regions = eval(cfg.get(section, options[5]))

    # forcing数据
    forcing_dir = cfg.get(section, options[6])
    forcing_type = cfg.get(section, options[7])
    forcing_url = cfg.get(section, options[8])
    if forcing_url == 'None':
        forcing_url = eval(forcing_url)
    forcing_lst = eval(cfg.get(section, options[9]))

    # streamflow数据
    streamflow_dir = cfg.get(section, options[10])
    streamflow_url = cfg.get(section, options[11])

    # attribute数据
    attr_dir = cfg.get(section, options[12])
    attr_url = cfg.get(section, options[13])
    attrBasin = eval(cfg.get(section, options[15]))
    attrLandcover = eval(cfg.get(section, options[16]))
    attrSoil = eval(cfg.get(section, options[17]))
    attrGeol = eval(cfg.get(section, options[18]))
    attrHydro = eval(cfg.get(section, options[19]))
    attrHydroModDams = eval(cfg.get(section, options[20]))
    attr_str_sel = eval(cfg.get(section, options[14]))

    opt_data = collections.OrderedDict(varT=forcing_lst, forcingDir=forcing_dir, forcingType=forcing_type,
                                       forcingUrl=forcing_url,
                                       varC=attr_str_sel, attrDir=attr_dir, attrUrl=attr_url,
                                       streamflowDir=streamflow_dir, streamflowUrl=streamflow_url,
                                       tRangeTrain=t_range_train, tRangeTest=t_range_test, regions=regions,
                                       doNorm=do_norm, rmNan=rm_nan, daObs=da_obs)
    return opt_data


def init_model_param(config_file):
    """根据配置文件读取有关模型的各项参数，返回optModel, optLoss, optTrain三组参数，分成几组的原因是为写成json文件时更清晰"""
    cfg = ConfigParser()
    cfg.read(config_file)
    section = 'model'
    options = cfg.options(section)

    # 首先读取几个训练使用的基本模型参数，主要是epoch和batch
    mini_batch = eval(cfg.get(section, options[0]))
    n_epoch = eval(cfg.get(section, options[1]))
    save_epoch = eval(cfg.get(section, options[2]))
    opt_train = collections.OrderedDict(miniBatch=mini_batch, nEpoch=n_epoch, saveEpoch=save_epoch)

    # 接下来是第二部分，读取数据配置项，
    opt_data = init_data_param(config_file)

    # 接着是模型输入输出的相关参数。根据opt_data判断输入输出变量个数，确定模型基本结构
    model_name = cfg.get(section, options[3])
    # 变量名不要修改
    varT = opt_data["varT"]
    varC = opt_data["varC"]
    nx = eval(cfg.get(section, options[4]))
    ny = cfg.get(section, options[5])
    hidden_size = cfg.get(section, options[6])
    do_relu = cfg.get(section, options[7])
    opt_model = collections.OrderedDict(name=model_name, nx=nx, ny=ny, hiddenSize=hidden_size, doReLU=do_relu)

    # 最后是loss的配置
    loss_name = cfg.get(section, options[8])
    prior = cfg.get(section, options[9])
    opt_loss = collections.OrderedDict(name=loss_name, prior=prior)
    return opt_train, opt_data, opt_model, opt_loss


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

    data_params = init_data_param(config_file)
    # 径流数据配置
    flow_dir = os.path.join(dir_db, data_params.get("streamflowDir"))
    flow_url = data_params.get("streamflowUrl")
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
    gage_id_file = os.path.join(attr_dir, 'spreadsheets-in-csv-format', 'conterm_basinid.txt')
    attr_url = data_params.get("attrUrl")
    # time range, TODO 直接在这里转换为时间对象
    t_range_train = data_params.get("tRangeTrain")
    t_range_test = data_params.get("tRangeTest")
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
    return collections.OrderedDict(root_dir=dir_db,
                                   t_range_train=t_range_train, t_range_test=t_range_test, regions=ref_nonref_regions,
                                   flow_dir=flow_dir, flow_url=flow_url,
                                   forcing_chosen=forcing_chosen, forcing_dir=forcing_dir, forcing_type=forcing_type,
                                   forcing_url=forcing_url,
                                   attr_chosen=attr_chosen, attr_dir=attr_dir, attr_url=attr_url,
                                   gage_id_file=gage_id_file, gage_region_dir=gage_region_dir,
                                   gage_point_file=gagesii_points_file, huc4_shp_file=huc4_shp_file
                                   )


def wrap_master(opt_data, opt_model, opt_loss, opt_train):
    """model的相关参数整合"""
    m_dict = OrderedDict(data=opt_data, model=opt_model, loss=opt_loss, train=opt_train)
    return m_dict


def read_master_file(out):
    """json格式的配置文件"""
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'r') as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + m_file)
    return m_dict


def write_master_file(m_dict):
    """将配置内容写入json格式的配置文件"""
    out = m_dict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'w') as fp:
        json.dump(m_dict, fp, indent=4)
    print('write master file ' + m_file)
    return out


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    """训练过程输出"""
    if not os.path.exists(out):
        os.makedirs(out)
    if not os.path.exists(os.path.join(out, 'master.json')):
        return ['None']
    mDict = read_master_file(out)
    target = mDict['data']['target']
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict['loss']['name']
    if epoch is None:
        epoch = mDict['train']['nEpoch']

    fileNameLst = list()
    for k in range(nt):
        testName = '_'.join(
            [subset, str(tRange[0]),
             str(tRange[1]), 'ep' + str(epoch)])
        fileName = '_'.join([target[k], testName])
        fileNameLst.append(fileName)
        if lossName == 'hydroDL.model.crit.SigmaLoss':
            fileName = '_'.join([target[k] + 'SigmaX', testName])
            fileNameLst.append(fileName)

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + '_' + suffix
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    return filePathLst
