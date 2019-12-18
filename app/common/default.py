import collections
import os
from configparser import ConfigParser


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
