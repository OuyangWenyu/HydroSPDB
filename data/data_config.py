import collections
import os
import definitions
from collections import OrderedDict
from configparser import ConfigParser


def add_model_param(data_config, model_dict_item, **kwargs):
    """model_dict has 5 items: dir, data, model, loss, train"""
    dict_chosen = data_config.model_dict[model_dict_item]
    for key in kwargs:
        dict_chosen[key] = kwargs[key]


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


def init_path(config_file):
    """根据配置文件读取数据源路径"""
    cfg = ConfigParser()
    cfg.read(config_file)
    sections = cfg.sections()
    data_input = cfg.get(sections[0], 'download')
    data_output = cfg.get(sections[0], 'output')
    data_temp = cfg.get(sections[0], 'temp')
    root = eval(cfg.get(sections[0], 'prefix'))
    data_input = os.path.join(root, data_input)
    data_output = os.path.join(root, data_output)
    data_temp = os.path.join(root, data_temp)
    if not os.path.isdir(data_input):
        os.mkdir(data_input)
    if not os.path.isdir(data_output):
        os.mkdir(data_output)
    if not os.path.isdir(data_temp):
        os.mkdir(data_temp)
    path_data = collections.OrderedDict(
        DB=os.path.join(data_input, cfg.get(sections[0], 'data')),
        Out=os.path.join(data_output, cfg.get(sections[0], 'data')),
        Temp=os.path.join(data_temp, cfg.get(sections[0], 'data')))
    if not os.path.isdir(path_data["DB"]):
        os.mkdir(path_data["DB"])
    if not os.path.isdir(path_data["Out"]):
        os.mkdir(path_data["Out"])
    if not os.path.isdir(path_data["Temp"]):
        os.mkdir(path_data["Temp"])
    print(path_data)
    return path_data


def wrap_master(opt_dir, opt_data, opt_model, opt_loss, opt_train):
    """model的相关参数整合"""
    m_dict = OrderedDict(dir=opt_dir, data=opt_data, model=opt_model, loss=opt_loss, train=opt_train)
    return m_dict


def name_pred(m_dict, out, t_range, epoch, subset=None, suffix=None):
    """训练过程输出"""
    loss_name = m_dict['loss']['name']
    file_name = '_'.join([str(t_range[0]), str(t_range[1]), 'ep' + str(epoch)])
    if loss_name == 'SigmaLoss':
        file_name = '_'.join('SigmaX', file_name)
    if suffix is not None:
        file_name = file_name + '_' + suffix
    file_path = os.path.join(out, file_name + '.csv')
    return file_path


class DataConfig(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.data_path = init_path(config_file)

    def init_data_param(self):
        pass

    def init_model_param(self):
        """根据配置文件读取有关模型的各项参数，返回optModel, optLoss, optTrain三组参数，分成几组的原因是为写成json文件时更清晰"""
        config_file = self.config_file
        cfg = ConfigParser()
        cfg.read(config_file)
        section = 'model'
        options = cfg.options(section)
        # train and test time range
        t_range_train = eval(cfg.get(section, options[0]))
        t_range_test = eval(cfg.get(section, options[1]))
        # data processing parameter
        do_norm = eval(cfg.get(section, options[2]))
        rm_nan = eval(cfg.get(section, options[3]))
        da_obs = eval(cfg.get(section, options[4]))
        opt_data = collections.OrderedDict(tRangeTrain=t_range_train, tRangeTest=t_range_test, doNorm=do_norm,
                                           rmNan=rm_nan, daObs=da_obs)

        # model parameters. 首先读取几个训练使用的基本模型参数，主要是epoch和batch
        mini_batch = eval(cfg.get(section, options[5]))
        n_epoch = eval(cfg.get(section, options[6]))
        save_epoch = eval(cfg.get(section, options[7]))
        opt_train = collections.OrderedDict(miniBatch=mini_batch, nEpoch=n_epoch, saveEpoch=save_epoch)

        # 接着是模型输入输出的相关参数。根据opt_data判断输入输出变量个数，确定模型基本结构
        model_name = cfg.get(section, options[8])
        # 变量名不要修改!!!!!!!!!!!!!!!!!!!!!!!!!!，因为后面eval执行会用到varT和varC这两个变量名。 除非修改配置文件
        hidden_size = eval(cfg.get(section, options[9]))
        do_relu = eval(cfg.get(section, options[10]))
        opt_model = collections.OrderedDict(name=model_name, hiddenSize=hidden_size, doReLU=do_relu)

        # 最后是loss的配置
        loss_name = cfg.get(section, options[11])
        prior = cfg.get(section, options[12])
        opt_loss = collections.OrderedDict(name=loss_name, prior=prior)

        return opt_data, opt_train, opt_model, opt_loss

    def read_data_config(self):
        pass
