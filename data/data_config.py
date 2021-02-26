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


def update_config_item(opt, **kwargs):
    for key in kwargs:
        if key in opt:
            try:
                opt[key] = kwargs[key]
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')


def init_path(config_file):
    """read path of data dir from config file"""
    path_data = collections.OrderedDict(
        DB=config_file.DATA_PATH,
        Out=config_file.OUT_PATH,
        Temp=config_file.TEMP_PATH)
    return path_data


def wrap_master(opt_dir, opt_data, opt_model, opt_loss, opt_train):
    """model params"""
    m_dict = OrderedDict(dir=opt_dir, data=opt_data, model=opt_model, loss=opt_loss, train=opt_train)
    return m_dict


def name_pred(m_dict, out, t_range, epoch, subset=None, suffix=None):
    """output file"""
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

    def init_model_param(self):
        cfg = self.config_file
        # model data parameters
        opt_data = collections.OrderedDict(tRangeTrain=cfg.MODEL.tRangeTrain, tRangeTest=cfg.MODEL.tRangeTest,
                                           doNorm=cfg.MODEL.doNorm, rmNan=cfg.MODEL.rmNan, daObs=cfg.MODEL.daObs)

        # model training parameters
        opt_train = collections.OrderedDict(miniBatch=cfg.MODEL.miniBatch, nEpoch=cfg.MODEL.nEpoch,
                                            saveEpoch=cfg.MODEL.saveEpoch)

        # model structure parameters
        opt_model = collections.OrderedDict(name=cfg.MODEL.name, hiddenSize=cfg.MODEL.hiddenSize,
                                            doReLU=cfg.MODEL.doReLU)

        # loss parameters
        opt_loss = collections.OrderedDict(name=cfg.MODEL.loss, prior=cfg.MODEL.prior)

        return opt_data, opt_train, opt_model, opt_loss

    def read_data_config(self):
        pass
