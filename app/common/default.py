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

    return collections.OrderedDict(
        name='hydroDL.data.gages2.DataframeGages2',
        subset='All',
        varT=gages2.FORCING_LST,
        varC=gages2.ATTR_STR_SEL,
        target=['streamflowTest'],
        tRange=gages2.tRangeTrain,
        doNorm=[True, True],
        rmNan=[True, False],
        daObs=0)


def init_model_param(config_file):
    """根据配置文件读取有关模型的各项参数，返回optModel, optLoss, optTrain三组参数"""

    return collections.OrderedDict(miniBatch=[100, 30], nEpoch=100, saveEpoch=10,
                                   name='hydroDL.model.rnn.CudnnLstmModel',
                                   nx=len(optDataParam['varT']) + len(optDataParam['varC']),
                                   ny=1,
                                   hiddenSize=256,
                                   doReLU=True, name='hydroDL.model.crit.RmseLoss', prior='gauss')


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
