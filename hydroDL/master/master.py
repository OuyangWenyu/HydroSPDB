"""调用训练，测试，读取模型等函数的函数"""
import os
import numpy as np
import pandas as pd

from data.data_config import name_pred
from utils import unserialize_json_ordered
from explore import stat
from hydroDL.model import *


def master_train(data_model):
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    if opt_loss['name'] == 'SigmaLoss':
        loss_fun = crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
        opt_model['ny'] = ny
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_model['name'] == 'CudnnLstmModel':
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'LstmCloseModel':
        model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                   fillObs=True)
    elif opt_model['name'] == 'AnnModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
    elif opt_model['name'] == 'AnnCloseModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                  fillObs=True)

    # train
    if opt_train['saveEpoch'] > opt_train['nEpoch']:
        opt_train['saveEpoch'] = opt_train['nEpoch']

    # train model
    out = model_dict['dir']['Out']
    model = model_run.model_train(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                                  mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'], save_folder=out)


def master_test(data_model):
    """:parameter
        data_model：测试使用的数据
        model_dict：测试时的模型配置
    """
    model_dict = data_model.data_source.data_config.model_dict
    opt_data = model_dict['data']
    # 测试和训练使用的batch_size, rho是一样的
    batch_size, rho = model_dict['train']['miniBatch']

    x, obs, c = data_model.load_data(model_dict)

    # generate file names and run model
    out = model_dict['dir']['Out']
    t_range = data_model.data_source.t_range
    epoch = model_dict['train']["nEpoch"]
    # do_mc = model_dict["do_mc"]  TODO： 明确do_mc参数是什么用的
    file_path = name_pred(model_dict, out, t_range, epoch)
    print('output files:', file_path)
    # 如果没有测试结果，那么就重新运行测试代码
    re_test = False
    if not os.path.isfile(file_path):
        re_test = True
    if re_test:
        print('Runing new results')
        model = model_run.model_load(out, epoch)
        model_run.model_test(model, x, c, file_path=file_path, batch_size=batch_size)
    else:
        print('Loaded previous results')

    # load previous result并反归一化为标准量纲
    data_pred = pd.read_csv(file_path, dtype=np.float, header=None).values
    is_sigma_x = False
    if model_dict['loss']['name'] == 'SigmaLoss':
        # TODO：sigmaloss下的情况都没做
        is_sigma_x = True
        pred = data_pred[:, :, ::2]
        sigma_x = data_pred[:, :, 1::2]
    else:
        # 扩充到三维才能很好地在后面调用stat.trans_norm函数反归一化
        pred = np.expand_dims(data_pred, axis=2)
    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = stat.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = stat.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    if is_sigma_x is True:
        return pred, obs, sigma_x
    else:
        return pred, obs


def train_natural_flow(data_model):
    model_dict = data_model.data_source.data_config.model_dict
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]
    opt_model['nx'] = nx
    opt_model['ny'] = ny
    # loss
    if opt_loss['name'] == 'SigmaLoss':
        loss_fun = crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELosstest':
        loss_fun = crit.NSELosstest()
        opt_model['ny'] = ny
    elif opt_loss['name'] == 'NSELoss':
        loss_fun = crit.NSELoss()
        opt_model['ny'] = ny
    else:
        print("Please specify the loss function!!!")

    # model
    if opt_model['name'] == 'CudnnLstmModel':
        model = rnn.CudnnLstmModel(nx=opt_model['nx'], ny=opt_model['ny'], hidden_size=opt_model['hiddenSize'])
    elif opt_model['name'] == 'LstmCloseModel':
        model = rnn.LstmCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                   fillObs=True)
    elif opt_model['name'] == 'AnnModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'])
    elif opt_model['name'] == 'AnnCloseModel':
        model = rnn.AnnCloseModel(nx=opt_model['nx'], ny=opt_model['ny'], hiddenSize=opt_model['hiddenSize'],
                                  fillObs=True)

    # train
    if opt_train['saveEpoch'] > opt_train['nEpoch']:
        opt_train['saveEpoch'] = opt_train['nEpoch']

    # train model
    out = model_dict['dir']['Out']
    model = model_run.model_train(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'],
                                  mini_batch=opt_train['miniBatch'], save_epoch=opt_train['saveEpoch'], save_folder=out)