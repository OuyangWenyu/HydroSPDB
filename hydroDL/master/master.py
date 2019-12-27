"""调用训练，测试，读取模型等函数的函数"""
import os
import numpy as np
import pandas as pd

from data.read_config import namePred
from utils import unserialize_json_ordered
from explore import stat
from hydroDL.model import *


def load_model(out, epoch=None):
    """
    根据out配置项读取
    :parameter
        out: model_dict"""
    if epoch is None:
        m_dict = unserialize_json_ordered(out)
        epoch = m_dict['train']['nEpoch']
    model = model_run.model_load(out, epoch)
    return model


def master_train(data_model, model_dict):
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data(model_dict)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if opt_loss['name'] == 'SigmaLoss':
        loss_fun = crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif opt_loss['name'] == 'RmseLoss':
        loss_fun = crit.RmseLoss()
        opt_model['ny'] = ny

    # model
    if opt_model['nx'] != nx:
        print('updated nx by input data')
        opt_model['nx'] = nx
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


def master_test(data_model, model_dict):
    """:parameter
        data_model：测试使用的数据
        model_dict：测试时的模型配置
    """
    m_dict = model_dict
    opt_data = m_dict['data']
    opt_test = m_dict['test']
    batch_size, rho = opt_test['miniBatch']

    df, x, obs, c = data_model.load_data(opt_data)

    # generate file names and run model
    out = data_model.data_source.all_configs['out']
    t_range = data_model.data_source.t_range
    epoch = model_dict["epoch"]
    do_mc = model_dict["do_mc"]
    suffix = model_dict["suffix"]
    file_path_lst = namePred(out, t_range, epoch=epoch, doMC=do_mc, suffix=suffix)
    print('output files:', file_path_lst)
    # 如果没有测试结果，那么就重新运行测试代码
    re_test = False
    for file_path in file_path_lst:
        if not os.path.isfile(file_path):
            re_test = True
    if re_test is True:
        print('Runing new results')
        model = load_model(model_dict, epoch=epoch)
        model_test(model, x, c, batch_size=batch_size, file_path_lst=file_path_lst)
    else:
        print('Loaded previous results')

    # load previous result
    m_dict = unserialize_json_ordered(out)
    data_pred = np.ndarray([obs.shape[0], obs.shape[1], len(file_path_lst)])
    for k in range(len(file_path_lst)):
        file_path = file_path_lst[k]
        data_pred[:, :, k] = pd.read_csv(file_path, dtype=np.float, header=None).values
    is_sigma_x = False
    if m_dict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss':
        is_sigma_x = True
        pred = data_pred[:, :, ::2]
        sigma_x = data_pred[:, :, 1::2]
    else:
        pred = data_pred

    if opt_data['doNorm'][1] is True:
        stat_dict = data_model.stat_dict
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        pred = stat.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
        obs = stat.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)

    if is_sigma_x is True:
        return df, pred, obs, sigma_x
    else:
        return df, pred, obs
