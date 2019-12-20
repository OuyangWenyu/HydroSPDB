import os
import hydroDL
import numpy as np

from data.read_config import read_master_file, write_master_file, namePred
import pandas as pd

from hydroDL.model import *


def load_model(out, epoch=None):
    if epoch is None:
        m_dict = read_master_file(out)
        epoch = m_dict['train']['nEpoch']
    model = train.load_model(out, epoch)
    return model


def train(data_model, model_dict):
    if model_dict is str:
        """如果参数直接给出了json配置文件，则读取json文件"""
        model_dict = read_master_file(model_dict)
    out = model_dict['out']
    opt_model = model_dict['model']
    opt_loss = model_dict['loss']
    opt_train = model_dict['train']

    # data
    x, y, c = data_model.load_data()
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
    write_master_file(model_dict)
    model = train.train_model(model, x, y, c, loss_fun, n_epoch=opt_train['nEpoch'], mini_batch=opt_train['miniBatch'],
                              save_epoch=opt_train['saveEpoch'], save_folder=out)


def test(out,
         *,
         t_range,
         subset,
         do_mc=False,
         suffix=None,
         batch_size=None,
         epoch=None,
         re_test=False):
    m_dict = read_master_file(out)

    opt_data = m_dict['data']
    opt_data['subset'] = subset
    opt_data['tRange'] = t_range
    opt_train = m_dict['train']
    batch_size, rho = opt_train['miniBatch']

    df, x, obs, c = load_data(opt_data)

    # generate file names and run model
    file_path_lst = namePred(
        out, t_range, subset, epoch=epoch, doMC=do_mc, suffix=suffix)
    print('output files:', file_path_lst)
    for file_path in file_path_lst:
        if not os.path.isfile(file_path):
            re_test = True
    if re_test is True:
        print('Runing new results')
        model = load_model(out, epoch=epoch)
        hydroDL.model.train.test_model(
            model, x, c, batch_size=batch_size, file_path_lst=file_path_lst)
    else:
        print('Loaded previous results')

    # load previous result
    m_dict = read_master_file(out)
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
        # 如果之前归一化了，这里为了展示原量纲数据，需要反归一化回来
        if eval(opt_data['name']) is app.streamflow.data.gages2.DataframeGages2:
            stat_dict = app.streamflow.data.gages2.statDict
            pred = refine.stat.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
            obs = refine.stat.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
        elif eval(opt_data['name']) is app.streamflow.data.camels.DataframeCamels:
            stat_dict = app.streamflow.data.camels.statDict
            pred = refine.stat.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
            obs = refine.stat.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
    if is_sigma_x is True:
        return df, pred, obs, sigma_x
    else:
        return df, pred, obs
