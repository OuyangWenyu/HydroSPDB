import os
import hydroDL
from collections import OrderedDict
import numpy as np
import json
from hydroDL import utils
import datetime as dt
import pandas as pd


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def read_master_file(out):
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'r') as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + m_file)
    return m_dict


def write_master_file(m_dict):
    out = m_dict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    m_file = os.path.join(out, 'master.json')
    with open(m_file, 'w') as fp:
        json.dump(m_dict, fp, indent=4)
    print('write master file ' + m_file)
    return out


def loadModel(out, epoch=None):
    if epoch is None:
        mDict = read_master_file(out)
        epoch = mDict['train']['nEpoch']
    model = hydroDL.model.train.load_model(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
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


def load_data(opt_data):
    if eval(opt_data['name']) is hydroDL.data.gages2.DataframeGages2:
        df = hydroDL.data.gages2.DataframeGages2(
            subset=opt_data['subset'],
            tRange=opt_data['tRange'])
    elif eval(opt_data['name']) is hydroDL.data.camels.DataframeCamels:
        df = hydroDL.data.camels.DataframeCamels(
            subset=opt_data['subset'], tRange=opt_data['tRange'])
    else:
        raise Exception('unknown database')
    x = df.get_data_ts(
        var_lst=opt_data['varT'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    y = df.get_data_obs(
        do_norm=opt_data['doNorm'][1], rm_nan=opt_data['rmNan'][1])
    c = df.get_data_const(
        var_lst=opt_data['varC'],
        do_norm=opt_data['doNorm'][0],
        rm_nan=opt_data['rmNan'][0])
    if opt_data['daObs'] > 0:
        nday = opt_data['daObs']
        sd = utils.time.t2dt(
            opt_data['tRange'][0]) - dt.timedelta(days=nday)
        ed = utils.time.t2dt(
            opt_data['tRange'][1]) - dt.timedelta(days=nday)
        if eval(opt_data['name']) is hydroDL.data.gages2.DataframeGages2:
            df = hydroDL.data.gages2.DataframeGages2(subset=opt_data['subset'], tRange=[sd, ed])
        elif eval(opt_data['name']) is hydroDL.data.camels.DataframeCamels:
            df = hydroDL.data.camels.DataframeCamels(subset=opt_data['subset'], tRange=[sd, ed])
        obs = df.get_data_obs(do_norm=opt_data['doNorm'][1], rm_nan=True)
        x = np.concatenate([x, obs], axis=2)
    return df, x, y, c


def train(m_dict):
    if m_dict is str:
        m_dict = read_master_file(m_dict)
    out = m_dict['out']
    opt_data = m_dict['data']
    opt_model = m_dict['model']
    opt_loss = m_dict['loss']
    opt_train = m_dict['train']

    # data
    df, x, y, c = load_data(opt_data)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if eval(opt_loss['name']) is hydroDL.model.crit.SigmaLoss:
        loss_fun = hydroDL.model.crit.SigmaLoss(prior=opt_loss['prior'])
        opt_model['ny'] = ny * 2
    elif eval(opt_loss['name']) is hydroDL.model.crit.RmseLoss:
        loss_fun = hydroDL.model.crit.RmseLoss()
        opt_model['ny'] = ny

    # model
    if opt_model['nx'] != nx:
        print('updated nx by input data')
        opt_model['nx'] = nx
    if eval(opt_model['name']) is hydroDL.model.rnn.CudnnLstmModel:
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=opt_model['nx'],
            ny=opt_model['ny'],
            hidden_size=opt_model['hiddenSize'])
    elif eval(opt_model['name']) is hydroDL.model.rnn.LstmCloseModel:
        model = hydroDL.model.rnn.LstmCloseModel(
            nx=opt_model['nx'],
            ny=opt_model['ny'],
            hiddenSize=opt_model['hiddenSize'],
            fillObs=True)
    elif eval(opt_model['name']) is hydroDL.model.rnn.AnnModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=opt_model['nx'],
            ny=opt_model['ny'],
            hiddenSize=opt_model['hiddenSize'])
    elif eval(opt_model['name']) is hydroDL.model.rnn.AnnCloseModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=opt_model['nx'],
            ny=opt_model['ny'],
            hiddenSize=opt_model['hiddenSize'],
            fillObs=True)

    # train
    if opt_train['saveEpoch'] > opt_train['nEpoch']:
        opt_train['saveEpoch'] = opt_train['nEpoch']

    # train model
    write_master_file(m_dict)
    model = hydroDL.model.train.train_model(
        model,
        x,
        y,
        c,
        loss_fun,
        n_epoch=opt_train['nEpoch'],
        mini_batch=opt_train['miniBatch'],
        save_epoch=opt_train['saveEpoch'],
        save_folder=out)


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
    opt_data['t_range'] = t_range
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
        model = loadModel(out, epoch=epoch)
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
        if eval(opt_data['name']) is hydroDL.data.gages2.DataframeGages2:
            stat_dict = hydroDL.data.gages2.statDict
            pred = hydroDL.utils.statistics.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
            obs = hydroDL.utils.statistics.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
        elif eval(opt_data['name']) is hydroDL.data.camels.DataframeCamels:
            stat_dict = hydroDL.data.camels.statDict
            pred = hydroDL.utils.statistics.trans_norm(pred, 'usgsFlow', stat_dict, to_norm=False)
            obs = hydroDL.utils.statistics.trans_norm(obs, 'usgsFlow', stat_dict, to_norm=False)
    if is_sigma_x is True:
        return df, pred, obs, sigma_x
    else:
        return df, pred, obs
