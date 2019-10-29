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
    model = hydroDL.model.train.loadModel(out, epoch)
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
            hiddenSize=opt_model['hiddenSize'])
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
    model = hydroDL.model.train.trainModel(
        model,
        x,
        y,
        c,
        loss_fun,
        nEpoch=opt_train['nEpoch'],
        miniBatch=opt_train['miniBatch'],
        saveEpoch=opt_train['saveEpoch'],
        saveFolder=out)


def test(out,
         *,
         tRange,
         subset,
         doMC=False,
         suffix=None,
         batchSize=None,
         epoch=None,
         reTest=False):
    mDict = read_master_file(out)

    optData = mDict['data']
    optData['subset'] = subset
    optData['tRange'] = tRange
    df, x, obs, c = load_data(optData)

    # generate file names and run model
    filePathLst = namePred(
        out, tRange, subset, epoch=epoch, doMC=doMC, suffix=suffix)
    print('output files:', filePathLst)
    for filePath in filePathLst:
        if not os.path.isfile(filePath):
            reTest = True
    if reTest is True:
        print('Runing new results')
        model = loadModel(out, epoch=epoch)
        hydroDL.model.train.testModel(
            model, x, c, batchSize=batchSize, filePathLst=filePathLst)
    else:
        print('Loaded previous results')

    # load previous result
    mDict = read_master_file(out)
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(
            filePath, dtype=np.float, header=None).values
    isSigmaX = False
    if mDict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss':
        isSigmaX = True
        pred = dataPred[:, :, ::2]
        sigmaX = dataPred[:, :, 1::2]
    else:
        pred = dataPred

    if optData['doNorm'][1] is True:
        if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
            target = optData['target']
            if type(target) is not list:
                target = [target]
            nTar = len(target)
            for k in range(nTar):
                pred[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    pred[:, :, k],
                    rootDB=optData['rootDB'],
                    fieldName=optData['target'][k],
                    fromRaw=False)
                obs[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    obs[:, :, k],
                    rootDB=optData['rootDB'],
                    fieldName=optData['target'][k],
                    fromRaw=False)
                if isSigmaX is True:
                    sigmaX[:, :, k] = hydroDL.data.dbCsv.transNormSigma(
                        sigmaX[:, :, k],
                        rootDB=optData['rootDB'],
                        fieldName=optData['target'][k],
                        fromRaw=False)
        elif eval(optData['name']) is hydroDL.data.camels.DataframeCamels:
            pred = hydroDL.data.camels.transNorm(
                pred, 'usgsFlow', toNorm=False)
            obs = hydroDL.data.camels.transNorm(obs, 'usgsFlow', toNorm=False)
    if isSigmaX is True:
        return df, pred, obs, sigmaX
    else:
        return df, pred, obs
