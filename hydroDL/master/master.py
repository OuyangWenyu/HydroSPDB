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

def readMasterFile(out):
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'r') as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + mFile)
    return mDict


def writeMasterFile(mDict):
    out = mDict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'w') as fp:
        json.dump(mDict, fp, indent=4)
    print('write master file ' + mFile)
    return out


def loadModel(out, epoch=None):
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict['train']['nEpoch']
    model = hydroDL.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    mDict = readMasterFile(out)
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


def loadData(optData):
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['rootDB'],
            subset=optData['subset'],
            tRange=optData['tRange'])
        x = df.getDataTs(
            varLst=optData['varT'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])
        y = df.getDataTs(
            varLst=optData['target'],
            doNorm=optData['doNorm'][1],
            rmNan=optData['rmNan'][1])
        c = df.getDataConst(
            varLst=optData['varC'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])
        if optData['daObs'] > 0:
            nday = optData['daObs']
            sd = utils.time.t2dt(
                optData['tRange'][0]) - dt.timedelta(days=nday)
            ed = utils.time.t2dt(
                optData['tRange'][1]) - dt.timedelta(days=nday)
            df = hydroDL.data.dbCsv.DataframeCsv(
                rootDB=optData['rootDB'],
                subset=optData['subset'],
                tRange=[sd, ed])
            obs = df.getDataTs(
                varLst=optData['target'],
                doNorm=optData['doNorm'][1],
                rmNan=optData['rmNan'][1])
            x = (x, obs)
    elif eval(optData['name']) is hydroDL.data.camels.DataframeCamels:
        df = hydroDL.data.camels.DataframeCamels(
            subset=optData['subset'], tRange=optData['tRange'])
        x = df.get_data_ts(
            varLst=optData['varT'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])
        y = df.get_data_obs(
            doNorm=optData['doNorm'][1], rmNan=optData['rmNan'][1])
        c = df.get_data_const(
            varLst=optData['varC'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])
        if optData['daObs'] > 0:
            nday = optData['daObs']
            sd = utils.time.t2dt(
                optData['tRange'][0]) - dt.timedelta(days=nday)
            ed = utils.time.t2dt(
                optData['tRange'][1]) - dt.timedelta(days=nday)
            df = hydroDL.data.camels.DataframeCamels(
                subset=optData['subset'], tRange=[sd, ed])
            obs = df.get_data_obs(doNorm=optData['doNorm'][1], rmNan=True)
            x = np.concatenate([x, obs], axis=2)
    else:
        raise Exception('unknown database')
    return df, x, y, c


def train(mDict):
    if mDict is str:
        mDict = readMasterFile(mDict)
    out = mDict['out']
    optData = mDict['data']
    optModel = mDict['model']
    optLoss = mDict['loss']
    optTrain = mDict['train']

    # data
    df, x, y, c = loadData(optData)
    nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if eval(optLoss['name']) is hydroDL.model.crit.SigmaLoss:
        lossFun = hydroDL.model.crit.SigmaLoss(prior=optLoss['prior'])
        optModel['ny'] = ny * 2
    elif eval(optLoss['name']) is hydroDL.model.crit.RmseLoss:
        lossFun = hydroDL.model.crit.RmseLoss()
        optModel['ny'] = ny

    # model
    if optModel['nx'] != nx:
        print('updated nx by input data')
        optModel['nx'] = nx
    if eval(optModel['name']) is hydroDL.model.rnn.CudnnLstmModel:
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'])
    elif eval(optModel['name']) is hydroDL.model.rnn.LstmCloseModel:
        model = hydroDL.model.rnn.LstmCloseModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'],
            fillObs=True)
    elif eval(optModel['name']) is hydroDL.model.rnn.AnnModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'])
    elif eval(optModel['name']) is hydroDL.model.rnn.AnnCloseModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'],
            fillObs=True)

    # train
    if optTrain['saveEpoch'] > optTrain['nEpoch']:
        optTrain['saveEpoch'] = optTrain['nEpoch']

    # train model
    writeMasterFile(mDict)
    model = hydroDL.model.train.trainModel(
        model,
        x,
        y,
        c,
        lossFun,
        nEpoch=optTrain['nEpoch'],
        miniBatch=optTrain['miniBatch'],
        saveEpoch=optTrain['saveEpoch'],
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
    mDict = readMasterFile(out)

    optData = mDict['data']
    optData['subset'] = subset
    optData['tRange'] = tRange
    df, x, obs, c = loadData(optData)

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
    mDict = readMasterFile(out)
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
