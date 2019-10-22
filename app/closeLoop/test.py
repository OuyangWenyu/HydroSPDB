import hydroDL
import os
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'closeLoop')
ty1 = [20150501, 20160501]
ty2 = [20160501, 20170501]
dLst = [1]

doLst = list()
doLst.append('train')
# doLst.append('test')
# doLst.append('post')

if 'train' in doLst:
    # load data
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
    c = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getDataTs('SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1] + c.shape[-1]
    ny = 1

    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
    lossFun = crit.RmseLoss()
    model = train.trainModel(
        model, x, y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
    modelName = 'test-LSTM'
    train.saveModel(outFolder, model, nEpoch, modelName=modelName)

    for k in dLst:
        sd = utils.time.t2dt(ty1[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty1[1]) - dt.timedelta(days=k)
        df2 = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        model = rnn.LstmCloseModel(nx=nx, ny=ny, hiddenSize=64)
        lossFun = crit.RmseLoss()
        model = train.trainModel(
            model, (x, obs), y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'test-LSTM-DA-' + str(k)
        train.saveModel(outFolder, model, nEpoch, modelName=modelName)

if 'test' in doLst:
    # load data
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
    x = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
    c = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
    nx = x.shape[-1] + c.shape[-1]
    yT = df.getDataTs('SMAP_AM', doNorm=False, rmNan=False)
    yT = yT[:, :, 0]

    # test
    ypLstmLst = list()
    ypAnnLst = list()
    modelName = 'LSTM'
    model = train.loadModel(outFolder, 100, modelName=modelName)
    yp = train.testModel(model, x, c, batchSize=100).squeeze()
    ypLstmLst.append(
        dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    for k in dLst:
        sd = utils.time.t2dt(ty2[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty2[1]) - dt.timedelta(days=k)
        df2 = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        modelName = 'LSTM-DA-' + str(k)
        model = train.loadModel(outFolder, nEpoch, modelName=modelName)
        yP = train.testModel(model, (x, obs), c, batchSize=100).squeeze()
        ypLstmLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'post' in doLst:
    # stat
    ypLst = [ypLstmLst, ypAnnLst]
    statDictLst = list()
    for i in range(0, len(ypLst)):
        tempLst = list()
        for j in range(0, len(ypLst[i])):
            tempLst.append(stat.statError(ypLst[i][j], yT))
        statDictLst.append(tempLst)
    keyLst = list(tempLst[0].keys())

    # plot box
    dataBox = list()
    caseLst1 = keyLst
    caseLst2 = ['LSTM', 'LSTM-DA']
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        dataBox = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(statDictLst)):
                temp.append(statDictLst[k][statStr])
            dataBox.append(temp)
        fig = plot.plotBoxFig(
            dataBox, caseLst1, caseLst2, sharey=True, title=statStr)
        fig.show()