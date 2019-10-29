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
dLst = [1, 5, 10, 20]

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')

if 'train' in doLst:
    # load data
    ty = ty1
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty)
    x = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
    c = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getDataTs('SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1] + c.shape[-1]
    ny = 1

    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
    lossFun = crit.RmseLoss()
    model = train.train_model(
        model, x, y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
    modelName = 'LSTM'
    train.save_model(outFolder, model, nEpoch, modelName=modelName)

    model = rnn.AnnModel(nx=nx, ny=ny, hiddenSize=64)
    lossFun = crit.RmseLoss()
    model = train.train_model(
        model, x, y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
    modelName = 'ANN'
    train.save_model(outFolder, model, nEpoch, modelName=modelName)

    for k in dLst:
        sd = utils.time.t2dt(ty[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty[1]) - dt.timedelta(days=k)
        df2 = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        model = rnn.LstmCloseModel(nx=nx, ny=ny, hiddenSize=64)
        lossFun = crit.RmseLoss()
        model = train.train_model(
            model, (x, obs), y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'LSTM-DA-' + str(k)
        train.save_model(outFolder, model, nEpoch, modelName=modelName)

        model = rnn.AnnCloseModel(nx=nx, ny=ny, hiddenSize=64)
        lossFun = crit.RmseLoss()
        model = train.train_model(
            model, (x, obs), y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'ANN-DA-' + str(k)
        train.save_model(outFolder, model, nEpoch, modelName=modelName)

if 'test' in doLst:
    # load data
    ty = ty2
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty)
    x = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
    c = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
    nx = x.shape[-1] + c.shape[-1]
    yT = df.getDataTs('SMAP_AM', doNorm=False, rmNan=False)
    yT = yT[:, :, 0]

    # test
    ypLstmLst = list()
    ypAnnLst = list()
    modelName = 'LSTM'
    model = train.load_model(outFolder, 100, modelName=modelName)
    yp = train.test_model(model, x, c, batchSize=100).squeeze()
    ypLstmLst.append(
        dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    modelName = 'ANN'
    model = train.load_model(outFolder, 100, modelName=modelName)
    yp = train.test_model(model, x, c, batchSize=100).squeeze()
    ypAnnLst.append(
        dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    for k in dLst:
        sd = utils.time.t2dt(ty[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty[1]) - dt.timedelta(days=k)
        df2 = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        modelName = 'LSTM-DA-' + str(k)
        model = train.load_model(outFolder, nEpoch, modelName=modelName)
        yP = train.test_model(model, (x, obs), c, batchSize=100).squeeze()
        ypLstmLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

        modelName = 'ANN-DA-' + str(k)
        model = train.load_model(outFolder, nEpoch, modelName=modelName)
        yP = train.test_model(model, (x, obs), c, batchSize=100).squeeze()
        ypAnnLst.append(
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
    cmap = plt.cm.jet
    cLst = cmap(np.linspace(0, 1, 5))

    caseLst1 = ['no DA']
    for k in dLst:
        caseLst1.append('DA-' + str(k))
    caseLst2 = ['LSTM', 'ANN']
    n1 = len(caseLst1)
    n2 = len(caseLst2)

    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        # statStr = 'RMSE'
        dataBox = list()
        for i in range(n1):
            temp = list()
            for j in range(n2):
                temp.append(statDictLst[j][i][statStr])
            dataBox.append(temp)
        fig = plot.plotBoxFig(
            dataBox,
            caseLst1,
            caseLst2,
            sharey=True,
            title=statStr)
        fig.show()