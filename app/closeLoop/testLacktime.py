import hydroDL
import os
from data import dbCsv
from hydroDL import rnn, crit, train
from hydroDL import stat
from visual import plot
import utils
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'closeLoop')
ty1 = [20150501, 20160501]
ty2 = [20160501, 20170501]

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')

# dLst = [1, 2, 3, 5, 7, 10, 15]
dLst = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

if 'train' in doLst:
    # load data
    df = app.streamflow.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1

    # train
    for k in dLst:
        sd = utils.time.t2dt(ty1[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty1[1]) - dt.timedelta(days=k)
        df = app.streamflow.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
        model = rnn.LstmCloseModel(
            nx=nx + 1, ny=ny, hiddenSize=64, fillObs=True)
        lossFun = crit.RmseLoss()
        model = train.train_model(
            model, (x, obs), y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'LSTM-DA-' + str(k)
        train.save_model(outFolder, model, nEpoch, modelName=modelName)

if 'test' in doLst:
    # load data
    df = app.streamflow.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1
    yT = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False)
    yT = yT[:, :, 0]

    # test
    ypLst = list()
    modelName = 'LSTM'
    model = train.load_model(outFolder, 100, modelName=modelName)
    yP = train.test_model(model, x, batchSize=100).squeeze()
    ypLst.append(
        dbCsv.transNorm(yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    for k in dLst:
        sd = utils.time.t2dt(ty2[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty2[1]) - dt.timedelta(days=k)
        df = app.streamflow.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
        modelName = 'LSTM-DA-' + str(k)
        model = train.load_model(outFolder, nEpoch, modelName=modelName)
        yP = train.test_model(model, (x, obs), batchSize=100).squeeze()
        ypLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'post' in doLst:
    statDictLst = list()
    for k in range(0, len(ypLst)):
        statDictLst.append(stat.statError(ypLst[k], yT))
    keyLst = ['RMSE', 'ubRMSE', 'Bias', 'Corr']
    caseLst = ['LSTM']
    for k in dLst:
        caseLst.append('DA-' + str(k))

    # plot box
    dataBox = list()
    cmap = plt.cm.jet
    cLst = cmap(np.linspace(0, 1, len(caseLst)))
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            temp.append(statDictLst[k][statStr])
        dataBox.append(temp)
    fig = plot.plot_box_fig(
        dataBox, keyLst, caseLst, sharey=False, colorLst=cLst)
    fig.show()