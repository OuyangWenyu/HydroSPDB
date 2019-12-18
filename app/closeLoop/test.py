import hydroDL
import os
from data import dbCsv
from hydroDL import rnn, crit, train
from hydroDL import stat
from visual import plot
import utils
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
    df = app.streamflow.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
    c = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getDataTs('SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1] + c.shape[-1]
    ny = 1

    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
    lossFun = crit.RmseLoss()
    model = train.train_model(
        model, x, y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
    modelName = 'test-LSTM'
    train.save_model(outFolder, model, nEpoch, modelName=modelName)

    for k in dLst:
        sd = utils.time.t2dt(ty1[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty1[1]) - dt.timedelta(days=k)
        df2 = app.streamflow.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        model = rnn.LstmCloseModel(nx=nx, ny=ny, hiddenSize=64)
        lossFun = crit.RmseLoss()
        model = train.train_model(
            model, (x, obs), y, c, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'test-LSTM-DA-' + str(k)
        train.save_model(outFolder, model, nEpoch, modelName=modelName)

if 'test' in doLst:
    # load data
    df = app.streamflow.data.dbCsv.DataframeCsv(
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
    model = train.load_model(outFolder, 100, modelName=modelName)
    yp = train.test_model(model, x, c, batchSize=100).squeeze()
    ypLstmLst.append(
        dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    for k in dLst:
        sd = utils.time.t2dt(ty2[0]) - dt.timedelta(days=k)
        ed = utils.time.t2dt(ty2[1]) - dt.timedelta(days=k)
        df2 = app.streamflow.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df2.getDataTs('SMAP_AM', doNorm=True, rmNan=False)

        modelName = 'LSTM-DA-' + str(k)
        model = train.load_model(outFolder, nEpoch, modelName=modelName)
        yP = train.test_model(model, (x, obs), c, batchSize=100).squeeze()
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
        fig = plot.plot_box_fig(
            dataBox, caseLst1, caseLst2, sharey=True, title=statStr)
        fig.show()