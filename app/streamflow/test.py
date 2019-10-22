import hydroDL
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.utils import interp

outFolder = r'/mnt/sdb/Data/Camels/test/'
nEpoch = 100
doLst = list()
doLst.append('train')
doLst.append('test')

if 'train' in doLst:
    df1 = camels.DataframeCamels(subset='all', tRange=[20050101, 20100101])
    x1 = df1.getDataTs(varLst=camels.forcingLst, doNorm=True, rmNan=True)
    y1 = df1.getDataObs(doNorm=True, rmNan=False)
    c1 = df1.getDataConst(varLst=camels.attrLstSel, doNorm=True, rmNan=True)
    yt1 = df1.getDataObs(doNorm=False, rmNan=False).squeeze()

    dfz1 = camels.DataframeCamels(subset='all', tRange=[20041231, 20091231])
    z1 = dfz1.getDataObs(doNorm=True, rmNan=True)
    # z1 = interp.interpNan1d(z1, mode='pre')
    xz1 = np.concatenate([x1, z1], axis=2)

    dfz2 = camels.DataframeCamels(subset='all', tRange=[20041225, 20091225])
    z2 = dfz2.getDataObs(doNorm=True, rmNan=True)
    # z2 = interp.interpNan1d(z2, mode='pre')
    xz2 = np.concatenate([x1, z2], axis=2)

    ny = 1
    nx = x1.shape[-1] + c1.shape[-1]
    lossFun = crit.RmseLoss()

    # model1 = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
    # model1 = train.trainModel(
    #     model1, x1, y1, c1, lossFun, nEpoch=nEpoch, miniBatch=(50, 365))
    # train.saveModel(outFolder, model1, nEpoch, modelName='LSTM')

    model2 = rnn.CudnnLstmModel(nx=nx + 1, ny=ny, hiddenSize=64)
    model2 = train.trainModel(
        model2, xz1, y1, c1, lossFun, nEpoch=nEpoch, miniBatch=(50, 365))
    train.saveModel(outFolder, model2, nEpoch, modelName='DA-1')

    model3 = rnn.CudnnLstmModel(nx=nx + 1, ny=ny, hiddenSize=64)
    model3 = train.trainModel(
        model3, xz2, y1, c1, lossFun, nEpoch=nEpoch, miniBatch=(50, 365))
    train.saveModel(outFolder, model3, nEpoch, modelName='DA-7')

if 'test' in doLst:
    df2 = camels.DataframeCamels(subset='all', tRange=[20050101, 20150101])
    x2 = df2.getDataTS(varLst=camels.forcingLst, doNorm=True, rmNan=True)
    c2 = df2.getDataConst(varLst=camels.attrLstSel, doNorm=True, rmNan=True)
    yt2 = df2.getDataObs(doNorm=False, rmNan=False).squeeze()

    dfz1 = camels.DataframeCamels(subset='all', tRange=[20041231, 20141231])
    z1 = dfz1.getDataObs(doNorm=True, rmNan=True)
    # z1 = interp.interpNan1d(z1, mode='pre')
    xz1 = np.concatenate([x2, z1], axis=2)

    dfz2 = camels.DataframeCamels(subset='all', tRange=[20041225, 20141225])
    z2 = dfz2.getDataObs(doNorm=True, rmNan=True)
    # z2 = interp.interpNan1d(z2, mode='pre')
    xz2 = np.concatenate([x2, z2], axis=2)

    model1 = train.loadModel(outFolder, nEpoch, modelName='LSTM')
    yp1 = train.testModel(model1, x2, c2)
    yp1 = camels.transNorm(yp1, 'usgsFlow', toNorm=False).squeeze()

    model2 = train.loadModel(outFolder, nEpoch, modelName='DA-1')
    yp2 = train.testModel(model2, xz1, c2)
    yp2 = camels.transNorm(yp2, 'usgsFlow', toNorm=False).squeeze()

    model3 = train.loadModel(outFolder, nEpoch, modelName='DA-7')
    yp3 = train.testModel(model3, xz2, c2)
    yp3 = camels.transNorm(yp3, 'usgsFlow', toNorm=False).squeeze()

    yLst = [yt2, yp1, yp2, yp3]

# plot box
statDictLst = [
    stat.statError(yp1, yt2),
    stat.statError(yp2, yt2),
    stat.statError(yp3, yt2)
]
keyLst = list(statDictLst[0].keys())

dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM', 'DA-1', 'DA-7'], sharey=False)
fig.show()

# plot time series
t = utils.time.tRange2Array([20050101, 20150101])
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
# iLst = [54, 219, 298, 325, 408]
for k in range(5):
    iGrid = np.random.randint(0, 671)
    # iGrid = iLst[k]
    yPlot = list()
    for y in yLst:
        yPlot.append(y[iGrid, :])
    if k == 0:
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='kbrg',
            markerLst='----',
            legLst=['USGS', 'LSTM', 'DA-1', 'DA-7'])
    else:
        plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrg', markerLst='----')
fig.show()