from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt

import numpy as np
import os

cid = 0
# train default model
out = os.path.join(pathCamels['Out'], 'All-90-95')
optData = default.optDataCamels
optModel = default.optLstm
optLoss = default.optLossRMSE
optTrain = default.optTrainCamels
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.run_train(masterDict, cuda_id=cid % 3, screen='test')
cid = cid + 1

# test
caseLst = ['All-90-95']
outLst = [os.path.join(pathCamels['Out'], x) for x in caseLst]
subset = 'All'
tRange = [19950101, 20000101]
predLst = list()
for out in outLst:
    df, pred, obs = master.test(out, t_range=tRange, subset=subset)
    predLst.append(pred)

# plot box
statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
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
fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM'], sharey=False)
fig.show()

# plot time series
t = utils.time.tRange2Array(tRange)
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
for k in range(5):
    iGrid = np.random.randint(0, 671)
    yPlot = [obs[iGrid, :]]
    for y in predLst:
        yPlot.append(y[iGrid, :])
    if k == 0:
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='kbrg',
            markerLst='----',
            legLst=['USGS', 'LSTM'])
    else:
        plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrg', markerLst='----')
fig.show()
