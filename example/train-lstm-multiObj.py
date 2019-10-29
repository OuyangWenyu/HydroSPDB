from hydroDL import pathSMAP
from hydroDL.master import default, wrapMaster, train, run_train, test
from hydroDL.post import plot, stat

import os

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/home/kxf227/work/GitHUB/pyRnnSMAP/example/'  # for coding. delete.

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB=pathSMAP['DB_L3_NA'],
    target=['SMAP_AM', 'SOILM_0-10_NOAH'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
optModel = default.optLstm
optLoss = default.optLossSigma
optTrain = default.update(default.optTrainSMAP, nEpoch=100)
out = os.path.join(cDir, 'output', 'CONUSv4f1_multi')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)
# runTrain(masterDict, cudaID=2, screen='LSTM-multi')

# test
df, yp, yt, sigma = test(out, tRange=[20160401, 20170401], subset='CONUSv4f1')

# plot ts MAP
dataGrid = list()
dataTs = list()
for k in range(2):
    statErr = stat.statError(yp[:, :, k], yt[:, :, k])
    dataGrid.append(statErr['RMSE'])
    dataTs.append([yp[:, :, k], yt[:, :, k]])
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE ', 'Correlation']
tsNameLst = ['LSTM', 'SMAP']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    multiTS=True,
    isGrid=True)

