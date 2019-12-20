import hydroDL
import os
from data import dbCsv
from hydroDL import train
from hydroDL import stat
from visual import plot

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'closeLoop')
ty1 = [20150401, 20160401]
ty2 = [20160401, 20170401]
ty3 = [20170401, 20180401]
df = app.streamflow.data.dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
x = df.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
yt = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False)
yt = yt[:, :, 0]

ypLst = list()
modelName = 'LSTM'
model = train.model_load(outFolder, nEpoch, modelName=modelName)
yp = train.model_test(model, x, batchSize=100).squeeze()
ypLst.append(
    dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
modelName = 'LSTM-DA'
model = train.model_load(outFolder, nEpoch, modelName=modelName)
yp = train.model_test(model, x, z=y, batchSize=100).squeeze()
ypLst.append(
    dbCsv.transNorm(yp, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

##

statErr1 = stat.statError(ypLst[0], yt)
statErr2 = stat.statError(ypLst[1], yt)
dataGrid = [statErr2['RMSE'], statErr2['RMSE'] - statErr1['RMSE']]
dataTs = [ypLst[0], ypLst[1], yt]
t = df.getT()
crd = df.getGeo()
mapNameLst = ['DA', 'DA-LSTM']
tsNameLst = ['LSTM', 'DA', 'SMAP']
colorMap = None
colorTs = None

plot.plotTsMap(
    dataGrid,
    dataTs,
    crd,
    t,
    colorMap=colorMap,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst)
