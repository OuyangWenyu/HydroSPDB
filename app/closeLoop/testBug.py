import hydroDL
import os
from data import dbCsv
from hydroDL import rnn, crit, train

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'closeLoop')
ty1 = [20150401, 20160401]
ty2 = [20160401, 20170401]
ty3 = [20170401, 20180401]

doLst = list()
doLst.append('train')
# doLst.append('test')
# doLst.append('post')

df = app.streamflow.data.dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
x = df.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
nx = x.shape[-1]
ny = 1

model3 = rnn.LstmCloseModel(nx=nx + 1, ny=ny, hiddenSize=64, opt=1)
lossFun = crit.RmseLoss()
model3 = train.model_train(
    model3, x, y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
modelName = 'LSTM-DA'
train.model_save(outFolder, model3, nEpoch, modelName=modelName)