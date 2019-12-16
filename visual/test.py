import app.common.default
import data.data_process
from hydroDL import pathSMAP, master
import os
from data import dbCsv

optData = data.default.update(
    data.default.optDataSMAP,
    rootDB=pathSMAP['DB_L3_Global'],
    subset='Globalv4f1_NorthAmerica',
    tRange=[20150401, 20160401],
    varT=dbCsv.varForcingGlobal)
optModel = data.default.optLstm
optLoss = data.default.optLossSigma
optTrain = data.default.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'test')
masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss, optTrain)
master.train(masterDict)
