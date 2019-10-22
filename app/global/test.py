from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv

optData = master.default.update(
    master.default.optDataSMAP,
    rootDB=pathSMAP['DB_L3_Global'],
    subset='Globalv4f1_NorthAmerica',
    tRange=[20150401, 20160401],
    varT=dbCsv.varForcingGlobal)
optModel = master.default.optLstm
optLoss = master.default.optLossSigma
optTrain = master.default.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'test')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.train(masterDict)
