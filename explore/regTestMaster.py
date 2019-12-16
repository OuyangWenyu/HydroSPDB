from hydroDL import pathSMAP, master
import os

optData = master.updateOpt(
    data.default.optDataCsv,
    path=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    dateRange=[20150401, 20160401])
optModel = data.default.optLstm
optLoss = data.default.optLoss
optTrain = data.default.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'explore')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
# master.train(masterDict, overwrite=True)

pred = master.test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1', epoch=500)
