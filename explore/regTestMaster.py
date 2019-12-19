import data.read_config
import data.data_process
from hydroDL import pathSMAP, master
import os

optData = master.updateOpt(
    data.read_config.optDataCsv,
    path=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    dateRange=[20150401, 20160401])
optModel = data.read_config.optLstm
optLoss = data.read_config.optLoss
optTrain = data.read_config.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'explore')
masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss, optTrain)
# master.train(masterDict, overwrite=True)

pred = master.test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1', epoch=500)
