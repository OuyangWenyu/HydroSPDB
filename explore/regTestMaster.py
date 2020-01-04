import data.data_config
import data.data_input
from hydroDL import pathSMAP, master
import os

optData = master.updateOpt(
    data.data_config.optDataCsv,
    path=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    dateRange=[20150401, 20160401])
optModel = data.data_config.optLstm
optLoss = data.data_config.optLoss
optTrain = data.data_config.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'explore')
masterDict = data.data_config.wrap_master(out, optData, optModel, optLoss, optTrain)
# master.train(masterDict, overwrite=True)

pred = master.master_test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1', epoch=500)
