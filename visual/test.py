"""可视化测试结果"""
import data.read_config
import data.data_input
from hydroDL import pathSMAP, master
import os
from data import dbCsv

optData = data.read_config.update(
    data.read_config.optDataSMAP,
    rootDB=pathSMAP['DB_L3_Global'],
    subset='Globalv4f1_NorthAmerica',
    tRange=[20150401, 20160401],
    varT=dbCsv.varForcingGlobal)
optModel = data.read_config.optLstm
optLoss = data.read_config.optLossSigma
optTrain = data.read_config.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'test')
masterDict = data.read_config.wrap_master(out, optData, optModel, optLoss, optTrain)
master.train(masterDict)
