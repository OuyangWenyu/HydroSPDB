import data.read_config
import data.data_input
from hydroDL import master
import os
from data import read_config

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/home/kxf227/work/GitHUB/pyRnnSMAP/example/'

# define training options
optData = read_config.update(
    read_config.optDataSMAP,
    rootDB=os.path.join(cDir, 'data'),
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = read_config.optLstm
optLoss = read_config.optLossSigma
optTrain = read_config.update(data.read_config.optTrainSMAP, nEpoch=5, saveEpoch=5)
out = os.path.join(cDir, 'output', 'CONUSv4f1_sigma')
masterDict = data.read_config.wrap_master(out, optData, optModel, optLoss, optTrain)

# train
master.master_train(masterDict)

# test
pred = master.master_test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1')
