from hydroDL import wrapMaster, train
from data import data_config
import os

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/home/kxf227/work/GitHUB/pyRnnSMAP/example/'

# define training options
optData = data_config.update(
    data_config.optDataSMAP,
    rootDB=os.path.join(cDir, 'data'),
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
optModel = data_config.optLstm
optLoss = data_config.optLossRMSE
optTrain = data_config.update(data_config.optTrainSMAP, nEpoch=100)
out = os.path.join(cDir, 'output', 'CONUSv4f1')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)
