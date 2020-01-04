import data.data_config
import data.data_input
from hydroDL import pathSMAP, master
import os

# define training options
out = os.path.join(pathSMAP['Out_L3_NA'], 'RegTest', 'CONUSv4f1_sigma')

optData = data.data_config.update(
    data.data_config.optDataCsv,
    rootDB=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = data.data_config.optLstm
optLoss = data.data_config.update(
    data.data_config.optLoss, name='hydroDL.model.crit.SigmaLoss')
optTrain = data.data_config.optTrain

masterDict = data.data_config.wrap_master(out, optData, optModel, optLoss, optTrain)

# train
master.run_train(masterDict, cudaID=0, screenName='sigmaTest')
