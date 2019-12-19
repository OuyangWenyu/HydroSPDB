import data.read_config
import data.data_process
from hydroDL import pathSMAP, master
import os

# define training options
out = os.path.join(pathSMAP['Out_L3_NA'], 'RegTest', 'CONUSv4f1_sigma')

optData = data.read_config.update(
    data.read_config.optDataCsv,
    rootDB=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = data.read_config.optLstm
optLoss = data.read_config.update(
    data.read_config.optLoss, name='hydroDL.model.crit.SigmaLoss')
optTrain = data.read_config.optTrain

masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss, optTrain)

# train
master.run_train(masterDict, cudaID=0, screenName='sigmaTest')
