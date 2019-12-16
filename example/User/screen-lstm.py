import app.common.default
import data.data_process
from hydroDL import pathSMAP, master
import os

# define training options
out = os.path.join(pathSMAP['Out_L3_NA'], 'RegTest', 'CONUSv4f1_sigma')

optData = data.default.update(
    data.default.optDataCsv,
    rootDB=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = data.default.optLstm
optLoss = data.default.update(
    data.default.optLoss, name='hydroDL.model.crit.SigmaLoss')
optTrain = data.default.optTrain

masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss, optTrain)

# train
master.run_train(masterDict, cudaID=0, screenName='sigmaTest')
