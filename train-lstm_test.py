from hydroDL import pathSMAP, master
import os

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

# define training options
optData = master.updateOpt(
    master.default.optDataCsv,
    path=os.path.join(cDir, 'input_VIC'),
    subset='CONUS_VIC',
    tRange=[20150401, 20160401])
optModel = master.default.optLstm
optLoss = master.default.optLoss
optTrain = master.default.optTrainSMAP
out = os.path.join(cDir, 'output_VIC', 'CONUS_VIC')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
master.train(masterDict, overwrite=True)
