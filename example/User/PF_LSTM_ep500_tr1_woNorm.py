from data import dbCsv
from hydroDL import rnn, crit, train

import os

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')

df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS_VICv16f1', tRange=[20150401, 20160401]
    )

Forcing = df.getDataTs(dbCsv.varForcing, doNorm=False, rmNan=True)
Parameters = df.getDataConst(dbCsv.varConst, doNorm=False, rmNan=True)

Target = df.getDataTs(['SOILM_lev1_VIC'], doNorm=False, rmNan=True)

nx = Forcing.shape[-1] + Parameters.shape[-1]
ny = 1

path_PF = 'multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SOILM_lev1_PF_LSTM_woNorm'
outFolder = os.path.join(cDir, path_PF)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=500
model_PF = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=256)
lossFun_PF = crit.RmseLoss()
model_PF = train.train_model(
    model_PF, Forcing, Target, Parameters, lossFun_PF, nEpoch=epoch, miniBatch=[100, 60], saveFolder=outFolder)
modelName = 'PF_LSTM'

train.save_model(outFolder, model_PF, epoch, modelName=modelName)

# # test and obtain output
# train.testModel(model_PF, Forcing, Parameters)
