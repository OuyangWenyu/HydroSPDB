from data import dbCsv
from hydroDL import rnn, crit, train

import os

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'H:\Wenping\7_DL_VIC\SUR_VIC'

rootDB = os.path.join(cDir, 'input_VIC')

df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS_VICv16f1', tRange=[20150401, 20160401]
    )

Forcing = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
Parameters = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)

Target = df.getDataTs(['SOILM_lev1_VIC'], doNorm=True, rmNan=True)

nx = Forcing.shape[-1] + Parameters.shape[-1]
ny = 1

path_PF = '\CONUS_v16f1_SOILM_lev1_PF_LSTM_windows'
outFolder = os.path.join(cDir, path_PF)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=500
model_PF = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=256)
lossFun_PF = crit.RmseLoss()
model_PF = train.model_train(
    model_PF, Forcing, Target, Parameters, lossFun_PF, nEpoch=epoch, miniBatch=[100, 60], saveFolder=outFolder)
modelName = 'PF_LSTM'

train.model_save(outFolder, model_PF, epoch, modelName=modelName)

# test and obtain output
train.model_test(model_PF, Forcing, Parameters)
