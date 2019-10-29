from hydroDL import master, utils
from hydroDL.data import dbCsv
from hydroDL.master import default, wrapMaster, run_train, test
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat

import os
import numpy as np
import statistics
import torch

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')

df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS_VICv16f1', tRange=[20150401, 20160401]
    )

Forcing = df.getDataTs(dbCsv.varForcing, doNorm=False, rmNan=True)
Raw_data = df.getDataConst(dbCsv.varRaw, doNorm=False, rmNan=True)

Target = df.getDataTs(['SOILM_lev1_VIC'], doNorm=False, rmNan=True)

nx = Forcing.shape[-1] + Raw_data.shape[-1]
ny = 1

path_RF = 'multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SOILM_lev1_RF_LSTM_woNorm'
outFolder = os.path.join(cDir, path_RF)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=500
model_RF = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=256)
lossFun_RF = crit.RmseLoss()
model_RF = train.train_model(
    model_RF, Forcing, Target, Raw_data, lossFun_RF, nEpoch=epoch, miniBatch=[100, 60], saveFolder=outFolder)
modelName = 'RF_LSTM'
train.save_model(outFolder, model_RF, epoch, modelName=modelName)