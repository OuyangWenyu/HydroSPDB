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

Forcing = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
Raw_data = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)

Target = df.getDataTs(['SOILM_lev1_VIC'], doNorm=True, rmNan=True)

nx = Forcing.shape[-1] + Raw_data.shape[-1]
ny = 1

path_F = 'multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SOILM_lev1_F_LSTM'
outFolder = os.path.join(cDir, path_F)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=500
model_F = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=256)
lossFun_F = crit.RmseLoss()
model_F = train.trainModel(
    model_F, Forcing, Target, Raw_data, lossFun_F, nEpoch=epoch, miniBatch=[100, 60], saveFolder=outFolder)
modelName = 'F_LSTM'
train.saveModel(outFolder, model_F, epoch, modelName=modelName)