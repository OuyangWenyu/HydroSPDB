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

varC=[
'DEPTH_1', 'DEPTH_2', 'DEPTH_3', 'Ds', 'Ds_MAX', 'EXPT_1', 'EXPT_2', 'EXPT_3',
'INFILT', 'Ws'
]

Parameters = df.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)

Forcing = df.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
Raw_data = df.getDataConst(dbCsv.varRaw, doNorm=True, rmNan=True)

Target = df.getDataTs(['SOILM_lev1_VIC'], doNorm=True, rmNan=True)

# transfer to tensor
Parameters_tensor = torch.from_numpy(Parameters)
Forcing_tensor = torch.from_numpy(Forcing)
Raw_tensor = torch.from_numpy(Raw_data)

nx = (Forcing.shape[-1] + Raw_data.shape[-1], Raw_data.shape[-1], len(varC))
ny = 1

filename1 = '/mnt/sdc/SUR_VIC/multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SOILM_lev1_R2PF_LSTM_badResults/R2PF_LSTM_Ep500.pt'
model_R2PF_loaded = torch.load(filename1)
model_R2PF_loaded.eval()
# train.testModel(model_R2PF_loaded, Forcing_tensor, Raw_tensor)

filename2 = '/mnt/sdc/SUR_VIC/multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SOILM_lev1_PF_LSTM/PF_LSTM_Ep500.pt'
model_PF_loaded = torch.load(filename2)
model_PF_loaded.eval()
