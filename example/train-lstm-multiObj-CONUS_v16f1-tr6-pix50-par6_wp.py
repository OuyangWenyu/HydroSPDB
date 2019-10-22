from hydroDL import pathSMAP
from hydroDL.master import default, wrapMaster, train, runTrain, test
from hydroDL.post import plot, stat

import os
import numpy as np

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB='/mnt/sdc/SUR_VIC/input_VIC/',
    varT=[
    'APCP_FORA', 'DLWRF_FORA', 'DSWRF_FORA', 'TMP_2_FORA', 'SPFH_2_FORA',
    'VGRD_10_FORA', 'UGRD_10_FORA', 'PEVAP_FORA', 'PRES_FORA'
],
    varC=[
    'DEPTH_1', 'Ds', 'Ds_MAX', 'EXPT_1', 'INFILT', 'Ws'
],

    target=['SOILM_lev1_VIC', 'SSRUN_VIC','EVP_VIC'],
    # target='SOILM_0-100_VIC',
    subset='CONUS_VICv16f1',
    tRange=[20100401, 20160401])
optModel = default.optLstm
optLoss = default.optLossSigma
optTrain = default.update(default.optTrainSMAP, miniBatch=[50,60], nEpoch=500, saveEpoch=100)
out = os.path.join(cDir, 'multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SSRUN_EVP_SOILM_lev1_tr6_pix50_par6')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)
# runTrain(masterDict, cudaID=2, screen='LSTM-multi')

'''
# test
df, yp, yt, sigma = test(out, tRange=[20160401, 20170401], subset='CONUS_VICv16f1')

# plot ts MAP
dataGrid = list()
dataTs = list()
for k in range(2):
    statErr = stat.statError(yp[:, :, k], yt[:, :, k])
    dataGrid.append(statErr['RMSE'])
    dataTs.append([yp[:, :, k], yt[:, :, k]])
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE-SOILM_lev1', 'RMSE-SSRUN', 'RMSE-EVP']
tsNameLst = ['LSTM', 'VIC']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    multiTS=True,
    isGrid=True)


dataGrid = list()
dataTs = list()
for k in range(2):
    statErr = stat.statError(yp[:, :, k], yt[:, :, k])
    dataGrid.append(statErr['RMSE'])
    dataTs.append([np.abs(yp[:, :, k]-yt[:, :, k]), sigma[:, :, k]])
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE-SOILM_lev1', 'RMSE-SSRUN', 'RMSE-EVP']
tsNameLst = ['Error', 'sigma']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    multiTS=True,
    isGrid=True)
'''

