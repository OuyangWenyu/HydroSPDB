from hydroDL import pathSMAP, master, utils
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

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB='/mnt/sdc/SUR_VIC/input_VIC/',
    varT=[
    'APCP_FORA', 'DLWRF_FORA', 'DSWRF_FORA', 'TMP_2_FORA', 'SPFH_2_FORA',
    'VGRD_10_FORA', 'UGRD_10_FORA', 'PEVAP_FORA', 'PRES_FORA'
],
    varC=[
    'DEPTH_1', 'DEPTH_2', 'DEPTH_3', 'Ds', 'Ds_MAX', 'EXPT_1', 'EXPT_2', 'EXPT_3',
    'INFILT', 'Ws'
],
    target=['SOILM_lev1_VIC'],
    # target=['SOILM_lev1_VIC', 'SSRUN_VIC','EVP_VIC'],
    # target='SOILM_0-100_VIC',
    subset='CONUS_VICv16f1',
    tRange=[20100401, 20160401])
'''
optData_2L = default.update(
    default.optDataSMAP_2L, # need to modify 'master/default.py & master.py', 'data/dbCsv.py'
    rootDB_2L='/mnt/sdc/SUR_VIC/input_VIC/',
    varConst=[],
    varR=[
    'NDVI', 'Capa', 'Bulk', 'Clay', 'Silt', 'Sand'
],
    target_2L=['SOILM_lev1_VIC'], # for 2L LSTM
    subset_2L='CONUS_VICv16f1',
    tRange_2L=[2010041, 20160401])
'''
optModel = default.optLstm
# optModel_2L = default.optLstm_2L 
optLoss = default.optLossSigma
# optLoss_2L = default.optLossSigma
optTrain = default.update(default.optTrainSMAP, miniBatch=[100,60], nEpoch=1) 
# optTrain_2L = default.update(default.optTrainSMAP_2L, miniBatch_2L=[100,1], nEpoch_2L=500)
out = os.path.join(cDir, 'multiOutput_CONUSv16f1_VIC/2L_CONUS_v16f1_SOILM_lev1_test')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
# print(masterDict)
# masterDict_2L = wrapMaster(out, optData_2L, optModel_2L, optLoss_2L, optTrain_2L)
# print('===============')
# print(masterDict_2L)

# train
#### master.train(masterDict)
# runTrain(masterDict, cudaID=2, screen='LSTM-multi')
# train(masterDict_2L)

'''
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
from hydroDL.model import rnn, crit, train
from hydroDL import master, utils
from hydroDL.master import default, wrapMaster, runTrain, test

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')
nEpoch = 500
out = os.path.join(cDir, 'multiOutput_CONUSv16f1_VIC/2L_CONUS_v16f1_SOILM_lev1_test')
tRange = [20100401, 20170401]

# test
# df, yp, yt, sigma = test(out, tRange=[20160401, 20170401], subset='CONUS_VICv16f1')
# load data
df, yp_1L, yt, sigma = master.test(
    out, tRange=tRange, subset='CONUS_VICv16f1', epoch=nEpoch)
yp_1L = yp_1L.squeeze()
yt = yt.squeeze()
'''

# 2L LSTM
# load data
rootDB = os.path.join(cDir, 'input_VIC')
df_2L = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS_VICv16f1', tRange=[20150401, 20160401])

varC=[
'DEPTH_1', 'DEPTH_2', 'DEPTH_3', 'Ds', 'Ds_MAX', 'EXPT_1', 'EXPT_2', 'EXPT_3',
'INFILT', 'Ws'
]

filename = '/mnt/sdc/SUR_VIC/output_VIC/CONUS_v16f1_SOILM_lev1_rho60_ep500_tr6/model_Ep500.pt'
# filename = '/mnt/sdc/SUR_VIC/multiOutput_CONUSv16f1_VIC/2L_CONUS_v16f1_SOILM_lev1_test/model_Ep500.pt'
path_2L = 'multiOutput_CONUSv16f1_VIC/2L_CONUS_v16f1_SOILM_lev1_test'

# x = df_2L.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True )
x_2L = df_2L.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True)
C2 = df_2L.getDataConst(dbCsv.varRaw, doNorm=True, rmNan=True)
# c = df_2L.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)

## y_mae = np.abs(yp_1L[:, :, 0]) - np.abs(yt[:, :, 0])
## print(y_mae)
#y_all = stat.statError(yp_1L[:, :, 0], yt[:, :, 0])
#y_rmse = y_all['RMSE']
# print(y_rmse)
#y = np.expand_dims(y_rmse, axis=1)
## y_tensor = torch.from_numpy(y_all['RMSE']).float()
## y_tensor.unsqueeze_(-1)
## # print(y_tensor)
## y=y_tensor.numpy()
## print(y)
## nx = x.shape[-1] + c.shape[-1]

y_2L = df_2L.getDataTs(['SOILM_lev1_VIC'], doNorm=True, rmNan=True)

nx_2L = (x_2L.shape[-1] + C2.shape[-1], C2.shape[-1], len(varC))
ny_2L = 1

# print(nx,ny)
# print(c[305])
# print(len(y))
# print(len(x[0]),len(c[0]))


outFolder = os.path.join(cDir, path_2L)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=500
model_2L = rnn.CudnnLstmModel_2L(nx=nx_2L, ny=ny_2L, hiddenSize=256,filename=filename)
lossFun_2L = crit.RmseLoss()
# print(model_2L)
# print(model_2L.parameters)

model_2L = train.trainModel(
    model_2L, x_2L, y_2L, C2, lossFun_2L, nEpoch=epoch)
modelName = 'test-2L'
train.saveModel(outFolder, model_2L, epoch, modelName=modelName)

# print(yp_2L)
# print(yt_2L)
# print(type(yp_2L))
# print(type(yt_2L))
### yp_2L = yp_2L.detach().cpu().numpy().swapaxes(0,1)
### yt_2L = yt_2L.detach().cpu().numpy().swapaxes(0,1)
# # yp_2L = yp_2L.squeeze()
# # yt_2L = yt_2L.squeeze()
# print(yp_2L)
# print(yt_2L)
# print(type(yp_2L))
# print(type(yt_2L))

### RMSE_2L = np.sqrt(np.nanmean((yp_2L - yt_2L)**2, axis=None))
### print(RMSE_2L)

# CORR_2L = scipy.stats.pearsonr(yp_2L, yt_2L)[0]
### yp_2L = yp_2L.squeeze()
### yt_2L = yt_2L.squeeze()
### CORR_2L = np.corrcoef(yp_2L, yt_2L)
### print(CORR_2L)



'''
# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB='/mnt/sdc/SUR_VIC/input_VIC/',
    varT=[],
    varC=['NDVI', 'Capa', 'Bulk', 'Silt', 'Sand', 'Clay' ],
    target=lossFuct,
    subset='CONUS_VICv16f1',
    tRange=[20100401, 20160401])
optModel = default.optAnn
optLoss = default.optLossSigma
optTrain = default.update(default.optTrainSMAP, miniBatch=[100,60], nEpoch=500)
out = os.path.join(cDir, 'multiOutput_CONUSv16f1_VIC/MLP_CONUS_v16f1_SOILM_lev1')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
'''


'''
# train
train(masterDict)
'''

'''
# plot ts MAP, Obs vs Out
# plot ts Map: Corr
dataGrid = list()
dataTs = list()
for k in range(3):
    statErr = stat.statError(yp[:, :, k], yt[:, :, k])
    dataGrid.append(statErr['Corr'])
    dataTs.append([yp[:, :, k], yt[:, :, k]])
t = df.getT()
crd = df.getGeo()
mapNameLst = ['Corr-SOILM_lev1', 'Corr-SSRUN', 'Corr-EVP']
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

# print(dataGrid[0])

# boxplot: CORR
fig1, ax1=plt.subplots()
ax1.set_title('CORR-SOILM_lev_1', fontsize=18, fontweight='bold')
ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax1.boxplot(dataGrid[0])
# plt.savefig(out + '/CORR_SOILM_lev_1.png')

fig2, ax2=plt.subplots()
ax2.set_title('CORR-SSRUN', fontsize=18, fontweight='bold')
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax2.boxplot(dataGrid[1])
# plt.savefig(out + '/CORR_SSUN.png')

fig3, ax3=plt.subplots()
ax3.set_title('CORR-EVP', fontsize=18, fontweight='bold')
ax3.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax3.boxplot(dataGrid[2])
# plt.savefig(out + '/CORR_EVP.png')

# plot ts Map: RMSE
dataGrid = list()
dataTs = list()
for k in range(3):
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

# boxplot: RMSE
fig1, ax1=plt.subplots()
ax1.set_title('RMSE-SOILM_lev_1', fontsize=18, fontweight='bold')
# ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax1.boxplot(dataGrid[0])
# plt.savefig(out + '/RMSE_SOILM_lev_1.png')

fig2, ax2=plt.subplots()
ax2.set_title('RMSE-SSRUN', fontsize=18, fontweight='bold')
# ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax2.boxplot(dataGrid[1])
# plt.savefig(out + '/RMSE_SSUN.png')

fig3, ax3=plt.subplots()
ax3.set_title('RMSE-EVP', fontsize=18, fontweight='bold')
# ax3.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 1.0])
ax3.boxplot(dataGrid[2])
# plt.savefig(out + '/RMSE_EVP.png')

# plot ts map, Error vs Sigma
dataGrid = list()
dataTs = list()
for k in range(3):
    statErr = stat.statError(yp[:, :, k], yt[:, :, k])
    dataGrid.append(statErr['RMSE'])
    dataTs.append([np.abs(yp[:, :, k]-yt[:, :, k]), sigma[:, :, k]])
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE-SOILM_lev1', 'RMSE-SSRUN', 'RMSE-EVP']
tsNameLst = ['Error', 'Sigma']
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

