import os
import numpy as np
from data import dbCsv
from hydroDL import stat
from hydroDL import rnn, crit, train
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')
nEpoch = 500
out = os.path.join(cDir, 'multiOutput_CONUSv16f1_VIC/CONUS_v16f1_SSRUN_EVP_SOILM_lev1_tr6_pix50')
tRange = [20160401, 20170401]

# test
# df, yp, yt, sigma = test(out, tRange=[20160401, 20170401], subset='CONUS_VICv16f1')

# load data
df, yp_1L, yt, sigma = master.master_test(
    out, tRange=tRange, subset='CONUS_VICv16f1', epoch=nEpoch)
yp_1L = yp_1L.squeeze()
yt = yt.squeeze()

# mlp
# load data
df_2L = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS_VICv16f1', tRange=[20160401, 20170401])
# x = df_2L.getDataTs(dbCsv.varForcing, doNorm=True, rmNan=True )
x = df_2L.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)
# c = df_2L.getDataConst(dbCsv.varConst, doNorm=True, rmNan=True)

# y_mae = np.abs(yp_1L[:, :, 0]) - np.abs(yt[:, :, 0])
# print(y_mae)

y_all = stat.statError(yp_1L[:, :, 0], yt[:, :, 0])
y_rmse = y_all['RMSE']
# print(y_rmse)
y = np.expand_dims(y_rmse, axis=1)
# y_tensor = torch.from_numpy(y_all['RMSE']).float()
# y_tensor.unsqueeze_(-1)
# # print(y_tensor)
# y=y_tensor.numpy()
# print(y)
# nx = x.shape[-1] + c.shape[-1]

y = df_2L.getDataTs(['SOILM_lev1_VIC'],doNorm=True,rmNan=True)

nx = x.shape[-1]
ny = 1

# print(nx,ny)
# print(c[305])
# print(len(y))
# print(len(x[0]),len(c[0]))

path_2L = 'multiOutput_CONUSv16f1_VIC/mlp_CONUS_v16f1_SOILM_lev1_test'
outFolder = os.path.join(cDir, path_2L)
if os.path.exists(outFolder) is False:
   os.mkdir(outFolder)

epoch=10
model = rnn.CudnnLstmModel_2L(nx=nx, ny=ny,  hiddenSize=6)
lossFun = crit.RmseLoss()

model, yp_2L, yt_2L = train.trainModel_2L(
    model, x, y, lossFun, nEpoch=epoch)
modelName = 'test-2L'
train.model_save(outFolder, model, epoch, modelName=modelName)

# print(yp_2L)
# print(yt_2L)
# print(type(yp_2L))
# print(type(yt_2L))
yp_2L = yp_2L.detach().cpu().numpy().swapaxes(0,1)
yt_2L = yt_2L.detach().cpu().numpy().swapaxes(0,1)
# # yp_2L = yp_2L.squeeze()
# # yt_2L = yt_2L.squeeze()
# print(yp_2L)
# print(yt_2L)
# print(type(yp_2L))
# print(type(yt_2L))

RMSE_2L = np.sqrt(np.nanmean((yp_2L - yt_2L)**2, axis=None))
print(RMSE_2L)

# CORR_2L = scipy.stats.pearsonr(yp_2L, yt_2L)[0]
yp_2L = yp_2L.squeeze()
yt_2L = yt_2L.squeeze()
CORR_2L = np.corrcoef(yp_2L, yt_2L)
print(CORR_2L)

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

