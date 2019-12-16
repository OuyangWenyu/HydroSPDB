import os
import matplotlib.pyplot as plt
import numpy as np
from hydroDL import stat
from visual import plot
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/mnt/sdc/SUR_VIC/'

rootDB = os.path.join(cDir, 'input_VIC')
nEpoch = 100
out = os.path.join(cDir, 'output_VIC/CONUS_v16f1_SOILM_lev1_rho60_ep500_tr6_Depth1_rm_Ds_Ws')
tRange = [20160401, 20170401]

# test
# df, yp, yt, sigma = test(out, tRange=[20160401, 20170401], subset='CONUS_VICv16f1')

# load data
df, yp, yt, sigma = master.test(
    out, tRange=tRange, subset='CONUS_VICv16f1', epoch=nEpoch)
yp = yp.squeeze()
yt = yt.squeeze()


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
