import utils.dataset_format
import utils.geo
from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    subset = 'CONUSv2f1'
    tRange = [20160401, 20180401]
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA2015')
    df, yf, obs = master.master_test(out, tRange=tRange, subset=subset, batchSize=100)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM2015')
    df, yp, obs = master.master_test(out, tRange=tRange, subset=subset)
    yf = yf.squeeze()
    yp = yp.squeeze()
    obs = obs.squeeze()

# figure out how many days observation lead
maskObs = 1 * ~np.isnan(obs.squeeze())
maskDay = np.zeros(maskObs.shape).astype(int)
ngrid, nt = maskObs.shape
for j in range(ngrid):
    temp = 0
    for i in range(nt):
        maskDay[j, i] = temp
        if maskObs[j, i] == 1:
            temp = 1
        else:
            if temp != 0:
                temp = temp + 1
ind = np.random.randint(0, ngrid)
print(np.array([maskObs[ind, :], maskDay[ind, :]]))
maskObsDay = maskObs * maskDay
unique, counts = np.unique(maskDay, return_counts=True)
print(np.asarray((unique, counts)).T)
print(counts / ngrid / nt)

fLst = [1, 2, 3]
statLstF = list()
statLstP = list()
maskF = (maskDay >= 1) & (maskDay <= 3)
statP = stat.statError(utils.dataset_format.fillNan(yp, maskF), utils.dataset_format.fillNan(obs, maskF))
statF = stat.statError(utils.dataset_format.fillNan(yf, maskF), utils.dataset_format.fillNan(obs, maskF))
for nf in fLst:
    xp = np.full([ngrid, nt], np.nan)
    xf = np.full([ngrid, nt], np.nan)
    y = np.full([ngrid, nt], np.nan)
    xf[maskObsDay == nf] = yf[maskObsDay == nf]
    xp[maskObsDay == nf] = yp[maskObsDay == nf]
    y[maskObsDay == nf] = obs[maskObsDay == nf]
    statLstF.append(stat.statError(xf, y))
    statLstP.append(stat.statError(xp, y))

# plot box - forecast
keyLst = stat.keyLst
dataBox = list()
caseLst = ['Predict'] + [str(nd) + 'd Forcast' for nd in fLst]
for k in range(len(keyLst)):
    key = keyLst[k]
    temp = list()
    data = statP[key]
    temp.append(data)
    print(key, np.nanmedian(statP[key]))
    print(key, np.nanmedian(statF[key]))
    for i in range(len(fLst)):
        data = statLstF[i][key]
        temp.append(data)
        # print(key, np.nanmedian(data))
    dataBox.append(temp)
plt.tight_layout()
fig = plot.plot_box_fig(dataBox, keyLst, caseLst, sharey=False, figsize=[8, 3])
fig.show()
fig.savefig(os.path.join(saveDir, 'box_forecast'))

# plot maps forecast_days
keyLst = ['RMSE', 'Corr']
cRangeLst = [[0, 0.05], [0.7, 1]]
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(len(fLst), len(keyLst), figsize=[8, 6])
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    for j in range(len(fLst)):
        data = statLstF[j][key]
        titleStr = key + ' of {}d Forecast'.format(fLst[j])
        grid, uy, ux = utils.geo.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid, ax=axes[j][i], lat=uy, lon=ux, title=titleStr, cRange=cRange)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast_days'))

# projection error
keyLst = ['RMSE', 'Corr']
cRangeLst = [[0, 0.05], [0.7, 1]]
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(1, len(keyLst), figsize=[8, 3])
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    data = statP[key]
    titleStr = 'Projection ' + key
    grid, uy, ux = utils.geo.array2grid(data, lat=lat, lon=lon)
    plot.plotMap(
        grid, ax=axes[i], lat=uy, lon=ux, title=titleStr, cRange=cRange)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_project'))

# change in forecast
keyLst = ['RMSE', 'Corr']
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(1, len(keyLst), figsize=[8, 3])
cRangeLst = [[-10, 40], [-3, 12]]
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    if key == 'RMSE':
        data = (statP[key] - statF[key]) / statP[key] * 100
    elif key == 'Corr':
        data = (statF[key] - statP[key]) * 100
    titleStr = 'Improvement in ' + key + ' (%)'
    grid, uy, ux = utils.geo.array2grid(data, lat=lat, lon=lon)
    plot.plotMap(
        grid, ax=axes[i], lat=uy, lon=ux, title=titleStr, cRange=cRangeLst[i])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast_impovement'))

key = 'RMSE'
data1 = (statP[key] - statF[key]) / statP[key] * 100
np.percentile(data1, 80)
key = 'Corr'
data2 = (statF[key] - statP[key]) * 100
np.percentile(data2, 4)

# plot map and time series
dataGrid = [data1, data2]
dataTs = [obs, yp, yf]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['diff ubRMSE', 'ratio Correlation']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    isGrid=True)

# import matplotlib
# matplotlib.rcParams.update({'font.size': 18})
# matplotlib.rcParams.update({'lines.linewidth': 2})
# matplotlib.rcParams.update({'lines.markersize': 12})
# matplotlib.rcParams.update({'legend.fontsize': 18})
