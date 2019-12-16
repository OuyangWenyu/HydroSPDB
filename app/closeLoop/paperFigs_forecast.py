from hydroDL import pathSMAP, master, utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA', 'paper')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    subset = 'CONUSv2f1'
    tRange = [20160401, 20180401]
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA2015')
    df, yf, obs = master.test(out, tRange=tRange, subset=subset, batchSize=100)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM2015')
    df, yp, obs = master.test(out, tRange=tRange, subset=subset)
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
statP = stat.statError(utils.fillNan(yp, maskF), utils.fillNan(obs, maskF))
statF = stat.statError(utils.fillNan(yf, maskF), utils.fillNan(obs, maskF))
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
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 11})
keyLst = stat.keyLst
keyLegLst = ['Bias', 'RMSE', 'ubRMSE', 'R']
dataBox = list()
caseLst = ['Project'] + [str(nd) + 'd Forecast' for nd in fLst]
for k in range(len(keyLst)):
    key = keyLst[k]
    temp = list()
    data = statP[key]
    temp.append(data)
    for i in range(len(fLst)):
        data = statLstF[i][key]
        temp.append(data)
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox, keyLegLst, sharey=False, figsize=[8, 4])
plt.suptitle('Error metrics of projection and forecast model')
plt.tight_layout()
plt.subplots_adjust(top=0.85, right=0.95)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_forecast.eps'))
fig.savefig(os.path.join(saveDir, 'box_forecast.png'))

fig = plot.plotBoxFig(
    dataBox, keyLst, caseLst, sharey=False, figsize=[8, 3], legOnly=True)
# plt.suptitle('Error matrices of project and forecast model')
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'box_forecast_leg.eps'))
fig.savefig(os.path.join(saveDir, 'box_forecast_leg.png'))

## map forecast
keyLst = ['RMSE', 'Corr']
cRangeLst = [[0, 0.05], [0.7, 1]]
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(len(fLst), len(keyLst), figsize=[8, 7])
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    for j in range(len(fLst)):
        data = statLstF[j][key]
        if key == 'Corr':
            titleStr = 'R of {}d Forecast'.format(fLst[j])
        else:
            titleStr = key + ' of {}d Forecast'.format(fLst[j])
        grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid, ax=axes[j][i], lat=uy, lon=ux, title=titleStr, cRange=cRange)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast.eps'))
fig.savefig(os.path.join(saveDir, 'map_forecast.png'))

# map improvement
keyLst = ['RMSE', 'Corr']
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(1, len(keyLst), figsize=[8, 3])
cRangeLst = [[0, 50], [0, 25]]
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    if key == 'RMSE':
        data = (statP[key] - statF[key]) / statP[key] * 100
        titleStr = 'Improvement in RMSE (%)'
    elif key == 'Corr':
        data = (statF[key] - statP[key]) / statP[key] * 100
        titleStr = 'Improvement in R (%)'
    grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
    plot.plotMap(
        grid, ax=axes[i], lat=uy, lon=ux, title=titleStr, cRange=cRangeLst[i])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_impovement.eps'))
fig.savefig(os.path.join(saveDir, 'map_impovement.png'))

import importlib
importlib.reload(plot)
