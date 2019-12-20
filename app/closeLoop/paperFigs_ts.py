from hydroDL import pathSMAP, master
import utils
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
statP = stat.statError(utils.fillNan(yp, maskF), utils.fillNan(obs, maskF))
statF = stat.statError(utils.fillNan(yf, maskF), utils.fillNan(obs, maskF))

# plot map and time series
import importlib
importlib.reload(plot)
dataGrid = [statP['RMSE'] - statF['RMSE'], statP['Corr'] - statF['Corr']]
prcp = df.get_data_ts('APCP_FORA').squeeze()
dataTs = [obs, yp, yf]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['dRMSE', 'dR']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    isGrid=True)

# plot pixel time series
import importlib
importlib.reload(plot)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})
matplotlib.rcParams.update({'legend.fontsize': 12})
indLst = [109, 943, 1113, 1023, 1188, 1442]

# pct = [10, 25, 50, 75, 90]
# indLst = list()
# diff = (statP['RMSE'] - statF['RMSE']) / statP['RMSE']
# for p in pct:
#     indLst.append(abs(diff - np.percentile(diff, p)).argmin())

nts = len(indLst)
fig, axes = plt.subplots(nts, 1, figsize=[8, 6])
t = df.getT()
for k in range(nts):
    ind = indLst[k]
    ax = axes[k]
    legLst = ['SMAP', 'project', 'forecast'] if k == 2 else None
    plot.plot_ts(
        t, [obs[ind, :], yp[ind, :], yf[ind, :]],
        ax=ax,
        cLst='krb',
        linewidth=1)
    if k != nts - 1:
        ax.set_xticklabels([])
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.subplots_adjust(vspace=0)
fig.show()
fig.savefig(os.path.join(saveDir, 'ts_pixel.eps'))
fig.savefig(os.path.join(saveDir, 'ts_pixel'))

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
plot.plot_ts(
    t, [obs[ind, :], yp[ind, :], yf[ind, :]],
    ax=ax,
    cLst='krb',
    legLst=['SMAP', 'project', 'forecast'],
    linewidth=1)
ax.set_ylim([-2, -1])
fig.show()
fig.savefig(os.path.join(saveDir, 'ts_pixel_leg.eps'))
fig.savefig(os.path.join(saveDir, 'ts_pixel_leg'))

#  add map of pts
import importlib
importlib.reload(plot)
lat, lon = df.getGeo()
pts = [lat[indLst], lon[indLst]]
diff = abs(yp - yf).mean(axis=1)
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
grid, uy, ux = utils.grid.array2grid(diff, lat=lat, lon=lon)
plot.plotMap(
    grid,
    ax=ax,
    lat=uy,
    lon=ux,
    pts=pts,
    title='Abs difference between project and forecast')
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'ts_pixel_map.eps'))
fig.savefig(os.path.join(saveDir, 'ts_pixel_map'))