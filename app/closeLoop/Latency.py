import utils.dataset_format
from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

dLst = [1, 2, 3, 5, 15, 30]
doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    outLst = [
        os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_d' + str(nd))
        for nd in dLst
    ]
    subset = 'CONUSv2f1'
    tRange = [20160501, 20170501]
    outLSTM = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1')
    df, yp, obs = master.master_test(
        outLSTM, tRange=tRange, subset=subset, batchSize=100)
    yp = yp.squeeze()
    yfLst = list()
    for out in outLst:
        df, yf, obs = master.master_test(
            out, tRange=tRange, subset=subset, batchSize=100)
        yfLst.append(yf.squeeze())
    obs = obs.squeeze()

# stat
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
maskObsDay = maskObs * maskDay
maskF = (maskDay >= 1) & (maskDay <= 3)
statP = stat.statError(yp, obs)
statLst = [
    stat.statError(utils.dataset_format.fillNan(x, maskF), utils.dataset_format.fillNan(obs, maskF))
    for x in yfLst
]

# if 'post' in doLst:
caseLst = ['Predict'] + [str(nd) + 'd latency' for nd in dLst]
keyLst = list(statLst[0].keys())
dataBox = list()
for iS in range(len(keyLst)):
    key = keyLst[iS]
    temp = list()
    temp.append(statP[key])
    print(key, np.nanmedian(statP[key]))
    for k in range(len(statLst)):
        data = statLst[k][key]
        temp.append(data)
        print(key, k, np.nanmedian(data))
    dataBox.append(temp)
fig = plot.plot_box_fig(dataBox, keyLst, caseLst, sharey=False, figsize=[8, 3])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'box_latency'))

# plot maps
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(2, 3, figsize=[8, 3])
ix = 0
for key in ['ubRMSE', 'Corr']:
    iy = 0
    for k in [3, 4, 5]:
        if key == 'ubRMSE':
            titleStr = 'ubRMSE({}d)/ubRMSE(1d)'.format(dLst[k])
            data = (statLst[k][key] / statLst[0][key])
            cRange = [0.5, 1.5]
        elif key == 'Corr':
            titleStr = 'R({}d)/R(1d)'.format(dLst[k])
            data = statLst[k][key] / statLst[0][key]
            cRange = [0.8, 1.2]
        grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid,
            ax=axes[ix][iy],
            lat=uy,
            lon=ux,
            title=titleStr,
            cRange=cRange)
        iy = iy + 1
    ix = ix + 1
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_memory'))

# plot maps
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(2, 3, figsize=[8, 3])
key = 'RMSE'
for k in [0, 1, 2, 3, 4, 5]:
    titleStr = 'RMSE(proj)- RMSE({}d)'.format(dLst[k])
    data = statP[key] - statLst[k][key]
    cRange = [0,0.03]
    grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
    iy, ix = utils.dataset_format.index2d(k, 2, 3)
    print(iy, ix)
    plot.plotMap(
        grid, ax=axes[iy][ix], lat=uy, lon=ux, title=titleStr, cRange=cRange)
# plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_memory_ratio'))

# plot map and time series
dataGrid = [statP[key] / statLst[2][key], statP[key] / statLst[4][key]]
dataTs = [obs, yp] + yfLst
crd = df.getGeo()
t = df.getT()
mapNameLst = ['R ratio 3d', 'R ratio 15d']
tsNameLst = ['obs', 'prj'] + ['for{}d'.format(x) for x in dLst]
c1 = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1]])
c2 = plt.cm.jet(np.linspace(0, 1, len(dLst)))
tsColor = np.concatenate([c1, c2], axis=0)
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    tsColor=tsColor,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    isGrid=True)