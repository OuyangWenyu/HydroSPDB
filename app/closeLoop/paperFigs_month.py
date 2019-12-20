from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA', 'paper')

# test
tAllR = [20150402, 20180401]
if 'test' in doLst:
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
    subset = 'CONUSv2f1'
    tRange = tAllR
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
maskObsDay = maskObs * maskDay
unique, counts = np.unique(maskObsDay, return_counts=True)
maskF = (maskDay >= 1) & (maskDay <= 3)
statP = stat.statError(utils.fillNan(yp, maskF), utils.fillNan(obs, maskF))
statF = stat.statError(utils.fillNan(yf, maskF), utils.fillNan(obs, maskF))

maskObsDay = maskObs * maskDay
print(np.array([maskObs[ind, :], maskDay[ind, :]]))
print(np.asarray((unique, counts)).T)
print(counts / ngrid / nt)

# see result for different seasons
tRangeLst = [[20180101, 20180201], [20180201, 20180301], [20180301, 20180401],
             [20160401, 20160501], [20160501, 20160601], [20160601, 20160701],
             [20160701, 20160801], [20160801, 20160901], [20160901, 20161001],
             [20161001, 20161101], [20161101, 20161201], [20161201, 20170101],
             [20170101, 20170201], [20170201, 20170301], [20170301, 20170401],
             [20170401, 20170501], [20170501, 20170601], [20170601, 20170701],
             [20170701, 20170801], [20170801, 20170901], [20170901, 20171001],
             [20171001, 20171101], [20171101, 20171201], [20171201, 20180101]]
tAllA = utils.time.tRange2Array(tAllR)
statPLst = list()
statFLst = list()
for k in range(12):
    tRLst = [tRangeLst[k], tRangeLst[k + 12]]
    temp = list()
    for tR in tRLst:
        tA = utils.time.tRange2Array(tR)
        ind0 = np.array(range(nt))
        ind1, ind2 = utils.time.intersect(tAllA, tA)
        temp.append(ind1)
    indT = np.concatenate(temp)
    yfTemp = utils.fillNan(yf, maskF)[:, indT]
    ypTemp = utils.fillNan(yp, maskF)[:, indT]
    obsTemp = utils.fillNan(obs, maskF)[:, indT]
    statPLst.append(stat.statError(ypTemp, obsTemp))
    statFLst.append(stat.statError(yfTemp, obsTemp))

## box plot of factors
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 11})

cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
indLst = [
    cropRate[:, 0] > 20, cropRate[:, 22] > 5, cropRate[:, 23] > 10,
    cropRate[:, 2] > 1
]
labMonth = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Agu', 'Sep', 'Oct',
    'Nov', 'Dec'
]
labCrop = ['Corn', 'Spring wheat', 'Winter wheat', 'Rice']
dataBox = list()
for k in range(12):
    tempLst = list()
    for iC in range(len(indLst)):
        data = statPLst[k]['RMSE'][indLst[iC]] - statFLst[k]['RMSE'][
            indLst[iC]]
        if len(data[~np.isnan(data)]) < 20:
            data = None
        tempLst.append(data)
    dataBox.append(tempLst)
fig = plot.plot_box_fig(
    dataBox,
    label1=labMonth,
    label2=None,
    sharey=True,
    figsize=[8, 3],
    colorLst='rgbk')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.ylim(-0.02, 0.04)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_crop.eps'))
fig.savefig(os.path.join(saveDir, 'box_crop.png'))

fig = plot.plot_box_fig(
    dataBox[10:11],
    label1=labMonth[10:11],
    label2=labCrop,
    sharey=True,
    figsize=[8, 3],
    colorLst='rgbk',
    legOnly=True)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'box_crop_leg.eps'))
fig.savefig(os.path.join(saveDir, 'box_crop_leg.png'))

# map crop
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(2, 2, figsize=[8, 4])
dataLst = [cropRate[:, 0], cropRate[:, 22], cropRate[:, 23], cropRate[:, 2]]
cRangeLst = [[0, 40], [0, 10], [0, 20], [0, 2]]
for k in range(4):
    iy, ix = utils.index2d(k, 2, 2)
    grid, uy, ux = utils.grid.array2grid(dataLst[k], lat=lat, lon=lon)
    plot.plotMap(
        grid,
        ax=axes[iy][ix],
        lat=uy,
        lon=ux,
        title=labCrop[k] + ' fraction (%)',
        cRange=cRangeLst[k])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_crop.eps'))
fig.savefig(os.path.join(saveDir, 'map_crop.png'))
