from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)

    subset = 'CONUSv2f1'
    tRange = [20150402, 20180401]
    yrStrLst = ['2015', '2016', '2017']
    yfLst = list()
    ypLst = list()
    for yrStr in yrStrLst:
        out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA' + yrStr)
        df, yf, obs = master.test(
            out, tRange=tRange, subset=subset, batchSize=100)
        out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                           'CONUSv2f1_LSTM' + yrStr)
        df, yp, obs = master.test(out, tRange=tRange, subset=subset)
        yf = yf.squeeze()
        yp = yp.squeeze()
        yfLst.append(yf)
        ypLst.append(yp)
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
statPLst = list()
statFLst = list()
for k in range(3):
    statP = stat.statError(
        utils.fillNan(ypLst[k], maskF), utils.fillNan(obs, maskF))
    statF = stat.statError(
        utils.fillNan(yfLst[k], maskF), utils.fillNan(obs, maskF))
    statPLst.append(statP)
    statFLst.append(statF)

cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
# croprate - 0 corn, 4 soybean, 22 spring wheat, 23 winter wheat
dataGrid = [(statPLst[0]['RMSE'] - statFLst[0]['RMSE']) / statPLst[0]['RMSE'],
            (statPLst[1]['RMSE'] - statFLst[1]['RMSE']) / statPLst[1]['RMSE'],
            (statPLst[2]['RMSE'] - statFLst[2]['RMSE']) / statPLst[2]['RMSE'],            
            ]
prcp = df.get_data_ts('APCP_FORA').squeeze()
dataTs = [[obs, ypLst[0], yfLst[0]], [obs, ypLst[1], yfLst[1]],
          [obs, ypLst[2], yfLst[2]], [prcp]]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['dRMSE 2015', 'dRMSE 2016', 'dRMSE 2017']
tsNameLst = ['obs', 'prj', 'fore']
tBar = [utils.time.t2dt(20160401), utils.time.t2dt(20170401)]
#plt.tight_layout()
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    isGrid=True,
    multiTS=True,
    linewidth=1,
    figsize=(10, 10),
    tBar=tBar)

# see result for different seasons
tRangeLst = [[20180101, 20180201], [20180201, 20180301], [20180301, 20180401],
             [20160401, 20160501], [20160501, 20160601], [20160601, 20160701],
             [20160701, 20160801], [20160801, 20160901], [20160901, 20161001],
             [20161001, 20161101], [20161101, 20161201], [20161201, 20170101],
             [20170101, 20170201], [20170201, 20170301], [20170301, 20170401],
             [20170401, 20170501], [20170501, 20170601], [20170601, 20170701],
             [20170701, 20170801], [20170801, 20170901], [20170901, 20171001],
             [20171001, 20171101], [20171101, 20171201], [20171201, 20180101]]
tAllR = [20150402, 20180401]
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

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

labCrop = ['Corn', 'Spring wheat', 'Winter wheat']
indCrop = [0, 22, 23]
cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
key = 'RMSE'
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(1, 3, figsize=[12, 5])
for k in range(3):
    grid, uy, ux = utils.grid.array2grid(
        cropRate[:, indCrop[k]], lat=lat, lon=lon)
    plot.plotMap(
        grid, ax=axes[k], lat=uy, lon=ux, title=labCrop[k] + ' percentage')
    plt.tight_layout()
    fig.show()

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
indLst = [cropRate[:, 0] > 30, cropRate[:, 22] > 5, cropRate[:, 23] > 10]
labMonth = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Agu', 'Sep', 'Oct',
    'Nov', 'Dec'
]
labCrop = ['Corn', 'Spring wheat', 'Winter wheat']
cLst = 'rgb'
dataBox = list()
for iC in range(len(indLst)):
    dataBox = list()
    for k in range(12):
        data = statPLst[k]['RMSE'][indLst[iC]] - statFLst[k]['RMSE'][
            indLst[iC]]
        if len(data[~np.isnan(data)]) < 20:
            data = []
        dataBox.append(data)
    fig = plot.plot_box_fig(
        dataBox,
        label1=labMonth,
        label2=[labCrop[iC]],
        sharey=True,
        figsize=[8, 3],
        colorLst=cLst[iC])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.ylim(-0.02, 0.04)
    fig.show()
