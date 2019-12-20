from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.gridspec as gridspec
import matplotlib

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
        df, yf, obs = master.master_test(
            out, tRange=tRange, subset=subset, batchSize=100)
        out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                           'CONUSv2f1_LSTM' + yrStr)
        df, yp, obs = master.master_test(out, tRange=tRange, subset=subset)
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

# # test error train on different year
trLst = [[20150402, 20160401], [20160401, 20170401], [20170401, 20180401]]
statPLst = list()
statFLst = list()
for k in range(3):
    trTrain = trLst[k]
    taTrain = utils.time.tRange2Array(trTrain)
    taAll = utils.time.tRange2Array([20150402, 20180401])
    indTrain, ind2 = utils.time.intersect(taAll, taTrain)
    indTest = np.delete(np.arange(len(taAll)), indTrain)
    tempYp = ypLst[k][:, indTest]
    tempYf = yfLst[k][:, indTest]
    tempMask = maskF[:, indTest]
    tempObs = obs[:, indTest]
    tempStatP = stat.statError(
        utils.fillNan(tempYp, tempMask), utils.fillNan(tempObs, tempMask))
    tempStatF = stat.statError(
        utils.fillNan(tempYf, tempMask), utils.fillNan(tempObs, tempMask))
    statPLst.append(tempStatP)
    statFLst.append(tempStatF)

# plot map and time series
import importlib
importlib.reload(plot)
dataGrid = [
    statPLst[0]['RMSE'] - statFLst[0]['RMSE'],
    statPLst[1]['RMSE'] - statFLst[1]['RMSE'],
    statPLst[2]['RMSE'] - statFLst[2]['RMSE']
]
prcp = df.get_data_ts('APCP_FORA').squeeze()
dataTs = [[obs, ypLst[0], yfLst[0]], [obs, ypLst[1], yfLst[1]],
          [obs, ypLst[2], yfLst[2]], [prcp]]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['dRMSE 2015', 'dRMSE 2016', 'dRMSE 2017']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    tBar=[utils.time.t2dt(20160401),
          utils.time.t2dt(20170401)],
    mapNameLst=mapNameLst,
    isGrid=True,
    multiTS=True)

# plot pixel time series
import importlib
importlib.reload(plot)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 6})
matplotlib.rcParams.update({'legend.fontsize': 12})
indLst = [1442, 1023]
indYLst = [0, 2]
nts = len(indLst)
hrLst = list()
for i in range(nts):
    hrLst = hrLst + [1, 0.7, 0.2]
del hrLst[-1]
gs = gridspec.GridSpec(nts * 3 - 1, 1, width_ratios=[1], height_ratios=hrLst)
fig = plt.figure(figsize=[12, 6])
plt.subplots_adjust(hspace=0)
plt.subplots_adjust(vspace=0)
t = df.getT()
prcp = df.get_data_ts('APCP_FORA').squeeze()
tBarLst = [20160401, [20160401, 20170401], 20170401]
for k in range(nts):
    ind = indLst[k]
    indY = indYLst[k]
    ax = fig.add_subplot(gs[k * 3, 0])
    tBar = utils.time.t2dt(tBarLst[indY])
    if k == 0:
        legLst1 = ['project', 'forecast', 'SMAP']
        legLst2 = ['prcp']
    else:
        legLst1 = None
        legLst2 = None
    plot.plot_ts(
        t, [ypLst[indY][ind, :], yfLst[indY][ind, :], obs[ind, :]],
        ax=ax,
        tBar=tBar,
        legLst=legLst1,
        linewidth=1)
    ax.set_xticklabels([])
    ax = fig.add_subplot(gs[k * 3 + 1, 0])
    plot.plot_ts(
        t, [prcp[ind, :]],
        ax=ax,
        cLst='c',
        legLst=legLst2,
        tBar=tBar,
        linewidth=1)
fig.show()
fig.savefig(os.path.join(saveDir, 'ts_extreme.eps'))
fig.savefig(os.path.join(saveDir, 'ts_extreme'))
