from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
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
prcp = df.getDataTs('APCP_FORA').squeeze()
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
    plot.plotTS(
        t, [ypLst[indY][ind, :], yfLst[indY][ind, :], obs[ind, :]],
        ax=ax,
        tBar=tBar,
        legLst=legLst1,
        linewidth=1)
    ax.set_xticklabels([])
    ax = fig.add_subplot(gs[k * 3 + 1, 0])
    plot.plotTS(
        t, [prcp[ind, :]],
        ax=ax,
        cLst='c',
        legLst=legLst2,
        tBar=tBar,
        linewidth=1)
fig.show()
fig.savefig(os.path.join(saveDir, 'ts_extreme.eps'))
fig.savefig(os.path.join(saveDir, 'ts_extreme'))

# # test error train on different year
trLst = [[20150402, 20160401], [20160402, 20170401], [20170402, 20180401]]
statPLst = list()
statFLst = list()
for j in range(3):
    tempPLst = list()
    tempFLst = list()
    for i in range(3):
        trTest = trLst[i]
        taTest = utils.time.tRange2Array(trTest)
        taAll = utils.time.tRange2Array([20150402, 20180401])
        ind, ind2 = utils.time.intersect(taAll, taTest)
        tempYp = ypLst[j][:, ind]
        tempYf = yfLst[j][:, ind]
        tempMask = maskF[:, ind]
        tempObs = obs[:, ind]
        tempStatP = stat.statError(
            utils.fillNan(tempYp, tempMask), utils.fillNan(tempObs, tempMask))
        tempStatF = stat.statError(
            utils.fillNan(tempYf, tempMask), utils.fillNan(tempObs, tempMask))
        tempPLst.append(tempStatP)
        tempFLst.append(tempStatF)
    statPLst.append(tempPLst)
    statFLst.append(tempFLst)

# # plot forecast error train on different year
# keyLst = ['RMSE', 'Corr']
# yrStrLst = ['2015', '2016', '2017']
# [lat, lon] = df.getGeo()
# fig, axes = plt.subplots(2, 2, figsize=[8, 4])
# key = 'RMSE'
# for j in range(2):
#     jLst = [0, 2]
#     jj = jLst[j]
#     iLst = [0, 1, 2]
#     iLst.remove(jj)
#     for i in range(2):
#         ii = iLst[i]
#         data = (statPLst[jj][ii][key] -
#                 statFLst[jj][ii][key]) / statPLst[jj][ii][key] * 100
#         titleStr = 'train on {}, test on {}'.format(yrStrLst[jj], yrStrLst[ii])
#         grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
#         plot.plotMap(
#             grid,
#             ax=axes[j][i],
#             lat=uy,
#             lon=ux,
#             title=titleStr,
#             cRange=[0, 60])
# # plt.suptitle('Percentage of RMSE improvement')
# plt.tight_layout()
# fig.show()
# fig.savefig(os.path.join(saveDir, 'map_impovement_multiyear'))

# plot map and time series
# import importlib
# importlib.reload(plot)
dataGrid = [
    statPLst[0]['RMSE'] - statFLst[0]['RMSE'],
    statPLst[1]['RMSE'] - statFLst[1]['RMSE'],
    statPLst[2]['RMSE'] - statFLst[2]['RMSE']
]
prcp = df.getDataTs('APCP_FORA').squeeze()
dataTs = [[obs, ypLst[0], yfLst[0]], [obs, ypLst[1], yfLst[1]],
          [obs, ypLst[2], yfLst[2]], [prcp]]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['d RMSE', 'crop']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    isGrid=True,
    multiTS=True)

# # plot maps forecast_diff
# keyLst = ['RMSE', 'Corr']
# [lat, lon] = df.getGeo()
# fig, axes = plt.subplots(1, len(keyLst), figsize=[8, 3])
# for i in range(len(keyLst)):
#     key = keyLst[i]
#     s1 = statF[key]
#     s2 = statP[key]
#     data = s1 / s2
#     titleStr = key + ' rate'
#     grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
#     plot.plotMap(grid, ax=axes[i], lat=uy, lon=ux, title=titleStr)
# plt.tight_layout()
# fig.show()
# fig.savefig(os.path.join(saveDir, 'map_forecast_diff'))
# for key in stat.keyLst:
#     print(key, np.nanmedian(statF[key]))
#     print(key, np.nanmedian(statP[key]))

# # plot map and time series
# # obsL4 = df.getDataTs('SMAP_L4', doNorm=False, rmNan=False).squeeze()
# dataGrid = [statP['RMSE'] / statF['RMSE'], statP['Corr'] / statF['Corr']]
# dataTs = [obs, yp, yf]
# crd = df.getGeo()
# t = df.getT()
# mapNameLst = ['ratio RMSE', 'ratio Correlation']
# tsNameLst = ['obs', 'prj', 'fore']
# plot.plotTsMap(
#     dataGrid,
#     dataTs,
#     lat=crd[0],
#     lon=crd[1],
#     t=t,
#     mapNameLst=mapNameLst,
#     tsNameLst=tsNameLst,
#     isGrid=True)

# # crop
# cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
# cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
# key = 'RMSE'
# diff = statF[key] / statP[key]
# fig, axes = plt.subplots(1, 2, figsize=[8, 3])
# grid, uy, ux = utils.grid.array2grid(diff, lat=lat, lon=lon)
# plot.plotMap(grid, ax=axes[0], lat=uy, lon=ux)
# grid, uy, ux = utils.grid.array2grid(cropRate[:, 21], lat=lat, lon=lon)
# plot.plotMap(grid, ax=axes[1], lat=uy, lon=ux)
# fig.show()
