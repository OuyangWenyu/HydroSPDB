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
import pandas as pd

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')

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
statP = stat.statError(utils.dataset_format.fillNan(yp, maskF), utils.dataset_format.fillNan(obs, maskF))
statF = stat.statError(utils.dataset_format.fillNan(yf, maskF), utils.dataset_format.fillNan(obs, maskF))

maskObsDay = maskObs * maskDay
print(np.array([maskObs[ind, :], maskDay[ind, :]]))
print(np.asarray((unique, counts)).T)
print(counts / ngrid / nt)

# see result for different seasons
tRangeLst = [[20160401, 20160701], [20160701, 20161001], [20161001, 20170101],
             [20170101, 20170401], [20170401, 20170701], [20170701, 20171001],
             [20171001, 20180101], [20180101, 20180401]]

tAllA = utils.hydro_time.t_range2_array(tAllR)
statPLst = list()
statFLst = list()
for k in range(4):
    tRLst = [tRangeLst[k], tRangeLst[k + 4]]
    temp = list()
    for tR in tRLst:
        tA = utils.hydro_time.t_range2_array(tR)
        ind0 = np.array(range(nt))
        ind1, ind2 = utils.hydro_time.intersect(tAllA, tA)
        temp.append(ind1)
    indT = np.concatenate(temp)
    yfTemp = utils.dataset_format.fillNan(yf, maskF)[:, indT]
    ypTemp = utils.dataset_format.fillNan(yp, maskF)[:, indT]
    obsTemp = utils.dataset_format.fillNan(obs, maskF)[:, indT]
    statPLst.append(stat.statError(ypTemp, obsTemp))
    statFLst.append(stat.statError(yfTemp, obsTemp))

# for k in range(len(tRangeLst)):
#     tR = tRangeLst[k]
#     tA = utils.time.tRange2Array(tR)
#     ind0 = np.array(range(nt))
#     ind1, ind2 = utils.time.intersect(tAllA, tA)
#     yfTemp = utils.fillNan(yf, maskF)[:, ind1]
#     ypTemp = utils.fillNan(yp, maskF)[:, ind1]
#     obsTemp = utils.fillNan(obs, maskF)[:, ind1]
#     statPLst.append(stat.statError(ypTemp, obsTemp))
#     statFLst.append(stat.statError(yfTemp, obsTemp))

# plot maps forecast_diff
tLeg = ['0401-0701', '0701-1001', '1001-0101', '0101-0401']
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(2, 2, figsize=[8, 5])
key = 'RMSE'
for k in range(4):
    s1 = statFLst[k][key]
    s2 = statPLst[k][key]
    data = s2 - s1
    cRange = [0, 0.03]
    keyLeg = 'RMSE(Proj)-RMSE(Fore)'
    titleStr = tLeg[k] + ' ' + keyLeg
    j, i = utils.dataset_format.index2d(k, 2, 2)
    grid, uy, ux = utils.geo.array2grid(data, lat=lat, lon=lon)
    plot.plotMap(
        grid, ax=axes[j][i], lat=uy, lon=ux, title=titleStr, cRange=cRange)
# plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_diff_RMSE_month'))

# plot map time series for each season / month
tM = [301, 501]
tRLst = [[20160000 + tM[0], 20160000 + tM[1]],
         [20170000 + tM[0], 20170000 + tM[1]]]
temp = list()
for tR in tRLst:
    tA = utils.hydro_time.t_range2_array(tR)
    ind0 = np.array(range(nt))
    ind1, ind2 = utils.hydro_time.intersect(tAllA, tA)
    temp.append(ind1)
indT = np.concatenate(temp)
yfTemp = utils.dataset_format.fillNan(yf, maskF)[:, indT]
ypTemp = utils.dataset_format.fillNan(yp, maskF)[:, indT]
obsTemp = utils.dataset_format.fillNan(obs, maskF)[:, indT]
statPtemp = stat.statError(ypTemp, obsTemp)
statFtemp = stat.statError(yfTemp, obsTemp)
dataGrid = [
    statPtemp['RMSE'] / statFtemp['RMSE'],
    statPtemp['Corr'] / statFtemp['Corr']
]
prcp = df.get_data_ts('APCP_FORA')
dataTs = [obs, yp, yf]
dataTs2 = [prcp]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['ratio RMSE', 'ratio Correlation']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    dataTs2=dataTs2,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    isGrid=True)

# plot map and time series
# obsL4 = df.getDataTs('SMAP_L4', doNorm=False, rmNan=False).squeeze()
dataGrid = [
    statPLst[0]['RMSE'] - statFLst[0]['RMSE'],
    # statPLst[1]['RMSE'] / statFLst[1]['RMSE'],
    statPLst[2]['RMSE'] - statFLst[2]['RMSE'],
    # statPLst[3]['RMSE'] / statFLst[3]['RMSE'],
]
prcp = df.get_data_ts('APCP_FORA')
dataTs = [obs, yp, yf]
dataTs2 = [prcp]
crd = df.getGeo()
t = df.getT()
mapNameLst = ['diff ubRMSE', 'ratio Correlation']
tsNameLst = ['obs', 'prj', 'fore']
plot.plotTsMap(
    dataGrid,
    dataTs,
    dataTs2=prcp,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    tsNameLst2=['prcp'],
    isGrid=True)

## box plot of factors
cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
key = 'RMSE'
diff = statP[key] - statF[key]
fig, axes = plt.subplots(1, 2, figsize=[8, 4])
grid, uy, ux = utils.geo.array2grid(cropRate[:, 4], lat=lat, lon=lon)
plot.plotMap(grid, ax=axes[0], lat=uy, lon=ux, title='Percentage of soybean')
grid, uy, ux = utils.geo.array2grid(cropRate[:, 22], lat=lat, lon=lon)
plot.plotMap(
    grid, ax=axes[1], lat=uy, lon=ux, title='Percentage of spring wheat')
fig.show()

indLst = [
    cropRate[:, 0] > 2, cropRate[:, 4] > 2, cropRate[:, 22] > 2,
    cropRate[:, 23] > 2
]
dataBox = list()
for iC in range(len(indLst)):
    tempLst = list()
    for k in range(4):
        data = statPLst[k]['RMSE'][indLst[iC]] - statFLst[k]['RMSE'][
            indLst[iC]]
        tempLst.append(data)
        print(key, k, np.nanmedian(data))
    dataBox.append(tempLst)
fig = plot.plot_box_fig(dataBox, sharey=True, figsize=[8, 3])
plt.tight_layout()
fig.show()

# plot time series
indLst = [1023]
strLst = ['east Texas']
tBar = [utils.hydro_time.t2dt(20160401)]
t = df.getT()
[lat, lon] = df.getGeo()
prcp = df.get_data_ts('APCP_FORA')
tsNameLst = ['obs', 'prj', 'fore']
for k in range(len(indLst)):
    fig, axes = plt.subplots(2, 1, figsize=[16, 6])
    ind = indLst[k]
    titleStr = 'typical {} pixel, lat {:.3}, lon {:.3}'.format(
        strLst[k], lat[ind], lon[ind])
    tsLst = [obs[ind, :], yp[ind, :], yf[ind, :]]
    tsLst2 = [prcp[ind, :]]
    plot.plot_ts(t, tsLst, ax=axes[1], legLst=tsNameLst, cLst='krb', tBar=tBar)
    plot.plot_ts(
        t,
        tsLst2,
        ax=axes[0],
        legLst=['prcp'],
        title=titleStr,
        cLst='c',
        tBar=tBar)
    axes[0].set_xticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()

# maps for argriculture
trLst = [[20160801, 20161101], [20170801, 20171101]]
trLab = '1001-1201'
cropLab='winter wheat'
temp = list()
for tR in trLst:
    tA = utils.hydro_time.t_range2_array(tR)
    ind0 = np.array(range(nt))
    ind1, ind2 = utils.hydro_time.intersect(tAllA, tA)
    temp.append(ind1)
indT = np.concatenate(temp)
yfTemp = utils.dataset_format.fillNan(yf, maskF)[:, indT]
ypTemp = utils.dataset_format.fillNan(yp, maskF)[:, indT]
obsTemp = utils.dataset_format.fillNan(obs, maskF)[:, indT]
statPTemp = stat.statError(ypTemp, obsTemp)
statFTemp = stat.statError(yfTemp, obsTemp)
cropFile = r'/mnt/sdb/Data/Crop/cropRate_CONUSv2f1.csv'
cropRate = pd.read_csv(cropFile, dtype=np.float, header=None).values
key = 'Corr'
diff = statPTemp[key] - statFTemp[key]
fig, axes = plt.subplots(1, 2, figsize=[8, 4])
grid, uy, ux = utils.geo.array2grid(diff, lat=lat, lon=lon)
plot.plotMap(
    grid,
    ax=axes[0],
    lat=uy,
    lon=ux,
    title='RMSE diff ' + trLab)
grid, uy, ux = utils.geo.array2grid(cropRate[:, 23], lat=lat, lon=lon)
plot.plotMap(
    grid,
    ax=axes[1],
    lat=uy,
    lon=ux,
    title='Percentage of ' + cropLab)
fig.show()

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 16})
