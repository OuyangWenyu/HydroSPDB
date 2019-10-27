from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
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
[lat, lon] = df.getGeo()
diff = statP[key] - statF[key]
fig, axes = plt.subplots(1, 2, figsize=[8, 4])
grid, uy, ux = utils.grid.array2grid(diff, lat=lat, lon=lon)
plot.plotMap(grid, ax=axes[0], lat=uy, lon=ux)
grid, uy, ux = utils.grid.array2grid(cropRate[:, 22], lat=lat, lon=lon)
plot.plotMap(grid, ax=axes[1], lat=uy, lon=ux, cRange=[0, 10])
fig.show()

indLst = [cropRate[:, 0] > 20, cropRate[:, 22] > 5, cropRate[:, 23] > 10]
labMonth = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Agu', 'Sep', 'Oct',
    'Nov', 'Dec'
]
labCrop = ['Corn', 'Spring wheat', 'Winter wheat']
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
fig = plot.plotBoxFig(
    dataBox,
    label1=labMonth,
    label2=labCrop,
    sharey=True,
    figsize=[8, 3],
    colorLst='rgb')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.ylim(-0.02, 0.04)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_month'))
