import data.data_process
from hydroDL import pathSMAP, master
import utils
from app.common import default
from hydroDL import stat
from visual import plot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

dLst = [1, 2, 3, 5, 15, 30]
doLst = list()
# doLst.append('train')
# doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')

# training
if 'train' in doLst:
    cid = 0
    for nd in dLst:
        optData = default.update(
            default.optDataSMAP,
            rootDB=pathSMAP['DB_L3_NA'],
            subset='CONUSv2f1',
            tRange=[20150501, 20160501],
            daObs=nd)
        optModel = default.optLstmClose
        optLoss = default.optLossRMSE
        optTrain = default.update(default.optTrainSMAP, nEpoch=300)
        out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                           'CONUSv2f1_d' + str(nd))
        masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss,
                                                   optTrain)
        master.run_train(masterDict, cudaID=cid % 3, screen='d' + str(nd))
        # master.train(masterDict)
        cid = cid + 1
    # vanila LSTM
    optData = default.update(
        default.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=[20150501, 20160501])
    optModel = default.optLstm
    optLoss = default.optLossRMSE
    optTrain = default.update(default.optTrainSMAP, nEpoch=300)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1')
    masterDict = data.data_process.wrap_master(out, optData, optModel, optLoss, optTrain)
    master.run_train(masterDict, cudaID=0, screen='LSTM')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    outLst = [
        os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_d' + str(nd))
        for nd in dLst
    ]
    subset = 'CONUSv2f1'
    tRange = [20160501, 20170501]
    predLst = list()
    outLSTM = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1')
    df, pred, obs = master.test(
        outLSTM, tRange=tRange, subset=subset, batchSize=100)
    predLst.append(pred.squeeze())
    for out in outLst:
        df, pred, obs = master.test(
            out, tRange=tRange, subset=subset, batchSize=100)
        predLst.append(pred.squeeze())
    obs = obs.squeeze()

# plot box - latency
# if 'post' in doLst:
caseLst = ['Predict'] + ['Nowcast ' + str(nd) + 'd latency' for nd in dLst]
statLst1 = [stat.statError(x, obs) for x in predLst]
keyLst = list(statLst1[0].keys())
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statLst1)):
        data = statLst1[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
fig = plot.plot_box_fig(dataBox, keyLst, caseLst, sharey=False)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_latency'))

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
unique, counts = np.unique(maskObsDay, return_counts=True)
print(np.asarray((unique, counts)).T)
print(counts / ngrid / nt)

fLst = [1, 2, 3]
statLst2 = list()
for pred in predLst:
    tempLst = list()
    for nf in fLst:
        x = np.full([ngrid, nt], np.nan)
        y = np.full([ngrid, nt], np.nan)
        x[maskObsDay == nf] = pred[maskObsDay == nf]
        y[maskObsDay == nf] = obs[maskObsDay == nf]
        tempLst.append(stat.statError(x, y))
    statLst2.append(tempLst)

# plot box - forecast
keyLst = list(statLst2[0][0].keys())
caseLst = ['Predict'] + [str(nd) + 'd Forcast' for nd in dLst]
dataBox = list()
for k in range(len(keyLst)):
    key = keyLst[k]
    temp = list()
    data = statLst1[0][key]
    temp.append(data)
    print(key, np.nanmean(data))
    for i in range(len(fLst)):
        data = statLst2[1][i][key]
        temp.append(data)
        print(key, np.nanmean(data))
    dataBox.append(temp)
fig = plot.plot_box_fig(dataBox, keyLst, caseLst, sharey=False)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_forecast'))

# plot line
fig, axes = plt.subplots(1, 4, figsize=[8, 6])
colorLst = 'rbkgcmy'
keyLst = list(statLst2[0][0].keys())
for k in range(len(keyLst)):
    key = keyLst[k]
    for j in range(len(dLst)):
        temp = list()
        for i in range(len(fLst)):
            temp.append(np.nanmean(statLst2[j][i][key]))
        axes[k].plot(fLst, temp, colorLst[j], label=caseLst[j])
    axes[k].legend(loc='best')
    axes[k].set_title(key)
    axes[k].set_xlabel('forecast day')
fig.show()

# plot maps
key = 'RMSE'
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(3, len(fLst), figsize=[8, 6])
for j in range(3):
    for i in range(len(fLst)):
        data = statLst2[j][i][key]
        grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid,
            ax=axes[i][j],
            lat=uy,
            lon=ux,
            title='latency-{} forcast-{} '.format(dLst[j], fLst[i]),
            cRange=[0, 0.05])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_all'))