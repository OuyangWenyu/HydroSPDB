import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from rnnSMAP import runTestLSTM
import shapefile
import time
import imp
import math
import torch
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# train on one HUC and test on CONUS. look at map of sigma

doOpt = []
# doOpt.append('test')
doOpt.append('loadData')
# doOpt.append('plotMapMC')
# doOpt.append('plotMapPaper')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'regionalization')
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['Bias', 'ubRMSE']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
yrLst = [2016, 2017]

hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS'
shapeLst = shapefile.Reader(hucShapeFile).shapes()
shapeHucLst = shapefile.Reader(hucShapeFile).records()

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})

#################################################
# test
if 'test' in doOpt:
    for k in range(9, 10):
        trainName = 'hucn1_' + str(k + 1).zfill(2) + '_v2f1'
        testName = trainName
        out = 'CONUSv2f1_y15_soilM'
        runTestLSTM.runCmdLine(
            rootDB=rootDB,
            rootOut=rootOut,
            out=out,
            testName=testName,
            yrLst=yrLst,
            cudaID=2,
            screenName=out)
        # if k % 3 == 2:
        # time.sleep(1000)

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLstAll = list()
    statSigmaLstAll = list()
    for k in range(18):
        trainName = 'hucn1_' + str(k + 1).zfill(2) + '_v2f1'
        testName = trainName
        outLst = [
            trainName + '_y15_Forcing', trainName + '_y15_soilM',
            'CONUSv2f1_y15_Forcing', 'CONUSv2f1_y15_soilM'
        ]
        statErrLst = list()
        statSigmaLst = list()
        for out in outLst:
            ds = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=yrLst)
            ds.readData(var='SMAP_AM', field='SMAP')
            ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
            statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
            statSigma = ds.statCalSigma(field='LSTM')
            statErrLst.append(statErr)
            statSigmaLst.append(statSigma)
        dsLst.append(ds)
        statErrLstAll.append(statErrLst)
        statSigmaLstAll.append(statSigmaLst)

### boxplot errors
hucStrLst = ['huc' + str(x + 1) for x in range(18)]
strErrLst = ['ubRMSE', 'RMSE', 'Bias', 'rho']
labelS = ['local-forcing', 'local-model', 'CONUS-forcing', 'CONUS-model']
for strErr in strErrLst:
    dataBox = list()
    for k in range(18):
        temp = list()
        for kk in range(4):
            temp.append(getattr(statErrLstAll[k][kk], strErr))
        dataBox.append(temp)
    fig = rnnSMAP.funPost.plotBox(
        dataBox,
        labelC=hucStrLst,
        figsize=(16, 6),
        colorLst='bgrk',
        labelS=labelS,
        title=strErr + ' of CONUS vs local')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout
    fig.show()
    saveFile = os.path.join(saveFolder, 'HUC_box_' + strErr)
    fig.savefig(saveFile, dpi=300)

### boxplot sigma
strSigmaLst = ['sigmaMC', 'sigmaX']
labelS = ['local-forcing', 'local-model', 'CONUS-forcing', 'CONUS-model']
for strSigma in strSigmaLst:
    dataBox = list()
    for k in range(18):
        temp = list()
        for kk in range(4):
            temp.append(getattr(statSigmaLstAll[k][kk], strSigma))
        dataBox.append(temp)
    fig = rnnSMAP.funPost.plotBox(
        dataBox,
        labelC=hucStrLst,
        figsize=(16, 6),
        colorLst='bgrk',
        labelS=labelS,
        title=strSigma + ' of CONUS vs local')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout
    fig.show()
    saveFile = os.path.join(saveFolder, 'HUC_box_' + strSigma)
    fig.savefig(saveFile, dpi=300)

## plot maps
crd = np.concatenate([dsLst[x].crd for x in range(18)])
statDiff = dict()
for strSigma in strSigmaLst:
    a = np.concatenate(
        [getattr(statSigmaLstAll[x][0], strSigma) for x in range(18)])
    b = np.concatenate(
        [getattr(statSigmaLstAll[x][2], strSigma) for x in range(18)])
    statDiff[strSigma] = a - b
for strErr in strErrLst:
    a = np.concatenate(
        [getattr(statErrLstAll[x][0], strErr) for x in range(18)])
    b = np.concatenate(
        [getattr(statErrLstAll[x][2], strErr) for x in range(18)])
    statDiff[strErr] = a - b

hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS'
shapeLst = shapefile.Reader(hucShapeFile).shapes()
(gridY, gridX, indY, indX) = rnnSMAP.funDB.crd2grid(crd[:, 0], crd[:, 1])
strLst = strSigmaLst + strErrLst
for s in strLst:
    grid = np.full([len(gridY), len(gridX)], np.nan)
    grid[indY, indX] = statDiff[s]
    titleStr = s + '(local) - ' + s + '(CONUS)'
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    rnnSMAP.funPost.plotMap(
        grid, crd=(gridY, gridX), title=titleStr, ax=ax, shape=shapeLst)
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'map_diff_' + s)
    fig.savefig(saveFile)
