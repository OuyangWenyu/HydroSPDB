import utils.dataset_format
from hydroDL import pathSMAP, master
import utils
from hydroDL import stat
import os
import numpy as np
import torch

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA')
yrLst = ['2015', '2016', '2017']
tRangeLst = [[20150402, 20160401], [20160402, 20170401], [20170402, 20180401]]

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
    subset = 'CONUSv2f1'
    tRange = [20150402, 20180401]
    yfLst = list()
    for yr in yrLst:
        out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA' + yr)
        df, yf, obs = master.master_test(
            out, tRange=tRange, subset=subset, batchSize=100)
        yfLst.append(yf)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM')
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
maskF = (maskDay >= 1) & (maskDay <= 3)

# figure out train and test time index
tR0 = [20150402, 20180401]
tA0 = utils.hydro_time.t_range2_array(tR0)
nt = len(tA0)
tTrainLst = list()
tTestLst = list()
for k in range(len(yrLst)):
    tR = tRangeLst[k]
    tA = utils.hydro_time.t_range2_array(tR)
    ind0 = np.array(range(nt))
    ind1, ind2 = utils.hydro_time.intersect(tA0, tA)
    tTestLst.append(np.delete(ind0, ind1))
    tTrainLst.append(ind1)

# calculate stat
for k in range(len(yrLst)):
    yfTemp = utils.dataset_format.fillNan(yfLst[k], maskF)
    yfTemp = yfTemp[:, tTestLst[k]]
    statP = stat.statError(yfTemp, utils.dataset_format.fillNan(obs, maskF))
    statF = stat.statError(yfTemp, utils.dataset_format.fillNan(obs, maskF))
