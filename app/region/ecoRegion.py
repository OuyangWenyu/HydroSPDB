from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat

# train for each cont
subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst1 = ['Local', 'COUNS']
caseLst2 = ['Forcing', 'Soilm']

# test and calculate stat
statLst = list()
tRange = [20160401, 20180401]
for k in range(len(subsetLst)):
    testName = subsetLst[k]
    tempLst = list()
    for case1 in caseLst1:
        for case2 in caseLst2:
            if case1 == 'Local':
                out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                                   subsetLst[k] + '_' + case2)
            elif case1 == 'CONUS':
                out = os.path.join(pathSMAP['Out_L3_NA'], 'CONUSv2f1_' + case2)
            df, yp, yt = master.test(out, tRange=tRange, subset=testName)
            temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
            tempLst.append(temp)
    statLst.append(tempLst)
