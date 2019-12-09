from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat

contLst = [
    'Africa',
    'Asia',
    'Australia',
    'Europe',
    'NorthAmerica',
    'SouthAmerica',
]
subsetLst = ['Globalv4f1_' + x for x in contLst]
outLst = [x + '_v4f1_y1' for x in contLst]
caseLst1 = ['Local', 'Global']
caseLst2 = ['Forcing', 'Soilm']
yrLst = [20160401, 20170401]

statLst = list()
for k in range(len(subsetLst)):
    testName = subsetLst[k]
    tempLst = list()
    for case1 in caseLst1:
        for case2 in caseLst2:
            if case1 == 'Local':
                outName = outLst[k] + '_' + case2
            elif case1 == 'Global':
                outName = 'Global_v4f1_y1' + '_' + case2
            out = os.path.join(pathSMAP['Out_L3_Global'], outName)
            df, yp, yt = master.test(
                out, tRange=yrLst, subset=testName, epoch=500)
            temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
            tempLst.append(temp)
    statLst.append(tempLst)

# plot box
keyLst = stat.keyLst
caseLst = list()
for case1 in caseLst1:
    for case2 in caseLst2:
        caseLst.append(case1 + ' ' + case2)

for k in range(len(keyLst)):
    dataBox = list()
    key = keyLst[k]
    for ss in statLst:
        temp = list()
        for s in ss:
            temp.append(s[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox, contLst, caseLst, title=key)
    fig.show()

# fig = plot.plotBoxFig(dataBox, keyLst, caseLst, sharey=False)
# fig.show()
# fig.savefig(os.path.join(saveDir, 'box_forecast'))