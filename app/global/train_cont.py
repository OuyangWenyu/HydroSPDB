from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv

# train for each cont
contLst = [
    'Africa',
    'Asia',
    'Australia',
    'Europe',
    'NorthAmerica',
    'SouthAmerica',
]
subsetLst = ['Globalv4f1_' + x for x in contLst]
subsetLst.append('Globalv4f1')
outLst = [x + '_v4f1_y1' for x in contLst]
outLst.append('Global_v4f1_y1')
caseLst = ['Forcing', 'Soilm']

cid = 0
for k in range(len(subsetLst)):
    for case in caseLst:
        if case == 'Forcing':
            varLst = dbCsv.varForcingGlobal
        else:
            varLst = dbCsv.varSoilmGlobal

        optData = master.default.update(
            master.default.optDataSMAP,
            rootDB=pathSMAP['DB_L3_Global'],
            subset=subsetLst[k],
            tRange=[20150401, 20160401],
            varT=varLst)
        optModel = master.default.optLstm
        optLoss = master.default.optLossSigma
        optTrain = master.default.optTrainSMAP
        out = os.path.join(pathSMAP['Out_L3_Global'], outLst[k] + '_' + case)

        masterDict = master.wrapMaster(out, optData, optModel, optLoss,
                                       optTrain)
        master.run_train(masterDict, cudaID=cid % 3, screen=outLst[k])
        cid = cid + 1
        # master.train(masterDict)

# some of them failed and rerun
# master.runTrain(
#     r'/mnt/sdb/rnnSMAP/Model_SMAPgrid/L3_Global/Africa_v4f1_y1_Forcing/',
#     cudaID=1,
#     screen='Africa_v4f1_y1_Forcing')
# master.runTrain(
#     r'/mnt/sdb/rnnSMAP/Model_SMAPgrid/L3_Global/Asia_v4f1_y1_Soilm/',
#     cudaID=0,
#     screen='Asia_v4f1_y1_Soilm')
# master.runTrain(
#     r'/mnt/sdb/rnnSMAP/Model_SMAPgrid/L3_Global/NorthAmerica_v4f1_y1_Soilm/',
#     cudaID=1,
#     screen='NorthAmerica_v4f1_y1_Soilm')
# master.runTrain(
#     r'/mnt/sdb/rnnSMAP/Model_SMAPgrid/L3_Global/Global_v4f1_y1_Forcing/',
#     cudaID=2,
#     screen='Global_v4f1_y1_Forcing')
