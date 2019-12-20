import data.read_config
import data.data_input
from hydroDL import pathSMAP, master
import os
from data import dbCsv

# train for each cont
subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst = ['Forcing', 'Soilm']

# train a CONUS model
cid = 1
for case in caseLst:
    if case == 'Forcing':
        varLst = dbCsv.varForcing
    else:
        varLst = dbCsv.varSoilM
    optData = data.read_config.update(
        data.read_config.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=[20150401, 20160401],
        varT=varLst)
    optModel = data.read_config.optLstm
    optLoss = data.read_config.optLossRMSE
    optTrain = data.read_config.optTrainSMAP
    out = os.path.join(pathSMAP['Out_L3_NA'], 'CONUSv2f1_' + case)
    masterDict = data.read_config.wrap_master(out, optData, optModel, optLoss, optTrain)
    master.run_train(masterDict, cudaID=cid % 3, screen=case)
    cid = cid + 1

# train for each region
cid = 0
for k in range(len(subsetLst)):
    for case in caseLst:
        if case == 'Forcing':
            varLst = dbCsv.varForcing
        else:
            varLst = dbCsv.varSoilM
        optData = data.read_config.update(
            data.read_config.optDataSMAP,
            rootDB=pathSMAP['DB_L3_NA'],
            subset=subsetLst[k],
            tRange=[20150401, 20160401],
            varT=varLst)
        optModel = data.read_config.optLstm
        optLoss = data.read_config.optLossRMSE
        optTrain = data.read_config.optTrainSMAP
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                           subsetLst[k] + '_' + case)
        masterDict = data.read_config.wrap_master(out, optData, optModel, optLoss,
                                                  optTrain)
        # master.runTrain(masterDict, cudaID=cid % 3, screen=subsetLst[k])
        cid = cid + 1
        # master.train(masterDict)

# retrain some models
rtEcoLst = [1, 2, 5, 7, 8, 10, 13, 16, 17]
rtCaseLst = [
    'Forcing', 'SoilM', 'SoilM', 'Forcing', 'SoilM', 'Forcing', 'Forcing',
    'Forcing', 'SoilM'
]
cid = 0
for kk in range(len(rtEcoLst)):
    k = rtEcoLst[kk] - 1
    case = rtCaseLst[kk]
    optData = data.read_config.update(
        data.read_config.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset=subsetLst[k],
        tRange=[20150401, 20160401],
        varT=varLst)
    optModel = data.read_config.optLstm
    optLoss = data.read_config.optLossRMSE
    optTrain = data.read_config.optTrainSMAP
    out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                       subsetLst[k] + '_' + case)
    masterDict = data.read_config.wrap_master(out, optData, optModel, optLoss, optTrain)
    master.run_train(masterDict, cudaID=cid % 3, screen=subsetLst[k])
    cid = cid + 1
    # master.train(masterDict)

# test
ss = ''
for k in range(len(subsetLst)):
    for case in caseLst:
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                           subsetLst[k] + '_' + case)
        try:
            master.master_test(out, tRange=[20160401, 20180401], subset=subsetLst[k])
        except:
            ss = ss + 'ecoRegion ' + str(k) + ' case ' + case + '; '


# retrain some models
# rtEcoLst = [1, 2, 5, 7, 8, 10, 13, 16, 17]
# rtCaseLst = [
#     'Forcing', 'Soilm', 'Soilm', 'Forcing', 'Soilm', 'Forcing', 'Forcing',
#     'Forcing', 'Soilm'
# ]
# rtEcoLst = [1, 4, 7, 16]
# rtCaseLst = ['Soilm', 'Soilm', 'Soilm', 'Soilm']
# cid = 2
# for kk in range(len(rtEcoLst)):
#     k = rtEcoLst[kk] - 1
#     case = rtCaseLst[kk]
#     if case == 'Forcing':
#         varLst = dbCsv.varForcing
#     else:
#         varLst = dbCsv.varSoilM
#     optData = master.default.update(
#         master.default.optDataSMAP,
#         rootDB=pathSMAP['DB_L3_NA'],
#         subset=subsetLst[k],
#         tRange=[20150401, 20160401],
#         varT=varLst)
#     optModel = master.default.optLstm
#     optLoss = master.default.optLossRMSE
#     optTrain = master.default.optTrainSMAP
#     out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
#                        subsetLst[k] + '_' + case)
#     masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
#     master.runTrain(masterDict, cudaID=cid % 3, screen=subsetLst[k])
#     cid = cid + 1
#     # master.train(masterDict)