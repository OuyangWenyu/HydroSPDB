import data.data_input
import data.data_config
from hydroDL import pathSMAP, master
from data import data_config
import os

# training
tLst = [[20150402, 20160401], [20160402, 20170401], [20170402, 20180401]]
yrLst = ['2015', '2016', '2017']
for k in range(len(tLst)):
    # optData = default.update(
    #     default.optDataSMAP,
    #     rootDB=pathSMAP['DB_L3_NA'],
    #     subset='CONUSv2f1',
    #     tRange=tLst[k],
    #     daObs=1)
    # optModel = default.optLstmClose
    # optLoss = default.optLossRMSE
    # optTrain = default.update(default.optTrainSMAP, nEpoch=300)
    # out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA' + yrLst[k])
    # masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    # master.runTrain(masterDict, cudaID=k % 3, screen='DA' + yrLst[k])

    # optData = default.update(
    #     default.optDataSMAP,
    #     rootDB=pathSMAP['DB_L3_NA'],
    #     subset='CONUSv2f1',
    #     tRange=tLst[k])
    # optModel = default.optLstm
    # optLoss = default.optLossRMSE
    # optTrain = default.update(default.optTrainSMAP, nEpoch=300)
    # out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM'+yrLst[k])
    # masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    # master.runTrain(masterDict, cudaID=k % 3, screen='LSTM' + yrLst[k])

    # k=0
    optData = data_config.update(
        data_config.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=tLst[k])
    optModel = data_config.update(
        data_config.optLstmClose, name='hydroDL.model.rnn.AnnModel')
    optLoss = data_config.optLossRMSE
    optTrain = data_config.update(data_config.optTrainSMAP, nEpoch=300)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                        'CONUSv2f1_NN' + yrLst[k])
    masterDict = data.data_config.wrap_master(out, optData, optModel, optLoss, optTrain)
    master.run_train(masterDict, cudaID=k % 3, screen='NN' + yrLst[k])
    # master.train(masterDict)

    optData = data_config.update(
        data_config.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        subset='CONUSv2f1',
        tRange=tLst[k],
        daObs=1)
    optModel = data_config.update(
        data_config.optLstmClose, name='hydroDL.model.rnn.AnnCloseModel')
    optLoss = data_config.optLossRMSE
    optTrain = data_config.update(data_config.optTrainSMAP, nEpoch=300)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA',
                       'CONUSv2f1_DANN' + yrLst[k])
    masterDict = data.data_config.wrap_master(out, optData, optModel, optLoss, optTrain)
    master.run_train(masterDict, cudaID=k % 3, screen='DANN' + yrLst[k])
    # master.train(masterDict)
