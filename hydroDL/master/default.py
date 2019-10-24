import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv, camels

# Gages2 default options
optDataGages2 = OrderedDict(
    name='hydroDL.data.dbCsv.DataframeCsv',
    rootDB=hydroDL.pathSMAP['DB_L3_Global'],
    subset='CONUSv4f1',
    varT=dbCsv.varForcing,
    varC=dbCsv.varConst,
    target=['SMAP_AM'],
    tRange=[20150401, 20160401],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0)
optTrainGages2 = OrderedDict(miniBatch=[100, 30], nEpoch=500, saveEpoch=100)
# Streamflow default options
optDataCamels = OrderedDict(
    name='hydroDL.data.camels.DataframeCamels',
    subset='All',
    varT=camels.forcingLst,
    varC=camels.attrLstSel,
    tRange=[19900101, 19950101],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0)
optTrainCamels = OrderedDict(miniBatch=[100, 200], nEpoch=1, saveEpoch=1)
""" model options """
optLstm = OrderedDict(
    name='hydroDL.model.rnn.CudnnLstmModel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),
    ny=1,
    hiddenSize=256,
    doReLU=True)
optLstmClose = OrderedDict(
    name='hydroDL.model.rnn.LstmCloseModel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),
    ny=1,
    hiddenSize=256,
    doReLU=True)
optLossRMSE = OrderedDict(name='hydroDL.model.crit.RmseLoss', prior='gauss')
optLossSigma = OrderedDict(name='hydroDL.model.crit.SigmaLoss', prior='gauss')


def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt
