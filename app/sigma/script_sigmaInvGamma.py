
import refine
import torch
import imp
from refine import runTrainLSTM
imp.reload(refine)
refine.reload()

opt = refine.classLSTM.optLSTM(
    rootDB=refine.kPath['DB_L3_NA'],
    rootOut=refine.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, loss='sigma',lossPrior='gauss'
)

trainName='CONUSv4f1'
opt['train'] = trainName
opt['out'] = trainName+'_test'
# runTrainLSTM.runCmdLine(opt=opt, cudaID=2, screenName=opt['out'])
refine.funLSTM.trainLSTM(opt)

testName='CONUSv4f1'