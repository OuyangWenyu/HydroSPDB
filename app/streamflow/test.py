"""利用GAGES-II数据训练LSTM，进行流域径流模拟，先针对671个CAMELS的reference sites进行训练。
直接使用GAGES-II自己的attributes；
forcing数据利用maurur的数据源通过matlab code进行basin average计算。然后就可以组成输入集；
输出集直接使用GAGES-II的即可。
先和dapeng用CAMELS计算的结果比较，看看结果是否合理，如果合理，直接使用GAGES-II的数据继续接下来的人类活动影响的研究即可。"""

# 首先，配置GAGES-II原始数据文件的路径。在hydroDL的init文件里直接配置输入输出路径
import os

from hydroDL import pathGages2
# 接下来是初始的默认模型配置，这部分直接在master文件夹下的default模块中配置
from hydroDL import master
# 接下来配置计算条件
from hydroDL.master import default

# 多GPU的话，可以并行计算
cid = 0
# train default model
out = os.path.join(pathGages2['Out'], 'All-90-95')
optData = default.optDataGages2
optModel = default.optLstm
optLoss = default.optLossRMSE
optTrain = default.optTrainGages2
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.runTrain(masterDict, cudaID=cid % 3, screen='test')
cid = cid + 1

# test
caseLst = ['All-90-95']
nDayLst = [1, 7]
outLst = [os.path.join(pathGages2['Out'], x) for x in caseLst]
subset = 'All'
tRange = [19950101, 20000101]
predLst = list()
for out in outLst:
    df, pred, obs = master.test(out, tRange=tRange, subset=subset)
    predLst.append(pred)
