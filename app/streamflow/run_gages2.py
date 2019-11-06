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
from hydroDL.master.master import namePred
# 导入统计计算和绘图的包
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
import numpy as np
from hydroDL import utils

cid = 0
# train config
out = os.path.join(pathGages2['Out'], 'All-90-95')
optData = default.optDataGages2
optModel = default.optLstm
optLoss = default.optLossRMSE
optTrain = default.optTrainGages2
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# test config
caseLst = ['All-90-95']
outLst = [os.path.join(pathGages2['Out'], x) for x in caseLst]
subset = 'All'
tRangeTest = [19950101, 20000101]
predLst = list()

# train model
# see whether there are previous results or not, if yes, there is no need to train again.
resultPathLst = namePred(out, tRangeTest, subset, epoch=optTrain['nEpoch'])
if not os.path.exists(resultPathLst[0]):
    master.run_train(masterDict, cuda_id=cid % 3, screen='test')
    cid = cid + 1

# test
for out in outLst:
    df, pred, obs = master.test(out, t_range=tRangeTest, subset=subset, epoch=optTrain['nEpoch'])
    predLst.append(pred)

# 补充一些计算结果输出，给出定量的指标，包括nse，rmse。
# 30% bottom 数据的分析，每个站点的<30分位数的径流数据和同时间的预测径流数据比较，计算box的几个指标
obs2d = obs.squeeze()
pred2d = pred.squeeze()
obs2d[np.where(np.isnan(obs2d))] = 0
pred2d[np.where(np.isnan(pred2d))] = 0

# 每个站点的分位数对应的数据量是不同的，因此不能直接调用二维的计算公式，得每个站点分别计算
obsBotIndex = [np.where(obs2di < np.percentile(obs2di, 30)) for obs2di in obs2d]
obsBot = np.array([obs2d[i][obsBotIndex[i]] for i in range(obs2d.shape[0])])
predBot = np.array([pred2d[j][obsBotIndex[j]] for j in range(pred2d.shape[0])])

statDictBot = np.array([stat.statError1d(predBot[i], obsBot[i]) for i in range(obsBot.shape[0])]).T
statDictBot[np.where(np.isnan(statDictBot))] = 0
keysBot = ["Bias", "RMSE", "NSE"]
stateBotPercentile = {keysBot[i]: np.percentile(statDictBot[i], [0, 25, 50, 75, 100]) for i in
                      range(statDictBot.shape[0])}
print(stateBotPercentile)

# top 10%的数据分析，同理
obs2d = obs.squeeze()
pred2d = pred.squeeze()
obs2d[np.where(np.isnan(obs2d))] = 0
pred2d[np.where(np.isnan(pred2d))] = 0

# 每个站点的分位数对应的数据量是不同的，因此不能直接调用二维的计算公式，得每个站点分别计算
obsBotIndex = [np.where(obs2di > np.percentile(obs2di, 90)) for obs2di in obs2d]
obsBot = np.array([obs2d[i][obsBotIndex[i]] for i in range(obs2d.shape[0])])
predBot = np.array([pred2d[j][obsBotIndex[j]] for j in range(pred2d.shape[0])])

statDictBot = np.array([stat.statError1d(predBot[i], obsBot[i]) for i in range(obsBot.shape[0])]).T
statDictBot[np.where(np.isnan(statDictBot))] = 0
keysBot = ["Bias", "RMSE", "NSE"]
stateBotPercentile = {keysBot[i]: np.percentile(statDictBot[i], [0, 25, 50, 75, 100]) for i in
                      range(statDictBot.shape[0])}
print(stateBotPercentile)

# plot box
statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
# 输出几组统计值的最值、中位数和四分位数
statePercentile = {key: np.percentile(value, [0, 25, 50, 75, 100]) for key, value in statDictLst[0].items()}
print(statePercentile)

# plot box
statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
keyLst = list(statDictLst[0].keys())
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM'], sharey=False)
fig.show()

# plot time series
t = utils.time.tRange2Array(tRangeTest)
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
for k in range(5):
    iGrid = np.random.randint(0, 671)
    yPlot = [obs[iGrid, :]]
    for y in predLst:
        yPlot.append(y[iGrid, :])
    if k == 0:
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='kbrg',
            markerLst='----',
            legLst=['USGS', 'LSTM'])
    else:
        plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrg', markerLst='----')
fig.show()
