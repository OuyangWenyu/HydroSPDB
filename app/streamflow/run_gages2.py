"""target：利用GAGES-II数据训练LSTM，并进行流域径流模拟。
   procedure： 标准机器学习pipeline，数据前处理——统计分析——模型训练及测试——可视化结果——参数调优"""
from app.common.default import init_path, init_data_param, init_model_param
from hydroDL import master

# 读取配置文件# 首先，配置GAGES-II原始数据文件的路径。在hydroDL的init文件里直接配置输入输出路径
import os
from collections import OrderedDict
from explore import gages2

print('loading package hydroDL')

config_file = "config.ini"
pathDataSource = init_path(config_file)

# 读取配置文件
optData = init_data_param(config_file)
optModel, optLoss, optTrain = init_model_param(config_file)

cid = 0
case = optData['subset'] + '-' + str(optData['tRange'][0])[2:4] + '-' + str(optData['tRange'][1])[2:4]
out = os.path.join(pathDataSource['Out'], case)
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

caseLst = [case]
outLst = [os.path.join(pathDataSource['Out'], x) for x in caseLst]
subset = 'All'
tRangeTest = [20010101, 20050101]
predLst = list()

# # train model
# # see whether there are previous results or not, if yes, there is no need to train again.
# resultPathLst = namePred(out, tRangeTest, subset, epoch=optTrain['nEpoch'])
# if not os.path.exists(resultPathLst[0]):
#     master.run_train(masterDict, cuda_id=cid % 3, screen='test')
#     cid = cid + 1
#
# # test
# for out in outLst:
#     df, pred, obs = master.test(out, t_range=tRangeTest, subset=subset, epoch=optTrain['nEpoch'])
#     predLst.append(pred)
#
# # 补充一些计算结果输出，给出定量的指标，包括nse，rmse。
# # 30% bottom 数据的分析，每个站点的<30分位数的径流数据和同时间的预测径流数据比较，计算box的几个指标
# obs2d = obs.squeeze()
# pred2d = pred.squeeze()
# obs2d[np.where(np.isnan(obs2d))] = 0
# pred2d[np.where(np.isnan(pred2d))] = 0
#
# # 每个站点的分位数对应的数据量是不同的，因此不能直接调用二维的计算公式，得每个站点分别计算
# obsBotIndex = [np.where(obs2di < np.percentile(obs2di, 30)) for obs2di in obs2d]
# obsBot = np.array([obs2d[i][obsBotIndex[i]] for i in range(obs2d.shape[0])])
# predBot = np.array([pred2d[j][obsBotIndex[j]] for j in range(pred2d.shape[0])])
#
# statDictBot = np.array([stat.statError1d(predBot[i], obsBot[i]) for i in range(obsBot.shape[0])]).T
# statDictBot[np.where(np.isnan(statDictBot))] = 0
# keysBot = ["Bias", "RMSE", "NSE"]
# stateBotPercentile = {keysBot[i]: np.percentile(statDictBot[i], [0, 25, 50, 75, 100]) for i in
#                       range(statDictBot.shape[0])}
# print(stateBotPercentile)
#
# # top 10%的数据分析，同理
# obs2d = obs.squeeze()
# pred2d = pred.squeeze()
# obs2d[np.where(np.isnan(obs2d))] = 0
# pred2d[np.where(np.isnan(pred2d))] = 0
#
# # 每个站点的分位数对应的数据量是不同的，因此不能直接调用二维的计算公式，得每个站点分别计算
# obsBotIndex = [np.where(obs2di > np.percentile(obs2di, 90)) for obs2di in obs2d]
# obsBot = np.array([obs2d[i][obsBotIndex[i]] for i in range(obs2d.shape[0])])
# predBot = np.array([pred2d[j][obsBotIndex[j]] for j in range(pred2d.shape[0])])
#
# statDictBot = np.array([stat.statError1d(predBot[i], obsBot[i]) for i in range(obsBot.shape[0])]).T
# statDictBot[np.where(np.isnan(statDictBot))] = 0
# keysBot = ["Bias", "RMSE", "NSE"]
# stateBotPercentile = {keysBot[i]: np.percentile(statDictBot[i], [0, 25, 50, 75, 100]) for i in
#                       range(statDictBot.shape[0])}
# print(stateBotPercentile)
#
# # plot box
# statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
# # 输出几组统计值的最值、中位数和四分位数
# statePercentile = {key: np.percentile(value, [0, 25, 50, 75, 100]) for key, value in statDictLst[0].items()}
# print(statePercentile)
#
# # plot box
# statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
# keyLst = list(statDictLst[0].keys())
# dataBox = list()
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(statDictLst)):
#         data = statDictLst[k][statStr]
#         data = data[~np.isnan(data)]
#         temp.append(data)
#     dataBox.append(temp)
# fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM'], sharey=False)
# fig.show()
#
# # plot time series
# t = utils.time.tRange2Array(tRangeTest)
# fig, axes = plt.subplots(5, 1, figsize=(12, 8))
# for k in range(5):
#     iGrid = np.random.randint(0, 671)
#     yPlot = [obs[iGrid, :]]
#     for y in predLst:
#         yPlot.append(y[iGrid, :])
#     if k == 0:
#         plot.plotTS(
#             t,
#             yPlot,
#             ax=axes[k],
#             cLst='kbrg',
#             markerLst='----',
#             legLst=['USGS', 'LSTM'])
#     else:
#         plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrg', markerLst='----')
# fig.show()
